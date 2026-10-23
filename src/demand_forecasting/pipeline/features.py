"""Pipeline step ``features``: silver panel -> gold ``features`` (history + future horizon).

**The leakage rule.** We forecast up to H = ``horizon_days`` ahead from a cut-off day T.
For a target day t <= T + H, the newest sales we may use are from day T >= t - H. So every
sales-based feature only looks at days ``t - H`` and earlier. One model then serves every
day of the horizon, and the same features can be computed for history and for the future
(docs/decisions/0004-direct-features-min-lag.md).

Features that *are* known ahead of time (calendar, planned promotions, list price, the
weather forecast) are used for the target day itself.

**Stockouts.** Sales on stockout days understate demand, so they are blanked (null) before
computing lags and rolling statistics; Spark's aggregates skip nulls.
"""

from __future__ import annotations

from itertools import chain

from pyspark.sql import Column, DataFrame, SparkSession, Window, WindowSpec
from pyspark.sql import functions as F

from demand_forecasting import storage as T
from demand_forecasting.config import Settings
from demand_forecasting.domain import CATEGORY_CODE, STORE_FORMAT_CODE

SERIES = ["store_id", "sku_id"]
EPOCH = "2000-01-01"

CATEGORICAL_FEATURES = ["store_format_code", "category_code"]


def lag_days(h: int) -> list[int]:
    return [h, h + 7, h + 14, h + 21, h + 28, 364]


def feature_columns(h: int) -> list[str]:
    lags = [f"lag_{k}" for k in lag_days(h)]
    rolling = [
        "roll_mean_7", "roll_mean_28", "roll_mean_91", "roll_std_28", "same_weekday_mean_4w",
        "zero_share_28", "stockout_share_28", "promo_mean_91", "nonpromo_mean_91",
    ]  # fmt: skip
    known = [
        "day_of_week", "day_of_month", "month", "week_of_year", "is_holiday", "is_holiday_eve",
        "is_after_holiday", "is_payday_window", "on_promo", "discount_pct", "list_price",
        "price_vs_recent", "temp_forecast_c", "rain_forecast_mm", "days_since_launch", "is_fresh",
        "lead_time_days",
    ]  # fmt: skip
    return lags + rolling + known + CATEGORICAL_FEATURES


def _code_map(mapping: dict[str, int]) -> Column:
    return F.create_map(*chain.from_iterable((F.lit(k), F.lit(v)) for k, v in mapping.items()))


def future_spine(panel: DataFrame, calendar: DataFrame, horizon: int) -> DataFrame:
    """Rows for the next ``horizon`` days for every series still on the shelf at the cut-off."""
    last_day = panel.agg(F.max("date").alias("last_day"))
    latest = (
        panel.join(last_day, F.col("date") == F.col("last_day"))
        .drop("date", "last_day")
        .select(
            *SERIES, "country_code", "store_format", "store_open_date", "aisle", "category", "is_fresh",
            "launch_date", "delist_date", "lead_time_days", "review_period_days", "list_price",
        )
    )  # fmt: skip
    future_days = (
        calendar.select("date")
        .distinct()
        .crossJoin(last_day)
        .where(
            (F.col("date") > F.col("last_day")) & (F.col("date") <= F.date_add("last_day", horizon))
        )
        .select("date")
    )
    return latest.crossJoin(future_days).where(
        F.col("delist_date").isNull() | (F.col("date") < F.col("delist_date"))
    )


def build_features(
    panel: DataFrame,
    calendar: DataFrame,
    weather: DataFrame,
    promotions: DataFrame,
    settings: Settings,
) -> DataFrame:
    h = settings.forecast.horizon_days
    history = panel.withColumn("is_future", F.lit(False))
    future = (
        future_spine(panel, calendar, h)
        .join(promotions.select("country_code", "sku_id", "date", "discount_pct"), ["country_code", "sku_id", "date"], "left")
        .withColumn("discount_pct", F.coalesce("discount_pct", F.lit(0.0)))
        .withColumn("on_promo", F.col("discount_pct") > 0)
        .withColumn("is_future", F.lit(True))
    )  # fmt: skip
    df = history.unionByName(future, allowMissingColumns=True)

    # Calendar context: holiday eve / day after, and a 3-day payday window.
    cw = Window.partitionBy("country_code").orderBy("date")
    cal = calendar.select(
        "country_code",
        "date",
        F.col("is_holiday").cast("int").alias("is_holiday"),
        F.coalesce(F.lead("is_holiday").over(cw), F.lit(False)).cast("int").alias("is_holiday_eve"),
        F.coalesce(F.lag("is_holiday").over(cw), F.lit(False))
        .cast("int")
        .alias("is_after_holiday"),
        F.max(F.col("is_payday").cast("int")).over(cw.rowsBetween(-2, 0)).alias("is_payday_window"),
    )
    df = df.join(cal, ["country_code", "date"], "left").join(
        weather.select("country_code", "date", "temp_forecast_c", "rain_forecast_mm"),
        ["country_code", "date"],
        "left",
    )

    # Sales history with stockout days blanked (censored), on an integer day axis so range
    # windows mean "calendar days back", robust to any gaps.
    df = df.withColumn("day", F.datediff("date", F.lit(EPOCH).cast("date"))).withColumn(
        "units_clean", F.when(~F.col("is_stockout"), F.col("units_sold").cast("double"))
    )
    w = Window.partitionBy(*SERIES).orderBy("day")

    def back(a: int, b: int) -> WindowSpec:
        """Frame covering days t-a .. t-b (a >= b >= h)."""
        return w.rangeBetween(-a, -b)

    u = F.col("units_clean")
    promo_u = F.when(F.col("on_promo"), u)
    nonpromo_u = F.when(~F.col("on_promo"), u)
    same_weekday = (
        Window.partitionBy(*SERIES, F.dayofweek("date")).orderBy("day").rangeBetween(-(h + 21), -h)
    )
    first_live = F.greatest("launch_date", "store_open_date")

    feats = [
        *[F.max(u).over(back(k, k)).alias(f"lag_{k}") for k in lag_days(h)],
        F.avg(u).over(back(h + 6, h)).alias("roll_mean_7"),
        F.avg(u).over(back(h + 27, h)).alias("roll_mean_28"),
        F.avg(u).over(back(h + 90, h)).alias("roll_mean_91"),
        F.stddev(u).over(back(h + 27, h)).alias("roll_std_28"),
        F.avg(u).over(same_weekday).alias("same_weekday_mean_4w"),
        F.avg((u == 0).cast("double")).over(back(h + 27, h)).alias("zero_share_28"),
        F.avg(F.col("is_stockout").cast("double")).over(back(h + 27, h)).alias("stockout_share_28"),
        F.avg(promo_u).over(back(h + 90, h)).alias("promo_mean_91"),
        F.avg(nonpromo_u).over(back(h + 90, h)).alias("nonpromo_mean_91"),
        F.try_divide(F.col("list_price"), F.avg("list_price").over(back(h + 90, h))).alias(
            "price_vs_recent"
        ),
        F.dayofweek("date").alias("day_of_week"),
        F.dayofmonth("date").alias("day_of_month"),
        F.month("date").alias("month"),
        F.weekofyear("date").alias("week_of_year"),
        F.least(F.greatest(F.datediff("date", first_live), F.lit(0)), F.lit(730)).alias(
            "days_since_launch"
        ),
        F.col("on_promo").cast("int").alias("on_promo_int"),
        F.col("is_fresh").cast("int").alias("is_fresh_int"),
        _code_map(STORE_FORMAT_CODE)[F.col("store_format")].alias("store_format_code"),
        _code_map(CATEGORY_CODE)[F.col("category")].alias("category_code"),
    ]
    out = df.select("*", *feats).drop("on_promo", "is_fresh")
    out = out.withColumnRenamed("on_promo_int", "on_promo").withColumnRenamed(
        "is_fresh_int", "is_fresh"
    )

    group_cols = [F.col(c).cast("string") for c in settings.forecast.group_by]
    if settings.forecast.store_buckets > 1:
        group_cols.append(
            F.pmod(F.hash("store_id"), F.lit(settings.forecast.store_buckets)).cast("string")
        )
    return out.withColumn("model_group", F.concat_ws("|", *group_cols)).drop("day", "units_clean")


def run(spark: SparkSession, settings: Settings) -> None:
    store = T.TableStore(spark, settings)
    features = build_features(
        store.read(T.DAILY_PANEL),
        store.read(T.CALENDAR),
        store.read(T.WEATHER),
        store.read(T.PROMOTIONS),
        settings,
    )
    store.write(features, T.FEATURES, cluster_by=["model_group", "date"])
