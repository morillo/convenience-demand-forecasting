"""Forecast accuracy metrics, computed in Spark at any level of aggregation.

All metrics are *volume weighted* (sums over rows, then a ratio), so a fast-selling SKU
counts more than a slow one, which matches the business cost of errors.

* **WAPE** = sum |forecast - actual| / sum actual. "We were off by X% of volume."
* **Bias** = sum (forecast - actual) / sum actual. > 0 over-forecasts (overstock risk),
  < 0 under-forecasts (stockout risk).
* **MAE** = mean |forecast - actual| in units per store x SKU x day.
* **Pinball loss** of the upper quantile q: the standard score for quantile forecasts.
* **Coverage** = share of days where actual <= upper forecast. Should be close to q.
* **WAPE vs true demand**: simulation-only. Compares against demand before stockouts.

Accuracy is measured on days *without* a stockout, because on stockout days the "actual"
sales are not the real demand.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

LEVELS: dict[str, list[str]] = {
    "overall": [],
    "origin_date": ["origin_date"],
    "country": ["country_code"],
    "aisle": ["aisle"],
    "horizon": ["horizon"],
}


def _safe_div(a: Column, b: Column) -> Column:
    return F.when(b != 0, a / b)


def accuracy(preds: DataFrame, upper_quantile: float) -> DataFrame:
    """``preds`` needs: model, forecast_mean, forecast_upper, units_sold, is_stockout,
    true_demand, and the LEVELS columns. Returns one row per level x value x model."""
    q = upper_quantile
    f, a, up = F.col("forecast_mean"), F.col("units_sold"), F.col("forecast_upper")
    clean = ~F.col("is_stockout")
    err = F.when(clean, F.abs(f - a))
    pinball = F.when(clean & up.isNotNull(), F.greatest(q * (a - up), (q - 1) * (a - up)))
    out = []
    for level, cols in LEVELS.items():
        value = F.concat_ws("|", *[F.col(c).cast("string") for c in cols]) if cols else F.lit("all")
        agg = (
            preds.withColumn("level_value", value)
            .groupBy("level_value", "model")
            .agg(
                F.sum(clean.cast("long")).alias("n_rows"),
                F.sum(F.when(clean, a)).alias("actual_units"),
                F.sum(err).alias("abs_error"),
                F.sum(F.when(clean, f - a)).alias("error"),
                F.avg(pinball).alias("pinball_upper"),
                F.avg(F.when(clean & up.isNotNull(), (a <= up).cast("double"))).alias(
                    "coverage_upper"
                ),
                F.sum(F.abs(f - F.col("true_demand"))).alias("abs_error_true"),
                F.sum("true_demand").alias("true_demand_units"),
            )
            .select(
                F.lit(level).alias("level"),
                "level_value",
                "model",
                "n_rows",
                _safe_div(F.col("abs_error"), F.col("actual_units")).alias("wape"),
                _safe_div(F.col("error"), F.col("actual_units")).alias("bias"),
                _safe_div(F.col("abs_error"), F.col("n_rows")).alias("mae"),
                "pinball_upper",
                "coverage_upper",
                _safe_div(F.col("abs_error_true"), F.col("true_demand_units")).alias(
                    "wape_vs_true_demand"
                ),
            )
        )
        out.append(agg)
    result = out[0]
    for df in out[1:]:
        result = result.unionByName(df)
    return result
