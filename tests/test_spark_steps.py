"""Spark-level tests on small hand-made DataFrames: leakage, quality gates, metrics, policy."""

from __future__ import annotations

import datetime as dt
import math
from statistics import NormalDist

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from demand_forecasting.config import load_settings
from demand_forecasting.pipeline import metrics, quality, replenish
from demand_forecasting.pipeline.features import build_features

D0 = dt.date(2024, 1, 1)


def panel(spark: SparkSession, n_days: int = 120, stockout_days: tuple[int, ...] = ()) -> DataFrame:
    rows = [
        (
            D0 + dt.timedelta(days=d), "S1", "P1", "US", "urban", D0 - dt.timedelta(days=400), "snacks", "chips",
            False, D0 - dt.timedelta(days=400), None, 2, 3, 1.5, 0.0, False, 1.5,
            d % 5,  # units_sold: a known, deterministic pattern
            0 if d in stockout_days else 5, d in stockout_days,
        )
        for d in range(n_days)
    ]  # fmt: skip
    schema = (
        "date date, store_id string, sku_id string, country_code string, store_format string, "
        "store_open_date date, aisle string, category string, is_fresh boolean, launch_date date, "
        "delist_date date, lead_time_days int, review_period_days int, list_price double, "
        "discount_pct double, on_promo boolean, unit_price double, units_sold int, "
        "on_hand_close int, is_stockout boolean"
    )
    return spark.createDataFrame(rows, schema)


def calendar_and_weather(spark: SparkSession, n_days: int) -> tuple[DataFrame, DataFrame]:
    days = [D0 + dt.timedelta(days=d) for d in range(n_days)]
    cal = spark.createDataFrame(
        [("US", d, False, d.day == 1) for d in days],
        "country_code string, date date, is_holiday boolean, is_payday boolean",
    )
    wx = spark.createDataFrame(
        [("US", d, 20.0, 0.0) for d in days],
        "country_code string, date date, temp_forecast_c double, rain_forecast_mm double",
    )
    return cal, wx  # fmt: skip


@pytest.fixture
def features(spark: SparkSession) -> DataFrame:
    settings = load_settings("local", preset="tiny")
    cal, wx = calendar_and_weather(spark, 134)
    promos = spark.createDataFrame(
        [], "country_code string, sku_id string, date date, discount_pct double"
    )
    return build_features(panel(spark, stockout_days=(100,)), cal, wx, promos, settings)


def test_lags_only_look_back_at_least_one_horizon(features: DataFrame) -> None:
    rows = {r["date"]: r for r in features.collect()}
    d = D0 + dt.timedelta(days=60)
    assert rows[d]["lag_14"] == float((60 - 14) % 5)
    assert rows[d]["lag_28"] == float((60 - 28) % 5)
    expected_mean7 = sum((k % 5) for k in range(60 - 20, 60 - 13)) / 7
    assert rows[d]["roll_mean_7"] == pytest.approx(expected_mean7)


def test_stockout_days_are_blanked_in_lags(features: DataFrame) -> None:
    rows = {r["date"]: r for r in features.collect()}
    assert rows[D0 + dt.timedelta(days=114)]["lag_14"] is None  # day 100 was a stockout


def test_future_rows_cover_exactly_the_horizon(features: DataFrame) -> None:
    future = features.where("is_future").orderBy("date").collect()
    assert len(future) == 14
    assert future[0]["date"] == D0 + dt.timedelta(days=120)
    # The furthest future day still has its lag_14 from known history (the last day).
    assert future[-1]["lag_14"] == float(119 % 5)
    assert all(r["units_sold"] is None for r in future)


def test_model_group_key(features: DataFrame) -> None:
    assert {r["model_group"] for r in features.select("model_group").distinct().collect()} == {
        "US|snacks"
    }


def test_quality_gates_detect_gaps_and_duplicates(spark: SparkSession) -> None:
    p = panel(spark, n_days=10)
    gappy = p.where(F.col("date") != F.lit(D0 + dt.timedelta(days=4)))
    assert quality.check_contiguous(p, "ok").passed
    assert not quality.check_contiguous(gappy, "gap").passed
    assert not quality.check_unique(
        p.unionByName(p.limit(1)), ["date", "store_id", "sku_id"], "dup"
    ).passed
    with pytest.raises(quality.DataQualityError):
        quality.enforce([quality.check_contiguous(gappy, "gap")])


def test_accuracy_metrics_by_hand(spark: SparkSession) -> None:
    rows = [  # model, forecast, upper, actual, stockout, true_demand
        ("m", 2.0, 3.0, 1, False, 1),
        ("m", 2.0, 3.0, 4, False, 4),
        ("m", 9.0, 9.0, 0, True, 6),  # stockout: excluded from WAPE/bias, kept for true-demand WAPE
    ]
    df = spark.createDataFrame(
        [(dt.date(2024, 1, 1), "US", "snacks", 1, *r) for r in rows],
        "origin_date date, country_code string, aisle string, horizon int, model string, forecast_mean double, "
        "forecast_upper double, units_sold int, is_stockout boolean, true_demand int",
    )
    overall = metrics.accuracy(df, 0.9).where("level = 'overall'").first()
    assert overall is not None
    assert overall["n_rows"] == 2
    assert overall["wape"] == pytest.approx((1 + 2) / 5)
    assert overall["bias"] == pytest.approx((1 - 2) / 5)
    assert overall["coverage_upper"] == pytest.approx(0.5)
    assert overall["wape_vs_true_demand"] == pytest.approx((1 + 2 + 3) / 11)


def test_order_up_to_formula(spark: SparkSession) -> None:
    fc = spark.createDataFrame(
        [(D0, "S1", "P1", h, 2.0, 3.0) for h in range(1, 15)],
        "origin_date date, store_id string, sku_id string, horizon int, forecast_mean double, forecast_upper double",
    )
    params = spark.createDataFrame(
        [("S1", "P1", 2, 3, False)],
        "store_id string, sku_id string, lead_time_days int, review_period_days int, is_fresh boolean",
    )
    row = replenish.order_up_to(
        fc, params, service_level=0.95, upper_quantile=0.9, horizon_days=14
    ).first()
    assert row is not None
    sigma_day = 1.0 / NormalDist().inv_cdf(0.9)
    assert row["protection_days"] == 5
    assert row["expected_demand"] == pytest.approx(10.0)
    assert row["safety_stock"] == pytest.approx(
        NormalDist().inv_cdf(0.95) * sigma_day * math.sqrt(5)
    )


def test_fresh_items_protect_only_the_review_period(spark: SparkSession) -> None:
    fc = spark.createDataFrame(
        [(D0, "S1", "P1", h, 2.0, 2.0) for h in range(1, 15)],
        "origin_date date, store_id string, sku_id string, horizon int, forecast_mean double, forecast_upper double",
    )
    params = spark.createDataFrame(
        [("S1", "P1", 1, 1, True)],
        "store_id string, sku_id string, lead_time_days int, review_period_days int, is_fresh boolean",
    )
    row = replenish.order_up_to(
        fc, params, service_level=0.95, upper_quantile=0.9, horizon_days=14
    ).first()
    assert (
        row is not None
        and row["protection_days"] == 1
        and row["expected_demand"] == pytest.approx(2.0)
    )


def test_safety_factor_calibration_and_default(spark: SparkSession) -> None:
    cycles = spark.createDataFrame(
        [("snacks", float(td), 10.0, 2.0) for td in range(8, 17)],
        "aisle string, true_demand double, expected_demand double, safety_stock double",
    )
    factors = replenish.calibrate_safety_factor(cycles, 0.5)
    k = factors.first()
    assert k is not None and k["safety_factor"] == pytest.approx(1.0, abs=0.51)
    target = spark.createDataFrame(
        [("snacks", 10.0, 2.0), ("frozen", 10.0, 2.0)],
        "aisle string, expected_demand double, safety_stock double",
    )
    applied = {r["aisle"]: r for r in replenish.apply_safety_factor(target, factors).collect()}
    assert applied["frozen"]["safety_factor"] == 1.0  # no calibration -> textbook formula
    assert applied["snacks"]["order_up_to_level"] == pytest.approx(
        10.0 + 2.0 * applied["snacks"]["safety_factor"]
    )
