"""Pipeline step ``forecast``: next-horizon forecasts and the recommended orders.

Scores the future rows of the features table with the production models, then turns the
forecasts into an order recommendation per store x SKU (see ``replenish.py``):

    recommended_order_qty = max(0, ceil(order_up_to_level - on_hand - on_order))

The safety stock is scaled by the per-aisle factor the backtest learned
(``safety_stock_calibration``), when available. ``on_hand`` is the closing stock on the last day (zero for fresh items, which are written off
overnight); ``on_order`` comes from open purchase orders that have not arrived yet.
"""

from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from demand_forecasting import storage as T
from demand_forecasting.config import Settings
from demand_forecasting.pipeline import replenish, training

log = logging.getLogger(__name__)

SERIES = ["store_id", "sku_id"]


def run(spark: SparkSession, settings: Settings) -> None:
    store = T.TableStore(spark, settings)
    features, models = store.read(T.FEATURES), store.read(T.MODELS)
    origin = training.last_history_date(features)
    run_id = models.select("run_id").first()
    assert run_id is not None, "models table is empty; run the train step first"

    preds = training.predict(
        training.scoring_rows(features, [origin], settings, future=True), models, settings
    )
    store.write(
        preds.withColumn("model_run_id", F.lit(run_id[0])),
        T.FORECASTS,
        cluster_by=["store_id", "date"],
    )
    forecasts = store.read(T.FORECASTS)

    products = store.read(T.PRODUCTS).select("sku_id", "sku_name", "aisle", "category", "is_fresh")
    params = store.read(T.ASSORTMENT).join(products, "sku_id")
    fc = settings.forecast
    target = replenish.order_up_to(
        forecasts,
        params,
        service_level=fc.service_level,
        upper_quantile=settings.model.upper_quantile,
        horizon_days=fc.horizon_days,
    )
    target = target.join(params.select(*SERIES, "aisle"), SERIES)
    if store.exists(T.SAFETY_STOCK_CALIBRATION):
        target = replenish.apply_safety_factor(target, store.read(T.SAFETY_STOCK_CALIBRATION))
    else:
        log.warning(
            "no safety-stock calibration found (run the backtest step); using the textbook formula"
        )
        target = target.withColumn("safety_factor", F.lit(1.0))
    origins = spark.createDataFrame([(origin,)], "origin_date date")
    on_hand = (
        store.read(T.DAILY_PANEL)
        .where(F.col("date") == F.lit(origin))
        .select(
            *SERIES,
            F.when(F.col("is_fresh"), F.lit(0)).otherwise(F.col("on_hand_close")).alias("on_hand"),
        )
    )
    on_order = replenish.on_order_at(store.read(T.PURCHASE_ORDERS), origins).drop("origin_date")
    plan = (
        target.join(params.select(*SERIES, "sku_name", "category", "is_fresh"), SERIES)
        .join(on_hand, SERIES, "left")
        .join(on_order, SERIES, "left")
        .fillna({"on_hand": 0, "on_order": 0})
        .withColumn(
            "recommended_order_qty",
            F.greatest(
                F.ceil(F.col("order_up_to_level") - F.col("on_hand") - F.col("on_order")), F.lit(0)
            ).cast("int"),
        )
        .withColumn("service_level_target", F.lit(fc.service_level))
        .withColumn("model_run_id", F.lit(run_id[0]))
    )
    store.write(plan, T.REPLENISHMENT_PLAN, cluster_by=["store_id"])
    totals = (
        store.read(T.REPLENISHMENT_PLAN)
        .agg(F.count("*").alias("series"), F.sum("recommended_order_qty").alias("units"))
        .first()
    )
    if totals is not None:
        log.info(
            "forecast from %s: %d series, %s units recommended",
            origin,
            totals["series"],
            totals["units"],
        )
