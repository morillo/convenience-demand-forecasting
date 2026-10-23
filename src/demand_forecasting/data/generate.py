"""Pipeline step ``generate``: build the mock chain and write the bronze tables.

Dimension tables (stores, products, calendar...) are small and built on the driver. The big
fact tables are simulated *in parallel, one store per Spark task*: each task rebuilds the
same deterministic ``World`` and simulates its store, so the work scales out with the cluster
and no large table is ever shipped from the driver.
"""

from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from demand_forecasting import storage as T
from demand_forecasting.config import Settings
from demand_forecasting.data.simulate import SIMULATION_SCHEMA, simulate_store
from demand_forecasting.data.world import get_world, public

log = logging.getLogger(__name__)


def run(spark: SparkSession, settings: Settings) -> None:
    store = T.TableStore(spark, settings)
    preset, seed, horizon = settings.preset.name, settings.seed, settings.forecast.horizon_days
    world = get_world(preset, seed, horizon)

    dims: list[tuple[T.Table, pd.DataFrame]] = [
        (T.STORES, public(world.stores)),
        (T.PRODUCTS, public(world.products)),
        (T.ASSORTMENT, public(world.assortment)),
        (T.CALENDAR, world.calendar),
        (T.WEATHER, world.weather),
        (T.PROMOTIONS, world.promotions),
        (T.PRICES, world.prices),
    ]
    for table, pdf in dims:
        store.write(spark.createDataFrame(pdf), table)
        log.info("wrote %s (%d rows)", table.key, len(pdf))

    def simulate(group: pd.DataFrame) -> pd.DataFrame:
        # Runs on a worker: rebuild (cached) world, simulate one store.
        return simulate_store(get_world(preset, seed, horizon), str(group["store_id"].iloc[0]))

    store_ids = spark.createDataFrame(world.stores[["store_id"]])
    truth = store_ids.groupBy("store_id").applyInPandas(simulate, SIMULATION_SCHEMA)
    store.write(truth, T.SIMULATION_TRUTH, cluster_by=["store_id", "date"])
    truth = store.read(T.SIMULATION_TRUTH)

    keys = ["date", "store_id", "sku_id"]
    # Operational tables: what a real chain's systems would record. Note what is *missing*:
    # true demand and the stockout flag. The pipeline has to infer stockouts from stock levels.
    pos = truth.where(F.col("units_sold") > 0).select(
        *keys,
        "units_sold",
        F.round(F.col("units_sold") * F.col("unit_price"), 2).alias("net_sales_amount"),
    )
    store.write(pos, T.POS_SALES, cluster_by=["store_id", "date"])
    store.write(
        truth.select(*keys, "on_hand_close", "units_received", "units_wasted"),
        T.INVENTORY,
        cluster_by=["store_id", "date"],
    )
    orders = truth.where(F.col("order_qty") > 0).select(
        "store_id",
        "sku_id",
        F.col("date").alias("order_date"),
        F.date_add(F.col("date"), F.col("lead_time_days")).alias("expected_arrival_date"),
        "order_qty",
    )
    store.write(orders, T.PURCHASE_ORDERS)
    log.info("generated preset %r: %d stores, %d SKUs, %d days", preset, settings.preset.n_stores,
             settings.preset.n_skus, settings.preset.n_days)  # fmt: skip
