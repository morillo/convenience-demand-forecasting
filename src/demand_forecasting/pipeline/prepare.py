"""Pipeline step ``prepare``: bronze -> silver ``daily_panel``.

The inventory snapshots define *which* store x SKU x days exist (a SKU is on the shelf), so
they are the spine. Point-of-sale data only has days with sales, so missing days become
``units_sold = 0``. Prices and promotions are joined on, and stockouts are inferred:

    is_stockout = (on_hand_close == 0)

An empty shelf at closing time means some customers may have walked away, so that day's
sales understate demand (they are *censored*). Downstream steps treat those days specially
(docs/decisions/0005-stockout-censoring.md).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from demand_forecasting import storage as T
from demand_forecasting.config import Settings
from demand_forecasting.pipeline import quality as Q

KEYS = ["date", "store_id", "sku_id"]
FAR_FUTURE = "2999-12-31"


def daily_list_price(prices: DataFrame) -> DataFrame:
    """Turn price-change events into validity intervals [effective_from, effective_to]."""
    w = Window.partitionBy("sku_id").orderBy("effective_from")
    return prices.withColumn(
        "effective_to",
        F.coalesce(F.date_sub(F.lead("effective_from").over(w), 1), F.lit(FAR_FUTURE).cast("date")),
    )


def build_panel(
    *,
    inventory: DataFrame,
    pos: DataFrame,
    stores: DataFrame,
    products: DataFrame,
    assortment: DataFrame,
    prices: DataFrame,
    promotions: DataFrame,
) -> DataFrame:
    price_intervals = daily_list_price(prices)
    return (
        inventory.join(pos.select(*KEYS, "units_sold"), on=KEYS, how="left")
        .withColumn("units_sold", F.coalesce("units_sold", F.lit(0)))
        .join(stores.select("store_id", "country_code", "store_format", F.col("open_date").alias("store_open_date")), "store_id")
        .join(products.select("sku_id", "aisle", "category", "is_fresh", "launch_date", "delist_date"), "sku_id")
        .join(assortment.select("store_id", "sku_id", "lead_time_days", "review_period_days"), ["store_id", "sku_id"])
        .join(
            price_intervals.withColumnRenamed("sku_id", "price_sku_id"),
            (F.col("sku_id") == F.col("price_sku_id")) & F.col("date").between(F.col("effective_from"), F.col("effective_to")),
            "left",
        )
        .drop("price_sku_id", "effective_from", "effective_to")
        .join(promotions.select("country_code", "sku_id", "date", "discount_pct"), ["country_code", "sku_id", "date"], "left")
        .withColumn("discount_pct", F.coalesce("discount_pct", F.lit(0.0)))
        .withColumn("on_promo", F.col("discount_pct") > 0)
        .withColumn("unit_price", F.round(F.col("list_price") * (1 - F.col("discount_pct")), 2))
        .withColumn("is_stockout", F.col("on_hand_close") == 0)
    )  # fmt: skip


def run(spark: SparkSession, settings: Settings) -> None:
    store = T.TableStore(spark, settings)
    inventory, pos = store.read(T.INVENTORY), store.read(T.POS_SALES)
    Q.enforce(
        [
            Q.check_unique(inventory, KEYS, "inventory_snapshots"),
            Q.check_unique(pos, KEYS, "pos_sales"),
            Q.check_non_negative(
                inventory,
                ["on_hand_close", "units_received", "units_wasted"],
                "inventory_snapshots",
            ),
            Q.check_non_negative(pos, ["units_sold"], "pos_sales"),
            Q.check_orphans(pos, inventory, KEYS, "pos_sales vs inventory_snapshots"),
        ]
    )
    panel = build_panel(
        inventory=inventory,
        pos=pos,
        stores=store.read(T.STORES),
        products=store.read(T.PRODUCTS),
        assortment=store.read(T.ASSORTMENT),
        prices=store.read(T.PRICES),
        promotions=store.read(T.PROMOTIONS),
    )
    store.write(panel, T.DAILY_PANEL, cluster_by=["store_id", "date"])
    written = store.read(T.DAILY_PANEL)
    Q.enforce(
        [
            Q.check_unique(written, KEYS, "daily_panel"),
            Q.check_contiguous(written, "daily_panel"),
            Q.check_non_negative(written, ["units_sold", "unit_price"], "daily_panel"),
        ]
    )
