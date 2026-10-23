"""Read and write Delta tables by logical name, wherever they physically live.

* ``path`` mode (local): ``<data_dir>/lake/<layer>/<table>`` Delta folders.
* ``catalog`` mode (Databricks): Unity Catalog table ``<catalog>.<schema>.<layer>_<table>``.

Pipeline code only ever says ``store.read(SALES)``; switching environments is a settings
change. Every write is a full, idempotent overwrite (docs/decisions/0006-storage.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from demand_forecasting.config import Settings


@dataclass(frozen=True)
class Table:
    layer: (
        str  # bronze = raw as received, silver = cleaned & conformed, gold = model-ready / outputs
    )
    name: str
    description: str

    @property
    def key(self) -> str:
        return f"{self.layer}_{self.name}"


# Bronze: what source systems would deliver (here: the mock data generator).
STORES = Table("bronze", "stores", "Store master: country, format, opening date, traffic level.")
PRODUCTS = Table(
    "bronze", "products", "SKU master: aisle, category, list price, launch/delist dates."
)
ASSORTMENT = Table(
    "bronze", "assortment", "Which store carries which SKU, with its replenishment rules."
)
CALENDAR = Table("bronze", "calendar", "One row per country and day: holidays and paydays.")
WEATHER = Table(
    "bronze", "weather", "Daily observed weather and the forecast available the day before."
)
PROMOTIONS = Table("bronze", "promotions", "Planned price promotions per country, SKU and day.")
PRICES = Table("bronze", "prices", "List price changes per SKU (effective-from dates).")
POS_SALES = Table(
    "bronze", "pos_sales", "Point-of-sale totals per store, SKU and day; only days with sales."
)
INVENTORY = Table(
    "bronze",
    "inventory_snapshots",
    "End-of-day stock, receipts and write-offs per store, SKU and day.",
)
PURCHASE_ORDERS = Table(
    "bronze", "purchase_orders", "Replenishment orders: order date, arrival date, quantity."
)
SIMULATION_TRUTH = Table(
    "bronze",
    "simulation_truth",
    "Simulation-only ground truth: true demand before stockouts cut it off, plus the incumbent's orders.",
)

# Silver: one clean, gap-free row per active store x SKU x day.
DAILY_PANEL = Table(
    "silver",
    "daily_panel",
    "Daily store x SKU panel with sales, stock, price, promo and stockout flag.",
)

# Gold: model inputs and business outputs.
FEATURES = Table(
    "gold", "features", "Model features per store x SKU x day, history plus the forecast horizon."
)
MODELS = Table("gold", "models", "Trained LightGBM models, one row per model group and model kind.")
BACKTEST_MODELS = Table(
    "gold", "backtest_models", "Models trained at each backtest origin (kept for audit)."
)
BACKTEST_PREDICTIONS = Table(
    "gold", "backtest_predictions", "Model and baseline predictions for each backtest origin."
)
BACKTEST_METRICS = Table(
    "gold", "backtest_metrics", "Accuracy metrics by level (overall, origin, country, aisle)."
)
POLICY_CYCLES = Table(
    "gold",
    "policy_cycles",
    "Backtest replenishment cycles: stock target vs. true demand per policy.",
)
SAFETY_STOCK_CALIBRATION = Table(
    "gold",
    "safety_stock_calibration",
    "Per-aisle safety-stock multiplier learned from the backtest.",
)
POLICY_EVALUATION = Table(
    "gold", "policy_evaluation", "Replenishment what-if: forecast-driven policy vs. incumbent."
)
FORECASTS = Table(
    "gold", "forecasts", "Forward forecast (mean and upper quantile) for the next horizon."
)
REPLENISHMENT_PLAN = Table(
    "gold", "replenishment_plan", "Recommended order quantity per store x SKU."
)

ALL_TABLES: tuple[Table, ...] = (
    STORES, PRODUCTS, ASSORTMENT, CALENDAR, WEATHER, PROMOTIONS, PRICES, POS_SALES, INVENTORY,
    PURCHASE_ORDERS, SIMULATION_TRUTH, DAILY_PANEL, FEATURES, MODELS, BACKTEST_MODELS, BACKTEST_PREDICTIONS,
    BACKTEST_METRICS, POLICY_CYCLES, SAFETY_STOCK_CALIBRATION, POLICY_EVALUATION, FORECASTS, REPLENISHMENT_PLAN,
)  # fmt: skip


def _plain_strings(df: DataFrame) -> DataFrame:
    """Re-cast string columns to the default string type before writing.

    Spark 4.2 marks strings coming back from pandas UDFs as ``string collate UTF8_BINARY``,
    and Delta then records that explicit collation in the table schema. Readers that do not
    support collations yet (delta-rs / the ``deltalake`` package, DuckDB, many BI tools)
    refuse such tables. A plain cast keeps the same values and drops the annotation.
    """
    return df.select(
        *[F.col(f.name).cast("string").alias(f.name) if isinstance(f.dataType, StringType) else F.col(f.name)
          for f in df.schema.fields]
    )  # fmt: skip


class TableStore:
    def __init__(self, spark: SparkSession, settings: Settings) -> None:
        self.spark = spark
        self.settings = settings

    def location(self, table: Table) -> str:
        s = self.settings.storage
        if s.mode == "catalog":
            return f"{s.catalog}.{s.schema}.{table.key}"
        return str(Path(s.data_dir, "lake", table.layer, table.name).resolve())

    def exists(self, table: Table) -> bool:
        if self.settings.storage.mode == "catalog":
            return self.spark.catalog.tableExists(self.location(table))
        return Path(self.location(table), "_delta_log").is_dir()

    def read(self, table: Table) -> DataFrame:
        if self.settings.storage.mode == "catalog":
            return self.spark.read.table(self.location(table))
        return self.spark.read.format("delta").load(self.location(table))

    def write(self, df: DataFrame, table: Table, *, cluster_by: list[str] | None = None) -> None:
        df = _plain_strings(df)
        writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        if cluster_by:
            # Liquid clustering: Delta co-locates rows with similar keys, so reads filtered by
            # store or date skip most files. Works locally and in Unity Catalog alike.
            writer = writer.clusterBy(*cluster_by)
        if self.settings.storage.mode == "catalog":
            writer.saveAsTable(self.location(table))
            comment = table.description.replace("'", "\\'")
            self.spark.sql(f"COMMENT ON TABLE {self.location(table)} IS '{comment}'")
        else:
            writer.save(self.location(table))
