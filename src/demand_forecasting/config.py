"""Typed settings for every pipeline step.

Settings are resolved in this order (later wins):

1. Defaults in the dataclasses below.
2. The profile file ``resources/<profile>.toml`` (``local`` or ``databricks``), chosen with
   ``--profile`` or the ``DF_PROFILE`` environment variable.
3. Command-line overrides (``--preset``, ``--catalog``, ``--schema``...).

Code never reads environment variables or hard-coded paths directly; it asks ``Settings``.
That is what lets the same pipeline run on a laptop and on Databricks unchanged.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from datetime import date
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

StorageMode = Literal["path", "catalog"]
PROFILES = ("local", "databricks")


@dataclass(frozen=True)
class DataPreset:
    """Size of the synthetic chain. ``n_days`` is the length of the sales history."""

    name: str
    n_countries: int
    n_stores: int
    n_skus: int
    n_days: int
    start_date: date = date(2023, 1, 2)  # a Monday, so weeks line up in charts


PRESETS: dict[str, DataPreset] = {
    # CI and unit tests: seconds.
    "tiny": DataPreset("tiny", n_countries=2, n_stores=4, n_skus=30, n_days=420),
    # Default local run: about 1.5M store x SKU x day rows, a few minutes on a laptop.
    "small": DataPreset("small", n_countries=3, n_stores=24, n_skus=120, n_days=730),
    # Laptop stress test / Databricks Free Edition upper end: about 25M rows.
    "medium": DataPreset("medium", n_countries=5, n_stores=80, n_skus=300, n_days=1095),
    # Paid Databricks workspace only: about 3 billion rows.
    "large": DataPreset("large", n_countries=12, n_stores=2500, n_skus=1200, n_days=1095),
}


@dataclass(frozen=True)
class StorageSettings:
    mode: StorageMode = "path"
    data_dir: str = "data"
    catalog: str = "workspace"
    schema: str = "demand_forecasting"


@dataclass(frozen=True)
class SparkSettings:
    master: str = "local[*]"
    driver_memory: str = "8g"


@dataclass(frozen=True)
class MlflowSettings:
    tracking_uri: str = "sqlite:///{data_dir}/mlflow.db"
    artifact_location: str | None = "{data_dir}/mlruns"
    experiment: str = "demand-forecasting"


@dataclass(frozen=True)
class ForecastSettings:
    # How far ahead we forecast. Every lag feature looks back at least this far, so one model
    # serves every day of the horizon without peeking at unknown sales (docs/decisions/0004).
    horizon_days: int = 14
    # Number of historical cut-off dates in the rolling-origin backtest.
    backtest_origins: int = 3
    # One LightGBM model is trained per group of these columns (docs/decisions/0002).
    group_by: tuple[str, ...] = ("country_code", "aisle")
    # Extra split of each group by hash(store_id) when a group outgrows max_rows_per_group.
    store_buckets: int = 1
    # Databricks serverless caps a pandas UDF at 1 GB of memory. ~1.5M rows x ~40 float
    # columns stays well inside that budget. The pipeline fails fast above it.
    max_rows_per_group: int = 1_500_000
    # Target probability of not running out of stock in a replenishment cycle.
    service_level: float = 0.95


@dataclass(frozen=True)
class ModelSettings:
    num_boost_round: int = 400
    early_stopping_rounds: int = 30
    validation_days: int = 28
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    tweedie_variance_power: float = 1.2
    upper_quantile: float = 0.9
    # EM passes that re-estimate demand on stockout days (0 = drop those days instead).
    censoring_em_iterations: int = 2
    # Spark already runs one training task per core; LightGBM threads inside a task would
    # oversubscribe the CPU.
    num_threads: int = 1


@dataclass(frozen=True)
class Settings:
    profile: str
    preset: DataPreset
    seed: int = 42
    storage: StorageSettings = field(default_factory=StorageSettings)
    spark: SparkSettings = field(default_factory=SparkSettings)
    mlflow: MlflowSettings = field(default_factory=MlflowSettings)
    forecast: ForecastSettings = field(default_factory=ForecastSettings)
    model: ModelSettings = field(default_factory=ModelSettings)

    @property
    def on_databricks(self) -> bool:
        return self.storage.mode == "catalog"


def _section[T: DataclassInstance](cls: type[T], values: dict[str, Any], section: str) -> T:
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in [{section}]")
    coerced = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**coerced)


def _resolve_data_dir(m: MlflowSettings, data_dir: str) -> MlflowSettings:
    """Fill the ``{data_dir}`` placeholder of the local MLflow paths."""
    location = m.artifact_location.format(data_dir=data_dir) if m.artifact_location else None
    return replace(
        m, tracking_uri=m.tracking_uri.format(data_dir=data_dir), artifact_location=location
    )


def load_settings(profile: str | None = None, **overrides: Any) -> Settings:
    """Build settings from a profile plus overrides (``None`` overrides are ignored)."""
    profile = profile or os.environ.get("DF_PROFILE", "local")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {PROFILES}")
    raw = tomllib.loads(
        resources.files("demand_forecasting.resources").joinpath(f"{profile}.toml").read_text()
    )
    data = raw.get("data", {})
    ov = {k: v for k, v in overrides.items() if v is not None}

    preset_name = ov.pop("preset", data.get("preset", "small"))
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}; expected one of {sorted(PRESETS)}")

    settings = Settings(
        profile=profile,
        preset=PRESETS[preset_name],
        seed=int(ov.pop("seed", data.get("seed", 42))),
        storage=_section(StorageSettings, raw.get("storage", {}), "storage"),
        spark=_section(SparkSettings, raw.get("spark", {}), "spark"),
        mlflow=_section(MlflowSettings, raw.get("mlflow", {}), "mlflow"),
        forecast=_section(ForecastSettings, raw.get("forecast", {}), "forecast"),
        model=_section(ModelSettings, raw.get("model", {}), "model"),
    )

    storage_ov = {k: ov.pop(k) for k in ("data_dir", "catalog", "schema") if k in ov}
    if storage_ov:
        settings = replace(settings, storage=replace(settings.storage, **storage_ov))
    if "experiment" in ov:
        settings = replace(
            settings, mlflow=replace(settings.mlflow, experiment=ov.pop("experiment"))
        )
    if ov:
        raise ValueError(f"Unknown override(s): {sorted(ov)}")
    mlflow_settings = _resolve_data_dir(settings.mlflow, settings.storage.data_dir)
    return replace(settings, mlflow=mlflow_settings)
