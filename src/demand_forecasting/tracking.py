"""MLflow experiment tracking, identical locally (SQLite file) and on Databricks (workspace).

Each backtest / training run records its settings (params), headline numbers (metrics) and
reports (artifacts), so any result can be traced back to exactly how it was produced.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from demand_forecasting.config import Settings


def _experiment_id(settings: Settings) -> str:
    m = settings.mlflow
    mlflow.set_tracking_uri(m.tracking_uri)
    existing = mlflow.get_experiment_by_name(m.experiment)
    if existing is not None:
        return existing.experiment_id
    location = None
    if m.artifact_location and not settings.on_databricks:
        location = Path(m.artifact_location).resolve().as_uri()
    return mlflow.create_experiment(m.experiment, artifact_location=location)


@contextmanager
def run(settings: Settings, name: str) -> Generator[str]:
    """Start an MLflow run with the common params logged; yields the run id."""
    with mlflow.start_run(experiment_id=_experiment_id(settings), run_name=name) as active:
        mlflow.set_tags(
            {"pipeline_step": name, "profile": settings.profile, "preset": settings.preset.name}
        )
        mlflow.log_params(
            {
                "preset": settings.preset.name,
                "seed": settings.seed,
                "horizon_days": settings.forecast.horizon_days,
                "backtest_origins": settings.forecast.backtest_origins,
                "group_by": ",".join(settings.forecast.group_by),
                "store_buckets": settings.forecast.store_buckets,
                "service_level": settings.forecast.service_level,
                **{f"model.{k}": v for k, v in asdict(settings.model).items()},
            }
        )
        yield active.info.run_id


def log_table(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def log_dict(data: dict[str, Any], name: str) -> None:
    mlflow.log_text(json.dumps(data, indent=2, default=str), f"{name}.json")


def log_metrics(metrics: dict[str, float | None]) -> None:
    """Log metrics, skipping missing values (None / NaN), e.g. coverage for point-only baselines."""
    clean = {
        _metric_name(k): float(v) for k, v in metrics.items() if v is not None and not math.isnan(v)
    }
    mlflow.log_metrics(clean)


def _metric_name(name: str) -> str:
    """MLflow allows only alphanumerics and _ - . space : / in metric names."""
    return re.sub(r"[^\w\-. :/]", "_", name)
