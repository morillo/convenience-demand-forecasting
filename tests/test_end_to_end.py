"""The whole pipeline on the tiny preset, exactly as `uv run dfc run-all --preset tiny` runs it."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from streamlit.testing.v1 import AppTest

from demand_forecasting import storage as T
from demand_forecasting.cli import RUN_ALL, STEPS
from demand_forecasting.config import Settings

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.e2e
def test_run_all_tiny(
    spark: SparkSession, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in RUN_ALL:
        STEPS[name][0](spark, settings)

    store = T.TableStore(spark, settings)
    for table in T.ALL_TABLES:
        assert store.exists(table), f"{table.key} was not written"

    h = settings.forecast.horizon_days
    fc = store.read(T.FORECASTS)
    series = fc.select("store_id", "sku_id").distinct().count()
    assert series > 0
    assert fc.count() <= series * h
    assert fc.where((F.col("forecast_mean") < 0) | F.col("forecast_mean").isNull()).count() == 0

    plan = store.read(T.REPLENISHMENT_PLAN)
    assert plan.count() == series
    assert plan.where(F.col("recommended_order_qty") < 0).count() == 0

    models = {
        r["model"] for r in store.read(T.BACKTEST_METRICS).select("model").distinct().collect()
    }
    assert models == {"lightgbm", "seasonal_naive", "moving_average_28"}

    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow.experiment)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert {r.data.tags["pipeline_step"] for r in runs} == {"backtest", "train"}

    # The results viewer renders every tab from these outputs without errors.
    monkeypatch.setenv("DF_DATA_DIR", settings.storage.data_dir)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert len(app.tabs) == 3
    assert app.metric, "summary KPIs should render"

    # The results viewer renders every tab from these outputs without errors.
    monkeypatch.setenv("DF_DATA_DIR", settings.storage.data_dir)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert len(app.tabs) == 3
    assert app.metric, "summary KPIs should render"
