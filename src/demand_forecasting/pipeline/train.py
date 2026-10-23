"""Pipeline step ``train``: fit the production models on the full history.

Same code path as the backtest, with a single origin: the last day of history. The models
are stored in the gold ``models`` Delta table (one row per model group and kind, LightGBM's
text format), stamped with the MLflow run id that produced them
(docs/decisions/0007-model-storage-and-tracking.md).
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from demand_forecasting import storage as T
from demand_forecasting import tracking
from demand_forecasting.config import Settings
from demand_forecasting.pipeline import training

log = logging.getLogger(__name__)


def run(spark: SparkSession, settings: Settings) -> None:
    store = T.TableStore(spark, settings)
    features = store.read(T.FEATURES)
    origin = training.last_history_date(features)

    with tracking.run(settings, "train") as run_id:
        models = training.fit(training.training_rows(features, [origin], settings), settings)
        store.write(
            models.withColumn("run_id", F.lit(run_id)).withColumn(
                "trained_at", F.current_timestamp()
            ),
            T.MODELS,
        )
        summary = (
            store.read(T.MODELS)
            .select(
                "model_group", "model_kind", "best_iteration", "n_train_rows", "feature_importance"
            )
            .toPandas()
        )
        if summary.empty:
            raise RuntimeError("No model group had enough training rows; is the dataset too small?")

        importance = (
            pd.DataFrame(
                [
                    json.loads(s)
                    for s in summary.loc[summary["model_kind"] == "mean", "feature_importance"]
                ]
            )
            .mean()
            .sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index(name="mean_gain_share")
        )
        tracking.log_metrics(
            {
                "model_groups": float(summary["model_group"].nunique()),
                "avg_best_iteration": float(summary["best_iteration"].mean()),
                "train_rows": float(
                    summary.loc[summary["model_kind"] == "mean", "n_train_rows"].sum()
                ),
            }
        )
        tracking.log_table(importance, "feature_importance")
        tracking.log_table(summary.drop(columns="feature_importance"), "model_groups")
        log.info(
            "trained %d model groups through %s (run %s)",
            summary["model_group"].nunique(),
            origin,
            run_id,
        )
        log.info("top features: %s", ", ".join(importance["feature"].head(8)))
