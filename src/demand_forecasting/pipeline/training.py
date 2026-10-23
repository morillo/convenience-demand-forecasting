"""Distributed training and scoring: one LightGBM model pair per (origin_date, model_group).

An *origin* is a forecast cut-off date T: the model may learn from rows up to T and predicts
T+1 .. T+H. The backtest uses several historical origins at once (each origin x group is an
independent Spark task); the production run uses a single origin, the last day of history.

Spark does the heavy lifting (filtering, shuffling rows to their group); each task receives
one group as a pandas DataFrame and calls the pure functions in ``model.py``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from demand_forecasting.config import Settings
from demand_forecasting.pipeline.features import feature_columns
from demand_forecasting.pipeline.model import (
    MODEL_SCHEMA,
    PREDICTION_SCHEMA,
    predict_group,
    train_group,
)

GROUP_KEYS = ["origin_date", "model_group"]


class GroupTooLargeError(RuntimeError):
    pass


def last_history_date(features: DataFrame) -> dt.date:
    value = features.where(~F.col("is_future")).agg(F.max("date")).first()
    assert value is not None and value[0] is not None, "features table has no history rows"
    return value[0]


def backtest_origins(last_date: dt.date, settings: Settings) -> list[dt.date]:
    """K non-overlapping test windows of H days at the end of the history."""
    h, k = settings.forecast.horizon_days, settings.forecast.backtest_origins
    return [last_date - dt.timedelta(days=h * i) for i in range(k, 0, -1)]


def _origins_df(features: DataFrame, origins: list[dt.date]) -> DataFrame:
    spark = features.sparkSession
    return spark.createDataFrame(pd.DataFrame({"origin_date": origins}))


def training_rows(features: DataFrame, origins: list[dt.date], settings: Settings) -> DataFrame:
    cols = [*feature_columns(settings.forecast.horizon_days), "store_id", "sku_id", "date", "model_group",
            "units_sold", "is_stockout"]  # fmt: skip
    return (
        features.where(~F.col("is_future"))
        .select(*cols)
        .crossJoin(F.broadcast(_origins_df(features, origins)))
        .where(F.col("date") <= F.col("origin_date"))
    )


def scoring_rows(
    features: DataFrame, origins: list[dt.date], settings: Settings, *, future: bool
) -> DataFrame:
    h = settings.forecast.horizon_days
    cols = [*feature_columns(h), "store_id", "sku_id", "date", "model_group"]
    return (
        features.where(F.col("is_future") == F.lit(future))
        .select(*cols)
        .crossJoin(F.broadcast(_origins_df(features, origins)))
        .where(
            (F.col("date") > F.col("origin_date")) & (F.col("date") <= F.date_add("origin_date", h))
        )
    )


def check_group_sizes(rows: DataFrame, settings: Settings) -> None:
    """Fail fast before a pandas task exceeds the serverless 1 GB UDF memory limit."""
    limit = settings.forecast.max_rows_per_group
    biggest = rows.groupBy(*GROUP_KEYS).count().agg(F.max("count")).first()
    n = 0 if biggest is None or biggest[0] is None else int(biggest[0])
    if n > limit:
        raise GroupTooLargeError(
            f"Largest model group has {n:,} training rows (limit {limit:,}). Split groups further: "
            "raise forecast.store_buckets or add a column to forecast.group_by."
        )


def fit(rows: DataFrame, settings: Settings) -> DataFrame:
    check_group_sizes(rows, settings)
    feats, ms, seed = feature_columns(settings.forecast.horizon_days), settings.model, settings.seed

    def train(pdf: pd.DataFrame) -> pd.DataFrame:
        return train_group(pdf, feats, ms, seed)

    return rows.groupBy(*GROUP_KEYS).applyInPandas(train, MODEL_SCHEMA)


def predict(rows: DataFrame, models: DataFrame, settings: Settings) -> DataFrame:
    feats = feature_columns(settings.forecast.horizon_days)

    def score(features_pdf: pd.DataFrame, models_pdf: pd.DataFrame) -> pd.DataFrame:
        return predict_group(features_pdf, models_pdf, feats)

    model_cols = models.select(*GROUP_KEYS, "model_kind", "model_string")
    return (
        rows.groupBy(*GROUP_KEYS)
        .cogroup(model_cols.groupBy(*GROUP_KEYS))
        .applyInPandas(score, PREDICTION_SCHEMA)
    )
