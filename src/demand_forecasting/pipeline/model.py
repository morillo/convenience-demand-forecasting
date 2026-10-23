"""Train and apply the LightGBM models of one model group. Pure pandas, no Spark.

Each group gets two models (docs/decisions/0003-tweedie-and-quantile.md):

* ``mean``: Tweedie objective. Retail demand is a count with many zeros and a long right tail;
  Tweedie handles both and predicts the expected units per day.
* ``upper``: quantile objective at ``upper_quantile`` (default P90). "On 9 days out of 10,
  demand will not exceed this." Used to size safety stock.

Stockout days (sales understate demand) are kept and *unconstrained* with EM
(``censoring.py``): fit on clean days, fill in sold-out days with the expected demand given
it was at least the sales, refit on all days, repeat. The best number of boosting rounds is
found by early stopping on the most recent ``validation_days`` of clean days; the final fits
use all rows with that many rounds, so the newest data is never wasted.

These functions run inside Spark ``applyInPandas`` tasks, one call per group.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_forecasting.config import ModelSettings
from demand_forecasting.pipeline.censoring import conditional_mean, sample_truncated
from demand_forecasting.pipeline.features import CATEGORICAL_FEATURES

MODEL_KINDS = ("mean", "upper")
# Below this many usable rows a group falls back to a moving-average forecast.
MIN_TRAIN_ROWS = 200

MODEL_SCHEMA = (
    "origin_date date, model_group string, model_kind string, model_string string, "
    "best_iteration int, n_train_rows long, feature_importance string, params string"
)
PREDICTION_SCHEMA = (
    "origin_date date, model_group string, store_id string, sku_id string, date date, "
    "horizon int, forecast_mean double, forecast_upper double, method string"
)
MODEL_COLUMNS = [c.split()[0] for c in MODEL_SCHEMA.split(", ")]
PREDICTION_COLUMNS = [c.split()[0] for c in PREDICTION_SCHEMA.split(", ")]


def lgb_params(kind: str, ms: ModelSettings, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "learning_rate": ms.learning_rate,
        "num_leaves": ms.num_leaves,
        "min_data_in_leaf": ms.min_data_in_leaf,
        "feature_fraction": ms.feature_fraction,
        "bagging_fraction": ms.bagging_fraction,
        "bagging_freq": 1,
        "num_threads": ms.num_threads,
        "seed": seed,
        "deterministic": True,
        "verbosity": -1,
    }
    if kind == "mean":
        params |= {"objective": "tweedie", "tweedie_variance_power": ms.tweedie_variance_power}
    else:
        params |= {"objective": "quantile", "alpha": ms.upper_quantile}
    return params


def _matrix(pdf: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = pdf[feature_cols].astype("float32")
    for c in CATEGORICAL_FEATURES:
        x[c] = pdf[c].fillna(-1).astype("int32")
    return x


def _dataset(x: pd.DataFrame, y: np.ndarray) -> lgb.Dataset:
    return lgb.Dataset(x, y, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)


def _best_rounds(
    params: dict[str, Any], x: pd.DataFrame, y: np.ndarray, dates: pd.Series, ms: ModelSettings
) -> int:
    """Early stopping on the last ``validation_days``; falls back to num_boost_round."""
    is_valid = (dates > dates.max() - pd.Timedelta(days=ms.validation_days)).to_numpy()
    if not is_valid.any() or (~is_valid).sum() < MIN_TRAIN_ROWS:
        return ms.num_boost_round
    probe = lgb.train(
        params,
        _dataset(x[~is_valid], y[~is_valid]),
        num_boost_round=ms.num_boost_round,
        valid_sets=[_dataset(x[is_valid], y[is_valid])],
        callbacks=[lgb.early_stopping(ms.early_stopping_rounds, verbose=False)],
    )
    return max(probe.best_iteration, 1)


def train_group(
    pdf: pd.DataFrame, feature_cols: list[str], ms: ModelSettings, seed: int
) -> pd.DataFrame:
    """Train both models for one (origin_date, model_group). Returns MODEL_SCHEMA rows."""
    rows_all = pdf[pdf["units_sold"].notna()].sort_values("date")
    censored = rows_all["is_stockout"].fillna(False).astype(bool).to_numpy()
    use_em = ms.censoring_em_iterations > 0
    train = rows_all if use_em else rows_all[~censored]
    censored_train = censored if use_em else np.zeros(len(train), dtype=bool)
    if (~censored_train).sum() < MIN_TRAIN_ROWS:
        return pd.DataFrame(columns=MODEL_COLUMNS)
    keys = {"origin_date": pdf["origin_date"].iloc[0], "model_group": pdf["model_group"].iloc[0]}

    x = _matrix(train, feature_cols)
    sold = train["units_sold"].to_numpy(dtype=float)
    dates = pd.to_datetime(train["date"])
    clean = ~censored_train

    # Mean model: rounds chosen on clean days, then EM over the censored days.
    mean_params = lgb_params("mean", ms, seed)
    mean_rounds = _best_rounds(mean_params, x[clean], sold[clean], dates[clean], ms)
    booster = lgb.train(mean_params, _dataset(x[clean], sold[clean]), num_boost_round=mean_rounds)
    y_mean = sold.copy()
    for _ in range(ms.censoring_em_iterations if censored_train.any() else 0):
        lam = np.asarray(booster.predict(x[censored_train]), dtype=float)
        y_mean[censored_train] = conditional_mean(sold[censored_train], lam)
        booster = lgb.train(mean_params, _dataset(x, y_mean), num_boost_round=mean_rounds)
    fitted = {"mean": (booster, mean_rounds, mean_params)}

    # Upper-quantile model: censored days get a draw from the truncated demand distribution.
    y_upper = sold.copy()
    if censored_train.any():
        lam = np.asarray(booster.predict(x[censored_train]), dtype=float)
        rng = np.random.default_rng([seed, zlib.crc32(str(keys["model_group"]).encode())])
        y_upper[censored_train] = sample_truncated(sold[censored_train], lam, rng)
    upper_params = lgb_params("upper", ms, seed)
    upper_rounds = _best_rounds(upper_params, x[clean], sold[clean], dates[clean], ms)
    fitted["upper"] = (
        lgb.train(upper_params, _dataset(x, y_upper), num_boost_round=upper_rounds),
        upper_rounds,
        upper_params,
    )

    rows: list[dict[str, Any]] = []
    for kind in MODEL_KINDS:
        model, rounds, params = fitted[kind]
        gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
        share = (gain / max(gain.sum(), 1e-12)).round(5).tolist()
        rows.append(
            keys
            | {
                "model_kind": kind,
                "model_string": model.model_to_string(),
                "best_iteration": rounds,
                "n_train_rows": len(train),
                "feature_importance": json.dumps(dict(zip(feature_cols, share, strict=True))),
                "params": json.dumps(params | {"settings": asdict(ms)}),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def predict_group(
    features: pd.DataFrame, models: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Score one (origin_date, model_group). Groups without a model use the 28-day average."""
    if features.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    out = features[["origin_date", "model_group", "store_id", "sku_id", "date"]].copy()
    delta = pd.to_datetime(out["date"]) - pd.to_datetime(out["origin_date"])
    out["horizon"] = delta.dt.days.astype("int32")
    if models.empty:
        out["forecast_mean"] = features["roll_mean_28"].fillna(0.0).to_numpy()
        out["forecast_upper"] = np.nan
        out["method"] = "fallback_moving_average"
        return out[PREDICTION_COLUMNS]
    x = _matrix(features, feature_cols)
    by_kind = dict(zip(models["model_kind"], models["model_string"], strict=True))
    for kind, col in (("mean", "forecast_mean"), ("upper", "forecast_upper")):
        booster = lgb.Booster(model_str=by_kind[kind])
        out[col] = np.clip(np.asarray(booster.predict(x), dtype=float), 0.0, None)
    out["method"] = "lightgbm"
    return out[PREDICTION_COLUMNS]
