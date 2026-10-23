from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.config import ModelSettings
from demand_forecasting.pipeline.features import CATEGORICAL_FEATURES
from demand_forecasting.pipeline.model import (
    MODEL_COLUMNS,
    PREDICTION_COLUMNS,
    predict_group,
    train_group,
)

FEATURES = ["x_signal", "x_noise", *CATEGORICAL_FEATURES]
FAST = ModelSettings(num_boost_round=40, early_stopping_rounds=5, num_leaves=7, min_data_in_leaf=10)


def synthetic_group(
    n_days: int = 300, n_series: int = 4, stockout_share: float = 0.2
) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    origin = dt.date(2024, 1, 1) + dt.timedelta(days=n_days)
    for s in range(n_series):
        signal = rng.uniform(0.5, 4.0, n_days)
        demand = rng.poisson(2.0 * signal)
        stockout = rng.random(n_days) < stockout_share
        sold = np.where(stockout, np.minimum(demand, rng.integers(0, 3, n_days)), demand)
        for d in range(n_days):
            rows.append(
                {
                    "origin_date": origin,
                    "model_group": "US|snacks",
                    "store_id": f"S{s}",
                    "sku_id": "P1",
                    "date": dt.date(2024, 1, 1) + dt.timedelta(days=d),
                    "x_signal": signal[d],
                    "x_noise": rng.normal(),
                    "store_format_code": s % 2,
                    "category_code": 3,
                    "units_sold": float(sold[d]),
                    "is_stockout": bool(stockout[d]),
                    "roll_mean_28": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_train_returns_mean_and_upper_models() -> None:
    models = train_group(synthetic_group(), FEATURES, FAST, seed=1)
    assert list(models.columns) == MODEL_COLUMNS
    assert sorted(models["model_kind"]) == ["mean", "upper"]
    assert (models["best_iteration"] >= 1).all()


def test_training_is_deterministic() -> None:
    a = train_group(synthetic_group(), FEATURES, FAST, seed=1)
    b = train_group(synthetic_group(), FEATURES, FAST, seed=1)
    assert a["model_string"].tolist() == b["model_string"].tolist()


def test_em_reduces_bias_from_stockouts() -> None:
    data = synthetic_group(stockout_share=0.4)
    expected = 2.0 * data["x_signal"].to_numpy()
    preds = {}
    for iterations in (0, 2):
        ms = ModelSettings(**{**FAST.__dict__, "censoring_em_iterations": iterations})
        models = train_group(data, FEATURES, ms, seed=1)
        preds[iterations] = predict_group(data, models, FEATURES)["forecast_mean"].to_numpy()
    bias = {k: (v.sum() - expected.sum()) / expected.sum() for k, v in preds.items()}
    assert abs(bias[2]) < abs(bias[0])


def test_predictions_are_non_negative_with_horizon() -> None:
    data = synthetic_group()
    out = predict_group(data, train_group(data, FEATURES, FAST, seed=1), FEATURES)
    assert list(out.columns) == PREDICTION_COLUMNS
    assert (out["forecast_mean"] >= 0).all() and (out["forecast_upper"] >= 0).all()
    assert out["horizon"].iloc[0] == (data["date"].iloc[0] - data["origin_date"].iloc[0]).days
    assert (out["method"] == "lightgbm").all()


def test_small_group_falls_back_to_moving_average() -> None:
    data = synthetic_group(n_days=20, n_series=1)
    models = train_group(data, FEATURES, FAST, seed=1)
    assert models.empty
    out = predict_group(data, models, FEATURES)
    assert (out["method"] == "fallback_moving_average").all()
    assert out["forecast_mean"].tolist() == pytest.approx([2.0] * len(data))
