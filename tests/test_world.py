from __future__ import annotations

import pandas as pd

from demand_forecasting.config import PRESETS
from demand_forecasting.data.world import World, public
from demand_forecasting.domain import CATEGORIES


def make(seed: int = 7) -> World:
    return World(PRESETS["tiny"], seed, horizon_days=14)


def test_world_is_deterministic() -> None:
    a, b = make(), make()
    pd.testing.assert_frame_equal(a.stores, b.stores)
    pd.testing.assert_frame_equal(a.promotions, b.promotions)
    pd.testing.assert_frame_equal(a.weather, b.weather)


def test_seed_changes_the_world() -> None:
    assert not make(1).products["_base_rate"].equals(make(2).products["_base_rate"])


def test_sizes_match_preset() -> None:
    w, p = make(), PRESETS["tiny"]
    assert len(w.stores) == p.n_stores
    assert len(w.products) == p.n_skus
    assert w.stores["country_code"].nunique() == p.n_countries
    assert len(w.all_dates) == p.n_days + 14


def test_every_category_present_when_room() -> None:
    w = World(PRESETS["small"], 7, 14)
    assert set(w.products["category"]) == {c.name for c in CATEGORIES}


def test_fresh_items_are_delivered_daily() -> None:
    w = make()
    a = w.assortment.merge(w.products[["sku_id", "is_fresh"]], on="sku_id")
    fresh = a[a["is_fresh"]]
    assert (fresh["lead_time_days"] == 1).all()
    assert (fresh["review_period_days"] == 1).all()


def test_weather_actuals_hidden_in_future_but_forecast_known() -> None:
    w = make()
    future = w.weather[pd.to_datetime(w.weather["date"]) > w.last_history_date]
    assert future["temp_c"].isna().all()
    assert future["temp_forecast_c"].notna().all()


def test_public_drops_internal_columns() -> None:
    assert not [c for c in public(make().stores).columns if c.startswith("_")]
