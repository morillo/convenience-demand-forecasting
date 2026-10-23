from __future__ import annotations

import pandas as pd
import pytest

from demand_forecasting.config import PRESETS
from demand_forecasting.data.simulate import simulate_store
from demand_forecasting.data.world import World


@pytest.fixture(scope="module")
def world() -> World:
    return World(PRESETS["tiny"], 11, horizon_days=14)


@pytest.fixture(scope="module")
def sim(world: World) -> pd.DataFrame:
    return pd.concat(
        [simulate_store(world, s) for s in world.stores["store_id"]], ignore_index=True
    )


def test_sales_never_exceed_demand_or_stock(sim: pd.DataFrame) -> None:
    assert (sim["units_sold"] <= sim["demand"]).all()
    assert (sim[["units_sold", "on_hand_close", "units_wasted", "order_qty"]] >= 0).all().all()


def test_lost_sales_only_when_shelf_empty(sim: pd.DataFrame) -> None:
    lost = sim["units_sold"] < sim["demand"]
    assert lost.any(), "the incumbent policy should cause some stockouts"
    assert (sim.loc[lost, "on_hand_close"] == 0).all()


def test_only_fresh_items_are_written_off(sim: pd.DataFrame, world: World) -> None:
    merged = sim.merge(world.products[["sku_id", "is_fresh"]], on="sku_id")
    assert (merged.loc[~merged["is_fresh"], "units_wasted"] == 0).all()


def test_rows_only_while_store_open_and_sku_listed(sim: pd.DataFrame, world: World) -> None:
    merged = sim.merge(world.stores[["store_id", "open_date"]], on="store_id").merge(
        world.products[["sku_id", "launch_date", "delist_date"]], on="sku_id"
    )
    assert (merged["date"] >= merged["open_date"]).all()
    assert (merged["date"] >= merged["launch_date"]).all()
    delisted = merged["delist_date"].notna()
    assert (merged.loc[delisted, "date"] < merged.loc[delisted, "delist_date"]).all()


def test_simulation_is_deterministic(world: World) -> None:
    pd.testing.assert_frame_equal(simulate_store(world, "S00001"), simulate_store(world, "S00001"))


def test_one_row_per_series_day(sim: pd.DataFrame) -> None:
    assert not sim.duplicated(["date", "store_id", "sku_id"]).any()
