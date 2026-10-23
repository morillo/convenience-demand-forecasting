"""Simulate one store: true demand, the incumbent ordering system, and what was actually sold.

For every SKU the store carries and every day it is active:

1. **Expected demand** (lambda) = SKU popularity x store traffic x format affinity
   x weekday x yearly season x temperature x rain x holiday x payday x promotion x price
   x store growth x launch ramp.
2. **True demand** is drawn from a negative-binomial distribution around lambda, which gives
   the bursty, often-zero sales typical of slow movers.
3. **Inventory**: the incumbent system orders up to "smoothed recent sales x (review period +
   lead time) x cover factor". It learns only from *sales*, so it under-reacts to promotions
   and heat waves. When the shelf is empty the sale is lost: ``units_sold < demand``.
   That censoring is the main trap the forecasting pipeline must handle.

The function is vectorised over SKUs and loops over days only for the inventory recursion.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from demand_forecasting.data.world import World, rng
from demand_forecasting.domain import (
    FORMAT_CATEGORY_AFFINITY,
    FORMAT_HOLIDAY_LIFT,
    FORMAT_WEEKDAY_PROFILE,
    Category,
    Country,
)

_DEMAND_STREAM = 7
SMOOTHING = 0.15  # incumbent's exponential smoothing of sales
STOCKOUT_UPLIFT = 1.10  # incumbent's correction after a sold-out day

SIMULATION_SCHEMA = (
    "date date, store_id string, sku_id string, expected_demand double, demand int, "
    "units_sold int, unit_price double, on_hand_close int, units_received int, "
    "units_wasted int, order_qty int, lead_time_days int"
)


@dataclass(frozen=True)
class _Store:
    """Everything about one store that the simulation steps share."""

    store_id: str
    attrs: dict[Hashable, Any]
    country: Country
    fmt: str
    assortment: pd.DataFrame
    sku_ids: np.ndarray
    sku_pos: np.ndarray  # column of each carried SKU in the world's product matrices
    products: pd.DataFrame
    categories: list[Category]
    dates: pd.DatetimeIndex

    def per_cat(self, attr: str) -> np.ndarray:
        return np.array([getattr(c, attr) for c in self.categories])


def _store(world: World, store_id: str) -> _Store:
    attrs = world.stores.set_index("store_id").to_dict("index")[store_id]
    assort = world.assortment[world.assortment["store_id"] == store_id]
    sku_ids = assort["sku_id"].to_numpy()
    position = {sku: i for i, sku in enumerate(world.products["sku_id"])}
    sku_pos = np.array([position[sku] for sku in sku_ids], dtype=np.int64)
    products = world.products.iloc[sku_pos]
    return _Store(
        store_id=store_id,
        attrs=attrs,
        country=next(c for c in world.countries if c.code == attrs["country_code"]),
        fmt=str(attrs["store_format"]),
        assortment=assort,
        sku_ids=sku_ids,
        sku_pos=sku_pos,
        products=products,
        categories=[world.categories[c] for c in products["category"]],
        dates=world.history_dates,
    )


def _activity(st: _Store) -> tuple[np.ndarray, np.ndarray, int]:
    """(days x SKUs) mask of store open, SKU launched and not yet delisted; plus the launch
    day of each SKU and the store's opening day (both relative to the first history day)."""
    start, n_days = st.dates[0], len(st.dates)
    day_idx = np.arange(n_days)
    store_open = int((pd.Timestamp(st.attrs["open_date"]) - start).days)
    launch = np.array([(pd.Timestamp(d) - start).days for d in st.products["launch_date"]])
    delist = np.array(
        [
            n_days if d is None or pd.isna(d) else (pd.Timestamp(d) - start).days
            for d in st.products["delist_date"]
        ]
    )
    active = (
        (day_idx[:, None] >= max(store_open, 0))
        & (day_idx[:, None] >= launch[None, :])
        & (day_idx[:, None] < delist[None, :])
    )
    return active, launch, store_open


def _expected_demand(
    world: World, st: _Store, active: np.ndarray, launch: np.ndarray, store_open: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expected units per (day, SKU) and the launch ramp factor (used for initial stock)."""
    n_days = len(st.dates)
    day_idx = np.arange(n_days)
    fmt, country = st.fmt, st.country

    base = st.products["_base_rate"].to_numpy() * float(st.attrs["traffic_index"])
    affinity = np.array([FORMAT_CATEGORY_AFFINITY.get((fmt, c.name), 1.0) for c in st.categories])
    weekday = np.array(FORMAT_WEEKDAY_PROFILE[fmt])[st.dates.dayofweek.to_numpy()]

    doy = st.dates.dayofyear.to_numpy()
    peak = 15 if country.southern_hemisphere else 200
    season_wave = np.cos(2 * np.pi * (doy - peak) / 365.25)  # +1 at the warmest time of year
    # Summer-peaking categories follow the wave; winter-peaking ones (negative temperature
    # sensitivity) follow its mirror image.
    direction = np.where(st.per_cat("temp_sensitivity") >= 0, 1.0, -1.0)
    season = 1 + st.per_cat("annual_amplitude")[None, :] * direction[None, :] * season_wave[:, None]

    weather = world.weather[world.weather["country_code"] == country.code].iloc[:n_days]
    temp_dev = weather["temp_c"].to_numpy() - country.mean_temp_c
    temp_eff = np.exp(
        np.clip(st.per_cat("temp_sensitivity")[None, :] * temp_dev[:, None], -1.5, 1.5)
    )
    rainy = (weather["rain_mm"].to_numpy() > 1.0).astype(float)
    rain_eff = np.clip(1 + st.per_cat("rain_sensitivity")[None, :] * rainy[:, None], 0.2, None)

    cal = world.calendar[world.calendar["country_code"] == country.code].iloc[:n_days]
    holiday = cal["is_holiday"].to_numpy().astype(float)
    holiday_eff = (
        1
        + st.per_cat("holiday_sensitivity")[None, :]
        * (FORMAT_HOLIDAY_LIFT[fmt] - 1)
        * holiday[:, None]
    )
    payday = cal["is_payday"].to_numpy().astype(float)
    payday_window = np.convolve(payday, np.ones(3), mode="full")[:n_days] > 0  # payday + 2 days
    payday_eff = 1 + st.per_cat("payday_lift")[None, :] * payday_window[:, None]

    discount = world.promo_matrix[country.code][:n_days][:, st.sku_pos]
    promo_eff = (1 - discount) ** (-st.per_cat("promo_elasticity")[None, :])
    list_price = world.price_matrix[:n_days][:, st.sku_pos]
    price_eff = (list_price / list_price[0][None, :]) ** -0.8

    growth = np.exp(float(st.attrs["_growth"]) * day_idx / 365.0)
    days_live = day_idx[:, None] - np.maximum(launch[None, :], store_open)
    ramp = np.clip(0.3 + 0.7 * days_live / 30.0, 0.3, 1.0)  # new items/stores take a month to ramp

    lam = (
        base[None, :] * affinity[None, :] * weekday[:, None] * season * temp_eff * rain_eff
        * holiday_eff * payday_eff * promo_eff * price_eff * growth[:, None] * ramp
    )  # fmt: skip
    return np.where(active, lam, 0.0), ramp


def _run_incumbent(
    st: _Store, lam: np.ndarray, ramp: np.ndarray, demand: np.ndarray, active: np.ndarray
) -> dict[str, np.ndarray]:
    """Day-by-day stock simulation under the incumbent ordering system."""
    n_days, n_skus = demand.shape
    lead = st.assortment["lead_time_days"].to_numpy()
    review = st.assortment["review_period_days"].to_numpy()
    cover = st.assortment["_cover_factor"].to_numpy()
    offset = st.assortment["_review_offset"].to_numpy()
    fresh = st.products["is_fresh"].to_numpy()
    # Fresh stock expires overnight, so it only needs to cover the days until the next
    # delivery; ambient stock must also cover the lead time.
    protect = np.where(fresh, review, review + lead)

    arrivals = np.zeros((n_days + int(lead.max()) + 1, n_skus), dtype=np.int64)
    on_order = np.zeros(n_skus, dtype=np.int64)
    stock = np.zeros(n_skus, dtype=np.int64)
    est = np.zeros(n_skus)
    was_active = np.zeros(n_skus, dtype=bool)
    out = {
        k: np.zeros((n_days, n_skus), dtype=np.int64)
        for k in ("sold", "close", "received", "wasted", "ordered")
    }
    cols = np.arange(n_skus)

    for d in range(n_days):
        act = active[d]
        new = act & ~was_active
        if new.any():  # first day on the shelf: initial delivery sized from the plan
            est[new] = lam[d, new] / ramp[d, new]
            stock[new] = np.ceil(est[new] * protect[new] * cover[new]).astype(np.int64)
        was_active = act

        out["received"][d] = arrivals[d]
        on_order -= arrivals[d]
        stock += arrivals[d]
        s = np.minimum(demand[d], stock)
        sold_out = (demand[d] >= stock) & act
        out["sold"][d] = s
        stock -= s
        out["close"][d] = stock
        if fresh.any():  # unsold fresh food is written off at closing time
            out["wasted"][d] = np.where(fresh, stock, 0)
            stock = np.where(fresh, 0, stock)
        # On sold-out days the system knows demand was at least the sales, and nudges its
        # estimate up instead of learning the censored number (a common stockout correction).
        observed = np.where(sold_out, np.maximum(s, est) * STOCKOUT_UPLIFT, s)
        est = np.where(act, SMOOTHING * observed + (1 - SMOOTHING) * est, est)

        is_review = act & ((d + offset) % review == 0)
        target = np.ceil(est * protect * cover).astype(np.int64)
        qty = np.where(is_review, np.maximum(target - stock - on_order, 0), 0)
        out["ordered"][d] = qty
        on_order += qty
        arrivals[d + lead, cols] += qty
    return out


def simulate_store(world: World, store_id: str) -> pd.DataFrame:
    """Simulate one store's history; returns rows matching SIMULATION_SCHEMA."""
    st = _store(world, store_id)
    active, launch, store_open = _activity(st)
    lam, ramp = _expected_demand(world, st, active, launch, store_open)

    # True demand: negative binomial = Poisson with a gamma-distributed rate.
    r = rng(world.seed, _DEMAND_STREAM, int(store_id[1:]))
    k = st.per_cat("dispersion")[None, :]
    demand = r.poisson(r.gamma(k, np.maximum(lam, 1e-9) / k)) * active

    sim = _run_incumbent(st, lam, ramp, demand, active)

    n_days = len(st.dates)
    discount = world.promo_matrix[st.country.code][:n_days][:, st.sku_pos]
    list_price = world.price_matrix[:n_days][:, st.sku_pos]
    d_idx, k_idx = np.nonzero(active)
    unit_price = (list_price * (1 - discount))[d_idx, k_idx]
    lead = st.assortment["lead_time_days"].to_numpy()
    return pd.DataFrame(
        {
            "date": st.dates[d_idx].date,
            "store_id": store_id,
            "sku_id": st.sku_ids[k_idx],
            "expected_demand": np.round(lam[d_idx, k_idx], 4),
            "demand": demand[d_idx, k_idx].astype("int32"),
            "units_sold": sim["sold"][d_idx, k_idx].astype("int32"),
            "unit_price": np.round(unit_price, 2),
            "on_hand_close": sim["close"][d_idx, k_idx].astype("int32"),
            "units_received": sim["received"][d_idx, k_idx].astype("int32"),
            "units_wasted": sim["wasted"][d_idx, k_idx].astype("int32"),
            "order_qty": sim["ordered"][d_idx, k_idx].astype("int32"),
            "lead_time_days": lead[k_idx].astype("int32"),
        }
    )
