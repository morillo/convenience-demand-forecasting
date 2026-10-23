"""The synthetic chain: stores, products, assortment, calendar, weather, promotions, prices.

A ``World`` is a pure function of ``(preset, seed, horizon_days)``: building it twice gives
identical tables. That property is what lets every Spark worker rebuild the world on its own
(instead of shipping large tables to it) and simulate its stores independently.

Random numbers come from ``numpy.random.default_rng([seed, STREAM, entity])``: one
independent stream per purpose and entity, so adding a store never changes another store's
data. See docs/mock-data.md for the story behind every knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd

from demand_forecasting.config import PRESETS, DataPreset
from demand_forecasting.domain import CATEGORIES, COUNTRIES, STORE_FORMATS, Category, Country

# Independent random streams (the numbers only need to differ from each other).
_STORES, _PRODUCTS, _ASSORTMENT, _WEATHER, _PROMOS, _PRICES, _DEMAND = range(1, 8)

FORMAT_SHARE = {"urban": 0.35, "suburban": 0.25, "highway": 0.15, "office": 0.15, "transit": 0.10}


def rng(seed: int, stream: int, entity: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, stream, entity])


@dataclass(frozen=True)
class World:
    preset: DataPreset
    seed: int
    horizon_days: int

    # ---- time -------------------------------------------------------------------------------
    @cached_property
    def history_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.preset.start_date, periods=self.preset.n_days, freq="D")

    @cached_property
    def all_dates(self) -> pd.DatetimeIndex:
        """History plus the forecast horizon (calendar, weather forecasts, planned promos)."""
        return pd.date_range(
            self.preset.start_date, periods=self.preset.n_days + self.horizon_days, freq="D"
        )

    @property
    def last_history_date(self) -> pd.Timestamp:
        return self.history_dates[-1]

    @cached_property
    def countries(self) -> tuple[Country, ...]:
        return COUNTRIES[: self.preset.n_countries]

    @cached_property
    def categories(self) -> dict[str, Category]:
        return {c.name: c for c in CATEGORIES}

    # ---- stores -----------------------------------------------------------------------------
    @cached_property
    def stores(self) -> pd.DataFrame:
        r = rng(self.seed, _STORES)
        n = self.preset.n_stores
        # Larger markets get more stores (weights 1, 1/2, 1/3...), but every country gets one.
        weights = 1.0 / np.arange(1, len(self.countries) + 1)
        country_idx = np.concatenate(
            [
                np.arange(len(self.countries)),
                r.choice(
                    len(self.countries),
                    size=max(n - len(self.countries), 0),
                    p=weights / weights.sum(),
                ),
            ]
        )[:n]
        formats = r.choice(list(FORMAT_SHARE), size=n, p=list(FORMAT_SHARE.values()))
        # ~10% of stores open during the history: the "new store" cold-start case.
        opens_late = r.random(n) < 0.10
        late_offset = r.integers(30, int(self.preset.n_days * 0.7), size=n)
        open_offset = np.where(opens_late, late_offset, -r.integers(200, 3000, size=n))
        start = pd.Timestamp(self.preset.start_date)
        return pd.DataFrame(
            {
                "store_id": [f"S{i + 1:05d}" for i in range(n)],
                "country_code": [self.countries[i].code for i in country_idx],
                "store_format": formats,
                "open_date": [(start + pd.Timedelta(days=int(o))).date() for o in open_offset],
                # Footfall multiplier: some stores are simply busier than others.
                "traffic_index": np.round(r.lognormal(0.0, 0.35, size=n), 3),
                # Underlying yearly growth (hidden from the model, it has to learn it).
                "_growth": r.normal(0.02, 0.04, size=n),
            }
        )

    # ---- products ---------------------------------------------------------------------------
    @cached_property
    def products(self) -> pd.DataFrame:
        r = rng(self.seed, _PRODUCTS)
        n = self.preset.n_skus
        weights = np.array([c.skus_weight for c in CATEGORIES])
        # Every category gets at least one SKU (when there is room), the rest by weight.
        counts = np.zeros(len(CATEGORIES), dtype=int)
        counts[: min(n, len(CATEGORIES))] = 1
        rest = n - counts.sum()
        if rest > 0:
            counts += r.multinomial(rest, weights / weights.sum())
        cat_of_sku = np.repeat(np.arange(len(CATEGORIES)), counts)
        start = pd.Timestamp(self.preset.start_date)
        rows: list[dict[str, object]] = []
        per_cat_counter: dict[str, int] = {}
        for i, ci in enumerate(cat_of_sku):
            cat = CATEGORIES[ci]
            per_cat_counter[cat.name] = per_cat_counter.get(cat.name, 0) + 1
            lo, hi = cat.base_rate
            # ~15% of SKUs launch during the history (new-product cold start).
            launch_offset = (
                int(r.integers(30, self.preset.n_days - 60))
                if r.random() < 0.15
                else -int(r.integers(30, 2000))
            )
            # ~6% are delisted (discontinued) during the history.
            delist_offset = (
                int(r.integers(max(launch_offset, 0) + 90, self.preset.n_days + 1))
                if r.random() < 0.06 and max(launch_offset, 0) + 90 < self.preset.n_days
                else None
            )
            rows.append(
                {
                    "sku_id": f"P{i + 1:06d}",
                    "sku_name": f"{cat.name.replace('_', ' ').title()} #{per_cat_counter[cat.name]:03d}",
                    "aisle": cat.aisle,
                    "category": cat.name,
                    "list_price": round(float(r.uniform(*cat.price)), 1) - 0.01,
                    "launch_date": (start + pd.Timedelta(days=launch_offset)).date(),
                    "delist_date": None
                    if delist_offset is None
                    else (start + pd.Timedelta(days=delist_offset)).date(),
                    "is_fresh": cat.fresh,
                    # Log-uniform popularity inside the category's range.
                    "_base_rate": float(np.exp(r.uniform(np.log(lo), np.log(hi)))),
                }
            )
        return pd.DataFrame(rows)

    # ---- assortment and replenishment rules --------------------------------------------------
    @cached_property
    def assortment(self) -> pd.DataFrame:
        r = rng(self.seed, _ASSORTMENT)
        stores, products = self.stores, self.products
        grid = stores[["store_id"]].merge(products[["sku_id", "is_fresh"]], how="cross")
        carried = r.random(len(grid)) < np.where(grid["is_fresh"], 0.95, 0.85)
        grid = grid.loc[carried].reset_index(drop=True)
        n = len(grid)
        # Each store has its own delivery rhythm for ambient goods; fresh is daily, next day.
        store_lead = dict(zip(stores["store_id"], r.choice([2, 3], size=len(stores)), strict=True))
        store_review = dict(
            zip(
                stores["store_id"],
                r.choice([2, 3, 7], size=len(stores), p=[0.3, 0.4, 0.3]),
                strict=True,
            )
        )
        fresh = grid["is_fresh"].to_numpy()
        grid["lead_time_days"] = np.where(fresh, 1, grid["store_id"].map(store_lead)).astype(int)
        grid["review_period_days"] = np.where(fresh, 1, grid["store_id"].map(store_review)).astype(
            int
        )
        # The incumbent ordering system's stocking generosity: below 1 = lean shelves and
        # frequent stockouts, above 1 = overstock. Hidden from the pipeline.
        grid["_cover_factor"] = np.where(
            fresh, r.uniform(1.0, 1.5, size=n), r.uniform(0.85, 1.6, size=n)
        )
        grid["_review_offset"] = r.integers(0, 7, size=n)
        return grid.drop(columns="is_fresh")

    # ---- calendar ---------------------------------------------------------------------------
    @cached_property
    def calendar(self) -> pd.DataFrame:
        frames = []
        dates = self.all_dates
        for c in self.countries:
            md = list(zip(dates.month, dates.day, strict=True))
            frames.append(
                pd.DataFrame(
                    {
                        "country_code": c.code,
                        "date": dates.date,
                        "is_holiday": [m_d in c.holidays for m_d in md],
                        "is_payday": np.isin(dates.day, c.paydays),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    # ---- weather ----------------------------------------------------------------------------
    @cached_property
    def weather(self) -> pd.DataFrame:
        frames = []
        dates = self.all_dates
        doy = dates.dayofyear.to_numpy()
        n_hist = self.preset.n_days
        for ci, c in enumerate(self.countries):
            r = rng(self.seed, _WEATHER, ci)
            peak = 15 if c.southern_hemisphere else 200  # warmest day of the year
            seasonal = c.mean_temp_c + c.temp_amplitude_c * np.cos(
                2 * np.pi * (doy - peak) / 365.25
            )
            # Weather is "sticky": an AR(1) anomaly makes warm spells last several days.
            anomaly = np.zeros(len(dates))
            shocks = r.normal(0, 1.8, size=len(dates))
            for i in range(1, len(dates)):
                anomaly[i] = 0.75 * anomaly[i - 1] + shocks[i]
            temp = seasonal + anomaly
            rainy = r.random(len(dates)) < c.rainy_day_share
            rain = np.where(rainy, r.gamma(1.2, 6.0, size=len(dates)), 0.0)
            # Forecasts issued the day before: actual plus forecast error.
            temp_fc = temp + r.normal(0, 1.5, size=len(dates))
            rain_fc = np.clip(rain + r.normal(0, 2.0, size=len(dates)), 0, None)
            observed = np.arange(len(dates)) < n_hist
            frames.append(
                pd.DataFrame(
                    {
                        "country_code": c.code,
                        "date": dates.date,
                        "temp_c": np.where(observed, np.round(temp, 1), np.nan),
                        "rain_mm": np.where(observed, np.round(rain, 1), np.nan),
                        "temp_forecast_c": np.round(temp_fc, 1),
                        "rain_forecast_mm": np.round(rain_fc, 1),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    # ---- promotions -------------------------------------------------------------------------
    @cached_property
    def promo_matrix(self) -> dict[str, np.ndarray]:
        """Per country: discount fraction, shape (len(all_dates), n_skus). 0 = no promo."""
        n_dates, n_skus = len(self.all_dates), len(self.products)
        out: dict[str, np.ndarray] = {}
        years = n_dates / 365.0
        for ci, c in enumerate(self.countries):
            r = rng(self.seed, _PROMOS, ci)
            m = np.zeros((n_dates, n_skus))
            promotable = r.random(n_skus) < 0.45
            for k in np.flatnonzero(promotable):
                for _ in range(r.poisson(5 * years)):
                    start = int(r.integers(0, n_dates))
                    length = int(r.choice([7, 14]))
                    disc = float(r.choice([0.10, 0.15, 0.20, 0.25, 0.30]))
                    m[start : start + length, k] = np.maximum(m[start : start + length, k], disc)
            out[c.code] = m
        return out

    @cached_property
    def promotions(self) -> pd.DataFrame:
        frames = []
        skus = self.products["sku_id"].to_numpy()
        dates = self.all_dates.date
        for code, m in self.promo_matrix.items():
            d_idx, k_idx = np.nonzero(m)
            frames.append(
                pd.DataFrame(
                    {
                        "country_code": code,
                        "sku_id": skus[k_idx],
                        "date": dates[d_idx],
                        "discount_pct": np.round(m[d_idx, k_idx], 2),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    # ---- list prices ------------------------------------------------------------------------
    @cached_property
    def price_matrix(self) -> np.ndarray:
        """List price per (date, sku). ~30% of SKUs get one price increase during the history."""
        r = rng(self.seed, _PRICES)
        base = self.products["list_price"].to_numpy()
        n_dates = len(self.all_dates)
        m = np.tile(base, (n_dates, 1))
        for k in range(len(base)):
            if r.random() < 0.30:
                day = int(r.integers(60, self.preset.n_days))
                m[day:, k] = np.round(base[k] * r.uniform(1.03, 1.12), 1) - 0.01
        return m

    @cached_property
    def prices(self) -> pd.DataFrame:
        m = self.price_matrix
        dates = self.all_dates.date
        rows = []
        for k, sku in enumerate(self.products["sku_id"]):
            change_days = np.flatnonzero(np.diff(m[:, k], prepend=np.nan) != 0)
            rows.extend(
                {"sku_id": sku, "effective_from": dates[d], "list_price": m[d, k]}
                for d in change_days
            )
        return pd.DataFrame(rows)


@lru_cache(maxsize=4)
def get_world(preset_name: str, seed: int, horizon_days: int) -> World:
    """Cached per Python process, so a Spark worker builds the world once per job."""
    return World(PRESETS[preset_name], seed, horizon_days)


def public(df: pd.DataFrame) -> pd.DataFrame:
    """Drop simulation-internal columns (prefixed ``_``) before publishing a table."""
    return df[[c for c in df.columns if not c.startswith("_")]]


# Format and categorical constants re-exported for the simulator.
__all__ = ["FORMAT_SHARE", "STORE_FORMATS", "World", "get_world", "public", "rng"]
