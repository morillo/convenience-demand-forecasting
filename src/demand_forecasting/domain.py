"""Business vocabulary of the mock convenience-store chain.

Everything here is *made up but plausible*: the countries, aisles, categories and the
behavioural knobs (how much a category reacts to heat, weekends, promotions...) that the
synthetic data generator uses. Keeping them in one place makes the simulation easy to read
and lets the feature code reuse the same stable integer codes for categorical columns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    southern_hemisphere: bool
    mean_temp_c: float
    temp_amplitude_c: float
    rainy_day_share: float
    # Fixed-date public holidays as (month, day). Simplified on purpose: real calendars have
    # moving holidays; the model only needs "some days per country behave differently".
    holidays: tuple[tuple[int, int], ...]
    # Paydays (day of month) drive a small lift in discretionary categories.
    paydays: tuple[int, ...]


# fmt: off
COUNTRIES: tuple[Country, ...] = (
    Country("US", "United States", False, 14.0, 11.0, 0.30, ((1, 1), (7, 4), (11, 27), (12, 25)), (1, 15)),
    Country("JP", "Japan", False, 16.0, 10.0, 0.33,
            ((1, 1), (1, 2), (1, 3), (4, 29), (5, 3), (5, 4), (5, 5), (8, 14), (8, 15)), (25,)),
    Country("TH", "Thailand", False, 28.5, 2.5, 0.40, ((1, 1), (4, 13), (4, 14), (4, 15), (12, 5)), (1, 30)),
    Country("MX", "Mexico", False, 18.0, 4.0, 0.25, ((1, 1), (9, 16), (11, 2), (12, 12), (12, 25)), (15, 30)),
    Country("AU", "Australia", True, 18.0, 6.0, 0.28, ((1, 1), (1, 26), (4, 25), (12, 25), (12, 26)), (1, 15)),
    Country("KR", "South Korea", False, 13.0, 13.0, 0.28, ((1, 1), (3, 1), (5, 5), (8, 15), (10, 3)), (10, 25)),
    Country("CA", "Canada", False, 7.0, 15.0, 0.33, ((1, 1), (7, 1), (12, 25), (12, 26)), (1, 15)),
    Country("TW", "Taiwan", False, 23.0, 6.0, 0.38, ((1, 1), (2, 28), (4, 4), (10, 10)), (5,)),
    Country("MY", "Malaysia", False, 27.5, 1.0, 0.45, ((1, 1), (5, 1), (8, 31), (9, 16)), (25,)),
    Country("PH", "Philippines", False, 27.5, 1.5, 0.45, ((1, 1), (4, 9), (6, 12), (12, 25), (12, 30)), (15, 30)),
    Country("SG", "Singapore", False, 27.5, 1.0, 0.42, ((1, 1), (5, 1), (8, 9), (12, 25)), (25,)),
    Country("NO", "Norway", False, 6.0, 9.0, 0.40, ((1, 1), (5, 1), (5, 17), (12, 25), (12, 26)), (20,)),
)
# fmt: on

# Store formats change *when* people shop: office stores are quiet at weekends, highway
# stores spike on holidays, transit stores follow commuter days.
STORE_FORMATS: tuple[str, ...] = ("urban", "suburban", "highway", "office", "transit")

# Weekday (Mon=0..Sun=6) multipliers per store format.
FORMAT_WEEKDAY_PROFILE: dict[str, tuple[float, ...]] = {
    "urban": (1.00, 1.00, 1.00, 1.02, 1.10, 1.15, 1.00),
    "suburban": (0.95, 0.95, 0.97, 1.00, 1.08, 1.20, 1.12),
    "highway": (0.92, 0.90, 0.92, 0.98, 1.15, 1.20, 1.18),
    "office": (1.15, 1.18, 1.18, 1.15, 1.10, 0.45, 0.35),
    "transit": (1.12, 1.12, 1.12, 1.12, 1.10, 0.70, 0.60),
}

# How strongly a store format reacts to public holidays (highway: road trips, office: closed).
FORMAT_HOLIDAY_LIFT: dict[str, float] = {
    "urban": 1.05,
    "suburban": 1.10,
    "highway": 1.35,
    "office": 0.40,
    "transit": 0.75,
}


@dataclass(frozen=True)
class Category:
    name: str
    aisle: str
    skus_weight: float  # share of the assortment that belongs to this category
    base_rate: tuple[float, float]  # (low, high) mean units/day per SKU in an average store
    price: tuple[float, float]  # (low, high) list price, USD-equivalent
    temp_sensitivity: float  # demand change per +1 C above the country mean (fraction)
    rain_sensitivity: float  # demand change on a rainy day (fraction)
    annual_amplitude: float  # strength of the yearly cycle beyond temperature
    promo_elasticity: float  # lift = (1 - discount) ** -elasticity
    holiday_sensitivity: float  # 1 = follows the store's holiday lift fully, 0 = ignores it
    payday_lift: float
    dispersion: float  # negative-binomial shape: lower = burstier, more zero days
    fresh: bool  # fresh items are delivered daily and unsold units are written off


# fmt: off
CATEGORIES: tuple[Category, ...] = (
    Category("soft_drinks", "cold_beverages", 0.08, (2.0, 9.0), (1.2, 2.8), 0.030, -0.05, 0.10, 1.8, 0.6, 0.03, 3.0, False),
    Category("water", "cold_beverages", 0.04, (2.0, 8.0), (0.8, 1.8), 0.035, -0.10, 0.10, 1.2, 0.6, 0.00, 4.0, False),
    Category("energy_drinks", "cold_beverages", 0.04, (1.0, 5.0), (2.0, 3.5), 0.010, 0.00, 0.00, 1.6, 0.2, 0.05, 2.5, False),
    Category("juices", "cold_beverages", 0.04, (0.6, 3.0), (1.5, 3.5), 0.015, 0.00, 0.05, 1.5, 0.4, 0.02, 2.0, False),
    Category("coffee", "hot_beverages", 0.03, (6.0, 25.0), (1.5, 4.5), -0.020, 0.08, 0.05, 0.8, 0.3, 0.00, 6.0, False),
    Category("tea", "hot_beverages", 0.02, (1.0, 5.0), (1.2, 3.0), -0.015, 0.06, 0.05, 0.8, 0.3, 0.00, 3.0, False),
    Category("chips", "snacks", 0.07, (0.8, 4.0), (1.5, 4.0), 0.000, 0.03, 0.05, 1.9, 0.7, 0.04, 1.8, False),
    Category("candy", "snacks", 0.07, (0.5, 3.5), (0.8, 2.5), -0.005, 0.03, 0.15, 2.0, 0.8, 0.05, 1.5, False),
    Category("gum_mints", "snacks", 0.03, (0.4, 2.5), (0.8, 2.0), 0.000, 0.00, 0.00, 1.2, 0.2, 0.02, 1.5, False),
    Category("nuts", "snacks", 0.02, (0.2, 1.2), (2.0, 5.0), 0.000, 0.00, 0.05, 1.6, 0.6, 0.03, 1.2, False),
    Category("sandwiches", "fresh_food", 0.04, (2.0, 8.0), (3.0, 6.5), 0.000, 0.00, 0.00, 1.4, 0.2, 0.03, 10.0, True),
    Category("hot_food", "fresh_food", 0.03, (3.0, 12.0), (1.5, 4.5), -0.010, 0.10, 0.05, 1.3, 0.4, 0.04, 10.0, True),
    Category("bakery", "fresh_food", 0.03, (1.5, 6.0), (1.0, 3.0), 0.000, 0.03, 0.00, 1.5, 0.3, 0.02, 8.0, True),
    Category("salads_fruit", "fresh_food", 0.02, (0.5, 3.0), (2.5, 5.5), 0.012, -0.05, 0.10, 1.3, 0.1, 0.01, 8.0, True),
    Category("ice_cream", "frozen", 0.04, (0.3, 3.0), (1.5, 5.0), 0.060, -0.20, 0.30, 1.8, 0.5, 0.02, 1.2, False),
    Category("frozen_meals", "frozen", 0.03, (0.2, 1.5), (3.0, 7.0), -0.010, 0.05, 0.05, 1.5, 0.2, 0.02, 1.2, False),
    Category("ice", "frozen", 0.01, (0.5, 4.0), (1.5, 3.5), 0.080, -0.30, 0.20, 0.5, 0.8, 0.00, 1.0, False),
    Category("milk", "dairy_grocery", 0.03, (1.0, 5.0), (1.2, 3.5), 0.000, 0.00, 0.00, 1.0, 0.4, 0.01, 4.0, False),
    Category("bread", "dairy_grocery", 0.02, (0.8, 3.5), (1.5, 3.5), 0.000, 0.00, 0.00, 1.0, 0.4, 0.01, 3.0, False),
    Category("eggs", "dairy_grocery", 0.01, (0.3, 1.5), (2.5, 4.5), 0.000, 0.00, 0.00, 1.0, 0.5, 0.01, 2.0, False),
    Category("canned_goods", "dairy_grocery", 0.03, (0.1, 0.8), (1.5, 4.0), 0.000, 0.00, 0.05, 1.2, 0.3, 0.01, 1.0, False),
    Category("otc_medicine", "health_personal_care", 0.04, (0.1, 0.8), (4.0, 10.0), -0.004, 0.02, 0.25, 0.4, 0.2, 0.00, 1.0, False),
    Category("toiletries", "health_personal_care", 0.04, (0.1, 0.7), (2.0, 7.0), 0.000, 0.00, 0.00, 0.8, 0.4, 0.01, 0.8, False),
    Category("batteries", "household", 0.02, (0.05, 0.4), (3.0, 9.0), 0.000, 0.00, 0.00, 1.0, 0.6, 0.00, 0.6, False),
    Category("phone_accessories", "household", 0.03, (0.05, 0.3), (5.0, 20.0), 0.000, 0.00, 0.00, 1.0, 0.6, 0.02, 0.6, False),
    Category("umbrellas", "household", 0.01, (0.02, 0.15), (5.0, 12.0), 0.000, 3.00, 0.00, 0.5, 0.3, 0.00, 0.5, False),
    Category("paper_products", "household", 0.03, (0.1, 0.8), (1.5, 6.0), 0.000, 0.00, 0.00, 1.2, 0.5, 0.01, 1.0, False),
)
# fmt: on

AISLES: tuple[str, ...] = tuple(dict.fromkeys(c.aisle for c in CATEGORIES))

# Format x category affinity: office workers buy coffee and sandwiches, drivers buy energy
# drinks and snacks. Missing pairs mean 1.0.
FORMAT_CATEGORY_AFFINITY: dict[tuple[str, str], float] = {
    ("office", "coffee"): 1.5,
    ("office", "sandwiches"): 1.4,
    ("office", "salads_fruit"): 1.4,
    ("office", "paper_products"): 0.6,
    ("highway", "energy_drinks"): 1.5,
    ("highway", "chips"): 1.3,
    ("highway", "ice"): 1.4,
    ("highway", "phone_accessories"): 1.6,
    ("transit", "coffee"): 1.3,
    ("transit", "gum_mints"): 1.4,
    ("transit", "umbrellas"): 1.8,
    ("suburban", "milk"): 1.4,
    ("suburban", "bread"): 1.4,
    ("suburban", "eggs"): 1.4,
    ("suburban", "paper_products"): 1.3,
}

# Stable integer codes for categorical model features. Appending new values is safe; never
# reorder existing ones, or saved models would read old codes with new meanings.
STORE_FORMAT_CODE: dict[str, int] = {f: i for i, f in enumerate(STORE_FORMATS)}
CATEGORY_CODE: dict[str, int] = {c.name: i for i, c in enumerate(CATEGORIES)}
