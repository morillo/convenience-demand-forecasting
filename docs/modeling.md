# Modelling, evaluation and replenishment

This is the technical walk-through. The "why" behind each choice is in [decisions/](decisions/README.md);
the plain-English version is in [overview.md](overview.md).

## 1. From raw data to a clean daily panel (`prepare`)
* The **inventory snapshots** define which store x SKU x days exist (the product is on the shelf);
  point-of-sale data only has days with sales, so missing days become `units_sold = 0`.
* List prices (valid-from intervals) and planned promotions are joined; `unit_price` = list price x (1 - discount).
* **Stockouts are inferred**: `is_stockout = on_hand_close == 0`. On the small preset 14% of store x SKU
  days end with an empty shelf; for fresh food it is about 40%.
* **Quality gates** stop the pipeline on duplicate keys, gaps in a daily series (which would shift
  every lag), negative quantities, or sales the inventory system does not know about.

## 2. Features (`features`)
One row per store x SKU x day, for the history **and** the next 14 days, grouped by `model_group`
(default `country|aisle`). The leakage rule: every sales-based feature ends at least 14 days before
the target day ([0004](decisions/0004-direct-features-min-lag.md)).

| Family | Features |
|---|---|
| Lags (sales on that exact earlier day) | `lag_14`, `lag_21`, `lag_28`, `lag_35`, `lag_42`, `lag_364` |
| Rolling statistics (windows ending 14 days back) | `roll_mean_7`, `roll_mean_28`, `roll_mean_91`, `roll_std_28`, `same_weekday_mean_4w`, `zero_share_28`, `stockout_share_28`, `promo_mean_91`, `nonpromo_mean_91` |
| Price and promotion (known in advance) | `on_promo`, `discount_pct`, `list_price`, `price_vs_recent` |
| Calendar (known in advance) | `day_of_week`, `day_of_month`, `month`, `week_of_year`, `is_holiday`, `is_holiday_eve`, `is_after_holiday`, `is_payday_window` |
| Weather forecast (known the day before) | `temp_forecast_c`, `rain_forecast_mm` |
| Lifecycle and logistics | `days_since_launch`, `is_fresh`, `lead_time_days` |
| Categorical | `store_format_code`, `category_code` (stable codes from `domain.py`) |

Sales on stockout days are **blanked** before computing lags and rolling statistics, so a stockout does not
masquerade as low demand. LightGBM handles the resulting missing values natively. That is also how new
products and stores are forecast: their lag features are empty, and the model leans on category, format,
price, calendar and `days_since_launch`.

## 3. Model (`train`, and inside `backtest`)
Per `(cut-off date, model_group)`, as an independent Spark task ([0002](decisions/0002-spark-plus-lightgbm-global-models.md)):

1. **Mean model**, LightGBM with a Tweedie objective ([0003](decisions/0003-tweedie-and-quantile.md)).
   The number of trees is chosen by early stopping on the last 28 days of clean data.
2. **Demand unconstraining** ([0005](decisions/0005-stockout-censoring.md)): on stockout days the
   target is replaced by `E[demand | demand >= sales]` under the model's current estimate, and the model is
   refit on all days; twice.
3. **Upper model**, LightGBM quantile objective at P90, trained with stockout days filled by draws from
   the same truncated distribution.

Default settings (`ModelSettings` in `config.py`): learning rate 0.05, 63 leaves, at most 400 trees,
early stopping after 30 rounds, feature and row subsampling 0.8, a fixed seed, deterministic mode.
Groups with fewer than 200 usable rows get a 28-day moving-average forecast instead (`method = fallback_moving_average`).

## 4. Evaluation (`backtest`)
Three cut-off dates, 14 days apart, at the end of the history; for each, train on data up to that day and
forecast the next 14 days. Two baselines are scored identically ([0008](decisions/0008-evaluation.md)).
Metrics are defined in [glossary.md](glossary.md) and implemented in `pipeline/metrics.py`.

## 5. Replenishment (`backtest` what-if and `forecast`)
Order-up-to policy with safety stock, calibrated per aisle on the backtest ([0009](decisions/0009-replenishment-policy.md)):

```
protection days P   = lead time + review period            (fresh: review period)
expected demand     = sum of daily mean forecasts, days 1..P
uncertainty         = sqrt( sum over days of ((P90 - mean) / 1.28)^2 )
safety stock        = 1.64 (95% service) x uncertainty x aisle safety factor
order-up-to level   = expected demand + safety stock
recommended order   = max(0, ceil(order-up-to level - on hand - on order))
```

## Results

`small` preset, seed 42, 24 stores x 120 SKUs x 730 days (1.55 million store x SKU x days), 3 backtest
cut-offs x 14 days. Reproduce with `uv run dfc run-all --preset small` (about 2 minutes on an Apple Silicon
laptop); every number below is in the `gold.backtest_metrics` / `gold.policy_evaluation` tables and in MLflow.

### Forecast accuracy

| Forecast | WAPE vs **true demand** (all days) | Bias vs true demand | WAPE vs observed sales (clean days) | P90 coverage |
|---|---|---|---|---|
| **LightGBM** | **0.593** | **-4.0%** | 0.634 | 92.2% (target 90%) |
| 28-day moving average | 0.636 | -12.9% | 0.636 | n/a |
| Same weekday last week | 0.766 | -8.3% | 0.834 | n/a |

How to read this:

* **Daily store x SKU error is large for every method.** Close to 40% of days sell nothing and most items
  sell a handful a day, so randomness dominates. Even a forecast that knew the simulator's exact expected
  demand would score a WAPE of about 0.63 against observed sales. The model is at that noise floor.
* **Bias is where the model earns its keep.** The moving average under-forecasts true demand by 13% because it
  learns from censored sales; the model is within 4%.
* **The "observed sales" column is biased**, because it only scores the days that did not sell out, which
  are the low-demand days. It is shown because it is what a real chain could compute; see
  [0005](decisions/0005-stockout-censoring.md).
* Error is nearly flat across the horizon: WAPE vs true demand 0.57-0.61 from day 1 to day 14.

By aisle (WAPE vs true demand, lower is better):

| Aisle | LightGBM | Moving average | Same weekday |
|---|---|---|---|
| Fresh food | **0.435** | 0.532 | 0.519 |
| Hot beverages | **0.448** | 0.469 | 0.595 |
| Cold beverages | **0.604** | 0.647 | 0.841 |
| Dairy & grocery | **0.722** | 0.726 | 0.930 |
| Snacks | **0.831** | 0.834 | 1.065 |
| Frozen | **1.159** | 1.176 | 1.377 |
| Health & personal care | 1.318 | **1.258** | 1.442 |
| Household | 1.588 | **1.471** | 1.673 |

The gains are largest where stockouts and drivers matter most (fresh food, beverages). On the slowest
movers (household, health & personal care: a few units a *week* per store) the model is slightly
**worse** than a moving average. Options there: route very slow series to the moving average or a
Croston/TSB intermittent-demand method, or pool them into a chain-level model. Listed as a next step.

### Replenishment what-if
Last cut-off, held out from the safety-stock calibration; 2,368 store x SKU ordering cycles; true demand as the judge.

| Policy | Cycles without stockout (target 95%) | Demand served (fill rate) | Avg units short per cycle | Avg excess units per cycle | Stock target (units) |
|---|---|---|---|---|---|
| **Order-up-to with LightGBM, calibrated** | **95.3%** | **98.8%** | 0.18 | 9.2 | 55,195 |
| Order-up-to with LightGBM, textbook safety stock | 91.4% | 97.8% | 0.31 | 7.3 | 50,218 |
| Order-up-to with 28-day moving average | 47.3% | 81.9% | 2.58 | 1.8 | 31,834 |
| Order-up-to with same weekday last week | 54.0% | 79.2% | 2.96 | 2.0 | 31,549 |
| *Incumbent system, actual simulated operation (backtest period)* | n/a | *90.5%* | n/a | n/a | n/a |

* The forecast-driven policy hits the service target, and would have served about 99% of demand, against 82%
  for the same policy driven by a moving average and 91% for the incumbent.
* **The cost is stock**: about 73% more units on the shelf than the moving-average policy, which never adds a
  safety buffer for its own bias. Whether that trade is right depends on the margin lost per stockout versus
  holding and waste costs, which the business must supply. The service level is one setting
  (`forecast.service_level`), and it can be set per category.
* Learned safety factors (95th percentile of the needed multiplier per aisle): 1.22-1.48x for most aisles,
  2.57x for household, the slow movers where the normal approximation is weakest.

### Known limitations
* Synthetic data: these numbers validate the pipeline and the method, not a specific chain's gains.
* Poisson approximation in the unconstraining step; slow movers keep a residual bias (health & personal care -12%).
* The policy evaluation is a single-cycle approximation that ignores stock arriving mid-cycle.
* No price optimisation, substitution between products, or cannibalisation by promotions of other SKUs.
