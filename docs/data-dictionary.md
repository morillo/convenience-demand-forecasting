# Data dictionary

Locally every table is a Delta folder `data/lake/<layer>/<table>`; on Databricks it is the Unity
Catalog table `<catalog>.<schema>.<layer>_<table>` (e.g. `workspace.demand_forecasting.gold_forecasts`).
All dates are calendar days in the store's country. Keys are in **bold**.

## Bronze: as delivered by source systems (here: the generator)

| Table | Grain | Columns |
|---|---|---|
| `stores` | store | **store_id**, country_code, store_format (urban, suburban, highway, office, transit), open_date, traffic_index |
| `products` | SKU | **sku_id**, sku_name, aisle, category, list_price (at the start), launch_date, delist_date (null = still listed), is_fresh |
| `assortment` | store x SKU | **store_id, sku_id**, lead_time_days, review_period_days |
| `calendar` | country x day (history + horizon) | **country_code, date**, is_holiday, is_payday |
| `weather` | country x day (history + horizon) | **country_code, date**, temp_c, rain_mm (observed; null in the future), temp_forecast_c, rain_forecast_mm (issued the day before) |
| `promotions` | country x SKU x promo day | **country_code, sku_id, date**, discount_pct (0.10 = 10% off) |
| `prices` | SKU x price change | **sku_id, effective_from**, list_price |
| `pos_sales` | store x SKU x day **with sales** | **date, store_id, sku_id**, units_sold, net_sales_amount |
| `inventory_snapshots` | store x SKU x day on the shelf | **date, store_id, sku_id**, on_hand_close (closing stock before write-off), units_received, units_wasted (fresh write-off) |
| `purchase_orders` | order | **store_id, sku_id, order_date**, expected_arrival_date, order_qty |
| `simulation_truth` | store x SKU x day | **date, store_id, sku_id**, expected_demand, demand (true, uncensored), plus the operational columns. **Simulation only; used for evaluation, never for training.** |

## Silver

| Table | Grain | Columns |
|---|---|---|
| `daily_panel` | store x SKU x day, gap-free | keys; store attributes (country_code, store_format, store_open_date); product attributes (aisle, category, is_fresh, launch_date, delist_date); lead_time_days, review_period_days; units_sold (0 on days without sales); on_hand_close, units_received, units_wasted; list_price, discount_pct, on_promo, unit_price; **is_stockout** (on_hand_close = 0) |

## Gold

| Table | Grain | Columns |
|---|---|---|
| `features` | store x SKU x day, history + 14 future days | every `daily_panel` column, `is_future`, the model features listed in [modeling.md](modeling.md#2-features-features), and `model_group` |
| `models` | model group x kind | origin_date (trained through), **model_group, model_kind** (mean / upper), model_string (LightGBM text format), best_iteration, n_train_rows, feature_importance (JSON, gain share), params (JSON), run_id (MLflow), trained_at |
| `forecasts` | store x SKU x future day | origin_date, model_group, **store_id, sku_id, date**, horizon (days ahead), forecast_mean, forecast_upper (P90), method (lightgbm / fallback_moving_average), model_run_id |
| `replenishment_plan` | store x SKU | **store_id, sku_id**, sku_name, aisle, category, is_fresh, origin_date, protection_days, expected_demand, demand_sigma, safety_factor, safety_stock, order_up_to_level, on_hand, on_order, **recommended_order_qty**, service_level_target, model_run_id |
| `backtest_models` | cut-off x model group x kind | as `models`, one set per backtest cut-off (audit trail) |
| `backtest_predictions` | cut-off x store x SKU x day x forecast method | origin_date, keys, horizon, model (lightgbm / moving_average_28 / seasonal_naive), forecast_mean, forecast_upper, units_sold, is_stockout, true_demand, country_code, aisle, run_id |
| `backtest_metrics` | level x value x model | level (overall / origin_date / country / aisle / horizon), level_value, model, n_rows, wape, bias, mae, pinball_upper, coverage_upper, wape_vs_true_demand, run_id |
| `policy_cycles` | policy x cut-off x store x SKU | policy, origin_date, keys, aisle, expected_demand, safety_stock, order_up_to_level, true_demand (over the protection period) |
| `policy_evaluation` | level x value x policy | level (all_origins, all_origins_by_aisle, last_origin, last_origin_by_aisle), level_value, policy, cycles, cycle_service_level, fill_rate, avg_excess_units, avg_short_units, stock_target_units, target_service_level, run_id |
| `safety_stock_calibration` | aisle | **aisle**, raw_factor, safety_factor (bounded 0.5-4), cycles, run_id |
