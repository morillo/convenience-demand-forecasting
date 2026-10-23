# Glossary

| Term | Meaning |
|---|---|
| **SKU** | Stock-keeping unit: one specific product (e.g. "Soft Drinks #012"). |
| **Series** | The daily sales of one SKU in one store. The chain has stores x SKUs of them. |
| **Aisle / category** | Product hierarchy: aisle (e.g. cold beverages) > category (e.g. water) > SKU. |
| **Horizon** | How many days ahead we forecast (14). "Days ahead" = 1 for tomorrow. |
| **Cut-off / origin** | The day a forecast is made. Only data up to that day may be used. |
| **Backtest** | Replaying the past: forecast from earlier cut-offs and compare with what happened. |
| **Baseline** | A simple rule the model must beat: *same weekday last week*, *28-day moving average*. |
| **Leakage** | Accidentally using information that would not exist at forecast time. It makes backtests look great and production fail. |
| **Feature** | An input to the model: a lag, a holiday flag, a price, the weather forecast. |
| **Lag** | A past value of the series, e.g. `lag_14` = sales 14 days before the target day. |
| **Stockout** | The shelf was empty at closing time, so some customers may not have found the product. |
| **Censored demand** | On stockout days, sales are a lower bound on demand ("at least 3"). |
| **Unconstraining / EM** | Estimating what demand probably was on censored days, refitting, and repeating (Expectation-Maximisation). |
| **True demand** | What customers wanted, including what they could not buy. Only known here because the data is simulated. |
| **Global model** | One model learning from many series together (e.g. all snacks in Japan). |
| **LightGBM** | A fast gradient-boosted decision tree library; the forecasting model. |
| **Tweedie** | A loss function suited to non-negative values with many zeros and occasional spikes. |
| **Quantile / P90** | A value demand stays below 90% of the time. Our "upper" forecast. |
| **Coverage** | How often actual demand was at or below the P90 forecast; should be about 90%. |
| **WAPE** | Weighted absolute percentage error = total absolute error / total actual units. 0.30 means "off by 30% of volume". |
| **Bias** | Total (forecast - actual) / total actual. Negative = forecasts too low (stockout risk). |
| **Pinball loss** | The standard error measure for quantile forecasts. |
| **Lead time** | Days between placing an order and the delivery arriving. |
| **Review period** | Days between ordering opportunities. |
| **Protection period** | Lead time + review period: how long stock ordered today must last (fresh: review period only). |
| **Safety stock** | Extra units held to absorb forecast error. |
| **Order-up-to level** | Target stock = expected demand over the protection period + safety stock. |
| **Service level (cycle)** | Share of ordering cycles with no stockout. Target 95%. |
| **Fill rate** | Share of demand actually served from stock. |
| **Incumbent** | The ordering system the (simulated) chain uses today, which only looks at recent sales. |
| **Medallion (bronze/silver/gold)** | Data layers: raw as received > cleaned and joined > model-ready and business outputs. |
| **Delta Lake** | Table format on files with transactions and version history ("time travel"). |
| **Spark / Spark Connect** | Distributed data engine / its client-server API, the only one Databricks serverless supports. |
| **Unity Catalog** | Databricks' governed catalog of tables, models and permissions. |
| **Serverless** | Databricks compute you do not configure or manage; the only kind on Free Edition. |
| **Asset Bundle** | Databricks' "infrastructure as code": `databricks.yml` defines jobs, schemas and experiments. |
| **MLflow** | Experiment tracking: records each run's settings, metrics and output files. |
| **Preset** | A named size for the mock chain: tiny, small, medium, large. |
