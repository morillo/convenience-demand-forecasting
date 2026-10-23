# Architecture decision records

Each file records one design decision: the situation, what we chose, what we rejected and
why, and what it costs us. They are short on purpose. When a decision changes, add a new
record that supersedes the old one instead of rewriting history.

| # | Decision |
|---|---|
| [0001](0001-local-first-mirroring-serverless.md) | Develop locally on a stack that mirrors Databricks serverless |
| [0002](0002-spark-plus-lightgbm-global-models.md) | Spark for data, one LightGBM model per group of series |
| [0003](0003-tweedie-and-quantile.md) | Tweedie for the expected demand, a quantile model for the upper bound |
| [0004](0004-direct-features-min-lag.md) | One model for all horizons: every sales feature lags by at least the horizon |
| [0005](0005-stockout-censoring.md) | Treat stockout days as censored demand and unconstrain them with EM |
| [0006](0006-storage.md) | Delta Lake medallion tables, path-based locally, Unity Catalog on Databricks |
| [0007](0007-model-storage-and-tracking.md) | Models in a Delta table, runs in MLflow |
| [0008](0008-evaluation.md) | Rolling-origin backtest against simple baselines, scored against true demand |
| [0009](0009-replenishment-policy.md) | Order-up-to policy with an empirically calibrated safety stock |
| [0010](0010-synthetic-data.md) | A deterministic, store-parallel simulator for the mock data |
