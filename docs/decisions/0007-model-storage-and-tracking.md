# 0007: Models in a Delta table, runs in MLflow

**Status:** accepted

## Context
A run produces one model pair per group: dozens locally, thousands at chain scale. They must
be loaded by Spark tasks, traced to the run that made them, and compared across runs.

## Decision
* **Models** are stored in the gold `models` Delta table, one row per group and kind, as
  LightGBM's text format, with the producing MLflow `run_id` and a timestamp. Scoring tasks
  receive exactly their group's models through `cogroup`. Nothing is broadcast from the driver
  and nothing is downloaded at scoring time, which serverless UDFs could not do anyway.
* **MLflow** tracks every `backtest` and `train` run: parameters (preset, horizon, model
  settings...), headline metrics (WAPE, bias, coverage, service level per policy), and
  artifacts (metric tables, feature importance, model-group summary). Locally that is a SQLite
  file (`data/mlflow.db`, viewable with `uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db`);
  on Databricks it is the workspace experiment created by the bundle.

## Alternatives considered
* **One MLflow model per group** in the registry: thousands of registry entries per run.
* **One pyfunc wrapping all groups**: registry-friendly, but a single huge artifact that every
  scoring task would have to download; that does not scale, and serverless UDFs have no internet access.

## Consequences
* "Promoting" a model is a table version: Delta time travel restores any earlier model set.
* A Unity Catalog model registry entry (for governance, approvals, lineage) is a next step:
  register a lightweight pyfunc that *points to* a `models` table version.
