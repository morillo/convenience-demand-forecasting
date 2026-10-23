# 0001: Develop locally on a stack that mirrors Databricks serverless

**Status:** accepted

## Context
The prototype is built on a laptop but must later run on Databricks, first on the Free
Edition. Free Edition offers **serverless compute only** (no classic clusters, no choice of
ML runtime) and throttles usage: going over the daily quota shuts compute down until the
next day. Serverless has its own rules: only the Spark Connect API (no RDDs, no
`sparkContext`), most `spark.conf` settings are blocked, Python UDFs are capped at
**1 GB of memory**, UDFs have no internet access, and Maven/JAR libraries are not allowed.

## Decision
Mirror the target as closely as a laptop allows, so "works locally" predicts "works on Databricks":

* **Same versions** as serverless environment 6 (Sep 2026): Python 3.12, Spark 4.2 (Databricks
  Runtime 19), and the library versions preinstalled in its ML base environment: lightgbm
  4.6.0, pandas 2.3.3, numpy 2.3.4, pyarrow 21.0.0, scipy 1.16.3, mlflow 3.12.0. They are
  pinned exactly in `pyproject.toml`; Dependabot is told not to bump them one by one.
* **Same API**: local Spark runs behind **Spark Connect** (`SparkSession.builder.remote("local[*]")`),
  the only mode serverless supports. Code that would fail on serverless fails locally first.
* **Same defaults**: ANSI SQL mode on, session time zone UTC.
* **Same limits**: training groups are size-checked against a row budget derived from the
  1 GB UDF cap before any pandas task starts (`forecast.max_rows_per_group`).
* **Pure-Python dependencies only**: the only JAR is Delta Lake, which Databricks provides.
* Everything environment-specific lives in a settings **profile** (`local` / `databricks`);
  pipeline code never branches on where it runs, except to obtain the Spark session.

## Alternatives considered
* **pandas/Polars locally, rewrite for Spark later.** Faster to start, but the rewrite is
  where migrations fail, and nothing would prove the Spark version works until it ran in the cloud.
* **Databricks Connect from the laptop to the workspace.** It runs the real engine, but every
  run burns the Free Edition quota and needs network access. We keep it as an option, not the default.
* **Classic local Spark session.** Allows APIs serverless rejects (RDDs, arbitrary configs),
  so problems would surface only after migration.

## Consequences
* Local runs need Java 17+ and, on macOS, Homebrew's `libomp` for LightGBM.
* Spark 4.2 marks strings coming out of pandas UDFs as `string collate UTF8_BINARY`. Delta records
  that in the schema and non-Spark readers (delta-rs, DuckDB, BI tools) refuse the table.
  `TableStore.write` re-casts strings to plain `string` on every write.
* Moving to a newer serverless environment means bumping the pinned set together, rerunning
  the tests, and updating `environment_version` in `resources/demand_forecasting.yml`.
