# 0006: Delta Lake medallion tables, path-based locally, Unity Catalog on Databricks

**Status:** accepted

## Context
The pipeline produces ~20 tables. They must be readable by Spark, by the results app, and
later by BI tools, and they must move to Unity Catalog without code changes.

## Decision
* **Delta Lake** everywhere (ACID writes, schema enforcement, time travel to any earlier version).
* **Medallion layers**: `bronze` = raw as delivered, `silver` = cleaned and conformed daily
  panel, `gold` = model inputs and business outputs.
* `TableStore` maps a logical table to its physical home. Locally that is
  `data/lake/<layer>/<name>` (a Delta folder); on Databricks it is
  `<catalog>.<schema>.<layer>_<name>` (one schema, layer as a name prefix, because Free Edition has one catalog
  and a schema per layer would triple the setup).
* Every write is a **full overwrite** (idempotent: rerunning a step gives the same result), with
  **liquid clustering** on the keys readers filter by (`store_id`, `date`, `model_group`).

## Alternatives considered
* **Parquet files**: no transactions, no schema evolution, no time travel.
* **A local Hive metastore** to use table names locally: extra moving parts (Derby, warehouse
  dirs) for no gain in a prototype.
* **Incremental `MERGE` per day**: what production needs (see the migration guide), but it makes
  every step stateful; full refresh keeps the prototype simple and reproducible.

## Consequences
* Delta time travel gives free audit history: "what did the forecast table say last Tuesday?".
* On a paid workspace with daily data feeds, the bronze and silver steps should become
  incremental (`MERGE` on the date partition) before data volume makes full refresh expensive.
