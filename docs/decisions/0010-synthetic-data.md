# 0010: A deterministic, store-parallel simulator for the mock data

**Status:** accepted

## Context
We need realistic data for a chain we cannot access, at sizes from "unit test" to "billions
of rows", with the known traps of real retail data, and the ground truth that real data never has.

## Decision
* A **`World`** (stores, products, assortment, calendar, weather, promotions, prices) is a pure function of
  `(preset, seed, horizon)`. Random numbers come from independent streams per purpose and entity,
  so adding a store never changes another store's data.
* **Facts are simulated per store in parallel** (`applyInPandas` by `store_id`): each task
  rebuilds the same world (cached per worker) and simulates its own store. Nothing large is
  shipped from the driver, so this scales with the cluster.
* Demand includes the effects a model must learn (see `docs/mock-data.md`), and an
  **incumbent ordering system** that learns only from sales. Its stockouts and fresh-food waste are the
  problem the forecast must fix.
* Operational tables hide what real systems do not record (true demand, lost sales). The
  `simulation_truth` table keeps it, for evaluation only.

## Alternatives considered
* **A public dataset (M5 / Favorita)**: real, but fixed in size, no inventory, no true demand, and licence constraints.
* **Simple random series**: no stockouts, no promotions, no reason for a model to beat a moving average.

## Consequences
* Model accuracy here is evidence that the *pipeline* works, not a promise about any real chain.
* Presets go from `tiny` (CI, seconds) to `large` (~3 billion rows, paid Databricks only).
