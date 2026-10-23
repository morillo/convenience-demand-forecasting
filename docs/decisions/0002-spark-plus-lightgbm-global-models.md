# 0002: Spark for data, one LightGBM model per group of series

**Status:** accepted

## Context
A global chain has tens of thousands of stores and thousands of SKUs per store, so
10^8 store x SKU series, forecast every day. The question we started with was "SparkML or scikit-learn?".

## Decision
Neither alone. **Spark** does everything that scales with data volume (simulation, joins,
features, metrics), and **LightGBM** (gradient-boosted trees) is the model. We train **one
*global* model per group of series**, by default per `country x aisle`, inside Spark with
`groupBy(...).applyInPandas(...)`: each group is an independent task that receives its rows
as a pandas DataFrame. Scoring uses `cogroup` so each task receives its features and its models together.

A "global" model learns from many series at once (all snacks in Japan, say). It can then
forecast slow sellers, new products and new stores, which have too little history for a
model of their own, from what similar series did.

## Alternatives considered
* **SparkML `GBTRegressor`**: distributed, but slow, no Tweedie or quantile objectives, and
  weaker than LightGBM on tabular data.
* **scikit-learn alone**: good models, but one machine cannot hold the data.
* **One model per series (ARIMA/ETS)**: 10^8 models, no sharing of information across
  series, and poor on intermittent demand. Useful later as a fallback for special cases.
* **One single global model**: does not fit in one task's memory at chain scale and cannot
  be trained in parallel.
* **Deep learning (DeepAR, TFT)**: needs GPUs, which are scarce and expensive, and rarely beats
  well-tuned gradient boosting on this kind of data. Worth testing later for promotion effects.

The M5 forecasting competition (Walmart store x SKU data) was won by global LightGBM models
with a Tweedie objective, which is public evidence for this design.

## Consequences
* Group size is the main scaling knob: `forecast.group_by` and `forecast.store_buckets` split
  groups until each fits the serverless 1 GB UDF limit, and the pipeline fails fast if one does not.
* Parallelism is "number of groups", so tiny presets do not use many cores. That is fine; large ones do.
* LightGBM runs single-threaded inside each task (`model.num_threads = 1`), because Spark
  already runs one task per core.
