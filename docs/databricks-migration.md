# Moving to Databricks

The pipeline is already packaged for Databricks: the bundle (`databricks.yml` +
`resources/demand_forecasting.yml`) builds the wheel and deploys one **serverless job** that runs the
same `dfc` steps. Nothing in the pipeline code changes; the `databricks` settings profile switches
storage to Unity Catalog and tracking to the workspace MLflow.

## What the bundle creates

| Resource | Name (target `free`, development mode) |
|---|---|
| Unity Catalog schema | `workspace.dev_<you>_demand_forecasting` (dev mode prefixes your user name) |
| MLflow experiment | `/Users/<you>/demand-forecasting-free` |
| Job | `[dev <you>] demand-forecasting-pipeline`: generate, prepare, features, then backtest and train in parallel, then forecast |
| Environment | serverless environment version 6 + the project wheel and its pinned dependencies |

Job parameters `catalog`, `schema`, `preset` and `experiment` can be changed per run in the UI
("Run now with different parameters") without redeploying.

## Step by step (Free Edition)

Prerequisites: the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) (>= 1.0) and `uv`.

1. **Log in** (opens the browser; creates or refreshes a CLI profile):
   ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile free-edition
   ```
2. **Validate** the bundle (checks YAML, variables and permissions; deploys nothing):
   ```bash
   databricks bundle validate -t free -p free-edition
   ```
3. **Deploy** (builds the wheel with `uv build`, uploads it, creates the schema, experiment and job):
   ```bash
   databricks bundle deploy -t free -p free-edition
   ```
4. **Run** (streams the task states; the serverless runtime for `small` has not been measured yet):
   ```bash
   databricks bundle run demand_forecasting_pipeline -t free -p free-edition
   ```
5. **Look at the results**: in *Catalog*, the `gold_*` tables in your schema (start with
   `gold_backtest_metrics`, `gold_policy_evaluation`, `gold_replenishment_plan`); in *Experiments*,
   the `backtest` and `train` runs.
6. **Clean up** when finished (deletes the job, experiment and schema with all its tables):
   ```bash
   databricks bundle destroy -t free -p free-edition
   ```

## Free Edition limits and how the design copes

| Limit (Free Edition / serverless) | Impact here |
|---|---|
| Serverless compute only, no ML runtime choice | Designed for it: environment version 6, wheel dependencies only |
| Daily compute quota; exceeding it shuts compute down until the next day | Use `preset` `small` (default) or `medium`; never `large` |
| Max 5 concurrent job tasks | The job runs at most 2 tasks at once (backtest and train) |
| Python UDF memory capped at 1 GB | Pre-flight check on model-group size; raise `forecast.store_buckets` if it fires |
| UDFs have no internet access; no Maven/JAR libraries | Models travel inside tables; only pure-Python wheels |
| Outbound internet limited to trusted domains | The job installs `lightgbm` and the other pinned packages from PyPI. **Not yet verified on Free Edition**; if blocked, use the workspace's PyPI mirror setting or upload the wheels to a volume |
| No GPUs | Not needed: LightGBM trains on CPU |

## Scaling beyond the prototype (paid workspace)

1. **Data**: replace `generate` with ingestion from real sources (POS, inventory, orders, master data)
   into the same bronze schemas; the rest of the pipeline is unchanged. Make bronze and silver
   incremental (`MERGE` per day) instead of full refresh ([0006](decisions/0006-storage.md)).
2. **Size**: `preset large` exercises ~3 billion rows. Model groups grow with stores, so set
   `forecast.store_buckets` so each group stays under `max_rows_per_group` (the pipeline tells you when).
3. **Compute**: serverless jobs scale automatically; for predictable nightly runs, a jobs cluster on a
   current ML runtime with memory-optimised workers is the alternative.
4. **Schedule**: add a daily trigger to the job (`schedule:` in `resources/demand_forecasting.yml`),
   and alerts on task failure.
5. **Governance**: register the model set in the Unity Catalog model registry, and grant read access on
   `gold_replenishment_plan` to the ordering system's service principal.
6. **Monitoring**: track forecast bias and stockout rate per aisle week over week. A rising bias means
   the model needs retraining or a new feature.
7. **Results app**: deploy `app/streamlit_app.py` as a Databricks App reading the gold tables through a
   SQL warehouse (Free Edition allows 3 apps). This needs a small data-access change, because the app reads local
   Delta folders today.

## Verification status

| Check | Status |
|---|---|
| Pipeline end to end on local Spark 4.2 + Delta 4.4 (tiny in CI, small locally) | verified |
| Spark Connect-only APIs (what serverless allows) | verified locally |
| Wheel builds (`uv build --wheel`) | verified in CI |
| `databricks bundle validate` | see the PR description for the latest result |
| Deploy and run on Free Edition | **not yet run**: needs your workspace login (step 1) |
