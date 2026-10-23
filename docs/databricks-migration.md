# Moving to Databricks

The pipeline is already packaged for Databricks: the bundle (`databricks.yml` +
`resources/demand_forecasting.yml`) builds the wheel and deploys one **serverless job** that runs the
same `dfc` steps. Nothing in the pipeline code changes; the `databricks` settings profile switches
storage to Unity Catalog and tracking to the workspace MLflow.

## What the bundle creates

| Resource | Name (target `free`, development mode) |
|---|---|
| Unity Catalog schema | `workspace.dev_<you>_demand_forecasting` (dev mode prefixes your user name) |
| MLflow experiment | `/Users/<you>/[dev <you>] demand-forecasting-free` (dev mode prefixes the name) |
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
4. **Run** (streams the task states; `small` takes about 12-15 minutes from start to finish on Free Edition,
   because backtest and train run in parallel; see below):
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

If your CLI profile name contains spaces, quote it: `-p "Free Edition"`. `databricks auth profiles` lists
your profiles and whether each login is still valid.

## Running from VS Code (Databricks extension)

The [Databricks extension](https://marketplace.visualstudio.com/items?itemName=databricks.databricks)
deploys and runs the same bundle with one click. It needs two things the CLI route does not:

1. **A workspace host on the target.** The extension only offers to sign in when the target has a
   `workspace.host`, and `databricks.yml` deliberately commits none. Create the git-ignored file
   `resources/workspace.local.yml` (the bundle's `include: resources/*.yml` picks it up):
   ```yaml
   targets:
     free:
       workspace:
         host: https://<your-workspace>.cloud.databricks.com
   ```
   With it in place, CLI commands also work without `-p`: the CLI picks the profile whose host matches.
2. **The repository root as the project folder.** Open the repository root in VS Code
   (`code <path-to-repo>`). If the Databricks panel offers to *migrate* or *initialize* a project, or shows
   `failed to initialize template, one or more files already exist: databricks.yml`, the extension is
   pointed at a subfolder: press `Cmd+Shift+P`, run **Select a Databricks project folder**, and choose
   the repository root.

Then, in the Databricks panel (its logo in the Activity Bar):

1. **Configuration**: check that *Target* is `free`, click **Login to Databricks**, and pick your CLI profile.
2. **Bundle Resource Explorer** > **Jobs** > `demand-forecasting-pipeline`: click **▶** (deploy the bundle
   and run the job). Task states update in the panel; **Databricks: Show bundle logs** has the full output.
3. **Bundle Variables** changes `preset` and the other variables for the next deploy.

The "Databricks Connect disabled" status-bar item is expected: the job runs entirely on serverless
compute, so Databricks Connect is not used. To stop the extension from offering to rebuild `.venv` for
Databricks Connect (the project's `.venv` is managed by `uv sync`), add
`"databricks.python.environmentSetup": "manual"` to your `.vscode/settings.json`.

## Free Edition limits and how the design copes

| Limit (Free Edition / serverless) | Impact here |
|---|---|
| Serverless compute only, no ML runtime choice | Designed for it: environment version 6, wheel dependencies only |
| Daily compute quota; exceeding it shuts compute down until the next day | Use `preset` `small` (default) or `medium`; never `large` |
| Max 5 concurrent job tasks | The job runs at most 2 tasks at once (backtest and train) |
| Python UDF memory capped at 1 GB | Pre-flight check on model-group size; raise `forecast.store_buckets` if it fires |
| UDFs have no internet access; no Maven/JAR libraries | Models travel inside tables; only pure-Python wheels |
| Outbound internet limited to trusted domains | The job installs `lightgbm` and the other pinned packages from PyPI; **verified to work on Free Edition** (2026-10-23) |
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
| Wheel builds (`uv build --wheel`) | verified in CI |
| `databricks bundle validate` / `deploy` | verified on Free Edition, 2026-10-23 |
| Full job run, `small` preset, on Free Edition serverless | **verified, 2026-10-23** (CLI: 15.4 min start to finish) |
| Deploy and run from the VS Code extension (v2.22.0) | **verified, 2026-10-23** (11.9 min start to finish) |

### Measured run on Free Edition (`small` preset)

| Task | Duration | | Result | Local Mac | Free Edition |
|---|---|---|---|---|---|
| generate | 2 min 40 s | | WAPE vs true demand (LightGBM) | 0.593 | 0.594 |
| prepare | 40 s | | WAPE vs true demand (moving average) | 0.636 | 0.636 |
| features | 56 s | | Cycles without stockout (calibrated) | 95.3% | 95.1% |
| backtest | 7 min 34 s | | Fill rate (calibrated) | 98.8% | 98.7% |
| train | 6 min 29 s | | Series forecast | 2,348 | 2,348 |
| forecast | 47 s | | | | |

Baselines match exactly (same data); LightGBM differs in the third decimal because of floating-point
differences between Apple Silicon and Databricks' Linux machines. Free Edition serverless runs these
small tasks slower than a 16-core laptop (backtest 7.5 min vs 50 s locally): it allocates little compute per
job, and environment start-up is included. That is fine for a nightly job and keeps quota use modest.

### Issues the cloud run found (and fixes)
* **Liquid clustering needs statistics on clustering columns**, and Delta only collects them for the first
  32 columns. Databricks enforces this (`DELTA_CLUSTERING_COLUMN_MISSING_STATS`) on the wide
  `gold_features` table; open-source Delta did not, so local tests passed. `TableStore.write` now puts
  clustering columns first for every clustered table.
* **Bundle resource keys must be unique across resource types** (`databricks bundle validate` caught a
  schema and an experiment sharing a key).
