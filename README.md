# Convenience-store demand forecasting

A prototype that forecasts daily demand for **every product in every store** of a (simulated) global
convenience-store chain, and turns those forecasts into **order recommendations**. It runs on a laptop
with Spark and is built to move unchanged to **Databricks serverless** (including the Free Edition).

* **New to the topic?** Start with [docs/overview.md](docs/overview.md), a plain-English summary with no jargon.
* **Want the numbers?** See [docs/modeling.md#results](docs/modeling.md#results).

| | |
|---|---|
| Data | Synthetic chain: 12 countries, 5 store formats, 27 categories in 8 aisles, promotions, weather, holidays, stockouts ([docs/mock-data.md](docs/mock-data.md)) |
| Engine | PySpark 4.2 (Spark Connect) + Delta Lake, mirroring Databricks serverless environment 6 |
| Model | One LightGBM model pair per country x aisle: Tweedie mean + P90 quantile, stockout-aware (EM unconstraining) |
| Evaluation | Rolling-origin backtest vs. two baselines, scored against true demand |
| Decision | Order-up-to policy with safety stock calibrated on the backtest |
| Tracking | MLflow (local SQLite or the Databricks workspace) |
| Deployment | Databricks Asset Bundle: one serverless job, one task per pipeline step |

## Setup

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), Java 17 or 21 (Spark), and on macOS
Homebrew's OpenMP runtime (LightGBM's macOS wheel needs it).

```bash
brew install openjdk@17 libomp     # macOS; on Linux: any JDK 17/21 (LightGBM's libgomp ships with the OS)
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"   # macOS; add to ~/.zshrc

git clone https://github.com/morillo/convenience-demand-forecasting.git
cd convenience-demand-forecasting
uv sync                                           # Python 3.12 + pinned dependencies into .venv
```

Setup: `uv sync`. The first Spark start downloads the Delta Lake jars (about 170 MB) into
`~/.cache/demand-forecasting/ivy` (override with `DF_IVY_DIR`).

## Quickstart

Run from the repository root:

```bash
uv run dfc run-all --preset tiny     # about 1 minute: whole pipeline on a 4-store chain
uv run dfc run-all                   # default `small` preset, about 2 minutes: 24 stores x 120 SKUs x 2 years
uv run streamlit run app/streamlit_app.py   # results viewer at http://localhost:8501
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db   # experiment runs at http://localhost:5000
```

Outputs land in `data/` (git-ignored; delete it to start over): Delta tables under `data/lake/`,
MLflow runs in `data/mlflow.db` and `data/mlruns/`. With `--data-dir PATH` all of it goes to `PATH`
instead; point the viewers there too (`DF_DATA_DIR=PATH uv run streamlit run ...`,
`--backend-store-uri sqlite:///PATH/mlflow.db`).

## Pipeline steps

`run-all` runs these in order; each can also be run alone (`uv run dfc <step>`), reading the previous
steps' tables.

| Step | What it does | Main outputs |
|---|---|---|
| `generate` | Builds the mock chain and simulates sales, stock and orders | `bronze_*` |
| `prepare` | Joins and cleans into a gap-free daily panel, infers stockouts, enforces data-quality gates | `silver_daily_panel` |
| `features` | Lags, rolling statistics, calendar, promotions, weather forecast, for history and the next 14 days | `gold_features` |
| `backtest` | Trains at 3 past cut-offs, scores vs. baselines, replenishment what-if, safety-stock calibration | `gold_backtest_*`, `gold_policy_*`, `gold_safety_stock_calibration` |
| `train` | Trains the production models on all history | `gold_models` |
| `forecast` | Forecasts the next 14 days and recommends order quantities | `gold_forecasts`, `gold_replenishment_plan` |

Options shared by every step (`uv run dfc --help`, `uv run dfc <step> --help`):

| Option | Meaning |
|---|---|
| `--preset tiny\|small\|medium\|large` | Size of the mock chain ([docs/mock-data.md](docs/mock-data.md#sizes)); `large` is for paid Databricks only |
| `--seed N` | Random seed of the mock data (default 42) |
| `--data-dir PATH` | Local output folder for tables and MLflow files (default `data`) |
| `--profile local\|databricks` | Settings profile (default: `$DF_PROFILE`, else `local`) |
| `--catalog`, `--schema`, `--experiment` | Unity Catalog location and MLflow experiment (Databricks profile) |

Everything else (horizon, model settings, service level, model grouping) is in the typed settings in
[`src/demand_forecasting/config.py`](src/demand_forecasting/config.py), optionally overridden in the
profile files under [`src/demand_forecasting/resources/`](src/demand_forecasting/resources/).

## Running on Databricks

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile free-edition
databricks bundle validate -t free -p free-edition
databricks bundle deploy   -t free -p free-edition
databricks bundle run demand_forecasting_pipeline -t free -p free-edition
```

The Free Edition limits and how the design handles them, what the bundle creates, scaling to a full
chain, and what has been verified so far are in [docs/databricks-migration.md](docs/databricks-migration.md).

## Development

```bash
uv run pytest                  # unit, Spark and end-to-end tests (about 1-2 minutes)
uv run pytest -m "not e2e"     # skip the end-to-end pipeline run
uv run ruff check && uv run ruff format --check
uv run pyright                 # same type checker as VS Code Pylance; must report 0 errors
uv run pre-commit install      # run the same checks (+ secret scan) on every commit
uv export --all-groups --no-hashes --no-emit-project -q > /tmp/reqs.txt && \
  uvx pip-audit -r /tmp/reqs.txt --no-deps --disable-pip --ignore-vuln PYSEC-2026-113   # dependency audit
```

CI (`.github/workflows/ci.yml`) runs lint, Pyright, the tests (including the end-to-end pipeline and a
headless render of the results app), the tiny quickstart in a fresh folder, a wheel build, a
vulnerability audit of every locked dependency, a secret scan and, on pull requests, a dependency
review. The one accepted advisory (pyarrow, pinned to match serverless) is explained in
[docs/decisions/0001](docs/decisions/0001-local-first-mirroring-serverless.md). Library versions deliberately mirror Databricks
serverless environment 6; bump them together, not one by one
([docs/decisions/0001](docs/decisions/0001-local-first-mirroring-serverless.md)).

## Documentation

| Document | For whom |
|---|---|
| [docs/overview.md](docs/overview.md) | Everyone: the problem, the approach and the results in plain English |
| [docs/glossary.md](docs/glossary.md) | Everyone: WAPE, P90, stockout, lead time... |
| [docs/architecture.md](docs/architecture.md) | Engineers: data flow, code layout, how work is distributed |
| [docs/mock-data.md](docs/mock-data.md) | Everyone: how the fake chain is built and what makes it realistic |
| [docs/modeling.md](docs/modeling.md) | Data scientists: features, model, evaluation, replenishment, results |
| [docs/data-dictionary.md](docs/data-dictionary.md) | Analysts: every table and column |
| [docs/databricks-migration.md](docs/databricks-migration.md) | Platform engineers: deploy, Free Edition limits, scaling |
| [docs/decisions/](docs/decisions/README.md) | Reviewers: why each design choice was made, and what was rejected |

## License

[MIT](LICENSE). The data is entirely synthetic; no real retailer's data or branding is used.
