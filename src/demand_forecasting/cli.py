"""Command-line entry point ``dfc``: one sub-command per pipeline step.

Locally:     uv run dfc run-all --preset small
Databricks:  the bundle's job tasks call the same entry point from the wheel, e.g.
             dfc features --profile databricks --catalog workspace --schema demand_forecasting
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence

from pyspark.sql import SparkSession

from demand_forecasting.config import PRESETS, PROFILES, Settings, load_settings
from demand_forecasting.data import generate
from demand_forecasting.pipeline import backtest, features, forecast, prepare, train
from demand_forecasting.spark import get_spark

Step = Callable[[SparkSession, Settings], None]

STEPS: dict[str, tuple[Step, str]] = {
    "generate": (generate.run, "Build the mock chain and write the bronze tables."),
    "prepare": (
        prepare.run,
        "Clean and join bronze into the silver daily panel (with data-quality gates).",
    ),
    "features": (features.run, "Compute model features for history and the forecast horizon."),
    "backtest": (
        backtest.run,
        "Rolling-origin backtest vs. baselines, plus the replenishment what-if.",
    ),
    "train": (train.run, "Train the production models on the full history."),
    "forecast": (forecast.run, "Forecast the next horizon and write the replenishment plan."),
}
RUN_ALL = ["generate", "prepare", "features", "backtest", "train", "forecast"]

log = logging.getLogger("demand_forecasting")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfc",
        description="Store x SKU demand forecasting pipeline (local Spark or Databricks).",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--profile", choices=PROFILES, help="Settings profile (default: $DF_PROFILE or 'local')."
    )
    common.add_argument(
        "--preset", choices=sorted(PRESETS), help="Mock data size (overrides the profile)."
    )
    common.add_argument("--seed", type=int, help="Random seed for the mock data.")
    common.add_argument("--data-dir", help="Local data folder (profile 'local' only).")
    common.add_argument("--catalog", help="Unity Catalog catalog (profile 'databricks').")
    common.add_argument("--schema", help="Unity Catalog schema (profile 'databricks').")
    common.add_argument("--experiment", help="MLflow experiment name or workspace path.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in STEPS.items():
        sub.add_parser(name, parents=[common], help=help_text, description=help_text)
    sub.add_parser(
        "run-all", parents=[common], help=f"Run every step in order: {' -> '.join(RUN_ALL)}."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("py4j", "pyspark", "mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings(
        args.profile,
        preset=args.preset,
        seed=args.seed,
        data_dir=args.data_dir,
        catalog=args.catalog,
        schema=args.schema,
        experiment=args.experiment,
    )
    steps = RUN_ALL if args.command == "run-all" else [args.command]
    s = settings.storage
    where = f"{s.catalog}.{s.schema}" if settings.on_databricks else s.data_dir
    log.info(
        "profile=%s preset=%s storage=%s steps=%s",
        settings.profile,
        settings.preset.name,
        where,
        steps,
    )

    spark = get_spark(settings)
    try:
        for name in steps:
            started = time.perf_counter()
            log.info("step %s: started", name)
            STEPS[name][0](spark, settings)
            log.info("step %s: done in %.1fs", name, time.perf_counter() - started)
    finally:
        if not settings.on_databricks:
            spark.stop()


if __name__ == "__main__":
    main()
