"""Create the Spark session: Databricks' own session there, a local Spark Connect one here.

Locally we deliberately use Spark Connect (``SparkSession.builder.remote("local[*]")``)
instead of a classic session. Databricks serverless only speaks Spark Connect, so anything
that would break there (RDD APIs, ``sparkContext``, most ``spark.conf`` settings) breaks on
the laptop first. See docs/decisions/0001-local-first-mirroring-serverless.md.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from importlib import resources
from pathlib import Path

from pyspark.sql import SparkSession

from demand_forecasting.config import Settings

# Delta Lake build for Spark 4.2 / Scala 2.13, same version as the delta-spark Python package.
DELTA_PACKAGE = "io.delta:delta-spark_4.2_2.13:4.4.1"


def ivy_cache_dir() -> Path:
    """Where the Delta jars are cached: $DF_IVY_DIR, else ~/.cache/demand-forecasting/ivy.
    Outside the repo, so any working directory (tests, CI) reuses one download."""
    return Path(os.environ.get("DF_IVY_DIR", Path.home() / ".cache" / "demand-forecasting" / "ivy"))


def get_spark(settings: Settings) -> SparkSession:
    if settings.on_databricks:
        # Databricks jobs and notebooks already have a session; never configure it here.
        return SparkSession.builder.getOrCreate()
    return _local_session(settings)


def _local_session(settings: Settings) -> SparkSession:
    if shutil.which("java") is None and "JAVA_HOME" not in os.environ:
        raise RuntimeError(
            "Spark needs Java 17 or 21. On macOS: brew install openjdk@17, then "
            'export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"'
        )
    # Python workers must be this interpreter (the project venv), not whatever python3 is first
    # on PATH, or they would miss pandas / LightGBM.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    ivy_settings = resources.files("demand_forecasting.resources").joinpath("ivysettings.xml")
    builder = (
        SparkSession.builder.remote(settings.spark.master)
        .appName("demand-forecasting")
        .config("spark.driver.memory", settings.spark.driver_memory)
        .config("spark.jars.packages", DELTA_PACKAGE)
        .config("spark.jars.ivySettings", str(ivy_settings))
        .config("spark.jars.ivy", str(ivy_cache_dir()))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        # Same defaults as Databricks serverless: ANSI SQL semantics and UTC session time zone.
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "false")
    )
    with warnings.catch_warnings():
        # Startup-only configs are applied to the local server, then the client tries to set
        # them again at runtime and warns that it cannot. Harmless; hide the noise.
        warnings.filterwarnings("ignore", message="Failed to set spark")
        return builder.getOrCreate()
