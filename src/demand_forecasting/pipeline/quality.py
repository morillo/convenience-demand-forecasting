"""Data-quality gates. A failed check stops the pipeline with a readable message.

Checks are deliberately few and aimed at what would silently corrupt a forecast:
duplicate rows, gaps in a series (which would shift every lag feature), negative quantities,
and sales recorded for store x SKU x days that the inventory system does not know about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

log = logging.getLogger(__name__)

SERIES = ["store_id", "sku_id"]


class DataQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckResult:
    name: str
    failing_rows: int

    @property
    def passed(self) -> bool:
        return self.failing_rows == 0


def check_unique(df: DataFrame, keys: list[str], name: str) -> CheckResult:
    dupes = df.groupBy(*keys).count().where(F.col("count") > 1).count()
    return CheckResult(f"{name}: unique {keys}", dupes)


def check_contiguous(df: DataFrame, name: str) -> CheckResult:
    """Every series has exactly one row per day between its first and last day."""
    gaps = (
        df.groupBy(*SERIES)
        .agg(F.count("*").alias("n"), F.min("date").alias("lo"), F.max("date").alias("hi"))
        .where(F.col("n") != F.datediff("hi", "lo") + 1)
        .count()
    )
    return CheckResult(f"{name}: no gaps in daily series", gaps)


def check_non_negative(df: DataFrame, columns: list[str], name: str) -> CheckResult:
    cond = F.lit(False)
    for c in columns:
        cond = cond | (F.col(c) < 0)
    return CheckResult(f"{name}: {columns} >= 0", df.where(cond).count())


def check_orphans(child: DataFrame, parent: DataFrame, keys: list[str], name: str) -> CheckResult:
    orphans = child.join(parent.select(*keys), on=keys, how="left_anti").count()
    return CheckResult(f"{name}: every row has a parent", orphans)


def enforce(results: list[CheckResult]) -> None:
    for r in results:
        log.info(
            "data quality %-55s %s", r.name, "PASS" if r.passed else f"FAIL ({r.failing_rows} rows)"
        )
    failed = [r for r in results if not r.passed]
    if failed:
        raise DataQualityError("; ".join(f"{r.name} ({r.failing_rows} rows)" for r in failed))
