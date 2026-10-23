"""Shared fixtures. One local Spark Connect session serves the whole test run (startup is the
slow part); every test that writes tables gets its own temporary data folder."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from demand_forecasting.config import Settings, load_settings
from demand_forecasting.spark import get_spark


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession]:
    session = get_spark(load_settings("local", preset="tiny"))
    yield session
    session.stop()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Tiny preset writing into a fresh temp folder (MLflow's relative SQLite path too)."""
    monkeypatch.chdir(tmp_path)
    return load_settings("local", preset="tiny", data_dir=str(tmp_path / "data"))
