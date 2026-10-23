from __future__ import annotations

import pytest

from demand_forecasting.config import PRESETS, load_settings


def test_local_profile_defaults() -> None:
    s = load_settings("local")
    assert s.storage.mode == "path"
    assert not s.on_databricks
    assert s.preset == PRESETS["small"]


def test_databricks_profile_with_overrides() -> None:
    s = load_settings(
        "databricks", catalog="main", schema="forecasting", preset="tiny", experiment="/Shared/x"
    )
    assert s.on_databricks
    assert (s.storage.catalog, s.storage.schema) == ("main", "forecasting")
    assert s.preset.name == "tiny"
    assert s.mlflow.experiment == "/Shared/x"


def test_none_overrides_are_ignored() -> None:
    assert load_settings("local", preset=None, catalog=None).preset.name == "small"


def test_profile_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DF_PROFILE", "databricks")
    assert load_settings().profile == "databricks"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"preset": "huge"}, "Unknown preset"), ({"colour": "blue"}, "Unknown override")],
)
def test_invalid_input_is_rejected(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_settings("local", **kwargs)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        load_settings("staging")


def test_mlflow_files_follow_the_data_dir() -> None:
    assert load_settings("local").mlflow.tracking_uri == "sqlite:///data/mlflow.db"
    moved = load_settings("local", data_dir="/tmp/run1").mlflow
    assert moved.tracking_uri == "sqlite:////tmp/run1/mlflow.db"
    assert moved.artifact_location == "/tmp/run1/mlruns"
    assert load_settings("databricks").mlflow.tracking_uri == "databricks"
