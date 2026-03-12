"""Unit tests for BackTestDatasetEnumeratorAdapter."""

from __future__ import annotations

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestDatasetEnumeratorAdapter
from tests.conftest import make_spec


def _make_spec_with_dataset_paths(
    dataset_paths: dict[str, str],
    meta: dict | None = None,
) -> object:
    return make_spec(
        meta=meta or {},
        execution_config={
            "executor_kind": "backtest",
            "default_resources": {"cpu": 1},
            "backtest_run_spec": {
                "backtestsys_root": "/tmp/BackTestSys",
                "base_config_path": "/tmp/config.xml",
                "output_root_dir": "/tmp/ocp_artifacts",
                "dataset_paths": dataset_paths,
            },
        },
    )


class TestBackTestDatasetEnumeratorAdapter:
    def test_single_dataset_returns_one_id(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_paths({"20250101": "/data/20250101.pkl"})
        result = adapter.enumerate(spec)
        assert result == ("20250101",)

    def test_multiple_datasets_returns_sorted_ids(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_paths({
            "20250131": "/data/20250131.pkl",
            "20250101": "/data/20250101.pkl",
            "20250115": "/data/20250115.pkl",
        })
        result = adapter.enumerate(spec)
        assert result == ("20250101", "20250115", "20250131")

    def test_meta_dataset_ids_restricts_subset(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_paths(
            dataset_paths={
                "20250101": "/data/20250101.pkl",
                "20250102": "/data/20250102.pkl",
                "20250103": "/data/20250103.pkl",
            },
            meta={"dataset_ids": ["20250102"]},
        )
        result = adapter.enumerate(spec)
        assert result == ("20250102",)

    def test_meta_dataset_ids_multiple(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_paths(
            dataset_paths={
                "20250101": "/data/20250101.pkl",
                "20250102": "/data/20250102.pkl",
                "20250103": "/data/20250103.pkl",
            },
            meta={"dataset_ids": ["20250103", "20250101"]},
        )
        result = adapter.enumerate(spec)
        assert result == ("20250103", "20250101")

    def test_missing_backtest_run_spec_raises(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = make_spec(
            execution_config={
                "executor_kind": "backtest",
                "default_resources": {"cpu": 1},
            }
        )
        with pytest.raises(
            ValueError,
            match="spec.execution_config.backtest_run_spec must be a dict",
        ):
            adapter.enumerate(spec)

    def test_empty_dataset_paths_raises(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_paths({})
        with pytest.raises(
            ValueError,
            match="dataset_paths.*must be a non-empty dict",
        ):
            adapter.enumerate(spec)

    def test_meta_dataset_ids_not_in_paths_raises(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_paths(
            dataset_paths={"20250101": "/data/20250101.pkl"},
            meta={"dataset_ids": ["20250199"]},
        )
        with pytest.raises(
            ValueError,
            match="meta.dataset_ids.*not found",
        ):
            adapter.enumerate(spec)
