"""Unit tests for BackTestDatasetEnumeratorAdapter."""

from __future__ import annotations

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestDatasetEnumeratorAdapter
from tests.conftest import make_spec


def _make_spec_with_dataset_inputs(
    dataset_inputs: dict[str, dict[str, str]],
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
                "dataset_inputs": dataset_inputs,
            },
        },
    )


class TestBackTestDatasetEnumeratorAdapter:
    def test_single_dataset_returns_one_id(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs(
            {
                "20250101": {
                    "market_data_path": "/data/20250101.pkl",
                    "order_file": "/orders/20250101.csv",
                    "cancel_file": "/cancels/20250101.csv",
                }
            }
        )
        result = adapter.enumerate(spec)
        assert result == ("20250101",)

    def test_multiple_datasets_returns_sorted_ids(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs(
            {
                "20250131": {
                    "market_data_path": "/data/20250131.pkl",
                    "order_file": "/orders/20250131.csv",
                    "cancel_file": "/cancels/20250131.csv",
                },
                "20250101": {
                    "market_data_path": "/data/20250101.pkl",
                    "order_file": "/orders/20250101.csv",
                    "cancel_file": "/cancels/20250101.csv",
                },
                "20250115": {
                    "market_data_path": "/data/20250115.pkl",
                    "order_file": "/orders/20250115.csv",
                    "cancel_file": "/cancels/20250115.csv",
                },
            }
        )
        result = adapter.enumerate(spec)
        assert result == ("20250101", "20250115", "20250131")

    def test_meta_dataset_ids_restricts_subset(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs(
            dataset_inputs={
                "20250101": {
                    "market_data_path": "/data/20250101.pkl",
                    "order_file": "/orders/20250101.csv",
                    "cancel_file": "/cancels/20250101.csv",
                },
                "20250102": {
                    "market_data_path": "/data/20250102.pkl",
                    "order_file": "/orders/20250102.csv",
                    "cancel_file": "/cancels/20250102.csv",
                },
                "20250103": {
                    "market_data_path": "/data/20250103.pkl",
                    "order_file": "/orders/20250103.csv",
                    "cancel_file": "/cancels/20250103.csv",
                },
            },
            meta={"dataset_ids": ["20250102"]},
        )
        result = adapter.enumerate(spec)
        assert result == ("20250102",)

    def test_meta_dataset_ids_multiple(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs(
            dataset_inputs={
                "20250101": {
                    "market_data_path": "/data/20250101.pkl",
                    "order_file": "/orders/20250101.csv",
                    "cancel_file": "/cancels/20250101.csv",
                },
                "20250102": {
                    "market_data_path": "/data/20250102.pkl",
                    "order_file": "/orders/20250102.csv",
                    "cancel_file": "/cancels/20250102.csv",
                },
                "20250103": {
                    "market_data_path": "/data/20250103.pkl",
                    "order_file": "/orders/20250103.csv",
                    "cancel_file": "/cancels/20250103.csv",
                },
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

    def test_empty_dataset_inputs_raises(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs({})
        with pytest.raises(
            ValueError,
            match="dataset_inputs.*must be a non-empty dict",
        ):
            adapter.enumerate(spec)

    def test_meta_dataset_ids_not_in_paths_raises(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs(
            dataset_inputs={
                "20250101": {
                    "market_data_path": "/data/20250101.pkl",
                    "order_file": "/orders/20250101.csv",
                    "cancel_file": "/cancels/20250101.csv",
                }
            },
            meta={"dataset_ids": ["20250199"]},
        )
        with pytest.raises(
            ValueError,
            match="meta.dataset_ids.*not found",
        ):
            adapter.enumerate(spec)

    def test_missing_required_input_field_raises(self) -> None:
        adapter = BackTestDatasetEnumeratorAdapter()
        spec = _make_spec_with_dataset_inputs(
            dataset_inputs={
                "20250101": {
                    "market_data_path": "/data/20250101.pkl",
                    "order_file": "/orders/20250101.csv",
                }
            }
        )
        with pytest.raises(
            ValueError,
            match=r"dataset_inputs\['20250101'\]\.cancel_file must be a non-empty string",
        ):
            adapter.enumerate(spec)
