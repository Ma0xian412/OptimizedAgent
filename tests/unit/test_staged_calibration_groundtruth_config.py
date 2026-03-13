from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    CalibrationConfig,
    DatasetDefinition,
    build_settings,
    validate_required_paths,
)


def _build_config(workspace_root: Path, datasets: tuple[DatasetDefinition, ...]) -> CalibrationConfig:
    return CalibrationConfig(
        workspace_root=workspace_root,
        backtestsys_root=workspace_root / "BackTestSys",
        base_config_path=workspace_root / "config.xml",
        mock_root=workspace_root / "mock_backtestsys",
        datasets=datasets,
        max_failures=2,
        baseline_trials=1,
        machine_delay_trials=4,
        contract_core_trials=4,
        verify_trials=1,
        default_resources={"cpu": 1, "max_runtime_seconds": 60},
    )


def _build_dataset_inputs(config: CalibrationConfig) -> dict[str, dict[str, str]]:
    return {
        item.dataset_id: {
            "market_data_path": str(config.mock_root / item.market_data_file),
            "order_file": str(config.mock_root / "replay_orders.csv"),
            "cancel_file": str(config.mock_root / "replay_cancels.csv"),
            "machine": item.machine,
            "contract": item.contract,
        }
        for item in config.datasets
    }


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def test_build_settings_includes_dataset_groundtruth_overrides(tmp_path: Path) -> None:
    datasets = (
        DatasetDefinition(
            "ds_01",
            "market_data_ds_01.csv",
            "m1",
            "c1",
            groundtruth_doneinfo_file="PubOrderDoneInfoLog_m1_20260312_IF2401.csv",
            groundtruth_executiondetail_file="PubExecutionDetailLog_m1_20260312_IF2401.csv",
        ),
        DatasetDefinition("ds_02", "market_data_ds_02.csv", "m2", "c2"),
    )
    config = _build_config(tmp_path, datasets)
    settings = build_settings(
        config=config,
        runtime_root=tmp_path / "runtime",
        spec_id="stage_a",
        dataset_inputs=_build_dataset_inputs(config),
        dataset_ids=["ds_01", "ds_02"],
        baseline_raw=None,
        max_trials=1,
        backtest_search_space=None,
        backtest_fixed_params=None,
        param_binding=None,
    )

    groundtruth = settings["objective_config"]["groundtruth"]
    assert groundtruth["doneinfo_path"].endswith("PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv")
    assert groundtruth["executiondetail_path"].endswith(
        "PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv"
    )
    assert groundtruth["datasets"] == {
        "ds_01": {
            "doneinfo_path": str(
                tmp_path / "mock_backtestsys" / "groundtruth" / "PubOrderDoneInfoLog_m1_20260312_IF2401.csv"
            ),
            "executiondetail_path": str(
                tmp_path / "mock_backtestsys" / "groundtruth" / "PubExecutionDetailLog_m1_20260312_IF2401.csv"
            ),
        }
    }


def test_build_settings_rejects_partial_dataset_groundtruth_override(tmp_path: Path) -> None:
    datasets = (
        DatasetDefinition(
            "ds_01",
            "market_data_ds_01.csv",
            "m1",
            "c1",
            groundtruth_doneinfo_file="PubOrderDoneInfoLog_m1_20260312_IF2401.csv",
        ),
    )
    config = _build_config(tmp_path, datasets)
    with pytest.raises(ValueError, match="requires both doneinfo and executiondetail"):
        build_settings(
            config=config,
            runtime_root=tmp_path / "runtime",
            spec_id="stage_a",
            dataset_inputs=_build_dataset_inputs(config),
            dataset_ids=["ds_01"],
            baseline_raw=None,
            max_trials=1,
            backtest_search_space=None,
            backtest_fixed_params=None,
            param_binding=None,
        )


def test_validate_required_paths_checks_dataset_override_files(tmp_path: Path) -> None:
    datasets = (
        DatasetDefinition(
            "ds_01",
            "market_data_ds_01.csv",
            "m1",
            "c1",
            groundtruth_doneinfo_file="PubOrderDoneInfoLog_m1_20260312_IF2401.csv",
            groundtruth_executiondetail_file="PubExecutionDetailLog_m1_20260312_IF2401.csv",
        ),
    )
    config = _build_config(tmp_path, datasets)
    _touch(config.backtestsys_root / "main.py")
    _touch(config.base_config_path)
    _touch(config.mock_root / "replay_orders.csv")
    _touch(config.mock_root / "replay_cancels.csv")
    _touch(config.mock_root / "contracts.xml")
    _touch(config.mock_root / "market_data_ds_01.csv")
    _touch(config.mock_root / "groundtruth" / "PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv")
    _touch(config.mock_root / "groundtruth" / "PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv")
    _touch(config.mock_root / "groundtruth" / "PubOrderDoneInfoLog_m1_20260312_IF2401.csv")

    with pytest.raises(FileNotFoundError, match="PubExecutionDetailLog_m1_20260312_IF2401.csv"):
        validate_required_paths(config)
