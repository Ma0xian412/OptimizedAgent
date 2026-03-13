from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    CalibrationConfig,
    DatasetDefinition,
    FloatRange,
    IntRange,
    build_dataset_inputs,
    build_settings,
    validate_required_paths,
)


def _build_config(workspace_root: Path, datasets: tuple[DatasetDefinition, ...]) -> CalibrationConfig:
    return CalibrationConfig(
        workspace_root=workspace_root,
        runtime_root=workspace_root / "runtime",
        backtestsys_root=workspace_root / "BackTestSys",
        base_config_path=workspace_root / "config.xml",
        python_executable="/usr/bin/python3",
        datasets=datasets,
        max_failures=2,
        baseline_trials=1,
        machine_delay_trials=4,
        contract_core_trials=4,
        verify_trials=1,
        default_resources={"cpu": 1, "max_runtime_seconds": 60},
        delay_range=IntRange(low=0, high=500000),
        time_scale_lambda_range=FloatRange(low=-0.5, high=0.5),
        cancel_bias_k_range=FloatRange(low=-1.0, high=1.0),
    )


def _dataset(prefix: Path, dataset_id: str) -> DatasetDefinition:
    return DatasetDefinition(
        dataset_id=dataset_id,
        market_data_path=prefix / f"{dataset_id}_market.csv",
        order_file=prefix / f"{dataset_id}_orders.csv",
        cancel_file=prefix / f"{dataset_id}_cancels.csv",
        machine=f"m_{dataset_id}",
        contract=f"c_{dataset_id}",
        groundtruth_doneinfo_path=prefix / f"{dataset_id}_done.csv",
        groundtruth_executiondetail_path=prefix / f"{dataset_id}_exec.csv",
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")


def test_build_settings_includes_dataset_groundtruth_for_all_datasets(tmp_path: Path) -> None:
    datasets = (_dataset(tmp_path, "ds_01"), _dataset(tmp_path, "ds_02"))
    config = _build_config(tmp_path, datasets)
    settings = build_settings(
        config=config,
        runtime_root=tmp_path / "runtime",
        spec_id="stage_a",
        dataset_inputs=build_dataset_inputs(config),
        dataset_ids=["ds_01", "ds_02"],
        baseline_raw=None,
        max_trials=1,
        backtest_search_space=None,
        backtest_fixed_params=None,
        param_binding=None,
    )

    groundtruth = settings["objective_config"]["groundtruth"]
    assert groundtruth["doneinfo_path"] == str(datasets[0].groundtruth_doneinfo_path)
    assert groundtruth["executiondetail_path"] == str(datasets[0].groundtruth_executiondetail_path)
    assert groundtruth["datasets"] == {
        "ds_01": {
            "doneinfo_path": str(datasets[0].groundtruth_doneinfo_path),
            "executiondetail_path": str(datasets[0].groundtruth_executiondetail_path),
        },
        "ds_02": {
            "doneinfo_path": str(datasets[1].groundtruth_doneinfo_path),
            "executiondetail_path": str(datasets[1].groundtruth_executiondetail_path),
        },
    }


def test_build_dataset_inputs_uses_dataset_specific_paths(tmp_path: Path) -> None:
    datasets = (_dataset(tmp_path, "ds_01"),)
    config = _build_config(tmp_path, datasets)
    dataset_inputs = build_dataset_inputs(config)
    assert dataset_inputs == {
        "ds_01": {
            "market_data_path": str(datasets[0].market_data_path),
            "order_file": str(datasets[0].order_file),
            "cancel_file": str(datasets[0].cancel_file),
            "machine": datasets[0].machine,
            "contract": datasets[0].contract,
        }
    }


def test_validate_required_paths_checks_dataset_groundtruth_files(tmp_path: Path) -> None:
    datasets = (_dataset(tmp_path, "ds_01"),)
    config = _build_config(tmp_path, datasets)
    _touch(config.backtestsys_root / "main.py")
    _touch(config.base_config_path)
    _touch(datasets[0].market_data_path)
    _touch(datasets[0].order_file)
    _touch(datasets[0].cancel_file)
    _touch(datasets[0].groundtruth_doneinfo_path)

    with pytest.raises(FileNotFoundError, match="ds_01_exec.csv"):
        validate_required_paths(config)
