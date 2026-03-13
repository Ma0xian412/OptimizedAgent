from __future__ import annotations

from pathlib import Path

from optimization_control_plane.adapters.backtestsys import staged_calibration
from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    BacktestDefaults,
    CalibrationConfig,
    DatasetDefinition,
    FloatRange,
    IntRange,
    StageResult,
)


def _build_config(workspace_root: Path) -> CalibrationConfig:
    dataset = DatasetDefinition(
        dataset_id="ds_01",
        market_data_path=workspace_root / "inputs" / "market_data_ds_01.csv",
        order_file=workspace_root / "inputs" / "replay_orders.csv",
        cancel_file=workspace_root / "inputs" / "replay_cancels.csv",
        machine="m1",
        contract="c1",
        groundtruth_doneinfo_path=workspace_root / "inputs" / "gt_done.csv",
        groundtruth_executiondetail_path=workspace_root / "inputs" / "gt_exec.csv",
    )
    return CalibrationConfig(
        workspace_root=workspace_root,
        runtime_root=workspace_root / "runtime",
        backtestsys_root=workspace_root / "BackTestSys",
        base_config_path=workspace_root / "config.xml",
        python_executable="/usr/bin/python3",
        datasets=(dataset,),
        max_failures=2,
        baseline_trials=1,
        machine_delay_trials=12,
        contract_core_trials=12,
        verify_trials=1,
        default_resources={"cpu": 1, "max_runtime_seconds": 60},
        delay_range=IntRange(low=0, high=500000),
        time_scale_lambda_range=FloatRange(low=-0.5, high=0.5),
        cancel_bias_k_range=FloatRange(low=-1.0, high=1.0),
    )


def test_baseline_stage_uses_cross_run_cache(monkeypatch: object, tmp_path: Path) -> None:
    call_count = {"value": 0}

    def fake_run_stage(
        runtime_root: Path,
        stage_name: str,
        settings: dict[str, object],
        search_space: object,
    ) -> StageResult:
        del runtime_root, settings, search_space
        call_count["value"] += 1
        if stage_name != "baseline":
            raise AssertionError("unexpected stage name")
        return StageResult(
            best_value=1.0,
            best_params={},
            best_attrs={"raw": {"curve": 2.0, "terminal": 3.0, "cancel": 4.0, "post": 5.0}},
        )

    monkeypatch.setattr(staged_calibration, "run_stage", fake_run_stage)
    config = _build_config(tmp_path)
    dataset_inputs = {
        "ds_01": {
            "market_data_path": str(tmp_path / "inputs" / "market_data_ds_01.csv"),
            "order_file": str(tmp_path / "inputs" / "replay_orders.csv"),
            "cancel_file": str(tmp_path / "inputs" / "replay_cancels.csv"),
            "machine": "m1",
            "contract": "c1",
        }
    }
    defaults = BacktestDefaults(time_scale_lambda=0.1, cancel_bias_k=0.2, delay_in=10, delay_out=10)

    first = staged_calibration._run_baseline_stage(
        config=config,
        runtime_root=tmp_path / "runtime" / "run_1",
        run_tag="run_1",
        dataset_inputs=dataset_inputs,
        defaults=defaults,
    )
    second = staged_calibration._run_baseline_stage(
        config=config,
        runtime_root=tmp_path / "runtime" / "run_2",
        run_tag="run_2",
        dataset_inputs=dataset_inputs,
        defaults=defaults,
    )

    assert first == second
    assert call_count["value"] == 1
