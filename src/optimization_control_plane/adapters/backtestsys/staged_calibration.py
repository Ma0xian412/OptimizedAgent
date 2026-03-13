from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from optimization_control_plane.adapters.backtestsys import BackTestCoreParamsSearchSpaceAdapter, BackTestDelaySearchSpaceAdapter
from optimization_control_plane.adapters.backtestsys.staged_calibration_observability import (
    StageProgressContext,
    StagedCalibrationProgressReporter,
)
from optimization_control_plane.adapters.backtestsys.staged_calibration_runtime_helpers import (
    build_baseline_cache_path,
    normalize_baseline_settings_for_cache,
    read_cached_baseline,
    run_observed_block,
    run_stage_for_progress,
    write_cached_baseline,
)
from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    CalibrationConfig,
    FixedBacktestSearchSpaceAdapter,
    StageResult,
    as_float,
    build_dataset_inputs,
    build_settings,
    extract_baseline_raw,
    group_dataset_ids,
    read_default_params,
    unique_machine_for_contract,
    validate_required_paths,
)

def run_staged_calibration(
    config: CalibrationConfig,
    *,
    progress_interval_seconds: float = 2.0,
    progress_format: str = "text",
) -> dict[str, object]:
    validate_required_paths(config)
    run_tag = dt.datetime.now(dt.UTC).strftime("iter_backtestsys_%Y%m%d_%H%M%S")
    runtime_root = config.runtime_root / run_tag
    runtime_root.mkdir(parents=True, exist_ok=True)
    reporter = StagedCalibrationProgressReporter(runtime_root, run_tag, output_format=progress_format)
    reporter.run_started(unit_total=4, dataset_count=len(config.datasets))
    dataset_inputs = build_dataset_inputs(config)
    defaults = read_default_params(config.base_config_path)
    baseline_raw = run_observed_block(
        reporter,
        StageProgressContext("baseline", 1, 4, config.baseline_trials),
        lambda: _run_baseline_stage(config, runtime_root, run_tag, dataset_inputs, defaults, reporter, progress_interval_seconds),
    )
    machine_delay_map = run_observed_block(
        reporter,
        StageProgressContext("machine_delay", 2, 4, None),
        lambda: _run_machine_delay_stage(
            config, runtime_root, run_tag, dataset_inputs, defaults, baseline_raw, reporter, progress_interval_seconds
        ),
    )
    contract_core_map = run_observed_block(
        reporter,
        StageProgressContext("contract_core", 3, 4, None),
        lambda: _run_contract_core_stage(
            config, runtime_root, run_tag, dataset_inputs, baseline_raw, machine_delay_map, reporter, progress_interval_seconds
        ),
    )
    final_result = run_observed_block(
        reporter,
        StageProgressContext("final_verify", 4, 4, config.verify_trials),
        lambda: _run_final_verify_stage(
            config,
            runtime_root,
            run_tag,
            dataset_inputs,
            defaults,
            baseline_raw,
            machine_delay_map,
            contract_core_map,
            reporter,
            progress_interval_seconds,
        ),
        resolve_best_value=lambda result: result.best_value,
    )
    payload = {
        "run_tag": run_tag,
        "runtime_root": str(runtime_root),
        "baseline_raw": baseline_raw,
        "machine_delay_map": machine_delay_map,
        "contract_core_map": contract_core_map,
        "final_best_value": final_result.best_value,
        "final_raw": final_result.best_attrs.get("raw"),
    }
    (runtime_root / "calibration_output.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reporter.run_finished(final_best_value=final_result.best_value, runtime_root=runtime_root)
    return payload
def _run_baseline_stage(
    config: CalibrationConfig,
    runtime_root: Path,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    defaults,
    reporter: StagedCalibrationProgressReporter | None = None,
    progress_interval_seconds: float = 2.0,
) -> dict[str, float]:
    settings = build_settings(
        config,
        runtime_root,
        spec_id=f"{run_tag}_baseline",
        dataset_inputs=dataset_inputs,
        dataset_ids=[item.dataset_id for item in config.datasets],
        baseline_raw=None,
        max_trials=config.baseline_trials,
        backtest_search_space=None,
        backtest_fixed_params=None,
        param_binding=None,
    )
    fixed_params = {
        "time_scale_lambda": defaults.time_scale_lambda,
        "cancel_bias_k": defaults.cancel_bias_k,
        "delay_in": defaults.delay_in,
        "delay_out": defaults.delay_out,
    }
    cache_path = build_baseline_cache_path(config.workspace_root, settings, fixed_params)
    cached = read_cached_baseline(cache_path)
    if cached is not None:
        if reporter is not None:
            reporter.baseline_cache_hit(StageProgressContext("baseline", 1, 1, config.baseline_trials), cache_path=cache_path)
        return cached
    stage_result = run_stage_for_progress(
        runtime_root=runtime_root,
        stage_name="baseline",
        settings=settings,
        search_space=FixedBacktestSearchSpaceAdapter(fixed_params),
        progress_reporter=reporter,
        progress_context=StageProgressContext("baseline", 1, 1, config.baseline_trials),
        progress_interval_seconds=progress_interval_seconds,
    )
    baseline_raw = extract_baseline_raw(stage_result.best_attrs)
    write_cached_baseline(cache_path, baseline_raw)
    return baseline_raw


def _run_machine_delay_stage(
    config: CalibrationConfig,
    runtime_root: Path,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    defaults,
    baseline_raw: dict[str, float],
    reporter: StagedCalibrationProgressReporter | None = None,
    progress_interval_seconds: float = 2.0,
) -> dict[str, int]:
    grouped = group_dataset_ids(config.datasets, key="machine")
    total = len(grouped)
    machine_delay_map: dict[str, int] = {}
    for index, (machine, dataset_ids) in enumerate(sorted(grouped.items()), start=1):
        settings = build_settings(
            config,
            runtime_root,
            spec_id=f"{run_tag}_machine_delay_{machine}",
            dataset_inputs=dataset_inputs,
            dataset_ids=dataset_ids,
            baseline_raw=baseline_raw,
            max_trials=config.machine_delay_trials,
            backtest_search_space={"delay": {"low": config.delay_range.low, "high": config.delay_range.high}},
            backtest_fixed_params={"time_scale_lambda": defaults.time_scale_lambda, "cancel_bias_k": defaults.cancel_bias_k},
            param_binding=None,
        )
        stage_name = f"machine_delay_{machine}"
        stage_result = run_stage_for_progress(
            runtime_root=runtime_root,
            stage_name=stage_name,
            settings=settings,
            search_space=BackTestDelaySearchSpaceAdapter(),
            progress_reporter=reporter,
            progress_context=StageProgressContext(stage_name, index, total, config.machine_delay_trials),
            progress_interval_seconds=progress_interval_seconds,
        )
        delay = stage_result.best_params.get("delay")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise ValueError(f"best delay for machine={machine} must be int")
        machine_delay_map[machine] = delay
    return machine_delay_map


def _run_contract_core_stage(
    config: CalibrationConfig,
    runtime_root: Path,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    baseline_raw: dict[str, float],
    machine_delay_map: dict[str, int],
    reporter: StagedCalibrationProgressReporter | None = None,
    progress_interval_seconds: float = 2.0,
) -> dict[str, dict[str, float]]:
    grouped = group_dataset_ids(config.datasets, key="contract")
    total = len(grouped)
    contract_core_map: dict[str, dict[str, float]] = {}
    for index, (contract, dataset_ids) in enumerate(sorted(grouped.items()), start=1):
        machine = unique_machine_for_contract(config.datasets, contract)
        settings = build_settings(
            config,
            runtime_root,
            spec_id=f"{run_tag}_contract_core_{contract}",
            dataset_inputs=dataset_inputs,
            dataset_ids=dataset_ids,
            baseline_raw=baseline_raw,
            max_trials=config.contract_core_trials,
            backtest_search_space={
                "time_scale_lambda": {"low": config.time_scale_lambda_range.low, "high": config.time_scale_lambda_range.high},
                "cancel_bias_k": {"low": config.cancel_bias_k_range.low, "high": config.cancel_bias_k_range.high},
            },
            backtest_fixed_params={"delay": machine_delay_map[machine]},
            param_binding=None,
        )
        stage_name = f"contract_core_{contract}"
        stage_result = run_stage_for_progress(
            runtime_root=runtime_root,
            stage_name=stage_name,
            settings=settings,
            search_space=BackTestCoreParamsSearchSpaceAdapter(),
            progress_reporter=reporter,
            progress_context=StageProgressContext(stage_name, index, total, config.contract_core_trials),
            progress_interval_seconds=progress_interval_seconds,
        )
        contract_core_map[contract] = {
            "time_scale_lambda": as_float(stage_result.best_params.get("time_scale_lambda"), contract, "time_scale_lambda"),
            "cancel_bias_k": as_float(stage_result.best_params.get("cancel_bias_k"), contract, "cancel_bias_k"),
        }
    return contract_core_map


def _run_final_verify_stage(
    config: CalibrationConfig,
    runtime_root: Path,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    defaults,
    baseline_raw: dict[str, float],
    machine_delay_map: dict[str, int],
    contract_core_map: dict[str, dict[str, float]],
    reporter: StagedCalibrationProgressReporter | None = None,
    progress_interval_seconds: float = 2.0,
) -> StageResult:
    settings = build_settings(
        config,
        runtime_root,
        spec_id=f"{run_tag}_final_verify",
        dataset_inputs=dataset_inputs,
        dataset_ids=[item.dataset_id for item in config.datasets],
        baseline_raw=baseline_raw,
        max_trials=config.verify_trials,
        backtest_search_space=None,
        backtest_fixed_params=None,
        param_binding={"mode": "calibrated_map", "machine_delay_map": machine_delay_map, "contract_core_map": contract_core_map},
    )
    fixed_params = {
        "time_scale_lambda": defaults.time_scale_lambda,
        "cancel_bias_k": defaults.cancel_bias_k,
        "delay_in": defaults.delay_in,
        "delay_out": defaults.delay_out,
    }
    return run_stage_for_progress(
        runtime_root=runtime_root,
        stage_name="final_verify",
        settings=settings,
        search_space=FixedBacktestSearchSpaceAdapter(fixed_params),
        progress_reporter=reporter,
        progress_context=StageProgressContext("final_verify", 1, 1, config.verify_trials),
        progress_interval_seconds=progress_interval_seconds,
    )


_build_baseline_cache_path = build_baseline_cache_path
_normalize_baseline_settings_for_cache = normalize_baseline_settings_for_cache
_read_cached_baseline = read_cached_baseline
_write_cached_baseline = write_cached_baseline
