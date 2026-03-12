from __future__ import annotations

import datetime as dt
import json

from optimization_control_plane.adapters.backtestsys import (
    BackTestCoreParamsSearchSpaceAdapter,
    BackTestDelaySearchSpaceAdapter,
)
from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    CalibrationConfig,
    FixedBacktestSearchSpaceAdapter,
    as_float,
    build_dataset_inputs,
    build_settings,
    extract_baseline_raw,
    group_dataset_ids,
    read_default_params,
    run_stage,
    unique_machine_for_contract,
    validate_required_paths,
)

_DELAY_RANGE = {"low": 0, "high": 500000}
_LAMBDA_RANGE = {"low": -0.5, "high": 0.5}
_CANCEL_BIAS_RANGE = {"low": -1.0, "high": 1.0}


def run_staged_calibration(config: CalibrationConfig) -> dict[str, object]:
    validate_required_paths(config)
    run_tag = dt.datetime.now(dt.UTC).strftime("iter_backtestsys_%Y%m%d_%H%M%S")
    runtime_root = config.workspace_root / "runtime" / run_tag
    runtime_root.mkdir(parents=True, exist_ok=True)
    dataset_inputs = build_dataset_inputs(config)
    defaults = read_default_params(config.base_config_path)
    baseline_raw = _run_baseline_stage(config, runtime_root, run_tag, dataset_inputs, defaults)
    machine_delay_map = _run_machine_delay_stage(
        config, runtime_root, run_tag, dataset_inputs, defaults, baseline_raw
    )
    contract_core_map = _run_contract_core_stage(
        config, runtime_root, run_tag, dataset_inputs, baseline_raw, machine_delay_map
    )
    final_result = _run_final_verify_stage(
        config,
        runtime_root,
        run_tag,
        dataset_inputs,
        defaults,
        baseline_raw,
        machine_delay_map,
        contract_core_map,
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
    (runtime_root / "calibration_output.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _run_baseline_stage(
    config: CalibrationConfig,
    runtime_root,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    defaults,
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
    result = run_stage(runtime_root, "baseline", settings, FixedBacktestSearchSpaceAdapter(fixed_params))
    return extract_baseline_raw(result.best_attrs)


def _run_machine_delay_stage(
    config: CalibrationConfig,
    runtime_root,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    defaults,
    baseline_raw: dict[str, float],
) -> dict[str, int]:
    grouped = group_dataset_ids(config.datasets, key="machine")
    machine_delay_map: dict[str, int] = {}
    for machine, dataset_ids in sorted(grouped.items()):
        settings = build_settings(
            config,
            runtime_root,
            spec_id=f"{run_tag}_machine_delay_{machine}",
            dataset_inputs=dataset_inputs,
            dataset_ids=dataset_ids,
            baseline_raw=baseline_raw,
            max_trials=config.machine_delay_trials,
            backtest_search_space={"delay": dict(_DELAY_RANGE)},
            backtest_fixed_params={
                "time_scale_lambda": defaults.time_scale_lambda,
                "cancel_bias_k": defaults.cancel_bias_k,
            },
            param_binding=None,
        )
        result = run_stage(runtime_root, f"machine_delay_{machine}", settings, BackTestDelaySearchSpaceAdapter())
        delay = result.best_params.get("delay")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise ValueError(f"best delay for machine={machine} must be int")
        machine_delay_map[machine] = delay
    return machine_delay_map


def _run_contract_core_stage(
    config: CalibrationConfig,
    runtime_root,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    baseline_raw: dict[str, float],
    machine_delay_map: dict[str, int],
) -> dict[str, dict[str, float]]:
    grouped = group_dataset_ids(config.datasets, key="contract")
    contract_core_map: dict[str, dict[str, float]] = {}
    for contract, dataset_ids in sorted(grouped.items()):
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
                "time_scale_lambda": dict(_LAMBDA_RANGE),
                "cancel_bias_k": dict(_CANCEL_BIAS_RANGE),
            },
            backtest_fixed_params={"delay": machine_delay_map[machine]},
            param_binding=None,
        )
        result = run_stage(runtime_root, f"contract_core_{contract}", settings, BackTestCoreParamsSearchSpaceAdapter())
        contract_core_map[contract] = {
            "time_scale_lambda": as_float(result.best_params.get("time_scale_lambda"), contract, "time_scale_lambda"),
            "cancel_bias_k": as_float(result.best_params.get("cancel_bias_k"), contract, "cancel_bias_k"),
        }
    return contract_core_map


def _run_final_verify_stage(
    config: CalibrationConfig,
    runtime_root,
    run_tag: str,
    dataset_inputs: dict[str, dict[str, str]],
    defaults,
    baseline_raw: dict[str, float],
    machine_delay_map: dict[str, int],
    contract_core_map: dict[str, dict[str, float]],
):
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
        param_binding={
            "mode": "calibrated_map",
            "machine_delay_map": machine_delay_map,
            "contract_core_map": contract_core_map,
        },
    )
    fixed_params = {
        "time_scale_lambda": defaults.time_scale_lambda,
        "cancel_bias_k": defaults.cancel_bias_k,
        "delay_in": defaults.delay_in,
        "delay_out": defaults.delay_out,
    }
    return run_stage(runtime_root, "final_verify", settings, FixedBacktestSearchSpaceAdapter(fixed_params))
