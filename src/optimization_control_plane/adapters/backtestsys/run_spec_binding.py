from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_RUN_SPEC_KEY = "backtest_run_spec"
_DATASET_INPUTS_KEY = "dataset_inputs"
_MARKET_DATA_PATH_KEY = "market_data_path"
_ORDER_FILE_KEY = "order_file"
_CANCEL_FILE_KEY = "cancel_file"
_MACHINE_KEY = "machine"
_CONTRACT_KEY = "contract"
_PARAM_BINDING_KEY = "param_binding"
_BINDING_MODE_KEY = "mode"
_BINDING_MODE_TRIAL_GLOBAL = "trial_global"
_BINDING_MODE_CALIBRATED_MAP = "calibrated_map"
_MACHINE_DELAY_MAP_KEY = "machine_delay_map"
_CONTRACT_CORE_MAP_KEY = "contract_core_map"


@dataclass(frozen=True)
class DatasetInput:
    market_data_path: str
    order_file: str
    cancel_file: str
    machine: str | None
    contract: str | None


@dataclass(frozen=True)
class ParamBindingConfig:
    mode: str
    machine_delay_map: dict[str, int]
    contract_core_map: dict[str, dict[str, float]]


def read_param_binding_config(
    run_cfg: dict[str, Any],
    *,
    as_int: Any,
    as_float: Any,
) -> ParamBindingConfig:
    raw = run_cfg.get(_PARAM_BINDING_KEY)
    if raw is None:
        return ParamBindingConfig(
            mode=_BINDING_MODE_TRIAL_GLOBAL,
            machine_delay_map={},
            contract_core_map={},
        )
    if not isinstance(raw, dict):
        raise ValueError(f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY} must be a dict")
    mode = raw.get(_BINDING_MODE_KEY, _BINDING_MODE_TRIAL_GLOBAL)
    if mode not in (_BINDING_MODE_TRIAL_GLOBAL, _BINDING_MODE_CALIBRATED_MAP):
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY}.{_BINDING_MODE_KEY} "
            f"must be one of [{_BINDING_MODE_TRIAL_GLOBAL}, {_BINDING_MODE_CALIBRATED_MAP}]"
        )
    if mode == _BINDING_MODE_TRIAL_GLOBAL:
        return ParamBindingConfig(mode=mode, machine_delay_map={}, contract_core_map={})
    machine_delay_map = _read_machine_delay_map(raw, as_int=as_int)
    contract_core_map = _read_contract_core_map(raw, as_float=as_float)
    return ParamBindingConfig(
        mode=mode,
        machine_delay_map=machine_delay_map,
        contract_core_map=contract_core_map,
    )


def resolve_dataset_input(run_cfg: dict[str, Any], dataset_id: str) -> DatasetInput:
    dataset_inputs = run_cfg.get(_DATASET_INPUTS_KEY)
    if not isinstance(dataset_inputs, dict) or not dataset_inputs:
        raise ValueError(f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY} must be a non-empty dict")
    raw_input = dataset_inputs.get(dataset_id)
    if not isinstance(raw_input, dict):
        raise ValueError(f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY}[{dataset_id}] must be a dict")
    return DatasetInput(
        market_data_path=_read_required_dataset_input_path(
            raw_input,
            dataset_id=dataset_id,
            key=_MARKET_DATA_PATH_KEY,
        ),
        order_file=_read_required_dataset_input_path(
            raw_input,
            dataset_id=dataset_id,
            key=_ORDER_FILE_KEY,
        ),
        cancel_file=_read_required_dataset_input_path(
            raw_input,
            dataset_id=dataset_id,
            key=_CANCEL_FILE_KEY,
        ),
        machine=_read_optional_dataset_input_label(
            raw_input,
            dataset_id=dataset_id,
            key=_MACHINE_KEY,
        ),
        contract=_read_optional_dataset_input_label(
            raw_input,
            dataset_id=dataset_id,
            key=_CONTRACT_KEY,
        ),
    )


def resolve_effective_params(
    *,
    dataset_id: str,
    trial_params: dict[str, object],
    dataset_input: DatasetInput,
    binding: ParamBindingConfig,
) -> dict[str, object]:
    if binding.mode == _BINDING_MODE_TRIAL_GLOBAL:
        return dict(trial_params)
    machine = _require_dataset_label(dataset_input.machine, dataset_id=dataset_id, key=_MACHINE_KEY)
    contract = _require_dataset_label(dataset_input.contract, dataset_id=dataset_id, key=_CONTRACT_KEY)
    delay = _resolve_machine_delay(binding.machine_delay_map, machine)
    core = _resolve_contract_core(binding.contract_core_map, contract)
    return {
        "time_scale_lambda": core["time_scale_lambda"],
        "cancel_bias_k": core["cancel_bias_k"],
        "delay_in": delay,
        "delay_out": delay,
    }


def _read_required_dataset_input_path(
    source: dict[str, Any],
    *,
    dataset_id: str,
    key: str,
) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY}[{dataset_id}].{key} must be a non-empty string"
        )
    return value


def _read_optional_dataset_input_label(
    source: dict[str, Any],
    *,
    dataset_id: str,
    key: str,
) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY}[{dataset_id}].{key} must be a non-empty string"
        )
    return value


def _read_machine_delay_map(binding_cfg: dict[str, Any], *, as_int: Any) -> dict[str, int]:
    raw = binding_cfg.get(_MACHINE_DELAY_MAP_KEY)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY}.{_MACHINE_DELAY_MAP_KEY} must be a non-empty dict"
        )
    result: dict[str, int] = {}
    for machine, raw_delay in raw.items():
        if not isinstance(machine, str) or not machine:
            raise ValueError(f"{_MACHINE_DELAY_MAP_KEY} keys must be non-empty strings")
        result[machine] = as_int(raw_delay, f"{_MACHINE_DELAY_MAP_KEY}[{machine}]")
    return result


def _read_contract_core_map(
    binding_cfg: dict[str, Any],
    *,
    as_float: Any,
) -> dict[str, dict[str, float]]:
    raw = binding_cfg.get(_CONTRACT_CORE_MAP_KEY)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY}.{_CONTRACT_CORE_MAP_KEY} must be a non-empty dict"
        )
    result: dict[str, dict[str, float]] = {}
    for contract, raw_core in raw.items():
        if not isinstance(contract, str) or not contract:
            raise ValueError(f"{_CONTRACT_CORE_MAP_KEY} keys must be non-empty strings")
        if not isinstance(raw_core, dict):
            raise ValueError(f"{_CONTRACT_CORE_MAP_KEY}[{contract}] must be a dict")
        result[contract] = {
            "time_scale_lambda": as_float(
                raw_core.get("time_scale_lambda"),
                f"{_CONTRACT_CORE_MAP_KEY}[{contract}].time_scale_lambda",
            ),
            "cancel_bias_k": as_float(
                raw_core.get("cancel_bias_k"),
                f"{_CONTRACT_CORE_MAP_KEY}[{contract}].cancel_bias_k",
            ),
        }
    return result


def _resolve_machine_delay(machine_delay_map: dict[str, int], machine: str) -> int:
    if machine not in machine_delay_map:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY}.{_MACHINE_DELAY_MAP_KEY}[{machine}] is required"
        )
    return machine_delay_map[machine]


def _resolve_contract_core(
    contract_core_map: dict[str, dict[str, float]],
    contract: str,
) -> dict[str, float]:
    if contract not in contract_core_map:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY}.{_CONTRACT_CORE_MAP_KEY}[{contract}] is required"
        )
    return contract_core_map[contract]


def _require_dataset_label(value: str | None, *, dataset_id: str, key: str) -> str:
    if value is None:
        raise ValueError(f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY}[{dataset_id}].{key} is required")
    return value
