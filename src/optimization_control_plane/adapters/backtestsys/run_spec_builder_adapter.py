from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import xml.etree.ElementTree as ET
from typing import Any

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    Job,
    ResourceRequest,
    RunSpec,
    stable_json_serialize,
)

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
_REQUIRED_PARAM_NAMES = (
    "time_scale_lambda",
    "cancel_bias_k",
    "delay_in",
    "delay_out",
)


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


class BackTestRunSpecBuilderAdapter:
    """Build executable RunSpec for BackTestSys by materializing trial config.xml."""

    def build(
        self,
        params: dict[str, object],
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> RunSpec:
        run_cfg = _read_run_spec_config(spec)
        trial_params = _normalize_params(params)
        binding = _read_param_binding_config(run_cfg)
        dataset_input = _resolve_dataset_input(run_cfg, dataset_id)
        effective_params = _resolve_effective_params(
            dataset_id=dataset_id,
            trial_params=trial_params,
            dataset_input=dataset_input,
            binding=binding,
        )
        digest = _build_digest(spec, dataset_id, effective_params)
        output_root = _read_required_string(run_cfg, "output_root_dir")
        config_path = os.path.join(output_root, "configs", f"{dataset_id}_{digest}.xml")
        result_dir = os.path.join(output_root, "results", f"{dataset_id}_{digest}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        _write_trial_config(
            base_config_path=_read_required_string(run_cfg, "base_config_path"),
            config_path=config_path,
            params=effective_params,
            dataset_input=dataset_input,
        )
        return RunSpec(
            job=_build_job(run_cfg, config_path, result_dir),
            result_path=result_dir,
            resource_request=_build_resource_request(spec.execution_config),
        )


def _read_run_spec_config(spec: ExperimentSpec) -> dict[str, Any]:
    value = spec.execution_config.get(_RUN_SPEC_KEY)
    if not isinstance(value, dict):
        raise ValueError(f"spec.execution_config.{_RUN_SPEC_KEY} must be a dict")
    return value


def _read_param_binding_config(run_cfg: dict[str, Any]) -> ParamBindingConfig:
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
    machine_delay_map = _read_machine_delay_map(raw)
    contract_core_map = _read_contract_core_map(raw)
    return ParamBindingConfig(
        mode=mode,
        machine_delay_map=machine_delay_map,
        contract_core_map=contract_core_map,
    )


def _normalize_params(params: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for name in _REQUIRED_PARAM_NAMES:
        if name not in params:
            raise ValueError(f"missing required param: {name}")
    normalized["time_scale_lambda"] = _as_float(params["time_scale_lambda"], "time_scale_lambda")
    normalized["cancel_bias_k"] = _as_float(params["cancel_bias_k"], "cancel_bias_k")
    normalized["delay_in"] = _as_int(params["delay_in"], "delay_in")
    normalized["delay_out"] = _as_int(params["delay_out"], "delay_out")
    return normalized


def _resolve_effective_params(
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


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"param {name} must be float-like")
    return float(value)


def _as_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"param {name} must be int")
    return value


def _build_digest(
    spec: ExperimentSpec,
    dataset_id: str,
    params: dict[str, object],
) -> str:
    payload = stable_json_serialize(
        {
            "spec_hash": spec.spec_hash,
            "dataset_id": dataset_id,
            "params": params,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _write_trial_config(
    *,
    base_config_path: str,
    config_path: str,
    params: dict[str, object],
    dataset_input: DatasetInput,
) -> None:
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"base config.xml not found: {base_config_path}")
    tree = ET.parse(base_config_path)
    root = tree.getroot()
    _set_xml_text(root, ("tape", "time_scale_lambda"), str(params["time_scale_lambda"]))
    _set_xml_text(root, ("exchange", "cancel_bias_k"), str(params["cancel_bias_k"]))
    _set_xml_text(root, ("runner", "delay_in"), str(params["delay_in"]))
    _set_xml_text(root, ("runner", "delay_out"), str(params["delay_out"]))
    _set_xml_text(root, ("data", "path"), dataset_input.market_data_path)
    _set_xml_text(root, ("strategy", "params", "order_file"), dataset_input.order_file)
    _set_xml_text(root, ("strategy", "params", "cancel_file"), dataset_input.cancel_file)
    tree.write(config_path, encoding="utf-8", xml_declaration=True)


def _set_xml_text(root: ET.Element, path: tuple[str, ...], value: str) -> None:
    current = root
    for tag in path:
        nxt = current.find(tag)
        if nxt is None:
            nxt = ET.SubElement(current, tag)
        current = nxt
    current.text = value


def _resolve_dataset_input(run_cfg: dict[str, Any], dataset_id: str) -> DatasetInput:
    dataset_inputs = run_cfg.get(_DATASET_INPUTS_KEY)
    if not isinstance(dataset_inputs, dict) or not dataset_inputs:
        raise ValueError(f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY} must be a non-empty dict")
    raw_input = dataset_inputs.get(dataset_id)
    if not isinstance(raw_input, dict):
        raise ValueError(f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY}[{dataset_id}] must be a dict")
    market_data_path = _read_required_dataset_input_path(
        raw_input,
        dataset_id=dataset_id,
        key=_MARKET_DATA_PATH_KEY,
    )
    order_file = _read_required_dataset_input_path(
        raw_input,
        dataset_id=dataset_id,
        key=_ORDER_FILE_KEY,
    )
    cancel_file = _read_required_dataset_input_path(
        raw_input,
        dataset_id=dataset_id,
        key=_CANCEL_FILE_KEY,
    )
    machine = _read_optional_dataset_input_label(
        raw_input,
        dataset_id=dataset_id,
        key=_MACHINE_KEY,
    )
    contract = _read_optional_dataset_input_label(
        raw_input,
        dataset_id=dataset_id,
        key=_CONTRACT_KEY,
    )
    return DatasetInput(
        market_data_path=market_data_path,
        order_file=order_file,
        cancel_file=cancel_file,
        machine=machine,
        contract=contract,
    )


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


def _read_machine_delay_map(binding_cfg: dict[str, Any]) -> dict[str, int]:
    raw = binding_cfg.get(_MACHINE_DELAY_MAP_KEY)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_PARAM_BINDING_KEY}.{_MACHINE_DELAY_MAP_KEY} must be a non-empty dict"
        )
    result: dict[str, int] = {}
    for machine, raw_delay in raw.items():
        if not isinstance(machine, str) or not machine:
            raise ValueError(f"{_MACHINE_DELAY_MAP_KEY} keys must be non-empty strings")
        result[machine] = _as_int(raw_delay, f"{_MACHINE_DELAY_MAP_KEY}[{machine}]")
    return result


def _read_contract_core_map(binding_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
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
            "time_scale_lambda": _as_float(
                raw_core.get("time_scale_lambda"),
                f"{_CONTRACT_CORE_MAP_KEY}[{contract}].time_scale_lambda",
            ),
            "cancel_bias_k": _as_float(
                raw_core.get("cancel_bias_k"),
                f"{_CONTRACT_CORE_MAP_KEY}[{contract}].cancel_bias_k",
            ),
        }
    return result


def _require_dataset_label(value: str | None, *, dataset_id: str, key: str) -> str:
    if value is None:
        raise ValueError(f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY}[{dataset_id}].{key} is required")
    return value


def _build_job(run_cfg: dict[str, Any], config_path: str, result_dir: str) -> Job:
    root_dir = _read_required_string(run_cfg, "backtestsys_root")
    python_executable = _read_optional_string(run_cfg, "python_executable", "python3")
    main_relpath = _read_optional_string(run_cfg, "main_relpath", "main.py")
    main_path = os.path.join(root_dir, main_relpath)
    return Job(
        command=[python_executable, main_path],
        args=["--config", config_path, "--save-result", result_dir],
        working_dir=root_dir,
    )


def _build_resource_request(execution_config: dict[str, Any]) -> ResourceRequest:
    default_resources = execution_config.get("default_resources", {})
    if not isinstance(default_resources, dict):
        raise ValueError("spec.execution_config.default_resources must be a dict")
    cpu_cores = _read_optional_int(default_resources, "cpu")
    memory_mb = _read_optional_memory_mb(default_resources)
    gpu_count = _read_optional_int(default_resources, "gpu")
    runtime = _read_optional_int(default_resources, "max_runtime_seconds")
    return ResourceRequest(
        cpu_cores=cpu_cores,
        memory_mb=memory_mb,
        gpu_count=gpu_count,
        max_runtime_seconds=runtime,
    )


def _read_optional_memory_mb(default_resources: dict[str, Any]) -> int | None:
    memory_mb = default_resources.get("memory_mb")
    if memory_mb is not None:
        return _as_positive_int(memory_mb, "memory_mb")
    memory_gb = default_resources.get("memory_gb")
    if memory_gb is None:
        return None
    return _as_positive_int(memory_gb, "memory_gb") * 1024


def _read_optional_int(source: dict[str, Any], key: str) -> int | None:
    raw = source.get(key)
    if raw is None:
        return None
    return _as_positive_int(raw, key)


def _as_positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive int")
    return value


def _read_required_string(source: dict[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_RUN_SPEC_KEY}.{key} must be a non-empty string")
    return value


def _read_optional_string(source: dict[str, Any], key: str, default: str) -> str:
    value = source.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_RUN_SPEC_KEY}.{key} must be a non-empty string")
    return value
