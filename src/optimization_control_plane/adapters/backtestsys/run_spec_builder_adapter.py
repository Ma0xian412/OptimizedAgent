from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from typing import Any

from optimization_control_plane.adapters.backtestsys.run_spec_binding import (
    DatasetInput,
    read_param_binding_config,
    resolve_dataset_input,
    resolve_effective_params,
)
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    Job,
    ResourceRequest,
    RunSpec,
    stable_json_serialize,
)

_RUN_SPEC_KEY = "backtest_run_spec"
_REQUIRED_PARAM_NAMES = ("time_scale_lambda", "cancel_bias_k", "delay_in", "delay_out")


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
        binding = read_param_binding_config(run_cfg, as_int=_as_int, as_float=_as_float)
        dataset_input = resolve_dataset_input(run_cfg, dataset_id)
        effective_params = resolve_effective_params(
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


def _build_digest(spec: ExperimentSpec, dataset_id: str, params: dict[str, object]) -> str:
    payload = stable_json_serialize(
        {"spec_hash": spec.spec_hash, "dataset_id": dataset_id, "params": params}
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


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"param {name} must be float-like")
    return float(value)


def _as_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"param {name} must be int")
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
