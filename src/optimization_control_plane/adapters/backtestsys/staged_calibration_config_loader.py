from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    CalibrationConfig,
    DatasetDefinition,
    FloatRange,
    IntRange,
)

_RESOURCE_KEYS = frozenset({"cpu", "memory_mb", "memory_gb", "gpu", "max_runtime_seconds"})


def load_calibration_config(config_path: Path) -> CalibrationConfig:
    _ensure_absolute_path(config_path, "--config")
    if config_path.suffix.lower() != ".xml":
        raise ValueError(f"--config must point to an .xml file: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    root = ET.parse(config_path).getroot()
    datasets = _read_datasets(root)
    config = CalibrationConfig(
        workspace_root=_read_required_abs_path(root, "workspace_root"),
        runtime_root=_read_required_abs_path(root, "runtime_root"),
        backtestsys_root=_read_required_abs_path(root, "backtestsys_root"),
        base_config_path=_read_required_abs_path(root, "base_config_path"),
        python_executable=_read_required_abs_path_str(root, "python_executable"),
        datasets=datasets,
        max_failures=_read_positive_int(root, "max_failures"),
        baseline_trials=_read_positive_int(root, "baseline_trials"),
        machine_delay_trials=_read_positive_int(root, "machine_delay_trials"),
        contract_core_trials=_read_positive_int(root, "contract_core_trials"),
        verify_trials=_read_positive_int(root, "verify_trials"),
        max_in_flight_trials=_read_optional_positive_int(root, "max_in_flight_trials", default=1),
        default_resources=_read_default_resources(root),
        delay_range=_read_int_range(root, ("search_ranges", "delay")),
        time_scale_lambda_range=_read_float_range(root, ("search_ranges", "time_scale_lambda")),
        cancel_bias_k_range=_read_float_range(root, ("search_ranges", "cancel_bias_k")),
    )
    _validate_workspace_relation(config, config_path)
    return config


def calibration_config_summary(config: CalibrationConfig, *, source: Path) -> dict[str, object]:
    return {
        "config_path": str(source),
        "workspace_root": str(config.workspace_root),
        "runtime_root": str(config.runtime_root),
        "backtestsys_root": str(config.backtestsys_root),
        "base_config_path": str(config.base_config_path),
        "python_executable": str(config.python_executable),
        "trials": {
            "baseline": config.baseline_trials,
            "machine_delay": config.machine_delay_trials,
            "contract_core": config.contract_core_trials,
            "verify": config.verify_trials,
            "max_failures": config.max_failures,
        },
        "parallelism": {"max_in_flight_trials": config.max_in_flight_trials},
        "default_resources": dict(config.default_resources),
        "search_ranges": {
            "delay": {"low": config.delay_range.low, "high": config.delay_range.high},
            "time_scale_lambda": {
                "low": config.time_scale_lambda_range.low,
                "high": config.time_scale_lambda_range.high,
            },
            "cancel_bias_k": {
                "low": config.cancel_bias_k_range.low,
                "high": config.cancel_bias_k_range.high,
            },
        },
        "dataset_count": len(config.datasets),
        "datasets": [_dataset_summary(dataset) for dataset in config.datasets],
    }


def _dataset_summary(dataset: DatasetDefinition) -> dict[str, object]:
    return {
        "dataset_id": dataset.dataset_id,
        "machine": dataset.machine,
        "contract": dataset.contract,
        "market_data_path": str(dataset.market_data_path),
        "order_file": str(dataset.order_file),
        "cancel_file": str(dataset.cancel_file),
        "groundtruth_doneinfo_path": str(dataset.groundtruth_doneinfo_path),
        "groundtruth_executiondetail_path": str(dataset.groundtruth_executiondetail_path),
    }


def _read_default_resources(root: ET.Element) -> dict[str, int]:
    node = root.find("default_resources")
    if node is None:
        raise ValueError("config.default_resources is required")
    result: dict[str, int] = {}
    for child in list(node):
        if child.tag not in _RESOURCE_KEYS:
            raise ValueError(f"config.default_resources.{child.tag} is not supported")
        result[child.tag] = _parse_positive_int(
            _read_node_text(child, f"config.default_resources.{child.tag}"), f"config.default_resources.{child.tag}"
        )
    if not result:
        raise ValueError("config.default_resources must contain at least one resource key")
    return result


def _read_datasets(root: ET.Element) -> tuple[DatasetDefinition, ...]:
    datasets_node = root.find("datasets")
    if datasets_node is None:
        raise ValueError("config.datasets is required")
    datasets: list[DatasetDefinition] = []
    for index, dataset_node in enumerate(datasets_node.findall("dataset")):
        prefix = f"config.datasets.dataset[{index}]"
        datasets.append(
            DatasetDefinition(
                dataset_id=_read_required_text(dataset_node, "dataset_id", f"{prefix}.dataset_id"),
                market_data_path=_read_required_abs_path(dataset_node, "market_data_path", prefix=prefix),
                order_file=_read_required_abs_path(dataset_node, "order_file", prefix=prefix),
                cancel_file=_read_required_abs_path(dataset_node, "cancel_file", prefix=prefix),
                machine=_read_required_text(dataset_node, "machine", f"{prefix}.machine"),
                contract=_read_required_text(dataset_node, "contract", f"{prefix}.contract"),
                groundtruth_doneinfo_path=_read_required_abs_path(
                    dataset_node, "groundtruth_doneinfo_path", prefix=prefix
                ),
                groundtruth_executiondetail_path=_read_required_abs_path(
                    dataset_node,
                    "groundtruth_executiondetail_path",
                    prefix=prefix,
                ),
            )
        )
    if not datasets:
        raise ValueError("config.datasets must contain at least one dataset")
    _validate_unique_dataset_ids(datasets)
    return tuple(datasets)


def _validate_unique_dataset_ids(datasets: list[DatasetDefinition]) -> None:
    seen: set[str] = set()
    for dataset in datasets:
        if dataset.dataset_id in seen:
            raise ValueError(f"config.datasets contains duplicate dataset_id: {dataset.dataset_id}")
        seen.add(dataset.dataset_id)


def _validate_workspace_relation(config: CalibrationConfig, config_path: Path) -> None:
    if config.workspace_root not in config_path.parents and config.workspace_root != config_path.parent:
        raise ValueError(
            "config.workspace_root must be an ancestor of --config file path: "
            f"{config.workspace_root} !<= {config_path}"
        )


def _read_required_abs_path(root: ET.Element, key: str, *, prefix: str = "config") -> Path:
    raw = _read_required_text(root, key, f"{prefix}.{key}")
    path = Path(raw)
    _ensure_absolute_path(path, f"{prefix}.{key}")
    return path


def _read_required_abs_path_str(root: ET.Element, key: str) -> str:
    path = _read_required_abs_path(root, key)
    return str(path)


def _read_required_text(root: ET.Element, key: str, field_name: str) -> str:
    node = root.find(key)
    if node is None:
        raise ValueError(f"{field_name} is required")
    return _read_node_text(node, field_name)


def _read_node_text(node: ET.Element, field_name: str) -> str:
    if node.text is None or node.text.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return node.text.strip()


def _read_positive_int(root: ET.Element, key: str) -> int:
    raw = _read_required_text(root, key, f"config.{key}")
    return _parse_positive_int(raw, f"config.{key}")


def _read_optional_positive_int(root: ET.Element, key: str, *, default: int) -> int:
    node = root.find(key)
    if node is None:
        return default
    return _parse_positive_int(_read_node_text(node, f"config.{key}"), f"config.{key}")


def _parse_positive_int(raw: str, field_name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive int, got: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive int, got: {value}")
    return value


def _read_int_range(root: ET.Element, path: tuple[str, str]) -> IntRange:
    node = _read_required_node(root, path, f"config.{'.'.join(path)}")
    low = _parse_int(_read_required_text(node, "low", f"config.{'.'.join(path)}.low"), f"config.{'.'.join(path)}.low")
    high = _parse_int(
        _read_required_text(node, "high", f"config.{'.'.join(path)}.high"), f"config.{'.'.join(path)}.high"
    )
    if low > high:
        raise ValueError(f"config.{'.'.join(path)} requires low <= high")
    return IntRange(low=low, high=high)


def _read_float_range(root: ET.Element, path: tuple[str, str]) -> FloatRange:
    node = _read_required_node(root, path, f"config.{'.'.join(path)}")
    low = _parse_float(_read_required_text(node, "low", f"config.{'.'.join(path)}.low"), f"config.{'.'.join(path)}.low")
    high = _parse_float(
        _read_required_text(node, "high", f"config.{'.'.join(path)}.high"), f"config.{'.'.join(path)}.high"
    )
    if low > high:
        raise ValueError(f"config.{'.'.join(path)} requires low <= high")
    return FloatRange(low=low, high=high)


def _read_required_node(root: ET.Element, path: tuple[str, ...], field_name: str) -> ET.Element:
    node: ET.Element | None = root
    for tag in path:
        node = node.find(tag) if node is not None else None
    if node is None:
        raise ValueError(f"{field_name} is required")
    return node


def _parse_int(raw: str, field_name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an int, got: {raw}") from exc


def _parse_float(raw: str, field_name: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a float, got: {raw}") from exc


def _ensure_absolute_path(path: Path, field_name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path: {path}")
