from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    GroundTruthData,
    stable_json_serialize,
)

_GROUNDTRUTH_KEY = "groundtruth"
_DATASETS_KEY = "datasets"
_DONEINFO_PATH_KEY = "doneinfo_path"
_EXECUTIONDETAIL_PATH_KEY = "executiondetail_path"
_DONEINFO_NAME_PATTERN = re.compile(
    r"^PubOrderDoneInfoLog_(?P<machine_name>.+)_(?P<time>\d{8})_(?P<contract_id>.+)\.csv$"
)
_EXECUTIONDETAIL_NAME_PATTERN = re.compile(
    r"^PubExecutionDetailLog_(?P<machine_name>.+)_(?P<time>\d{8})_(?P<contract_id>.+)\.csv$"
)
_FINGERPRINT_KIND = "backtest_groundtruth_v1"
_SHA_PREFIX = "sha256:"


class BackTestGroundTruthProviderAdapter:
    """Load BackTestSys ground truth and compute stable fingerprint."""

    def load(self, spec: ExperimentSpec, dataset_id: str) -> GroundTruthData:
        groundtruth_cfg = _read_groundtruth_config(spec, dataset_id)
        doneinfo_path = _read_csv_path(groundtruth_cfg, _DONEINFO_PATH_KEY)
        executiondetail_path = _read_csv_path(groundtruth_cfg, _EXECUTIONDETAIL_PATH_KEY)

        doneinfo_identity = _parse_filename_identity(
            filename=doneinfo_path.name,
            pattern=_DONEINFO_NAME_PATTERN,
            table_name="PubOrderDoneInfoLog",
        )
        executiondetail_identity = _parse_filename_identity(
            filename=executiondetail_path.name,
            pattern=_EXECUTIONDETAIL_NAME_PATTERN,
            table_name="PubExecutionDetailLog",
        )
        identity = _validate_same_identity(doneinfo_identity, executiondetail_identity)

        payload = {
            "DoneInfo": _read_csv_rows(doneinfo_path, "DoneInfo"),
            "ExecutionDetail": _read_csv_rows(executiondetail_path, "ExecutionDetail"),
        }
        return GroundTruthData(
            payload=payload,
            fingerprint=_build_fingerprint(identity),
        )


def _read_groundtruth_config(spec: ExperimentSpec, dataset_id: str) -> dict[str, Any]:
    groundtruth_cfg = spec.objective_config.get(_GROUNDTRUTH_KEY)
    if not isinstance(groundtruth_cfg, dict):
        raise ValueError(f"spec.objective_config.{_GROUNDTRUTH_KEY} must be a dict")
    if dataset_id == "":
        return dict(groundtruth_cfg)
    datasets_cfg = groundtruth_cfg.get(_DATASETS_KEY)
    if datasets_cfg is None:
        return dict(groundtruth_cfg)
    if not isinstance(datasets_cfg, dict):
        raise ValueError(f"{_GROUNDTRUTH_KEY}.{_DATASETS_KEY} must be a dict")
    dataset_cfg = datasets_cfg.get(dataset_id)
    if not isinstance(dataset_cfg, dict):
        raise ValueError(
            f"{_GROUNDTRUTH_KEY}.{_DATASETS_KEY}[{dataset_id}] must be a dict"
        )
    return dict(dataset_cfg)


def _read_csv_path(config: dict[str, Any], key: str) -> Path:
    raw_path = config.get(key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{_GROUNDTRUTH_KEY}.{key} must be a non-empty string")
    path = Path(raw_path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{_GROUNDTRUTH_KEY}.{key} must be a .csv file path")
    if not path.is_file():
        raise FileNotFoundError(f"ground truth file not found: {path}")
    return path


def _parse_filename_identity(
    *,
    filename: str,
    pattern: re.Pattern[str],
    table_name: str,
) -> dict[str, str]:
    match = pattern.match(filename)
    if match is None:
        raise ValueError(
            f"{table_name} filename must match required pattern, got: {filename}"
        )
    machine_name = match.group("machine_name").strip()
    day = match.group("time").strip()
    contract_id = match.group("contract_id").strip()
    if not machine_name or not day or not contract_id:
        raise ValueError(f"{table_name} filename identity parts must be non-empty")
    return {
        "machine_name": machine_name,
        "time": day,
        "contract_id": contract_id,
    }


def _validate_same_identity(
    doneinfo_identity: dict[str, str],
    executiondetail_identity: dict[str, str],
) -> dict[str, str]:
    if doneinfo_identity != executiondetail_identity:
        raise ValueError(
            "ground truth filename identity mismatch between "
            f"DoneInfo and ExecutionDetail: "
            f"{doneinfo_identity} != {executiondetail_identity}"
        )
    return dict(doneinfo_identity)


def _read_csv_rows(csv_path: Path, table_name: str) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"{table_name} csv must contain a header row")
        return [dict(row) for row in reader]


def _build_fingerprint(identity: dict[str, str]) -> str:
    payload = stable_json_serialize(
        {
            "kind": _FINGERPRINT_KIND,
            "machine_name": identity["machine_name"],
            "time": identity["time"],
            "contract_id": identity["contract_id"],
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_SHA_PREFIX}{digest}"
