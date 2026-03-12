from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, RunSpec, stable_json_serialize

_RUN_KEY_KIND = "backtest_run_key_v1"
_CONFIG_FLAG = "--config"


class BackTestRunKeyBuilderAdapter:
    """Build stable run key for BackTestSys execution payload."""

    def build(
        self,
        run_spec: RunSpec,
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> str:
        config_path = _extract_config_path(run_spec)
        config_payload = _read_config_payload(config_path)
        payload = {
            "kind": _RUN_KEY_KIND,
            "spec_hash": spec.spec_hash,
            "dataset_id": dataset_id,
            "meta": spec.meta,
            "config": config_payload,
        }
        digest = hashlib.sha256(stable_json_serialize(payload).encode("utf-8")).hexdigest()
        return f"run:{digest[:24]}"


def _extract_config_path(run_spec: RunSpec) -> str:
    args = run_spec.job.args
    for idx, arg in enumerate(args):
        if arg != _CONFIG_FLAG:
            continue
        if idx + 1 >= len(args):
            raise ValueError("run_spec.job.args missing value for --config")
        config_path = args[idx + 1]
        if not config_path:
            raise ValueError("run_spec.job.args config path must be non-empty")
        return config_path
    raise ValueError("run_spec.job.args must include --config <path>")


def _read_config_payload(config_path: str) -> dict[str, object]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    root = ET.parse(config_path).getroot()
    payload = _element_to_payload(root)
    if not isinstance(payload, dict):
        raise ValueError("config root must contain nested elements")
    return payload


def _element_to_payload(element: ET.Element) -> object:
    children = [child for child in list(element) if not callable(child.tag)]
    if not children:
        return (element.text or "").strip()
    grouped: dict[str, list[object]] = {}
    for child in children:
        grouped.setdefault(str(child.tag), []).append(_element_to_payload(child))
    result: dict[str, object] = {}
    for key in sorted(grouped):
        values = grouped[key]
        result[key] = values[0] if len(values) == 1 else values
    return result
