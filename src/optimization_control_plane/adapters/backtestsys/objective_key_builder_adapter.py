from __future__ import annotations

import hashlib
from typing import Any

from optimization_control_plane.domain.models import stable_json_serialize

_OBJECTIVE_KEY_KIND = "backtest_objective_key_v1"


class BackTestObjectiveKeyBuilderAdapter:
    """Build stable objective cache key for BackTestSys."""

    def build(self, run_key: str, objective_config: dict[str, object]) -> str:
        name = _read_required_string(objective_config, "name")
        version = _read_required_string(objective_config, "version")
        direction = _read_required_string(objective_config, "direction")
        params = _read_optional_params(objective_config)
        payload = {
            "kind": _OBJECTIVE_KEY_KIND,
            "run_key": run_key,
            "name": name,
            "version": version,
            "direction": direction,
            "params": params,
        }
        digest = hashlib.sha256(stable_json_serialize(payload).encode("utf-8")).hexdigest()
        return f"obj:{digest[:24]}"


def _read_required_string(source: dict[str, object], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"objective_config.{key} must be a non-empty string")
    return value


def _read_optional_params(source: dict[str, object]) -> dict[str, Any]:
    raw = source.get("params")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("objective_config.params must be a dict")
    return dict(raw)
