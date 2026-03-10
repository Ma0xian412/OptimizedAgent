from __future__ import annotations

import hashlib
from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, RunSpec, stable_json_serialize


class BackTestSysRunKeyBuilder:
    """Build stable run_key from run_spec + spec meta."""

    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str:
        payload = stable_json_serialize(
            {
                "kind": run_spec.kind,
                "config": run_spec.config,
                "meta": spec.meta,
            }
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"run:{digest}"


class BackTestSysObjectiveKeyBuilder:
    """Build stable objective_key from run_key + objective identity."""

    def build(self, run_key: str, objective_config: dict[str, Any]) -> str:
        payload = stable_json_serialize(
            {
                "run_key": run_key,
                "name": objective_config.get("name"),
                "version": objective_config.get("version"),
                "params": objective_config.get("params", {}),
            }
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"obj:{digest}"
