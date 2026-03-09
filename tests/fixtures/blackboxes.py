from __future__ import annotations

from typing import Any

_instance_seq = 0


def reset_instance_counter() -> None:
    global _instance_seq
    _instance_seq = 0


def echo_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": {"metric_1": float(config.get("x", 0.0))},
        "diagnostics": {"echo": True},
        "artifact_refs": [],
    }


def callable_target(config: dict[str, Any]) -> dict[str, Any]:
    val = float(config.get("score", 0.0))
    return {
        "metrics": {"metric_1": val},
        "diagnostics": {"kind": "callable"},
        "artifact_refs": [],
    }


def return_invalid_result(config: dict[str, Any]) -> dict[str, Any]:
    _ = config
    return {"diagnostics": {"bad": True}}


class StatefulClassTarget:
    def __init__(self) -> None:
        global _instance_seq
        _instance_seq += 1
        self._instance_id = _instance_seq
        self._calls = 0

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        _ = config
        self._calls += 1
        return {
            "metrics": {
                "metric_1": float(self._instance_id),
                "call_count": float(self._calls),
            },
            "diagnostics": {
                "instance_id": self._instance_id,
                "call_count": self._calls,
            },
            "artifact_refs": [],
        }
