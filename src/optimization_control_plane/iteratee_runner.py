from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExecutionRequest

_DEFAULT_LR = 1e-3
_DEFAULT_BATCH_SIZE = 64.0
_DEFAULT_TARGET_X = 3.0
_DEFAULT_TARGET_Y = 20.0
_Y_SCALE = 400.0
_LR_SCALE = 10.0


def run_iteratee(request: ExecutionRequest) -> dict[str, Any]:
    """Example real iteratee entrypoint used by PythonCallableExecutionBackend."""
    cfg = request.run_spec.config
    x = _read_float(cfg, "x")
    y = _read_float(cfg, "y")
    lr = float(cfg.get("lr", _DEFAULT_LR))
    batch_size = float(cfg.get("batch_size", _DEFAULT_BATCH_SIZE))
    model = str(cfg.get("model", "baseline"))

    model_penalty = 0.1 if model == "enhanced" else 0.0
    score = ((x - _DEFAULT_TARGET_X) ** 2) + (((y - _DEFAULT_TARGET_Y) ** 2) / _Y_SCALE)
    score += lr * _LR_SCALE
    score += model_penalty

    return {
        "metrics": {
            "metric_1": float(score),
            "loss": float(score),
        },
        "diagnostics": {
            "executor_kind": request.run_spec.kind,
            "trial_id": request.trial_id,
            "batch_size": batch_size,
        },
        "artifact_refs": [],
    }


def _read_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if value is None:
        raise ValueError(f"missing required run_spec.config key: {key}")
    return float(value)
