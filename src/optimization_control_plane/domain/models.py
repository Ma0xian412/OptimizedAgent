from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from optimization_control_plane.domain.enums import EventKind, SamplingMode


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_spec.target_id must be a non-empty string")
        if not isinstance(self.config, dict):
            raise ValueError("target_spec.config must be a dict")
        object.__setattr__(self, "target_id", self.target_id.strip())
        object.__setattr__(self, "config", dict(self.config))

    def to_dict(self) -> dict[str, Any]:
        return {"target_id": self.target_id, "config": dict(self.config)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetSpec:
        target_id = payload.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_spec.target_id must be a non-empty string")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("target_spec.config must be a dict")
        return cls(target_id=target_id, config=dict(config))

    def __hash__(self) -> int:
        return hash((self.target_id, stable_json_serialize(self.config)))


def validate_target_spec(
    value: Any,
    *,
    source: str = "target_spec",
) -> TargetSpec:
    if not isinstance(value, TargetSpec):
        raise ValueError(f"{source} must be a TargetSpec")
    if not isinstance(value.target_id, str) or not value.target_id.strip():
        raise ValueError(f"{source}.target_id must be a non-empty string")
    if not isinstance(value.config, dict):
        raise ValueError(f"{source}.config must be a dict")
    return value


@dataclass(frozen=True)
class ExperimentSpec:
    spec_id: str
    spec_hash: str
    meta: dict[str, Any]
    target_spec: TargetSpec
    objective_config: dict[str, Any]
    execution_config: dict[str, Any]


@dataclass(frozen=True)
class StudyHandle:
    study_id: str
    name: str
    spec_hash: str
    direction: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class TrialHandle:
    study_id: str
    trial_id: str
    number: int
    state: str


@dataclass(frozen=True)
class RunSpec:
    kind: str
    config: dict[str, Any]
    resources: dict[str, Any]
    target_spec: TargetSpec


@dataclass(frozen=True)
class Checkpoint:
    step: int
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]
    artifact_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    attrs: dict[str, Any]
    artifact_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    trial_id: str
    run_key: str
    objective_key: str
    cohort_id: str | None
    priority: int
    run_spec: RunSpec


@dataclass(frozen=True)
class RunHandle:
    handle_id: str
    request_id: str
    state: str


@dataclass(frozen=True)
class ExecutionEvent:
    kind: EventKind
    handle_id: str
    step: int | None = None
    checkpoint: Checkpoint | None = None
    run_result: RunResult | None = None
    reason: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class SamplerProfile:
    mode: SamplingMode
    startup_trials: int
    batch_size: int
    pending_policy: str
    recommended_max_inflight: int | None


def stable_json_serialize(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_spec_hash(
    spec_id: str,
    meta: dict[str, Any],
    target_spec: TargetSpec,
    objective_config: dict[str, Any],
    execution_config: dict[str, Any],
) -> str:
    payload = stable_json_serialize({
        "spec_id": spec_id,
        "meta": meta,
        "target_spec": target_spec.to_dict(),
        "objective_config": objective_config,
        "execution_config": execution_config,
    })
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
