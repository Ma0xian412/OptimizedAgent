from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from optimization_control_plane.domain.enums import EventKind, JobStatus, SamplingMode


@dataclass(frozen=True)
class ExperimentSpec:
    spec_id: str
    spec_hash: str
    meta: dict[str, Any]
    objective_config: dict[str, Any]
    execution_config: dict[str, Any]


@dataclass(frozen=True)
class GroundTruthData:
    payload: object
    fingerprint: str


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
class Job:
    command: list[str] | None = None
    script_path: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None


@dataclass(frozen=True)
class ResourceRequest:
    cpu_cores: int | None = None
    memory_mb: int | None = None
    gpu_count: int | None = None
    max_runtime_seconds: int | None = None


@dataclass(frozen=True)
class RunSpec:
    job: Job
    result_path: str
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)


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
    state: JobStatus


@dataclass(frozen=True)
class ExecutionEvent:
    kind: EventKind
    handle_id: str
    step: int | None = None
    checkpoint: Checkpoint | None = None
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
    objective_config: dict[str, Any],
    execution_config: dict[str, Any],
) -> str:
    payload = stable_json_serialize({
        "spec_id": spec_id,
        "meta": meta,
        "objective_config": objective_config,
        "execution_config": execution_config,
    })
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
