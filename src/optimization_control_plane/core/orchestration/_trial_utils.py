from __future__ import annotations

import hashlib

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ObjectiveResult,
    RunSpec,
    stable_json_serialize,
)


def validate_run_spec(run_spec: RunSpec) -> None:
    command = run_spec.job.command
    script_path = run_spec.job.script_path
    has_command = command is not None and len(command) > 0
    has_script = script_path is not None and script_path.strip() != ""
    if has_command == has_script:
        raise ValueError("run_spec.job must set exactly one of command or script_path")
    if has_command and any((not isinstance(part, str) or part == "") for part in command or []):
        raise ValueError("run_spec.job.command must contain non-empty strings")
    if has_script and (not isinstance(script_path, str) or script_path.strip() == ""):
        raise ValueError("run_spec.job.script_path must be a non-empty string")
    if run_spec.job.working_dir is not None and run_spec.job.working_dir.strip() == "":
        raise ValueError("run_spec.job.working_dir must be non-empty when provided")
    for key, value in run_spec.job.env.items():
        if not isinstance(key, str) or not isinstance(value, str) or key == "" or value == "":
            raise ValueError("run_spec.job.env keys and values must be non-empty strings")
    for name, amount in (
        ("cpu_cores", run_spec.resource_request.cpu_cores),
        ("memory_mb", run_spec.resource_request.memory_mb),
        ("gpu_count", run_spec.resource_request.gpu_count),
        ("max_runtime_seconds", run_spec.resource_request.max_runtime_seconds),
    ):
        if amount is not None and amount <= 0:
            raise ValueError(f"run_spec.resource_request.{name} must be > 0 when provided")


def scope_objective_key(raw_key: str, groundtruth_fingerprint: str) -> str:
    return f"{raw_key}::gt={groundtruth_fingerprint}"


def with_shared_run_attrs(result: ObjectiveResult, leader_trial_ids: set[str]) -> ObjectiveResult:
    if not leader_trial_ids:
        return result
    attrs = dict(result.attrs)
    attrs["shared_run"] = True
    attrs["shared_run_leader_trial_ids"] = sorted(leader_trial_ids)
    return ObjectiveResult(value=result.value, attrs=attrs, artifact_refs=result.artifact_refs)


def build_trial_objective_key(
    *,
    params: dict[str, object],
    dataset_ids: tuple[str, ...],
    spec: ExperimentSpec,
    groundtruth_fingerprint: str,
) -> str:
    payload = stable_json_serialize(
        {
            "kind": "trial_objective",
            "spec_id": spec.spec_id,
            "spec_hash": spec.spec_hash,
            "params": params,
            "dataset_ids": sorted(dataset_ids),
            "objective_config": spec.objective_config,
            "groundtruth_fingerprint": groundtruth_fingerprint,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"trial_obj:{digest[:24]}"
