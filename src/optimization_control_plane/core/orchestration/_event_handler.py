"""Event handling logic extracted from TrialOrchestrator to respect file-size limits."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
    TrialBinding,
)
from optimization_control_plane.domain.enums import EventKind, TrialState
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExperimentSpec,
    GroundTruthData,
    ObjectiveResult,
    SamplerProfile,
)
from optimization_control_plane.domain.state import StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.result_store import ResultStore

logger = logging.getLogger(__name__)


@dataclass
class EventHandlerDeps:
    study_id: str
    spec: ExperimentSpec
    groundtruth: GroundTruthData
    profile: SamplerProfile
    objective_def: ObjectiveDefinition
    backend: OptimizerBackend
    execution_backend: ExecutionBackend
    run_cache: RunCache
    objective_cache: ObjectiveCache
    result_store: ResultStore
    inflight_registry: InflightRegistry
    study_state: StudyRuntimeState
    metrics: Metrics


def _handle_checkpoint(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    scorer = deps.objective_def.progress_scorer
    if scorer is None or event.checkpoint is None:
        return

    partial = scorer.score(event.checkpoint, deps.spec)
    if partial is None:
        return

    ctx = entry.leader.trial_ctx
    ctx.report(partial, event.step or 0)

    if ctx.should_prune():
        logger.info(
            "pruning trial",
            extra=_log_ctx(deps, entry.leader, event, run_key=entry.run_key),
        )
        deps.execution_backend.cancel(entry.handle, reason="pruned")


def _handle_completed(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    run_key = entry.run_key
    run_result = event.run_result
    assert run_result is not None, "COMPLETED event must carry run_result"

    deps.run_cache.put(run_key, run_result)
    deps.result_store.write_run_record(run_key, run_result)

    obj = deps.objective_def.objective_evaluator.evaluate(run_result, deps.spec, deps.groundtruth)
    deps.objective_cache.put(entry.leader.objective_key, obj)

    all_bindings = deps.inflight_registry.pop_all_trials_for_run_key(run_key)
    leader_trial_id = entry.leader.trial_id

    for binding in all_bindings:
        attrs = _build_attrs(obj.attrs, binding, leader_trial_id)
        deps.result_store.write_trial_result(
            binding.trial_id,
            ObjectiveResult(
                value=obj.value,
                attrs=attrs,
                artifact_refs=obj.artifact_refs,
            ),
        )
        deps.backend.tell(
            deps.study_id, binding.trial_id,
            TrialState.COMPLETE, obj.value, attrs,
        )
        deps.metrics.inc("trials_completed_total")
        logger.info(
            "trial completed",
            extra=_log_ctx(deps, binding, event, run_key=run_key),
        )

    deps.study_state.completed_trials += len(all_bindings)
    deps.study_state.active_executions -= 1


def _handle_cancelled(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    run_key = entry.run_key
    all_bindings = deps.inflight_registry.pop_all_trials_for_run_key(run_key)
    leader_trial_id = entry.leader.trial_id

    is_pruned = event.reason == "pruned"
    state = TrialState.PRUNED if is_pruned else TrialState.FAIL
    counter = "trials_pruned_total" if is_pruned else "trials_failed_total"
    error = TrialState.PRUNED if is_pruned else (event.reason or "CANCELLED")

    for binding in all_bindings:
        attrs = _build_attrs({}, binding, leader_trial_id)
        deps.result_store.write_trial_failure(
            binding.trial_id,
            _build_failure_payload(error, state, attrs),
        )
        deps.backend.tell(deps.study_id, binding.trial_id, state, None, attrs)
        deps.metrics.inc(counter)
        logger.info(
            "trial %s", "pruned" if is_pruned else "cancelled",
            extra=_log_ctx(deps, binding, event, run_key=run_key),
        )

    if is_pruned:
        deps.study_state.pruned_trials += len(all_bindings)
    else:
        deps.study_state.failed_trials += len(all_bindings)
    deps.study_state.active_executions -= 1


def _handle_failed(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    run_key = entry.run_key
    all_bindings = deps.inflight_registry.pop_all_trials_for_run_key(run_key)
    leader_trial_id = entry.leader.trial_id

    for binding in all_bindings:
        attrs = _build_attrs({}, binding, leader_trial_id)
        deps.result_store.write_trial_failure(
            binding.trial_id,
            _build_failure_payload(event.error_code or "UNKNOWN", TrialState.FAIL, attrs),
        )
        deps.backend.tell(
            deps.study_id, binding.trial_id,
            TrialState.FAIL, None, attrs,
        )
        deps.metrics.inc("trials_failed_total")
        logger.info(
            "trial failed",
            extra=_log_ctx(deps, binding, event, run_key=run_key),
        )

    deps.study_state.failed_trials += len(all_bindings)
    deps.study_state.active_executions -= 1


EVENT_HANDLERS = {
    EventKind.CHECKPOINT: _handle_checkpoint,
    EventKind.COMPLETED: _handle_completed,
    EventKind.CANCELLED: _handle_cancelled,
    EventKind.FAILED: _handle_failed,
}


def _build_attrs(
    base: dict[str, Any],
    binding: TrialBinding,
    leader_trial_id: str,
) -> dict[str, Any]:
    attrs = dict(base)
    if binding.trial_id != leader_trial_id:
        attrs["shared_run"] = True
        attrs["shared_run_leader_trial_id"] = leader_trial_id
    return attrs


def _build_failure_payload(
    error: str,
    state: TrialState,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "error": error,
        "state": state,
        "attrs": attrs,
    }


def _log_ctx(
    deps: EventHandlerDeps,
    binding: TrialBinding,
    event: ExecutionEvent,
    run_key: str,
) -> dict[str, Any]:
    return {
        "study_id": deps.study_id,
        "trial_id": binding.trial_id,
        "trial_number": binding.trial_number,
        "run_key": run_key,
        "objective_key": binding.objective_key,
        "handle_id": event.handle_id,
        "event_kind": event.kind.value,
        "sampling_mode": deps.profile.mode.value,
    }
