"""Event handling logic extracted from TrialOrchestrator."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
    RunBinding,
    TrialCohort,
)
from optimization_control_plane.core.orchestration._trial_utils import with_shared_run_attrs
from optimization_control_plane.domain.enums import EventKind, JobStatus, TrialState
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
from optimization_control_plane.ports.run_result_loader import RunResultLoader

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
    run_result_loader: RunResultLoader
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
        deps.execution_backend.cancel(entry.handle, reason="pruned")


def _handle_completed(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    run_result = deps.run_result_loader.load(entry.leader.run_spec)
    deps.run_cache.put(entry.run_key, run_result)
    deps.result_store.write_run_record(entry.run_key, run_result)
    objective = deps.objective_def.objective_evaluator.evaluate(run_result, deps.spec, deps.groundtruth)
    bindings = deps.inflight_registry.pop_all_trials_for_run_key(entry.run_key)
    for binding in bindings:
        deps.objective_cache.put(binding.per_run_objective_key, objective)
        deps.inflight_registry.record_run_complete(
            trial_id=binding.trial_id,
            run_key=binding.run_key,
            dataset_id=binding.dataset_id,
            objective_result=objective,
            leader_trial_id=entry.leader.trial_id,
        )
        _finalize_trial_if_ready(deps, binding, event)
    deps.study_state.active_executions -= 1


def _handle_cancelled(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    bindings = deps.inflight_registry.pop_all_trials_for_run_key(entry.run_key)
    is_pruned = event.reason == "pruned"
    state = TrialState.PRUNED if is_pruned else TrialState.FAIL
    error = TrialState.PRUNED.value if is_pruned else (event.reason or "CANCELLED")
    for binding in bindings:
        deps.inflight_registry.record_run_failure(
            trial_id=binding.trial_id,
            run_key=binding.run_key,
            dataset_id=binding.dataset_id,
            state=state,
            error=error,
            leader_trial_id=entry.leader.trial_id,
        )
        _finalize_trial_if_ready(deps, binding, event)
    deps.study_state.active_executions -= 1


def _handle_failed(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    bindings = deps.inflight_registry.pop_all_trials_for_run_key(entry.run_key)
    error = event.error_code or "UNKNOWN"
    for binding in bindings:
        deps.inflight_registry.record_run_failure(
            trial_id=binding.trial_id,
            run_key=binding.run_key,
            dataset_id=binding.dataset_id,
            state=TrialState.FAIL,
            error=error,
            leader_trial_id=entry.leader.trial_id,
        )
        _finalize_trial_if_ready(deps, binding, event)
    deps.study_state.active_executions -= 1


EVENT_HANDLERS = {
    EventKind.CHECKPOINT: _handle_checkpoint,
    EventKind.COMPLETED: _handle_completed,
    EventKind.CANCELLED: _handle_cancelled,
    EventKind.FAILED: _handle_failed,
}


def _finalize_trial_if_ready(
    deps: EventHandlerDeps,
    binding: RunBinding,
    event: ExecutionEvent,
) -> None:
    cohort = deps.inflight_registry.get_trial_cohort(binding.trial_id)
    if cohort is None or not cohort.is_complete:
        return
    finished = deps.inflight_registry.pop_trial_cohort(binding.trial_id)
    if finished is None:
        return
    if finished.successful_results:
        _finalize_trial_success(deps, finished, binding, event)
        return
    _finalize_trial_failure(deps, finished, binding, event)


def _finalize_trial_success(
    deps: EventHandlerDeps,
    cohort: TrialCohort,
    binding: RunBinding,
    event: ExecutionEvent,
) -> None:
    aggregated = deps.objective_def.trial_result_aggregator.aggregate(cohort.successful_results, deps.spec)
    final_result = with_shared_run_attrs(aggregated, cohort.shared_run_leader_trial_ids)
    deps.objective_cache.put(cohort.trial_objective_key, final_result)
    deps.result_store.write_trial_result(cohort.trial_id, final_result)
    deps.backend.tell(
        deps.study_id,
        cohort.trial_id,
        TrialState.COMPLETE,
        final_result.value,
        final_result.attrs,
    )
    deps.study_state.completed_trials += 1
    deps.metrics.inc("trials_completed_total")
    logger.info("trial completed", extra=_log_ctx(deps, binding, event, cohort.trial_objective_key))


def _finalize_trial_failure(
    deps: EventHandlerDeps,
    cohort: TrialCohort,
    binding: RunBinding,
    event: ExecutionEvent,
) -> None:
    all_pruned = all(failure.state == TrialState.PRUNED for failure in cohort.failures)
    state = TrialState.PRUNED if all_pruned else TrialState.FAIL
    error = TrialState.PRUNED.value if all_pruned else _join_failure_errors(cohort)
    attrs = _build_trial_attrs(cohort)
    deps.result_store.write_trial_failure(
        cohort.trial_id,
        _build_failure_payload(error=error, state=state, attrs=attrs),
    )
    deps.backend.tell(deps.study_id, cohort.trial_id, state, None, attrs)
    if state == TrialState.PRUNED:
        deps.study_state.pruned_trials += 1
        deps.metrics.inc("trials_pruned_total")
    else:
        deps.study_state.failed_trials += 1
        deps.metrics.inc("trials_failed_total")
    logger.info("trial failed", extra=_log_ctx(deps, binding, event, cohort.trial_objective_key))


def _build_trial_attrs(cohort: TrialCohort) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "successful_run_count": len(cohort.successful_results),
        "failed_run_count": len(cohort.failures),
    }
    if cohort.shared_run_leader_trial_ids:
        attrs["shared_run"] = True
        attrs["shared_run_leader_trial_ids"] = sorted(cohort.shared_run_leader_trial_ids)
    return attrs


def _join_failure_errors(cohort: TrialCohort) -> str:
    errors = sorted({failure.error for failure in cohort.failures})
    if not errors:
        return "NO_SUCCESSFUL_RUNS"
    return ",".join(errors)


def _build_failure_payload(
    *,
    error: str,
    state: TrialState,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    return {"error": error, "state": state, "attrs": attrs}


def _log_ctx(
    deps: EventHandlerDeps,
    binding: RunBinding,
    event: ExecutionEvent,
    objective_key: str,
) -> dict[str, Any]:
    return {
        "study_id": deps.study_id,
        "trial_id": binding.trial_id,
        "trial_number": binding.trial_number,
        "dataset_id": binding.dataset_id,
        "run_key": binding.run_key,
        "objective_key": objective_key,
        "handle_id": event.handle_id,
        "event_kind": event.kind.value,
        "job_status": _event_kind_to_job_status(event.kind).value,
        "sampling_mode": deps.profile.mode.value,
    }


def _event_kind_to_job_status(kind: EventKind) -> JobStatus:
    mapping = {
        EventKind.CHECKPOINT: JobStatus.RUNNING,
        EventKind.COMPLETED: JobStatus.COMPLETED,
        EventKind.FAILED: JobStatus.FAILED,
        EventKind.CANCELLED: JobStatus.CANCELLED,
    }
    return mapping[kind]
