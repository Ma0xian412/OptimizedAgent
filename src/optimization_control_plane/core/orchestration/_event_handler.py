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
from optimization_control_plane.core.orchestration.trial_batching import TrialBatchRegistry
from optimization_control_plane.domain.enums import EventKind, TrialState
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExperimentSpec,
    ObjectiveResult,
    SamplerProfile,
)
from optimization_control_plane.domain.state import StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.objective import TrialLossAggregator
from optimization_control_plane.ports.result_store import ResultStore

logger = logging.getLogger(__name__)


@dataclass
class EventHandlerDeps:
    study_id: str
    spec: ExperimentSpec
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
    trial_batches: TrialBatchRegistry | None = None


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
    obj = deps.objective_def.objective_evaluator.evaluate(run_result, deps.spec)
    deps.objective_cache.put(entry.leader.objective_key, obj)
    all_bindings = deps.inflight_registry.pop_all_trials_for_run_key(run_key)
    if deps.trial_batches is None:
        _complete_single_run(deps, event, run_key, all_bindings, obj, entry.leader.trial_id)
    else:
        _complete_multi_run(deps, event, run_key, all_bindings, obj)
    deps.study_state.active_executions -= 1


def _handle_cancelled(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    run_key = entry.run_key
    all_bindings = deps.inflight_registry.pop_all_trials_for_run_key(run_key)
    leader_trial_id = entry.leader.trial_id

    _handle_terminal_failure(
        deps=deps,
        event=event,
        run_key=run_key,
        all_bindings=all_bindings,
        leader_trial_id=leader_trial_id,
        is_pruned=(event.reason == "pruned"),
        error=("PRUNED" if event.reason == "pruned" else (event.reason or "CANCELLED")),
    )
    deps.study_state.active_executions -= 1


def _handle_failed(deps: EventHandlerDeps, event: ExecutionEvent) -> None:
    entry = deps.inflight_registry.get_by_handle(event.handle_id)
    run_key = entry.run_key
    all_bindings = deps.inflight_registry.pop_all_trials_for_run_key(run_key)
    leader_trial_id = entry.leader.trial_id

    _handle_terminal_failure(
        deps=deps,
        event=event,
        run_key=run_key,
        all_bindings=all_bindings,
        leader_trial_id=leader_trial_id,
        is_pruned=False,
        error=(event.error_code or "UNKNOWN"),
    )
    deps.study_state.active_executions -= 1


EVENT_HANDLERS = {
    EventKind.CHECKPOINT: _handle_checkpoint,
    EventKind.COMPLETED: _handle_completed,
    EventKind.CANCELLED: _handle_cancelled,
    EventKind.FAILED: _handle_failed,
}


def _complete_single_run(
    deps: EventHandlerDeps,
    event: ExecutionEvent,
    run_key: str,
    all_bindings: list[TrialBinding],
    obj: ObjectiveResult,
    leader_trial_id: str,
) -> None:
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
        deps.backend.tell(deps.study_id, binding.trial_id, TrialState.COMPLETE, obj.value, attrs)
        deps.metrics.inc("trials_completed_total")
        logger.info("trial completed", extra=_log_ctx(deps, binding, event, run_key=run_key))
    deps.study_state.completed_trials += len(all_bindings)


def _complete_multi_run(
    deps: EventHandlerDeps,
    event: ExecutionEvent,
    run_key: str,
    all_bindings: list[TrialBinding],
    obj: ObjectiveResult,
) -> None:
    trial_batches = deps.trial_batches
    if trial_batches is None:
        return
    aggregator = deps.objective_def.trial_loss_aggregator
    if aggregator is None:
        raise ValueError("trial_loss_aggregator is required when trial_batches is enabled")
    for binding in all_bindings:
        trial_batches.add_objective(binding.trial_id, obj)
        _try_finalize_aggregated_trial(
            deps=deps,
            event=event,
            binding=binding,
            run_key=run_key,
            trial_batches=trial_batches,
            aggregator=aggregator,
        )


def _handle_terminal_failure(
    *,
    deps: EventHandlerDeps,
    event: ExecutionEvent,
    run_key: str,
    all_bindings: list[TrialBinding],
    leader_trial_id: str,
    is_pruned: bool,
    error: str | TrialState,
) -> None:
    state = TrialState.PRUNED if is_pruned else TrialState.FAIL
    counter = "trials_pruned_total" if is_pruned else "trials_failed_total"
    trial_batches = deps.trial_batches
    for binding in all_bindings:
        attrs = _build_attrs({}, binding, leader_trial_id)
        if trial_batches is None:
            should_emit = True
        else:
            should_emit = trial_batches.mark_terminal(binding.trial_id, state, str(error))
        if not should_emit:
            continue
        deps.result_store.write_trial_failure(
            binding.trial_id,
            _build_failure_payload(str(error), state, attrs),
        )
        deps.backend.tell(deps.study_id, binding.trial_id, state, None, attrs)
        deps.metrics.inc(counter)
        logger.info(
            "trial %s", "pruned" if is_pruned else "failed",
            extra=_log_ctx(deps, binding, event, run_key=run_key),
        )
        if is_pruned:
            deps.study_state.pruned_trials += 1
        else:
            deps.study_state.failed_trials += 1


def _try_finalize_aggregated_trial(
    *,
    deps: EventHandlerDeps,
    event: ExecutionEvent,
    binding: TrialBinding,
    run_key: str,
    trial_batches: TrialBatchRegistry,
    aggregator: TrialLossAggregator,
) -> None:
    if not trial_batches.is_ready(binding.trial_id):
        return
    state = trial_batches.get(binding.trial_id)
    final_objective = aggregator.aggregate(state.objectives, deps.spec, state.split)
    attrs = {
        **dict(final_objective.attrs),
        "split": state.split,
        "run_count": len(state.objectives),
    }
    wrapped_result = ObjectiveResult(
        value=final_objective.value,
        attrs=attrs,
        artifact_refs=list(final_objective.artifact_refs),
    )
    deps.result_store.write_trial_result(binding.trial_id, wrapped_result)
    deps.backend.tell(
        deps.study_id,
        binding.trial_id,
        TrialState.COMPLETE,
        wrapped_result.value,
        wrapped_result.attrs,
    )
    trial_batches.mark_terminal(binding.trial_id, TrialState.COMPLETE)
    trial_batches.track_best_train(binding.trial_id, wrapped_result.value)
    deps.metrics.inc("trials_completed_total")
    deps.study_state.completed_trials += 1
    logger.info(
        "trial completed after aggregation",
        extra={**_log_ctx(deps, binding, event, run_key=run_key), "run_count": len(state.objectives)},
    )


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
