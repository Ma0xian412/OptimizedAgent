"""Request planning logic extracted from TrialOrchestrator."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
    TrialBinding,
)
from optimization_control_plane.domain.enums import DispatchDecision, TrialState
from optimization_control_plane.domain.models import (
    ExecutionRequest,
    ExperimentSpec,
    ResolvedTarget,
    SamplerProfile,
    validate_resolved_target,
    validate_target_spec,
)
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.policies import DispatchPolicy
from optimization_control_plane.ports.result_store import ResultStore

logger = logging.getLogger(__name__)

RequestBufferItem = tuple[ExecutionRequest, TrialBinding]


def _plan_and_fill(
    *,
    study_id: str,
    spec: ExperimentSpec,
    profile: SamplerProfile,
    objective_def: ObjectiveDefinition,
    backend: OptimizerBackend,
    execution_backend: ExecutionBackend,
    dispatch_policy: DispatchPolicy,
    run_cache: RunCache,
    objective_cache: ObjectiveCache,
    result_store: ResultStore,
    inflight_registry: InflightRegistry,
    study_state: StudyRuntimeState,
    resource_state: ResourceState,
    request_buffer: list[RequestBufferItem],
    target: int,
    stop_requested: bool,
    metrics: Metrics,
    max_trials: int | None,
    max_failures: int | None,
) -> None:
    """Ask trials and fill execution slots up to target."""
    target_spec = validate_target_spec(spec.target_spec, source="spec.target_spec")
    resolved_target = _resolve_target_for_experiment(target_spec, spec)
    while not stop_requested and _slots_available(study_state, request_buffer, target):
        if _should_stop_asking(study_state, max_trials, max_failures):
            break

        trial = backend.ask(study_id)
        study_state.asked_trials += 1
        metrics.inc("trials_asked_total")
        ctx = backend.open_trial_context(study_id, trial.trial_id)

        params = objective_def.search_space.sample(ctx, spec)
        run_spec = objective_def.run_spec_builder.build(
            resolved_target, params, spec.execution_config
        )
        run_key = objective_def.run_key_builder.build(run_spec, spec)
        obj_key = objective_def.objective_key_builder.build(
            run_key, spec.objective_config
        )

        log_extra = _log_extra(
            study_id,
            trial.trial_id,
            trial.number,
            run_key,
            obj_key,
            profile,
            resolved_target.target_id,
        )

        obj = objective_cache.get(obj_key)
        if obj is not None:
            metrics.inc("objective_cache_hit_total")
            result_store.write_trial_result(trial.trial_id, obj)
            backend.tell(study_id, trial.trial_id, TrialState.COMPLETE, obj.value, obj.attrs)
            study_state.completed_trials += 1
            metrics.inc("trials_completed_total")
            logger.info("objective cache hit", extra=log_extra)
            continue

        run_result = run_cache.get(run_key)
        if run_result is not None:
            metrics.inc("run_cache_hit_total")
            obj = objective_def.objective_evaluator.evaluate(run_result, spec)
            objective_cache.put(obj_key, obj)
            result_store.write_trial_result(trial.trial_id, obj)
            backend.tell(study_id, trial.trial_id, TrialState.COMPLETE, obj.value, obj.attrs)
            study_state.completed_trials += 1
            metrics.inc("trials_completed_total")
            logger.info("run cache hit", extra=log_extra)
            continue

        binding = TrialBinding(
            trial_id=trial.trial_id,
            trial_number=trial.number,
            objective_key=obj_key,
            trial_ctx=ctx,
        )

        if inflight_registry.has(run_key):
            inflight_registry.attach_follower(run_key, binding)
            study_state.attached_follower_trials += 1
            logger.info("attached as follower", extra=log_extra)
            continue

        request = ExecutionRequest(
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            trial_id=trial.trial_id,
            run_key=run_key,
            objective_key=obj_key,
            cohort_id=None,
            priority=0,
            run_spec=run_spec,
        )

        decision = dispatch_policy.classify(request, profile, study_state, resource_state)
        if decision == DispatchDecision.SUBMIT_NOW:
            _submit_leader(
                execution_backend, inflight_registry, study_state, metrics,
                request, binding, run_key, log_extra,
            )
        else:
            request_buffer.append((request, binding))
            study_state.buffered_requests += 1
            logger.info("request buffered", extra=log_extra)


def _submit_leader(
    execution_backend: ExecutionBackend,
    inflight_registry: InflightRegistry,
    study_state: StudyRuntimeState,
    metrics: Metrics,
    request: ExecutionRequest,
    binding: TrialBinding,
    run_key: str,
    log_extra: dict[str, Any],
) -> None:
    handle = execution_backend.submit(request)
    inflight_registry.register_leader(run_key, handle, binding)
    study_state.active_executions += 1
    metrics.inc("execution_submitted_total")
    logger.info(
        "submitted execution",
        extra={**log_extra, "handle_id": handle.handle_id, "request_id": request.request_id},
    )


def _slots_available(
    state: StudyRuntimeState,
    buffer: list[RequestBufferItem],
    target: int,
) -> bool:
    return (state.active_executions + len(buffer)) < target


def _should_stop_asking(
    state: StudyRuntimeState,
    max_trials: int | None,
    max_failures: int | None,
) -> bool:
    if max_trials is not None and state.asked_trials >= max_trials:
        return True
    return max_failures is not None and state.failed_trials >= max_failures


def _resolve_target_for_experiment(
    target_spec: Any,
    spec: ExperimentSpec,
) -> ResolvedTarget:
    del spec
    # Iteration-2 compatibility path: target resolution still happens in planning.
    resolved_target = ResolvedTarget.from_target_spec(
        validate_target_spec(target_spec, source="spec.target_spec")
    )
    return validate_resolved_target(
        resolved_target,
        source="resolved_target",
    )


def _log_extra(
    study_id: str,
    trial_id: str,
    trial_number: int,
    run_key: str,
    objective_key: str,
    profile: SamplerProfile,
    target_id: str,
) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "trial_id": trial_id,
        "trial_number": trial_number,
        "run_key": run_key,
        "objective_key": objective_key,
        "target_id": target_id,
        "sampling_mode": profile.mode.value,
    }
