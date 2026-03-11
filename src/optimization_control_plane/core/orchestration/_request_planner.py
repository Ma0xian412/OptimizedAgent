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
from optimization_control_plane.core.orchestration.trial_batching import (
    DatasetPlan,
    TrialBatchRegistry,
    with_dataset_path,
)
from optimization_control_plane.domain.enums import DispatchDecision, TrialState
from optimization_control_plane.domain.models import (
    ExecutionRequest,
    ExperimentSpec,
    ObjectiveResult,
    GroundTruthData,
    SamplerProfile,
)
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.objective import TrialLossAggregator
from optimization_control_plane.ports.policies import DispatchPolicy
from optimization_control_plane.ports.result_store import ResultStore

logger = logging.getLogger(__name__)

RequestBufferItem = tuple[ExecutionRequest, TrialBinding]


def _plan_and_fill(
    *,
    study_id: str,
    spec: ExperimentSpec,
    groundtruth: GroundTruthData,
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
    dataset_plan: DatasetPlan | None = None,
    trial_batches: TrialBatchRegistry | None = None,
) -> None:
    """Ask trials and fill execution slots up to target."""
    if dataset_plan is None or trial_batches is None:
        _plan_single_run(
            study_id=study_id,
            spec=spec,
            groundtruth=groundtruth,
            profile=profile,
            objective_def=objective_def,
            backend=backend,
            execution_backend=execution_backend,
            dispatch_policy=dispatch_policy,
            run_cache=run_cache,
            objective_cache=objective_cache,
            result_store=result_store,
            inflight_registry=inflight_registry,
            study_state=study_state,
            resource_state=resource_state,
            request_buffer=request_buffer,
            target=target,
            stop_requested=stop_requested,
            metrics=metrics,
            max_trials=max_trials,
            max_failures=max_failures,
        )
        return
    _plan_multi_run(
        study_id=study_id,
        spec=spec,
        groundtruth=groundtruth,
        profile=profile,
        objective_def=objective_def,
        backend=backend,
        execution_backend=execution_backend,
        dispatch_policy=dispatch_policy,
        run_cache=run_cache,
        objective_cache=objective_cache,
        result_store=result_store,
        inflight_registry=inflight_registry,
        study_state=study_state,
        resource_state=resource_state,
        request_buffer=request_buffer,
        target=target,
        stop_requested=stop_requested,
        metrics=metrics,
        max_trials=max_trials,
        max_failures=max_failures,
        dataset_plan=dataset_plan,
        trial_batches=trial_batches,
    )


def _plan_multi_run(
    *,
    study_id: str,
    spec: ExperimentSpec,
    groundtruth: GroundTruthData,
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
    dataset_plan: DatasetPlan,
    trial_batches: TrialBatchRegistry,
) -> None:
    aggregator = objective_def.trial_loss_aggregator
    if aggregator is None:
        raise ValueError("trial_loss_aggregator is required when dataset_plan is enabled")
    train_shards = dataset_plan.train_shards
    while not stop_requested and _slots_available(study_state, request_buffer, target):
        if _should_stop_asking(study_state, max_trials, max_failures):
            break
        trial = backend.ask(study_id)
        study_state.asked_trials += 1
        metrics.inc("trials_asked_total")
        ctx = backend.open_trial_context(study_id, trial.trial_id)
        params = objective_def.search_space.sample(ctx, spec)
        base_run_spec = objective_def.run_spec_builder.build(params, spec)
        trial_batches.register(
            trial_id=trial.trial_id,
            trial_number=trial.number,
            trial_ctx=ctx,
            split="train",
            expected_runs=len(train_shards),
            base_run_spec=base_run_spec,
        )
        for shard in train_shards:
            run_spec = with_dataset_path(base_run_spec, shard)
            run_key = objective_def.run_key_builder.build(run_spec, spec)
            raw_obj_key = objective_def.objective_key_builder.build(run_key, spec.objective_config)
            obj_key = _scope_objective_key(raw_obj_key, groundtruth.fingerprint)
            log_extra = {
                **_log_extra(study_id, trial.trial_id, trial.number, run_key, obj_key, profile),
                "split": shard.split,
                "shard_id": shard.shard_id,
            }
            obj = objective_cache.get(obj_key)
            if obj is not None:
                metrics.inc("objective_cache_hit_total")
                trial_batches.add_objective(trial.trial_id, obj)
                logger.info("objective cache hit (sub-run)", extra=log_extra)
                continue
            run_result = run_cache.get(run_key)
            if run_result is not None:
                metrics.inc("run_cache_hit_total")
                obj = objective_def.objective_evaluator.evaluate(run_result, spec, groundtruth)
                objective_cache.put(obj_key, obj)
                trial_batches.add_objective(trial.trial_id, obj)
                logger.info("run cache hit (sub-run)", extra=log_extra)
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
                logger.info("attached as follower (sub-run)", extra=log_extra)
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
                    execution_backend=execution_backend,
                    inflight_registry=inflight_registry,
                    study_state=study_state,
                    metrics=metrics,
                    request=request,
                    binding=binding,
                    run_key=run_key,
                    log_extra=log_extra,
                )
            else:
                request_buffer.append((request, binding))
                study_state.buffered_requests += 1
                logger.info("request buffered", extra=log_extra)
        _finalize_if_ready(
            trial_id=trial.trial_id,
            study_id=study_id,
            spec=spec,
            trial_batches=trial_batches,
            backend=backend,
            result_store=result_store,
            study_state=study_state,
            metrics=metrics,
            aggregator=aggregator,
        )


def _finalize_if_ready(
    *,
    trial_id: str,
    study_id: str,
    spec: ExperimentSpec,
    trial_batches: TrialBatchRegistry,
    backend: OptimizerBackend,
    result_store: ResultStore,
    study_state: StudyRuntimeState,
    metrics: Metrics,
    aggregator: TrialLossAggregator,
) -> None:
    if not trial_batches.is_ready(trial_id):
        return
    state = trial_batches.get(trial_id)
    final_objective = aggregator.aggregate(state.objectives, spec, state.split)
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
    result_store.write_trial_result(trial_id, wrapped_result)
    backend.tell(study_id, trial_id, TrialState.COMPLETE, wrapped_result.value, wrapped_result.attrs)
    trial_batches.mark_terminal(trial_id, TrialState.COMPLETE)
    trial_batches.track_best_train(trial_id, wrapped_result.value)
    study_state.completed_trials += 1
    metrics.inc("trials_completed_total")
    logger.info(
        "trial completed after aggregation",
        extra={"trial_id": trial_id, "run_count": len(state.objectives), "value": wrapped_result.value},
    )


def _plan_single_run(
    *,
    study_id: str,
    spec: ExperimentSpec,
    groundtruth: GroundTruthData,
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
    """Legacy single-run planning path."""
    while not stop_requested and _slots_available(study_state, request_buffer, target):
        if _should_stop_asking(study_state, max_trials, max_failures):
            break

        trial = backend.ask(study_id)
        study_state.asked_trials += 1
        metrics.inc("trials_asked_total")
        ctx = backend.open_trial_context(study_id, trial.trial_id)

        params = objective_def.search_space.sample(ctx, spec)
        run_spec = objective_def.run_spec_builder.build(params, spec)
        run_key = objective_def.run_key_builder.build(run_spec, spec)
        raw_obj_key = objective_def.objective_key_builder.build(
            run_key, spec.objective_config
        )
        obj_key = _scope_objective_key(raw_obj_key, groundtruth.fingerprint)

        log_extra = _log_extra(study_id, trial.trial_id, trial.number, run_key, obj_key, profile)

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
            obj = objective_def.objective_evaluator.evaluate(run_result, spec, groundtruth)
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


def _log_extra(
    study_id: str,
    trial_id: str,
    trial_number: int,
    run_key: str,
    objective_key: str,
    profile: SamplerProfile,
) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "trial_id": trial_id,
        "trial_number": trial_number,
        "run_key": run_key,
        "objective_key": objective_key,
        "sampling_mode": profile.mode.value,
    }


def _scope_objective_key(raw_key: str, groundtruth_fingerprint: str) -> str:
    return f"{raw_key}::gt={groundtruth_fingerprint}"
