"""Request planning logic extracted from TrialOrchestrator."""
from __future__ import annotations

import uuid

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration._run_binding_factory import (
    build_bindings,
    enumerate_dataset_ids,
)
from optimization_control_plane.core.orchestration._trial_utils import (
    build_trial_objective_key,
    with_shared_run_attrs,
)
from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
    RunBinding,
    TrialCohort,
)
from optimization_control_plane.domain.enums import DispatchDecision, TrialState
from optimization_control_plane.domain.models import (
    ExecutionRequest,
    ExperimentSpec,
    GroundTruthData,
    ObjectiveResult,
    SamplerProfile,
)
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.policies import DispatchPolicy
from optimization_control_plane.ports.result_store import ResultStore

RequestBufferItem = tuple[ExecutionRequest, RunBinding]


def _plan_and_fill(
    *,
    study_id: str,
    spec: ExperimentSpec,
    groundtruth_by_dataset: dict[str, GroundTruthData],
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
    while not stop_requested and _slots_available(study_state, request_buffer, target):
        if _should_stop_asking(study_state, max_trials, max_failures):
            return
        trial = backend.ask(study_id)
        study_state.asked_trials += 1
        metrics.inc("trials_asked_total")
        trial_ctx = backend.open_trial_context(study_id, trial.trial_id)
        params = objective_def.search_space.sample(trial_ctx, spec)
        dataset_ids = enumerate_dataset_ids(objective_def, spec)
        trial_objective_key = build_trial_objective_key(
            params=params,
            dataset_ids=dataset_ids,
            spec=spec,
            groundtruth_fingerprints=_groundtruth_fingerprints(
                dataset_ids=dataset_ids,
                groundtruth_by_dataset=groundtruth_by_dataset,
            ),
        )
        cached_trial_obj = objective_cache.get(trial_objective_key)
        if cached_trial_obj is not None:
            _tell_complete_trial(
                backend=backend,
                result_store=result_store,
                study_id=study_id,
                trial_id=trial.trial_id,
                objective_result=cached_trial_obj,
            )
            study_state.completed_trials += 1
            metrics.inc("trials_completed_total")
            metrics.inc("objective_cache_hit_total")
            continue

        bindings = build_bindings(
            objective_def=objective_def,
            spec=spec,
            groundtruth_by_dataset=groundtruth_by_dataset,
            params=params,
            dataset_ids=dataset_ids,
            trial_id=trial.trial_id,
            trial_number=trial.number,
            trial_objective_key=trial_objective_key,
            trial_ctx=trial_ctx,
        )
        inflight_registry.register_trial_cohort(
            TrialCohort(
                trial_id=trial.trial_id,
                trial_number=trial.number,
                trial_ctx=trial_ctx,
                trial_objective_key=trial_objective_key,
                run_bindings=bindings,
            )
        )
        _plan_trial_bindings(
            study_id=study_id,
            spec=spec,
            groundtruth_by_dataset=groundtruth_by_dataset,
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
            metrics=metrics,
            bindings=bindings,
        )


def _plan_trial_bindings(
    *,
    study_id: str,
    spec: ExperimentSpec,
    groundtruth_by_dataset: dict[str, GroundTruthData],
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
    metrics: Metrics,
    bindings: tuple[RunBinding, ...],
) -> None:
    for binding in bindings:
        cached_run_obj = objective_cache.get(binding.per_run_objective_key)
        if cached_run_obj is not None:
            objective_cache.put(binding.per_run_objective_key, cached_run_obj)
            inflight_registry.record_run_complete(
                trial_id=binding.trial_id,
                run_key=binding.run_key,
                dataset_id=binding.dataset_id,
                objective_result=cached_run_obj,
            )
            metrics.inc("objective_cache_hit_per_run_total")
            continue

        run_result = run_cache.get(binding.run_key)
        if run_result is not None:
            groundtruth = _groundtruth_for_dataset(
                dataset_id=binding.dataset_id,
                groundtruth_by_dataset=groundtruth_by_dataset,
            )
            evaluated = objective_def.objective_evaluator.evaluate(run_result, spec, groundtruth)
            objective_cache.put(binding.per_run_objective_key, evaluated)
            inflight_registry.record_run_complete(
                trial_id=binding.trial_id,
                run_key=binding.run_key,
                dataset_id=binding.dataset_id,
                objective_result=evaluated,
            )
            metrics.inc("run_cache_hit_total")
            continue

        if inflight_registry.has(binding.run_key):
            inflight_registry.attach_follower(binding.run_key, binding)
            study_state.attached_follower_trials += 1
            continue
        request = ExecutionRequest(
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            trial_id=binding.trial_id,
            run_key=binding.run_key,
            objective_key=binding.per_run_objective_key,
            cohort_id=binding.trial_id,
            priority=0,
            run_spec=binding.run_spec,
        )
        decision = dispatch_policy.classify(request, profile, study_state, resource_state)
        if decision == DispatchDecision.SUBMIT_NOW:
            handle = execution_backend.submit(request)
            inflight_registry.register_leader(binding.run_key, handle, binding)
            study_state.active_executions += 1
            metrics.inc("execution_submitted_total")
            continue
        request_buffer.append((request, binding))
        study_state.buffered_requests += 1

    _try_finalize_ready_trial(
        study_id=study_id,
        spec=spec,
        objective_def=objective_def,
        objective_cache=objective_cache,
        result_store=result_store,
        backend=backend,
        inflight_registry=inflight_registry,
        study_state=study_state,
        metrics=metrics,
        trial_id=bindings[0].trial_id,
    )


def _try_finalize_ready_trial(
    *,
    study_id: str,
    spec: ExperimentSpec,
    objective_def: ObjectiveDefinition,
    objective_cache: ObjectiveCache,
    result_store: ResultStore,
    backend: OptimizerBackend,
    inflight_registry: InflightRegistry,
    study_state: StudyRuntimeState,
    metrics: Metrics,
    trial_id: str,
) -> None:
    cohort = inflight_registry.get_trial_cohort(trial_id)
    if cohort is None or not cohort.is_complete:
        return
    finished = inflight_registry.pop_trial_cohort(trial_id)
    if finished is None:
        return
    aggregated = objective_def.trial_result_aggregator.aggregate(finished.successful_results, spec)
    final_result = with_shared_run_attrs(aggregated, finished.shared_run_leader_trial_ids)
    objective_cache.put(finished.trial_objective_key, final_result)
    _tell_complete_trial(
        backend=backend,
        result_store=result_store,
        study_id=study_id,
        trial_id=finished.trial_id,
        objective_result=final_result,
    )
    study_state.completed_trials += 1
    metrics.inc("trials_completed_total")


def _tell_complete_trial(
    *,
    backend: OptimizerBackend,
    result_store: ResultStore,
    study_id: str,
    trial_id: str,
    objective_result: ObjectiveResult,
) -> None:
    result_store.write_trial_result(trial_id, objective_result)
    backend.tell(
        study_id=study_id,
        trial_id=trial_id,
        state=TrialState.COMPLETE,
        value=objective_result.value,
        attrs=objective_result.attrs,
    )


def _slots_available(state: StudyRuntimeState, buffer: list[RequestBufferItem], target: int) -> bool:
    return (state.active_executions + len(buffer)) < target


def _should_stop_asking(state: StudyRuntimeState, max_trials: int | None, max_failures: int | None) -> bool:
    if max_trials is not None and state.asked_trials >= max_trials:
        return True
    return max_failures is not None and state.failed_trials >= max_failures


def _groundtruth_for_dataset(
    *,
    dataset_id: str,
    groundtruth_by_dataset: dict[str, GroundTruthData],
) -> GroundTruthData:
    groundtruth = groundtruth_by_dataset.get(dataset_id)
    if groundtruth is None:
        raise KeyError(f"missing groundtruth for dataset_id={dataset_id}")
    return groundtruth


def _groundtruth_fingerprints(
    *,
    dataset_ids: tuple[str, ...],
    groundtruth_by_dataset: dict[str, GroundTruthData],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for dataset_id in dataset_ids:
        groundtruth = _groundtruth_for_dataset(
            dataset_id=dataset_id,
            groundtruth_by_dataset=groundtruth_by_dataset,
        )
        result[dataset_id] = groundtruth.fingerprint
    return result
