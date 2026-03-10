"""TrialOrchestrator — the single entry-point that drives the optimisation control loop."""
from __future__ import annotations

import logging
from uuid import uuid4
from typing import Any

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._event_handler import (
    EVENT_HANDLERS,
    EventHandlerDeps,
)
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration._request_planner import (
    RequestBufferItem,
    _plan_and_fill,
)
from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
)
from optimization_control_plane.core.orchestration.trial_batching import (
    DatasetPlan,
    TrialBatchRegistry,
    build_dataset_plan,
    with_dataset_path,
)
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import (
    ExecutionRequest,
    ExecutionEvent,
    ExperimentSpec,
    RunResult,
    RunSpec,
    SamplerProfile,
    StudyHandle,
    compute_spec_hash,
)
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.policies import DispatchPolicy, ParallelismPolicy
from optimization_control_plane.ports.result_store import ResultStore

logger = logging.getLogger(__name__)

_EVENT_LOOP_TIMEOUT = 1.0
_SPEC_SETTINGS_KEYS = ("spec_id", "meta", "objective_config", "execution_config")


class TrialOrchestrator:
    def __init__(
        self,
        backend: OptimizerBackend,
        objective_def: ObjectiveDefinition,
        execution_backend: ExecutionBackend,
        parallelism_policy: ParallelismPolicy,
        dispatch_policy: DispatchPolicy,
        run_cache: RunCache,
        objective_cache: ObjectiveCache,
        result_store: ResultStore,
    ) -> None:
        self._backend = backend
        self._objective_def = objective_def
        self._execution_backend = execution_backend
        self._parallelism_policy = parallelism_policy
        self._dispatch_policy = dispatch_policy
        self._run_cache = run_cache
        self._objective_cache = objective_cache
        self._result_store = result_store

        self._study_handle: StudyHandle | None = None
        self._spec: ExperimentSpec | None = None
        self._profile: SamplerProfile | None = None
        self._study_state = StudyRuntimeState()
        self._resource_state: ResourceState | None = None
        self._inflight = InflightRegistry()
        self._request_buffer: list[RequestBufferItem] = []
        self._stop_requested = False
        self._metrics = Metrics()
        self._max_trials: int | None = None
        self._max_failures: int | None = None
        self._dataset_plan: DatasetPlan | None = None
        self._trial_batches = TrialBatchRegistry()

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    @property
    def study_state(self) -> StudyRuntimeState:
        return self._study_state

    def start(
        self,
        spec: ExperimentSpec | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        resolved_settings = dict(settings or {})
        resolved_spec = self._resolve_start_spec(spec=spec, settings=resolved_settings)

        self._study_handle = self._backend.open_or_resume_experiment(resolved_spec)
        self._spec = self._backend.get_spec(self._study_handle.study_id)
        self._profile = self._backend.get_sampler_profile(self._study_handle.study_id)

        max_in_flight = resolved_settings.get("parallelism", {}).get("max_in_flight_trials", 1)
        self._reset_runtime_state(max_in_flight)

        stop_cfg = resolved_settings.get("stop", {})
        self._max_trials = stop_cfg.get("max_trials")
        self._max_failures = stop_cfg.get("max_failures")
        self._dataset_plan = build_dataset_plan(resolved_settings)
        if self._dataset_plan is not None and self._objective_def.trial_loss_aggregator is None:
            raise ValueError("trial_loss_aggregator is required when dataset_plan is enabled")

        logger.info(
            "orchestrator started",
            extra={
                "study_id": self._study_handle.study_id,
                "spec_hash": self._spec.spec_hash,
                "sampling_mode": self._profile.mode.value,
                "max_in_flight": max_in_flight,
                "train_shards": (
                    len(self._dataset_plan.train_shards)
                    if self._dataset_plan is not None else 0
                ),
                "test_shards": (
                    len(self._dataset_plan.test_shards)
                    if self._dataset_plan is not None else 0
                ),
            },
        )
        self._run_loop()

    def _run_loop(self) -> None:
        assert self._study_handle is not None
        assert self._spec is not None
        assert self._profile is not None
        assert self._resource_state is not None

        study_id = self._study_handle.study_id

        while not self._stop_requested:
            self._sync_gauges()

            if self._reached_limit():
                if not self._inflight.handles():
                    break
                self._drain_inflight(study_id)
                continue

            target = self._parallelism_policy.target_in_flight(
                self._profile, self._study_state, self._resource_state,
            )

            self._plan_requests(study_id, target)
            self._try_release_buffer(study_id)

            if not self._inflight.handles():
                if self._reached_limit():
                    break
                continue

            event = self._execution_backend.wait_any(
                self._inflight.handles(), timeout=_EVENT_LOOP_TIMEOUT,
            )
            if event is not None:
                self._handle_event(event)

        self._drain_remaining(study_id)
        self._run_final_test_report()
        logger.info("orchestrator stopped", extra={"study_id": study_id})

    def stop(self) -> None:
        self._stop_requested = True
        logger.info("graceful stop requested")

    def _plan_requests(self, study_id: str | None = None, target: int | None = None) -> None:
        assert self._spec is not None
        assert self._profile is not None
        assert self._resource_state is not None

        sid = study_id or (self._study_handle.study_id if self._study_handle else "")
        t = target or self._parallelism_policy.target_in_flight(
            self._profile, self._study_state, self._resource_state,
        )

        _plan_and_fill(
            study_id=sid,
            spec=self._spec,
            profile=self._profile,
            objective_def=self._objective_def,
            backend=self._backend,
            execution_backend=self._execution_backend,
            dispatch_policy=self._dispatch_policy,
            run_cache=self._run_cache,
            objective_cache=self._objective_cache,
            result_store=self._result_store,
            inflight_registry=self._inflight,
            study_state=self._study_state,
            resource_state=self._resource_state,
            request_buffer=self._request_buffer,
            target=t,
            stop_requested=self._stop_requested,
            metrics=self._metrics,
            max_trials=self._max_trials,
            max_failures=self._max_failures,
            dataset_plan=self._dataset_plan,
            trial_batches=(self._trial_batches if self._dataset_plan is not None else None),
        )

    def _handle_event(self, event: ExecutionEvent) -> None:
        assert self._spec is not None
        assert self._profile is not None
        assert self._study_handle is not None

        deps = EventHandlerDeps(
            study_id=self._study_handle.study_id,
            spec=self._spec,
            profile=self._profile,
            objective_def=self._objective_def,
            backend=self._backend,
            execution_backend=self._execution_backend,
            run_cache=self._run_cache,
            objective_cache=self._objective_cache,
            result_store=self._result_store,
            inflight_registry=self._inflight,
            study_state=self._study_state,
            metrics=self._metrics,
            trial_batches=(self._trial_batches if self._dataset_plan is not None else None),
        )

        handler = EVENT_HANDLERS.get(event.kind)
        if handler is not None:
            handler(deps, event)
        self._sync_gauges()

    def _drain_inflight(self, study_id: str) -> None:
        if not self._inflight.handles():
            return
        event = self._execution_backend.wait_any(
            self._inflight.handles(), timeout=_EVENT_LOOP_TIMEOUT,
        )
        if event is not None:
            self._handle_event(event)

    def _drain_remaining(self, study_id: str) -> None:
        while self._inflight.handles():
            self._drain_inflight(study_id)

    def _try_release_buffer(self, study_id: str) -> None:
        assert self._profile is not None
        if not self._parallelism_policy.should_release_buffer(
            self._profile, self._study_state
        ):
            return
        # V1: buffer release path (kept for interface completeness)

    def _reached_limit(self) -> bool:
        if self._stop_requested:
            return True
        if self._max_trials is not None and self._study_state.asked_trials >= self._max_trials:
            return True
        return self._max_failures is not None and self._study_state.failed_trials >= self._max_failures

    def _sync_gauges(self) -> None:
        self._metrics.set_gauge(
            "inflight_leader_executions_gauge",
            self._inflight.active_leader_count,
        )
        self._metrics.set_gauge(
            "attached_follower_trials_gauge",
            self._inflight.total_follower_count,
        )
        self._metrics.set_gauge(
            "buffered_requests_gauge",
            len(self._request_buffer),
        )
        if self._resource_state is not None:
            self._resource_state.free_slots = (
                self._resource_state.configured_slots - self._study_state.active_executions
            )

    def _reset_runtime_state(self, max_in_flight: int) -> None:
        self._study_state = StudyRuntimeState()
        self._resource_state = ResourceState(
            configured_slots=max_in_flight,
            free_slots=max_in_flight,
        )
        self._inflight = InflightRegistry()
        self._request_buffer = []
        self._stop_requested = False
        self._metrics = Metrics()
        self._trial_batches = TrialBatchRegistry()

    def _resolve_start_spec(
        self,
        *,
        spec: ExperimentSpec | None,
        settings: dict[str, Any],
    ) -> ExperimentSpec:
        if spec is None and not settings:
            raise ValueError("start() requires at least one of spec or settings")

        settings_spec: ExperimentSpec | None = None
        if settings:
            settings_spec = self._build_spec_from_settings(settings)

        if spec is None:
            assert settings_spec is not None
            return settings_spec
        if settings_spec is None:
            return spec
        if spec != settings_spec:
            raise ValueError(
                "provided spec does not match spec constructed from settings: "
                f"given_hash={spec.spec_hash}, settings_hash={settings_spec.spec_hash}"
            )
        return spec

    def _build_spec_from_settings(self, settings: dict[str, Any]) -> ExperimentSpec:
        payload = self._read_spec_payload(settings)
        missing = [key for key in _SPEC_SETTINGS_KEYS if key not in payload]
        if missing:
            raise ValueError(
                "settings must include spec fields to construct ExperimentSpec: "
                f"missing={missing}"
            )

        spec_id = payload["spec_id"]
        if not isinstance(spec_id, str) or not spec_id:
            raise ValueError("settings spec_id must be a non-empty string")
        meta = self._read_required_dict(payload, "meta")
        objective_config = self._augment_objective_config(payload, settings)
        execution_config = self._read_required_dict(payload, "execution_config")
        computed_hash = compute_spec_hash(spec_id, meta, objective_config, execution_config)
        provided_hash = payload.get("spec_hash")
        if provided_hash is not None and provided_hash != computed_hash:
            raise ValueError(
                "settings provided spec_hash does not match computed spec_hash: "
                f"provided={provided_hash}, computed={computed_hash}"
            )
        spec_hash = provided_hash if isinstance(provided_hash, str) else computed_hash
        return ExperimentSpec(
            spec_id=spec_id,
            spec_hash=spec_hash,
            meta=meta,
            objective_config=objective_config,
            execution_config=execution_config,
        )

    @staticmethod
    def _read_spec_payload(settings: dict[str, Any]) -> dict[str, Any]:
        payload = settings.get("spec", settings)
        if not isinstance(payload, dict):
            raise ValueError("settings.spec must be a dict when provided")
        return payload

    @staticmethod
    def _read_required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"settings {key} must be a dict")
        return dict(value)

    def _augment_objective_config(
        self,
        payload: dict[str, Any],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        objective_config = self._read_required_dict(payload, "objective_config")
        sampler = settings.get("sampler")
        pruner = settings.get("pruner")
        if isinstance(sampler, dict):
            objective_config["sampler"] = dict(sampler)
        if isinstance(pruner, dict):
            objective_config["pruner"] = dict(pruner)
        return objective_config

    def _run_final_test_report(self) -> None:
        if self._dataset_plan is None or not self._dataset_plan.test_shards:
            return
        if self._spec is None:
            return
        base_run_spec = self._trial_batches.best_train_run_spec
        if base_run_spec is None:
            logger.warning("skip final test report: no successful train trial")
            return
        aggregator = self._objective_def.trial_loss_aggregator
        if aggregator is None:
            return
        handles = []
        for shard in self._dataset_plan.test_shards:
            run_spec = with_dataset_path(base_run_spec, shard)
            request = self._build_final_test_request(run_spec)
            handles.append(self._execution_backend.submit(request))
        objectives = []
        pending = list(handles)
        while pending:
            event = self._execution_backend.wait_any(pending, timeout=_EVENT_LOOP_TIMEOUT)
            if event is None:
                continue
            pending = [handle for handle in pending if handle.handle_id != event.handle_id]
            if event.kind != EventKind.COMPLETED or event.run_result is None:
                logger.warning(
                    "test report sub-run not completed",
                    extra={"event_kind": event.kind.value, "handle_id": event.handle_id},
                )
                continue
            objectives.append(self._objective_def.objective_evaluator.evaluate(event.run_result, self._spec))
        if not objectives:
            logger.warning("skip final test report: no completed test sub-run")
            return
        report = aggregator.aggregate(objectives, self._spec, split="test")
        self._result_store.write_run_record(
            "final_report:test",
            RunResult(
                metrics={"test_objective": report.value, "test_run_count": len(objectives)},
                diagnostics={
                    "report_type": "final_test",
                    "best_train_trial_id": self._trial_batches.best_train_trial_id,
                },
                artifact_refs=list(report.artifact_refs),
            ),
        )
        logger.info(
            "final test report generated",
            extra={
                "best_train_trial_id": self._trial_batches.best_train_trial_id,
                "test_objective": report.value,
                "test_run_count": len(objectives),
            },
        )

    def _build_final_test_request(self, run_spec: RunSpec) -> ExecutionRequest:
        assert self._spec is not None
        run_key = self._objective_def.run_key_builder.build(run_spec, self._spec)
        objective_key = self._objective_def.objective_key_builder.build(
            run_key,
            self._spec.objective_config,
        )
        return ExecutionRequest(
            request_id=f"final_{uuid4().hex[:12]}",
            trial_id=f"final_test_{uuid4().hex[:8]}",
            run_key=run_key,
            objective_key=objective_key,
            cohort_id="final_test",
            priority=0,
            run_spec=run_spec,
        )
