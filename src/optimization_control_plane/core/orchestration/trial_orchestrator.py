"""TrialOrchestrator — the single entry-point that drives the optimisation control loop."""
from __future__ import annotations

import logging
from typing import Any

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._event_handler import (
    EVENT_HANDLERS,
    EventHandlerDeps,
)
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration._request_planner import (
    RequestBufferItem,
    plan_and_fill,
)
from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
)
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExperimentSpec,
    SamplerProfile,
    StudyHandle,
)
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState
from optimization_control_plane.ports.cache import ObjectiveCache, RunCache
from optimization_control_plane.ports.execution_backend import ExecutionBackend
from optimization_control_plane.ports.optimizer_backend import OptimizerBackend
from optimization_control_plane.ports.policies import DispatchPolicy, ParallelismPolicy
from optimization_control_plane.ports.result_store import ResultStore

logger = logging.getLogger(__name__)

_EVENT_LOOP_TIMEOUT = 1.0


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

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    @property
    def study_state(self) -> StudyRuntimeState:
        return self._study_state

    def start(self, spec: ExperimentSpec, settings: dict[str, Any]) -> None:
        self._study_handle = self._backend.open_or_resume_experiment(spec, settings)
        self._spec = self._backend.get_spec(self._study_handle.study_id)
        self._profile = self._backend.get_sampler_profile(self._study_handle.study_id)

        max_in_flight = settings.get("parallelism", {}).get("max_in_flight_trials", 1)
        self._resource_state = ResourceState(
            configured_slots=max_in_flight,
            free_slots=max_in_flight,
        )

        stop_cfg = settings.get("stop", {})
        self._max_trials = stop_cfg.get("max_trials")
        self._max_failures = stop_cfg.get("max_failures")

        logger.info(
            "orchestrator started",
            extra={
                "study_id": self._study_handle.study_id,
                "spec_hash": self._spec.spec_hash,
                "sampling_mode": self._profile.mode.value,
                "max_in_flight": max_in_flight,
            },
        )

    def run_loop(self) -> None:
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

            self.plan_requests(study_id, target)
            self._try_release_buffer(study_id)

            if not self._inflight.handles():
                if self._reached_limit():
                    break
                continue

            event = self._execution_backend.wait_any(
                self._inflight.handles(), timeout=_EVENT_LOOP_TIMEOUT,
            )
            if event is not None:
                self.handle_event(event)

        self._drain_remaining(study_id)
        logger.info("orchestrator stopped", extra={"study_id": study_id})

    def stop(self) -> None:
        self._stop_requested = True
        logger.info("graceful stop requested")

    def plan_requests(self, study_id: str | None = None, target: int | None = None) -> None:
        assert self._spec is not None
        assert self._profile is not None
        assert self._resource_state is not None

        sid = study_id or (self._study_handle.study_id if self._study_handle else "")
        t = target or self._parallelism_policy.target_in_flight(
            self._profile, self._study_state, self._resource_state,
        )

        plan_and_fill(
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
        )

    def handle_event(self, event: ExecutionEvent) -> None:
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
            self.handle_event(event)

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
