"""IT: planner fail-fast and target propagation."""
from __future__ import annotations

import os
from typing import Any

import pytest

from optimization_control_plane.adapters.execution import FakeExecutionBackend
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter
from optimization_control_plane.adapters.policies import SubmitNowDispatchPolicy
from optimization_control_plane.adapters.storage import (
    FileObjectiveCache,
    FileResultStore,
    FileRunCache,
)
from optimization_control_plane.core import ObjectiveDefinition
from optimization_control_plane.core.orchestration._metrics import Metrics
from optimization_control_plane.core.orchestration._request_planner import _plan_and_fill
from optimization_control_plane.core.orchestration.inflight_registry import InflightRegistry
from optimization_control_plane.domain.enums import SamplingMode
from optimization_control_plane.domain.models import ResolvedTarget, SamplerProfile, TargetSpec
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_spec,
)


class AskTrackingBackend:
    def __init__(self) -> None:
        self.ask_calls = 0

    def ask(self, study_id: str) -> Any:
        self.ask_calls += 1
        raise AssertionError("ask must not be called for invalid target")


def _build_objective_definition() -> ObjectiveDefinition:
    return ObjectiveDefinition(
        search_space=StubSearchSpace({"x": 1.0}),
        run_spec_builder=StubRunSpecBuilder(),
        run_key_builder=StubRunKeyBuilder(),
        objective_key_builder=StubObjectiveKeyBuilder(),
        progress_scorer=None,
        objective_evaluator=StubObjectiveEvaluator(),
    )


def _unsafe_target_spec(target_id: Any, config: Any) -> Any:
    target = object.__new__(TargetSpec)
    object.__setattr__(target, "target_id", target_id)
    object.__setattr__(target, "config", config)
    return target


def test_plan_and_fill_submitted_request_keeps_target_spec(tmp_path: Any) -> None:
    db = os.path.join(str(tmp_path), "test.db")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    execution_backend = FakeExecutionBackend()
    spec = make_spec(
        target_spec={"target_id": "target_gamma", "config": {"market": "crypto"}}
    )
    study = backend.open_or_resume_experiment(spec)
    profile = backend.get_sampler_profile(study.study_id)
    resolved_target = ResolvedTarget(
        target_id="resolved_gamma",
        config={"market": "crypto", "region": "global"},
    )

    _plan_and_fill(
        study_id=study.study_id,
        spec=spec,
        resolved_target=resolved_target,
        profile=profile,
        objective_def=_build_objective_definition(),
        backend=backend,
        execution_backend=execution_backend,
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
        objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
        result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
        inflight_registry=InflightRegistry(),
        study_state=StudyRuntimeState(),
        resource_state=ResourceState(configured_slots=1, free_slots=1),
        request_buffer=[],
        target=1,
        stop_requested=False,
        metrics=Metrics(),
        max_trials=None,
        max_failures=None,
    )

    submitted = execution_backend.submitted_requests()
    assert len(submitted) == 1
    assert submitted[0].run_spec.resolved_target is resolved_target
    assert submitted[0].run_spec.resolved_target.target_id == "resolved_gamma"
    assert submitted[0].run_spec.resolved_target.config == {
        "market": "crypto",
        "region": "global",
    }


def _unsafe_resolved_target(target_id: Any, config: Any) -> Any:
    target = object.__new__(ResolvedTarget)
    object.__setattr__(target, "target_id", target_id)
    object.__setattr__(target, "config", config)
    return target


@pytest.mark.parametrize(
    ("resolved_target", "error"),
    [
        (None, "resolved_target must be a ResolvedTarget"),
        (_unsafe_resolved_target("", {}), "resolved_target.target_id must be a non-empty string"),
        (_unsafe_resolved_target("resolved_x", "oops"), "resolved_target.config must be a dict"),
    ],
)
def test_plan_and_fill_invalid_resolved_target_fails_before_ask_or_submit(
    tmp_path: Any,
    resolved_target: Any,
    error: str,
) -> None:
    spec = make_spec(target_spec=_unsafe_target_spec("target_x", {}))
    backend = AskTrackingBackend()
    execution_backend = FakeExecutionBackend()
    metrics = Metrics()
    profile = SamplerProfile(
        mode=SamplingMode.ASYNC_FILL,
        startup_trials=0,
        batch_size=1,
        pending_policy="none",
        recommended_max_inflight=None,
    )

    with pytest.raises(ValueError, match=error):
        _plan_and_fill(
            study_id="study_x",
            spec=spec,
            resolved_target=resolved_target,
            profile=profile,
            objective_def=_build_objective_definition(),
            backend=backend,  # type: ignore[arg-type]
            execution_backend=execution_backend,
            dispatch_policy=SubmitNowDispatchPolicy(),
            run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
            objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
            result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
            inflight_registry=InflightRegistry(),
            study_state=StudyRuntimeState(),
            resource_state=ResourceState(configured_slots=1, free_slots=1),
            request_buffer=[],
            target=1,
            stop_requested=False,
            metrics=metrics,
            max_trials=None,
            max_failures=None,
        )

    assert backend.ask_calls == 0
    assert execution_backend.submitted_requests() == []
    snapshot = metrics.snapshot()
    assert snapshot["trials_asked_total"] == 0
    assert snapshot["execution_submitted_total"] == 0
