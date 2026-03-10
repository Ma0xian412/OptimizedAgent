"""IT: contract acceptance for package/project target kinds."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from optimization_control_plane.adapters.execution import FakeExecutionBackend, FakeRunScript
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter
from optimization_control_plane.adapters.policies import (
    AsyncFillParallelismPolicy,
    SubmitNowDispatchPolicy,
)
from optimization_control_plane.adapters.storage import (
    FileObjectiveCache,
    FileResultStore,
    FileRunCache,
)
from optimization_control_plane.adapters.target_resolution import SimpleTargetResolver
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import ExperimentSpec, ResolvedTarget, RunResult, RunSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    make_settings,
    make_spec,
)


class TrackingSearchSpace:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, Any]:
        self._trace.append("sample")
        return {"x": 1.0}


class TrackingRunSpecBuilder:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def build(
        self,
        resolved_target: ResolvedTarget,
        params: dict[str, Any],
        execution_config: dict[str, Any],
    ) -> RunSpec:
        self._trace.append("build")
        return RunSpec(
            kind="backtest",
            config=dict(params),
            resources=dict(execution_config.get("default_resources", {})),
            resolved_target=resolved_target,
        )


def _target_spec_payload(
    kind: str,
    *,
    ref: str = "target_ref",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "target_id": "logical_target",
        "config": {
            "envelope": {
                "kind": kind,
                "ref": ref,
                "config": dict(config or {"region": "global"}),
            }
        },
    }


def _build_orchestrator(
    tmp_path: str,
    trace: list[str],
) -> tuple[TrialOrchestrator, OptunaBackendAdapter, FakeExecutionBackend]:
    db = os.path.join(tmp_path, "test.db")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    execution_backend = FakeExecutionBackend()
    execution_backend.set_default_script(FakeRunScript(
        run_result=RunResult(metrics={"metric_1": 0.42}, diagnostics={}, artifact_refs=[])
    ))
    objective_def = ObjectiveDefinition(
        search_space=TrackingSearchSpace(trace),
        run_spec_builder=TrackingRunSpecBuilder(trace),
        run_key_builder=StubRunKeyBuilder(),
        objective_key_builder=StubObjectiveKeyBuilder(),
        progress_scorer=None,
        objective_evaluator=StubObjectiveEvaluator(),
    )
    orchestrator = TrialOrchestrator(
        backend=backend,
        objective_def=objective_def,
        execution_backend=execution_backend,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(tmp_path, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp_path, "data")),
        result_store=FileResultStore(os.path.join(tmp_path, "data")),
        target_resolver=SimpleTargetResolver(),
    )
    return orchestrator, backend, execution_backend


def _assert_single_trial_flow(
    tmp_path: Any,
    *,
    kind: str,
    ref: str,
    expected_target_id: str,
) -> None:
    trace: list[str] = []
    orch, backend, execution_backend = _build_orchestrator(str(tmp_path), trace)
    target_spec = _target_spec_payload(kind, ref=ref, config={"region": "apac"})
    spec = make_spec(target_spec=target_spec)
    settings = make_settings(
        target_spec=target_spec,
        stop={"max_trials": 1},
        parallelism={"max_in_flight_trials": 1},
    )

    original_ask = backend.ask
    original_submit = execution_backend.submit
    original_wait_any = execution_backend.wait_any
    original_tell = backend.tell

    def traced_ask(study_id: str) -> Any:
        trace.append("ask")
        return original_ask(study_id)

    def traced_submit(request: Any) -> Any:
        trace.append("submit")
        return original_submit(request)

    def traced_wait_any(handles: list[Any], timeout: float | None = None) -> Any:
        event = original_wait_any(handles, timeout)
        if event is not None and event.kind == EventKind.COMPLETED:
            trace.append("complete")
        return event

    def traced_tell(
        study_id: str,
        trial_id: str,
        state: str,
        value: float | None,
        attrs: dict[str, Any] | None,
    ) -> Any:
        trace.append("tell")
        return original_tell(study_id, trial_id, state, value, attrs)

    with (
        patch.object(backend, "ask", side_effect=traced_ask),
        patch.object(execution_backend, "submit", side_effect=traced_submit),
        patch.object(execution_backend, "wait_any", side_effect=traced_wait_any),
        patch.object(backend, "tell", side_effect=traced_tell),
    ):
        orch.start(spec=spec, settings=settings)

    submitted = execution_backend.submitted_requests()
    assert len(submitted) == 1
    resolved_target = submitted[0].run_spec.resolved_target
    assert resolved_target.target_id == expected_target_id
    assert resolved_target.config == {"region": "apac"}
    assert trace == ["ask", "sample", "build", "submit", "complete", "tell"]


def test_package_target_runs_without_core_changes(tmp_path: Any) -> None:
    _assert_single_trial_flow(
        tmp_path,
        kind="package",
        ref="pkg_alpha",
        expected_target_id="pkg::pkg_alpha",
    )


def test_project_target_runs_without_core_changes(tmp_path: Any) -> None:
    _assert_single_trial_flow(
        tmp_path,
        kind="project",
        ref="proj_beta",
        expected_target_id="proj::proj_beta",
    )


def test_unknown_kind_fails_in_resolver_before_ask(tmp_path: Any) -> None:
    trace: list[str] = []
    orch, backend, execution_backend = _build_orchestrator(str(tmp_path), trace)
    target_spec = _target_spec_payload("workspace", ref="bad_kind")
    spec = make_spec(target_spec=target_spec)
    settings = make_settings(
        target_spec=target_spec,
        stop={"max_trials": 1},
        parallelism={"max_in_flight_trials": 1},
    )

    with (
        patch.object(backend, "ask", wraps=backend.ask) as ask_spy,
        patch.object(execution_backend, "submit", wraps=execution_backend.submit) as submit_spy,
    ):
        with pytest.raises(ValueError, match="envelope.kind must be one of"):
            orch.start(spec=spec, settings=settings)

    assert ask_spy.call_count == 0
    assert submit_spy.call_count == 0
    assert execution_backend.submitted_requests() == []


@pytest.mark.parametrize(
    ("target_spec", "error"),
    [
        (
            {
                "target_id": "logical_target",
                "config": {"envelope": {"kind": "package", "config": {"region": "apac"}}},
            },
            "target_spec.config.envelope.ref must be a non-empty string",
        ),
        (
            {
                "target_id": "logical_target",
                "config": {"envelope": {"kind": "project", "ref": "proj_x"}},
            },
            "target_spec.config.envelope.config must be a dict",
        ),
    ],
)
def test_missing_ref_or_config_fails_in_resolver(
    tmp_path: Any,
    target_spec: dict[str, Any],
    error: str,
) -> None:
    trace: list[str] = []
    orch, backend, execution_backend = _build_orchestrator(str(tmp_path), trace)
    spec = make_spec(target_spec=target_spec)
    settings = make_settings(
        target_spec=target_spec,
        stop={"max_trials": 1},
        parallelism={"max_in_flight_trials": 1},
    )

    with (
        patch.object(backend, "ask", wraps=backend.ask) as ask_spy,
        patch.object(execution_backend, "submit", wraps=execution_backend.submit) as submit_spy,
    ):
        with pytest.raises(ValueError, match=error):
            orch.start(spec=spec, settings=settings)

    assert ask_spy.call_count == 0
    assert submit_spy.call_count == 0
