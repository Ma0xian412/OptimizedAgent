"""IT: start() resolves spec from user inputs."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from optimization_control_plane.adapters.execution import FakeExecutionBackend
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
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.models import TargetSpec
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    StubTargetResolver,
    make_settings,
    make_spec,
)


def _build_orchestrator(
    tmp_path: str,
    *,
    execution_backend: FakeExecutionBackend | None = None,
    resolver: StubTargetResolver | None = None,
) -> TrialOrchestrator:
    db = os.path.join(tmp_path, "test.db")
    return TrialOrchestrator(
        backend=OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}"),
        objective_def=ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            progress_scorer=None,
            objective_evaluator=StubObjectiveEvaluator(),
        ),
        execution_backend=execution_backend or FakeExecutionBackend(),
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(tmp_path, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp_path, "data")),
        result_store=FileResultStore(os.path.join(tmp_path, "data")),
        target_resolver=resolver or StubTargetResolver(),
    )


def test_start_accepts_settings_only(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))

    settings = make_settings(stop={"max_trials": 0})
    expected = make_spec()
    with patch.object(TrialOrchestrator, "_run_loop", return_value=None):
        orch.start(settings=settings)

    assert orch._spec == expected


def test_start_accepts_spec_only(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))

    spec = make_spec()
    with patch.object(TrialOrchestrator, "_run_loop", return_value=None):
        orch.start(spec=spec)

    assert orch._spec == spec


def test_start_rejects_mismatched_spec_and_settings(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    spec = make_spec()
    settings = make_settings(spec_id="another_spec")

    with pytest.raises(ValueError, match="provided spec does not match"):
        orch.start(spec=spec, settings=settings)


def test_start_with_settings_requires_target_spec(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    settings = make_settings()
    settings.pop("target_spec")

    with pytest.raises(ValueError, match="missing=\\['target_spec'\\]"):
        orch.start(settings=settings)


def test_start_rejects_legacy_spec_payload_without_target_spec(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    legacy = make_settings()
    payload = {
        "spec_id": legacy["spec_id"],
        "meta": legacy["meta"],
        "objective_config": legacy["objective_config"],
        "execution_config": legacy["execution_config"],
    }
    settings = {"spec": payload}

    with pytest.raises(ValueError, match="missing=\\['target_spec'\\]"):
        orch.start(settings=settings)


def test_start_does_not_accept_target_spec_in_execution_config(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    settings = make_settings()
    target_spec = settings.pop("target_spec")
    execution_config = dict(settings["execution_config"])
    execution_config["target_spec"] = target_spec
    settings["execution_config"] = execution_config

    with pytest.raises(ValueError, match="missing=\\['target_spec'\\]"):
        orch.start(settings=settings)


def _unsafe_target_spec(target_id: Any, config: Any) -> Any:
    target = object.__new__(TargetSpec)
    object.__setattr__(target, "target_id", target_id)
    object.__setattr__(target, "config", config)
    return target


@pytest.mark.parametrize(
    ("target_spec", "error"),
    [
        (None, "spec.target_spec must be a TargetSpec"),
        (_unsafe_target_spec("", {}), "spec.target_spec.target_id must be a non-empty string"),
        (_unsafe_target_spec("target_x", "not_dict"), "spec.target_spec.config must be a dict"),
    ],
)
def test_start_with_spec_rejects_invalid_target_spec(
    tmp_path: Any,
    target_spec: Any,
    error: str,
) -> None:
    orch = _build_orchestrator(str(tmp_path))
    spec = make_spec()
    object.__setattr__(spec, "target_spec", target_spec)

    with pytest.raises(ValueError, match=error):
        orch.start(spec=spec)


def test_start_calls_resolver_once_per_experiment(tmp_path: Any) -> None:
    resolver = StubTargetResolver()
    orch = _build_orchestrator(str(tmp_path), resolver=resolver)
    spec = make_spec()
    settings = make_settings(stop={"max_trials": 3}, parallelism={"max_in_flight_trials": 1})

    orch.start(spec=spec, settings=settings)

    assert len(resolver.calls) == 1


def test_start_resolver_failure_fails_before_ask_or_submit(tmp_path: Any) -> None:
    execution_backend = FakeExecutionBackend()
    resolver = StubTargetResolver(fail_with=ValueError("cannot resolve target"))
    orch = _build_orchestrator(
        str(tmp_path),
        execution_backend=execution_backend,
        resolver=resolver,
    )
    spec = make_spec()
    settings = make_settings(stop={"max_trials": 3}, parallelism={"max_in_flight_trials": 1})

    with patch.object(orch._backend, "ask", wraps=orch._backend.ask) as ask_spy:  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="cannot resolve target"):
            orch.start(spec=spec, settings=settings)

    assert len(resolver.calls) == 1
    assert ask_spy.call_count == 0
    assert execution_backend.submitted_requests() == []
