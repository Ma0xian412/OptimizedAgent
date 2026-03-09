"""IT: start() supports settings/spec input modes."""
from __future__ import annotations

import os
from typing import Any

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
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_settings,
    make_spec,
)


def _build_orchestrator(tmp_path: str) -> TrialOrchestrator:
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
        execution_backend=FakeExecutionBackend(),
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(tmp_path, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp_path, "data")),
        result_store=FileResultStore(os.path.join(tmp_path, "data")),
    )


def test_start_accepts_settings_only(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    spec = make_spec()
    settings = make_settings(
        spec=spec,
        stop={"max_trials": 0},
        parallelism={"max_in_flight_trials": 1},
    )

    orch.start(settings=settings)

    assert orch.metrics.snapshot()["trials_asked_total"] == 0


def test_start_rejects_mismatched_spec(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    spec = make_spec(spec_id="a")
    mismatched = make_spec(spec_id="b")
    settings = make_settings(spec=mismatched, stop={"max_trials": 0})

    with pytest.raises(ValueError, match="spec mismatch"):
        orch.start(spec=spec, settings=settings)


def test_start_requires_settings_spec_when_both_provided(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    spec = make_spec()
    settings = make_settings(stop={"max_trials": 0})

    with pytest.raises(ValueError, match="spec construction fields"):
        orch.start(spec=spec, settings=settings)
