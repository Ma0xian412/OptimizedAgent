"""IT: start() reinitializes runtime state per run."""
from __future__ import annotations

import os
from typing import Any

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
    StubGroundTruthProvider,
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
        groundtruth_provider=StubGroundTruthProvider(),
        execution_backend=FakeExecutionBackend(),
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(tmp_path, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp_path, "data")),
        result_store=FileResultStore(os.path.join(tmp_path, "data")),
    )


def test_start_resets_runtime_state(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    spec = make_spec()
    settings = make_settings(
        parallelism={"max_in_flight_trials": 3},
        stop={"max_trials": 0},
    )

    orch.start(spec, settings)
    first_state = orch.study_state
    first_inflight = orch._inflight
    first_buffer = orch._request_buffer

    orch.study_state.completed_trials = 7
    orch.stop()

    orch.start(spec, settings)

    assert orch.study_state is not first_state
    assert orch.study_state.completed_trials == 0
    assert orch._inflight is not first_inflight
    assert orch._request_buffer is not first_buffer
    assert orch._request_buffer == []
    assert orch.metrics.snapshot()["trials_asked_total"] == 0
    assert orch._stop_requested is False
