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
from tests.conftest import (
    StubDatasetEnumerator,
    StubGroundTruthProvider,
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    StubTrialResultAggregator,
    make_settings,
    make_spec,
)


def _build_orchestrator(tmp_path: str) -> TrialOrchestrator:
    db = os.path.join(tmp_path, "test.db")
    return TrialOrchestrator(
        backend=OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}"),
        objective_def=ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            dataset_enumerator=StubDatasetEnumerator(),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            trial_result_aggregator=StubTrialResultAggregator(),
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


def test_start_rejects_missing_groundtruth(tmp_path: Any) -> None:
    orch = _build_orchestrator(str(tmp_path))
    settings = make_settings(
        objective_config={
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "nop"},
        }
    )
    with pytest.raises(ValueError, match="spec.objective_config.groundtruth"):
        orch.start(settings=settings)
