"""IT: orchestrator loads groundtruth per dataset."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

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
from optimization_control_plane.domain.models import ExperimentSpec, GroundTruthData
from tests.conftest import (
    StubDatasetEnumerator,
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunResultLoader,
    StubRunSpecBuilder,
    StubSearchSpace,
    StubTrialResultAggregator,
    make_settings,
)


class DatasetAwareGroundTruthProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self, spec: ExperimentSpec, dataset_id: str) -> GroundTruthData:
        del spec
        if not dataset_id:
            raise ValueError("dataset_id must be provided")
        self.calls.append(dataset_id)
        return GroundTruthData(
            payload={"dataset_id": dataset_id},
            fingerprint=f"sha256:gt:{dataset_id}",
        )


def _build_orchestrator(tmp_path: str, provider: DatasetAwareGroundTruthProvider) -> TrialOrchestrator:
    db = os.path.join(tmp_path, "test.db")
    return TrialOrchestrator(
        backend=OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}"),
        objective_def=ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            dataset_enumerator=StubDatasetEnumerator(("ds_if2401", "ds_if2402")),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            trial_result_aggregator=StubTrialResultAggregator(),
            progress_scorer=None,
            objective_evaluator=StubObjectiveEvaluator(),
        ),
        groundtruth_provider=provider,
        execution_backend=FakeExecutionBackend(),
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_result_loader=StubRunResultLoader(),
        run_cache=FileRunCache(os.path.join(tmp_path, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp_path, "data")),
        result_store=FileResultStore(os.path.join(tmp_path, "data")),
    )


def test_start_loads_groundtruth_per_dataset(tmp_path: Any) -> None:
    provider = DatasetAwareGroundTruthProvider()
    orch = _build_orchestrator(str(tmp_path), provider)
    settings = make_settings(stop={"max_trials": 0})

    with patch.object(TrialOrchestrator, "_run_loop", return_value=None):
        orch.start(settings=settings)

    assert provider.calls == ["ds_if2401", "ds_if2402"]
    assert set(orch._groundtruth_by_dataset.keys()) == {"ds_if2401", "ds_if2402"}
