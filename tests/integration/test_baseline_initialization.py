"""Integration: baseline loss is initialized before optimisation loop."""
from __future__ import annotations

import os
from typing import Any

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
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.models import ObjectiveResult, RunResult
from tests.conftest import (
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_settings,
    make_spec,
)


class _BaselineAwareEvaluator:
    def __init__(self) -> None:
        self.base_loss: float | None = None

    def set_base_loss(self, loss: float, attrs: dict[str, Any] | None = None) -> None:
        self.base_loss = float(loss)

    def evaluate(self, run_result: RunResult, spec: Any) -> ObjectiveResult:
        return ObjectiveResult(
            value=float(run_result.metrics["metric_1"]),
            attrs={},
            artifact_refs=[],
        )


class _MeanAggregator:
    def aggregate(
        self,
        run_objectives: list[ObjectiveResult],
        spec: Any,
        split: str,
    ) -> ObjectiveResult:
        value = sum(item.value for item in run_objectives) / len(run_objectives)
        return ObjectiveResult(value=float(value), attrs={"split": split}, artifact_refs=[])


def test_baseline_loss_initialized_before_trials(tmp_path: Any) -> None:
    db = os.path.join(str(tmp_path), "test.db")
    data_dir = os.path.join(str(tmp_path), "data")
    evaluator = _BaselineAwareEvaluator()
    obj_def = ObjectiveDefinition(
        search_space=StubSearchSpace({"x": 1.0}),
        run_spec_builder=StubRunSpecBuilder(),
        run_key_builder=StubRunKeyBuilder(),
        objective_key_builder=StubObjectiveKeyBuilder(),
        progress_scorer=None,
        objective_evaluator=evaluator,
        trial_loss_aggregator=_MeanAggregator(),
    )
    exec_be = FakeExecutionBackend()
    exec_be.set_default_script(
        FakeRunScript(
            run_result=RunResult(metrics={"metric_1": 1.5}, diagnostics={}, artifact_refs=[]),
        )
    )
    orch = TrialOrchestrator(
        backend=OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}"),
        objective_def=obj_def,
        execution_backend=exec_be,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(data_dir),
        objective_cache=FileObjectiveCache(data_dir),
        result_store=FileResultStore(data_dir),
    )
    spec = make_spec()
    settings = make_settings(
        stop={"max_trials": 2},
        parallelism={"max_in_flight_trials": 2},
        dataset_plan={
            "files": [
                {"id": "a", "path": "/tmp/a.csv"},
                {"id": "b", "path": "/tmp/b.csv"},
                {"id": "c", "path": "/tmp/c.csv"},
            ],
            "train_ratio": 2,
            "test_ratio": 1,
            "seed": 3,
        },
    )
    orch.start(spec, settings)
    assert evaluator.base_loss == 1.5
