"""Integration: baseline loss is initialized before optimisation loop."""
from __future__ import annotations

import os
from typing import Any

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
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.core.orchestration.trial_batching import (
    build_dataset_plan,
    with_dataset_path,
)
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
        self.base_attrs: dict[str, Any] = {}

    def set_base_loss(self, loss: float, attrs: dict[str, Any] | None = None) -> None:
        self.base_loss = float(loss)
        self.base_attrs = dict(attrs or {})

    def evaluate(self, run_result: RunResult, spec: Any) -> ObjectiveResult:
        attrs = {}
        for name in ("curve", "terminal", "cancel", "post"):
            if name in run_result.metrics:
                attrs[name] = float(run_result.metrics[name])
        return ObjectiveResult(
            value=float(run_result.metrics["metric_1"]),
            attrs=attrs,
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
    run_key_builder = StubRunKeyBuilder()
    run_spec_builder = StubRunSpecBuilder()
    obj_def = ObjectiveDefinition(
        search_space=StubSearchSpace({"x": 1.0}),
        run_spec_builder=run_spec_builder,
        run_key_builder=run_key_builder,
        objective_key_builder=StubObjectiveKeyBuilder(),
        progress_scorer=None,
        objective_evaluator=evaluator,
        trial_loss_aggregator=_MeanAggregator(),
    )
    exec_be = FakeExecutionBackend()
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
    dataset_plan = build_dataset_plan(settings)
    assert dataset_plan is not None
    baseline_metrics = [
        {"metric_1": 1.0, "curve": 1.0, "terminal": 2.0, "cancel": 3.0, "post": 4.0},
        {"metric_1": 2.0, "curve": 3.0, "terminal": 4.0, "cancel": 5.0},
        {"metric_1": 4.0, "curve": 5.0, "terminal": 6.0, "cancel": 7.0, "post": 8.0},
    ]
    base_run_spec = run_spec_builder.build({}, spec)
    for shard, metrics in zip(dataset_plan.all_shards(), baseline_metrics, strict=True):
        run_spec = with_dataset_path(base_run_spec, shard)
        run_key = run_key_builder.build(run_spec, spec)
        exec_be.set_script(
            run_key,
            FakeRunScript(
                run_result=RunResult(metrics=metrics, diagnostics={}, artifact_refs=[]),
            ),
        )
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
    orch.start(spec, settings)
    assert evaluator.base_loss == pytest.approx(7.0 / 3.0)
    assert evaluator.base_attrs["base_split"] == "all"
    assert evaluator.base_attrs["base_run_count"] == 3
    baseline_components = evaluator.base_attrs["baseline_components"]
    assert baseline_components["curve"] == pytest.approx(3.0)
    assert baseline_components["terminal"] == pytest.approx(4.0)
    assert baseline_components["cancel"] == pytest.approx(5.0)
    assert baseline_components["post"] == pytest.approx(6.0)
