"""Integration: one trial fans out to multiple train shards and aggregates loss."""
from __future__ import annotations

import json
import os
from pathlib import Path
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
    StubGroundTruthProvider,
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_settings,
    make_spec,
)


class MeanAggregator:
    def aggregate(
        self,
        run_objectives: list[ObjectiveResult],
        spec: Any,
        split: str,
    ) -> ObjectiveResult:
        value = sum(item.value for item in run_objectives) / len(run_objectives)
        return ObjectiveResult(value=float(value), attrs={"split": split}, artifact_refs=[])


def test_multi_run_trial_aggregation_and_final_test_report(tmp_path: Any) -> None:
    db = os.path.join(str(tmp_path), "test.db")
    data_dir = os.path.join(str(tmp_path), "data")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    exec_be = FakeExecutionBackend()
    exec_be.set_default_script(FakeRunScript(
        run_result=RunResult(metrics={"metric_1": 0.2}, diagnostics={}, artifact_refs=[]),
    ))
    obj_def = ObjectiveDefinition(
        search_space=StubSearchSpace({"x": 1.0}),
        run_spec_builder=StubRunSpecBuilder(),
        run_key_builder=StubRunKeyBuilder(),
        objective_key_builder=StubObjectiveKeyBuilder(),
        progress_scorer=None,
        objective_evaluator=StubObjectiveEvaluator(metric_name="metric_1"),
        trial_loss_aggregator=MeanAggregator(),
    )
    orch = TrialOrchestrator(
        backend=backend,
        objective_def=obj_def,
        groundtruth_provider=StubGroundTruthProvider(),
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
        parallelism={"max_in_flight_trials": 4},
        dataset_plan={
            "files": [
                {"id": "a", "path": "/tmp/a.csv"},
                {"id": "b", "path": "/tmp/b.csv"},
                {"id": "c", "path": "/tmp/c.csv"},
            ],
            "train_ratio": 2,
            "test_ratio": 1,
            "seed": 1,
        },
    )
    orch.start(spec, settings)
    metrics = orch.metrics.snapshot()
    assert metrics["trials_completed_total"] == 2
    assert metrics["execution_submitted_total"] == 2
    assert _has_final_test_report(Path(data_dir))


def _has_final_test_report(data_dir: Path) -> bool:
    for file in (data_dir / "run_records").glob("*.json"):
        payload = json.loads(file.read_text(encoding="utf-8"))
        diagnostics = payload.get("diagnostics", {})
        if diagnostics.get("report_type") == "final_test":
            return True
    return False
