"""IT-4: Pruning closed loop."""
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
    JsonRunResultLoader,
)
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import Checkpoint, RunResult
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


class AlwaysPruneScorer:
    """Returns a score for every checkpoint so pruner can act."""
    def score(self, checkpoint: Checkpoint, spec: Any) -> float | None:
        return float(checkpoint.metrics.get("loss", 100.0))


class TestPruning:
    def test_pruning_loop(self, tmp_path: Any) -> None:
        db = os.path.join(str(tmp_path), "test.db")
        backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")

        exec_be = FakeExecutionBackend()
        exec_be.set_default_script(FakeRunScript(
            checkpoints=[
                Checkpoint(step=1, metrics={"loss": 100.0}),
                Checkpoint(step=2, metrics={"loss": 100.0}),
                Checkpoint(step=3, metrics={"loss": 100.0}),
            ],
            final_event=EventKind.COMPLETED,
            run_result=RunResult(metrics={"metric_1": 0.5}, diagnostics={}, artifact_refs=[]),
        ))

        obj_def = ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            dataset_enumerator=StubDatasetEnumerator(),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            trial_result_aggregator=StubTrialResultAggregator(),
            progress_scorer=AlwaysPruneScorer(),
            objective_evaluator=StubObjectiveEvaluator(),
        )

        settings = make_settings(
            stop={"max_trials": 3},
            parallelism={"max_in_flight_trials": 1},
            pruner={"type": "median", "n_startup_trials": 0, "n_warmup_steps": 0},
        )

        orch = TrialOrchestrator(
            backend=backend,
            objective_def=obj_def,
            groundtruth_provider=StubGroundTruthProvider(),
            execution_backend=exec_be,
            parallelism_policy=AsyncFillParallelismPolicy(),
            dispatch_policy=SubmitNowDispatchPolicy(),
            run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
            objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
            result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
            run_result_loader=JsonRunResultLoader(),
        )

        spec = make_spec(objective_config={
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
            "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "median", "n_startup_trials": 0, "n_warmup_steps": 0},
        })
        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        total = m["trials_completed_total"] + m["trials_pruned_total"] + m["trials_failed_total"]
        assert total == 3
