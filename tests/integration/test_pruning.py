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
)
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import Checkpoint, RunResult
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
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
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            progress_scorer=AlwaysPruneScorer(),
            objective_evaluator=StubObjectiveEvaluator(),
        )

        spec = make_spec()
        settings = make_settings(
            spec=spec,
            stop={"max_trials": 3},
            parallelism={"max_in_flight_trials": 1},
            pruner={"type": "median", "n_startup_trials": 0, "n_warmup_steps": 0},
        )

        orch = TrialOrchestrator(
            backend=backend,
            objective_def=obj_def,
            execution_backend=exec_be,
            parallelism_policy=AsyncFillParallelismPolicy(),
            dispatch_policy=SubmitNowDispatchPolicy(),
            run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
            objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
            result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
        )

        orch.start(spec=spec, settings=settings)

        m = orch.metrics.snapshot()
        total = m["trials_completed_total"] + m["trials_pruned_total"] + m["trials_failed_total"]
        assert total == 3
