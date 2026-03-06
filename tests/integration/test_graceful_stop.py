"""IT-5: Graceful stop."""
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
from optimization_control_plane.domain.models import RunResult
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_settings,
    make_spec,
)


class TestGracefulStop:
    def test_stop_prevents_new_trials(self, tmp_path: Any) -> None:
        db = os.path.join(str(tmp_path), "test.db")
        backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")

        exec_be = FakeExecutionBackend()
        exec_be.set_default_script(FakeRunScript(
            run_result=RunResult(metrics={"metric_1": 0.5}, diagnostics={}, artifact_refs=[]),
        ))

        obj_def = ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            progress_scorer=None,
            objective_evaluator=StubObjectiveEvaluator(),
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

        spec = make_spec()
        settings = make_settings(
            stop={"max_trials": 1000},
            parallelism={"max_in_flight_trials": 1},
        )
        orch.start(spec, settings)
        orch.stop()
        orch.run_loop()

        m = orch.metrics.snapshot()
        assert m["trials_asked_total"] == 0
