"""IT-2/3: cache hit skips execution; run cache hit only triggers evaluator."""
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
    FileRunResultLoader,
    FileResultStore,
    FileRunCache,
)
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.models import RunResult
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


def _build_orchestrator(
    tmp: str,
    *,
    search_params: dict[str, Any] | None = None,
) -> tuple[TrialOrchestrator, FakeExecutionBackend]:
    db = os.path.join(tmp, "test.db")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    exec_be = FakeExecutionBackend()
    exec_be.set_default_script(FakeRunScript(
        run_result=RunResult(metrics={"metric_1": 0.1}, diagnostics={}, artifact_refs=[]),
    ))

    obj_def = ObjectiveDefinition(
        search_space=StubSearchSpace(search_params or {"x": 1.0}),
        dataset_enumerator=StubDatasetEnumerator(),
        run_spec_builder=StubRunSpecBuilder(),
        run_key_builder=StubRunKeyBuilder(),
        objective_key_builder=StubObjectiveKeyBuilder(),
        trial_result_aggregator=StubTrialResultAggregator(),
        progress_scorer=None,
        objective_evaluator=StubObjectiveEvaluator(),
    )

    orch = TrialOrchestrator(
        backend=backend,
        objective_def=obj_def,
        groundtruth_provider=StubGroundTruthProvider(),
        execution_backend=exec_be,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_result_loader=FileRunResultLoader(),
        run_cache=FileRunCache(os.path.join(tmp, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp, "data")),
        result_store=FileResultStore(os.path.join(tmp, "data")),
    )
    return orch, exec_be


class TestObjectiveCacheHit:
    def test_objective_cache_skips_execution(self, tmp_path: Any) -> None:
        orch, exec_be = _build_orchestrator(str(tmp_path))
        spec = make_spec()
        settings = make_settings(stop={"max_trials": 4}, parallelism={"max_in_flight_trials": 1})
        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        assert m["objective_cache_hit_total"] >= 1
        assert m["execution_submitted_total"] == 1


class TestRunCacheHit:
    def test_run_cache_hit_triggers_evaluator_only(self, tmp_path: Any) -> None:
        orch, exec_be = _build_orchestrator(str(tmp_path))
        spec = make_spec()
        settings = make_settings(stop={"max_trials": 3}, parallelism={"max_in_flight_trials": 1})
        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        assert m["execution_submitted_total"] == 1
        assert m["trials_completed_total"] == 3
