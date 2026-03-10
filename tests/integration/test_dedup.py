"""IT-6: duplicate run_key attach follower."""
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
from optimization_control_plane.domain.models import RunResult
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    StubTargetResolver,
    make_settings,
    make_spec,
)


def _load_trial_results(base_dir: str) -> list[dict[str, Any]]:
    directory = Path(base_dir) / "data" / "trial_results"
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


class IncreasingSearchSpace:
    def __init__(self) -> None:
        self._x = 0

    def sample(self, ctx: Any, spec: Any) -> dict[str, float]:
        self._x += 1
        return {"x": float(self._x)}


class TestDedup:
    def test_same_run_key_deduped(self, tmp_path: Any) -> None:
        """With constant search space, all trials produce the same run_key.
        Only 1 execution should happen; the rest become followers or cache hits."""
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
        resolver = StubTargetResolver()

        orch = TrialOrchestrator(
            backend=backend,
            objective_def=obj_def,
            execution_backend=exec_be,
            parallelism_policy=AsyncFillParallelismPolicy(),
            dispatch_policy=SubmitNowDispatchPolicy(),
            run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
            objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
            result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
            target_resolver=resolver,
        )

        spec = make_spec()
        settings = make_settings(
            stop={"max_trials": 5},
            parallelism={"max_in_flight_trials": 3},
        )
        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        assert m["execution_submitted_total"] == 1
        assert m["trials_completed_total"] == 5
        assert len(resolver.calls) == 1
        records = _load_trial_results(str(tmp_path))
        assert any(record["attrs"].get("shared_run") is True for record in records)

    def test_different_run_key_does_not_attach_followers(self, tmp_path: Any) -> None:
        db = os.path.join(str(tmp_path), "test.db")
        backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
        exec_be = FakeExecutionBackend()
        exec_be.set_default_script(
            FakeRunScript(run_result=RunResult(metrics={"metric_1": 0.5}, diagnostics={}, artifact_refs=[]))
        )
        obj_def = ObjectiveDefinition(
            search_space=IncreasingSearchSpace(),
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
            target_resolver=StubTargetResolver(),
        )
        spec = make_spec()
        settings = make_settings(stop={"max_trials": 5}, parallelism={"max_in_flight_trials": 3})
        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        assert m["execution_submitted_total"] == 5
