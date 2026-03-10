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


def _load_records(base_dir: str, subdir: str) -> list[dict[str, Any]]:
    directory = Path(base_dir) / "data" / subdir
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def _settings_for_spec(spec: Any, *, max_trials: int) -> dict[str, Any]:
    return make_settings(
        spec_id=spec.spec_id,
        meta=spec.meta,
        target_spec=spec.target_spec.to_dict(),
        objective_config=spec.objective_config,
        execution_config=spec.execution_config,
        stop={"max_trials": max_trials},
        parallelism={"max_in_flight_trials": max_trials},
    )


def _run_constant_search(tmp_path: Any, spec: Any, *, max_trials: int) -> dict[str, int]:
    db = os.path.join(str(tmp_path), "test.db")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    exec_be = FakeExecutionBackend()
    exec_be.set_default_script(
        FakeRunScript(run_result=RunResult(metrics={"metric_1": 0.5}, diagnostics={}, artifact_refs=[]))
    )
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
        target_resolver=StubTargetResolver(),
    )
    orch.start(spec, _settings_for_spec(spec, max_trials=max_trials))
    return orch.metrics.snapshot()


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
        records = _load_records(str(tmp_path), "trial_results")
        followers = [record for record in records if record["attrs"].get("shared_run") is True]
        assert len(records) == 5
        assert len(followers) == 4
        leader_ids = {record["attrs"].get("shared_run_leader_trial_id") for record in followers}
        assert len(leader_ids) == 1
        run_records = _load_records(str(tmp_path), "run_records")
        assert len(run_records) == 1

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
        records = _load_records(str(tmp_path), "trial_results")
        assert len(records) == 5
        assert all(record["attrs"].get("shared_run") is not True for record in records)
        run_records = _load_records(str(tmp_path), "run_records")
        assert len(run_records) == 5

    def test_different_target_never_shares_leader(self, tmp_path: Any) -> None:
        spec_a = make_spec(target_spec={"target_id": "target_dedup_a", "config": {"market": "us"}})
        spec_b = make_spec(target_spec={"target_id": "target_dedup_b", "config": {"market": "us"}})
        first = _run_constant_search(tmp_path, spec_a, max_trials=3)
        second = _run_constant_search(tmp_path, spec_b, max_trials=3)

        assert first["execution_submitted_total"] == 1
        assert second["execution_submitted_total"] == 1
        run_records = _load_records(str(tmp_path), "run_records")
        assert len(run_records) == 2
        assert len({record["run_key"] for record in run_records}) == 2
        assert {record["target_id"] for record in run_records} == {
            "target_dedup_a",
            "target_dedup_b",
        }
