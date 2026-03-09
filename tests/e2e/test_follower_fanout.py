"""E2E-3/4/5: Leader fan-out to followers — complete, pruned, failed."""
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
from optimization_control_plane.domain.enums import EventKind
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


def _load_records(base_dir: str, subdir: str) -> list[dict[str, Any]]:
    directory = Path(base_dir) / "data" / subdir
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def _make_orchestrator(
    tmp_path: str,
    script: FakeRunScript,
    progress_scorer: Any = None,
) -> tuple[TrialOrchestrator, FakeExecutionBackend]:
    db = os.path.join(tmp_path, "test.db")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    exec_be = FakeExecutionBackend()
    exec_be.set_default_script(script)

    obj_def = ObjectiveDefinition(
        search_space=StubSearchSpace({"x": 1.0}),
        run_spec_builder=StubRunSpecBuilder(),
        run_key_builder=StubRunKeyBuilder(),
        objective_key_builder=StubObjectiveKeyBuilder(),
        progress_scorer=progress_scorer,
        objective_evaluator=StubObjectiveEvaluator(),
    )

    orch = TrialOrchestrator(
        backend=backend,
        objective_def=obj_def,
        execution_backend=exec_be,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(tmp_path, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp_path, "data")),
        result_store=FileResultStore(os.path.join(tmp_path, "data")),
    )
    return orch, exec_be


class TestLeaderCompleteFollowerFanout:
    def test_followers_inherit_complete(self, tmp_path: Any) -> None:
        script = FakeRunScript(
            run_result=RunResult(
                metrics={"metric_1": 0.5}, diagnostics={}, artifact_refs=[]
            ),
        )
        orch, _ = _make_orchestrator(str(tmp_path), script)

        spec = make_spec()
        settings = make_settings(
            spec=spec,
            stop={"max_trials": 5},
            parallelism={"max_in_flight_trials": 5},
        )

        orch.start(spec=spec, settings=settings)

        m = orch.metrics.snapshot()
        assert m["trials_completed_total"] == 5
        assert m["execution_submitted_total"] == 1
        records = _load_records(str(tmp_path), "trial_results")
        assert len(records) == 5
        assert any(record["attrs"].get("shared_run") is True for record in records)
        assert any(
            record["attrs"].get("shared_run_leader_trial_id") is not None
            for record in records
            if record["attrs"].get("shared_run") is True
        )


class TestLeaderFailedFollowerFanout:
    def test_followers_inherit_fail(self, tmp_path: Any) -> None:
        script = FakeRunScript(
            final_event=EventKind.FAILED,
            fail_error_code="OOM",
        )
        orch, _ = _make_orchestrator(str(tmp_path), script)

        spec = make_spec()
        settings = make_settings(
            spec=spec,
            stop={"max_trials": 5, "max_failures": 5},
            parallelism={"max_in_flight_trials": 5},
        )

        orch.start(spec=spec, settings=settings)

        m = orch.metrics.snapshot()
        assert m["trials_failed_total"] >= 1
        assert m["execution_submitted_total"] >= 1


class TestLeaderPrunedFollowerFanout:
    def test_followers_inherit_pruned(self, tmp_path: Any) -> None:
        script = FakeRunScript(
            final_event=EventKind.CANCELLED,
            fail_reason="pruned",
        )
        orch, _ = _make_orchestrator(str(tmp_path), script)

        spec = make_spec()
        settings = make_settings(
            spec=spec,
            stop={"max_trials": 5, "max_failures": 5},
            parallelism={"max_in_flight_trials": 5},
        )

        orch.start(spec=spec, settings=settings)

        m = orch.metrics.snapshot()
        assert m["trials_pruned_total"] >= 1
        assert m["execution_submitted_total"] >= 1
        records = _load_records(str(tmp_path), "trial_failures")
        assert len(records) == 5
        assert all(record["error"] == "PRUNED" for record in records)
        assert all(record["state"] == "PRUNED" for record in records)
        assert any(record["attrs"].get("shared_run") is True for record in records)
