"""IT: cache/key semantics stay target-aware and objective-aware."""
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
from optimization_control_plane.domain.models import ExperimentSpec, RunResult
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_settings,
    make_spec,
)


def _settings_for_spec(spec: ExperimentSpec, max_trials: int = 1) -> dict[str, Any]:
    return make_settings(
        spec_id=spec.spec_id,
        meta=spec.meta,
        target_spec=spec.target_spec.to_dict(),
        objective_config=spec.objective_config,
        execution_config=spec.execution_config,
        stop={"max_trials": max_trials},
        parallelism={"max_in_flight_trials": 1},
    )


def _run_once(tmp: str, spec: ExperimentSpec, max_trials: int = 1) -> dict[str, int]:
    db = os.path.join(tmp, "test.db")
    backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")
    exec_be = FakeExecutionBackend()
    exec_be.set_default_script(
        FakeRunScript(
            run_result=RunResult(metrics={"metric_1": 0.2}, diagnostics={}, artifact_refs=[]),
        )
    )
    orch = TrialOrchestrator(
        backend=backend,
        objective_def=ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            progress_scorer=None,
            objective_evaluator=StubObjectiveEvaluator(),
        ),
        execution_backend=exec_be,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(os.path.join(tmp, "data")),
        objective_cache=FileObjectiveCache(os.path.join(tmp, "data")),
        result_store=FileResultStore(os.path.join(tmp, "data")),
    )
    orch.start(spec, _settings_for_spec(spec, max_trials=max_trials))
    return orch.metrics.snapshot()


def _load_records(tmp: str, subdir: str) -> list[dict[str, Any]]:
    directory = Path(tmp) / "data" / subdir
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


class TestCacheKeySemantics:
    def test_same_target_same_params_second_run_hits_cache(self, tmp_path: Any) -> None:
        spec = make_spec(
            target_spec={"target_id": "target_cache_a", "config": {"market": "us", "venue": "paper"}}
        )
        first = _run_once(str(tmp_path), spec)
        second = _run_once(str(tmp_path), spec)

        assert first["execution_submitted_total"] == 1
        assert second["execution_submitted_total"] == 0
        assert second["objective_cache_hit_total"] >= 1

    def test_different_target_same_params_does_not_hit_previous_cache(self, tmp_path: Any) -> None:
        spec_a = make_spec(
            target_spec={"target_id": "target_cache_a", "config": {"market": "us", "venue": "paper"}}
        )
        spec_b = make_spec(
            target_spec={"target_id": "target_cache_b", "config": {"market": "us", "venue": "paper"}}
        )
        _run_once(str(tmp_path), spec_a)
        second = _run_once(str(tmp_path), spec_b)

        assert second["execution_submitted_total"] == 1
        assert second["objective_cache_hit_total"] == 0
        assert second["run_cache_hit_total"] == 0

    def test_same_target_same_run_result_different_objective_config_splits_objective_key(
        self,
        tmp_path: Any,
    ) -> None:
        obj_v1 = {
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {"alpha": 1},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "nop"},
        }
        obj_v2 = {
            "name": "test_loss",
            "version": "v2",
            "direction": "minimize",
            "params": {"alpha": 1},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "nop"},
        }
        spec_v1 = make_spec(
            target_spec={"target_id": "target_cache_c", "config": {"market": "us"}},
            objective_config=obj_v1,
        )
        spec_v2 = make_spec(
            target_spec={"target_id": "target_cache_c", "config": {"market": "us"}},
            objective_config=obj_v2,
        )
        _run_once(str(tmp_path), spec_v1)
        second = _run_once(str(tmp_path), spec_v2)

        assert second["execution_submitted_total"] == 0
        assert second["objective_cache_hit_total"] == 0
        assert second["run_cache_hit_total"] >= 1

    def test_run_record_contains_explicit_target_id(self, tmp_path: Any) -> None:
        spec = make_spec(
            target_spec={"target_id": "target_audit_1", "config": {"market": "crypto"}}
        )
        _run_once(str(tmp_path), spec)

        run_records = _load_records(str(tmp_path), "run_records")
        assert len(run_records) == 1
        assert run_records[0]["target_id"] == "target_audit_1"

    def test_different_targets_produce_isolated_run_records(self, tmp_path: Any) -> None:
        spec_a = make_spec(
            target_spec={"target_id": "target_iso_a", "config": {"market": "us"}}
        )
        spec_b = make_spec(
            target_spec={"target_id": "target_iso_b", "config": {"market": "us"}}
        )
        _run_once(str(tmp_path), spec_a)
        _run_once(str(tmp_path), spec_b)

        run_records = _load_records(str(tmp_path), "run_records")
        assert len(run_records) == 2
        assert {record["target_id"] for record in run_records} == {
            "target_iso_a",
            "target_iso_b",
        }
        assert len({record["run_key"] for record in run_records}) == 2
