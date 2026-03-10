"""E2E-1: RandomSampler + ASYNC_FILL full flow."""
from __future__ import annotations

import hashlib
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
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ResolvedTarget,
    RunResult,
    RunSpec,
    TargetSpec,
    compute_spec_hash,
    stable_json_serialize,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext
from tests.conftest import (
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubTargetResolver,
    make_settings,
)


class RandomSearchSpace:
    """Samples from TrialContext so each trial gets distinct params."""

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, Any]:
        x = ctx.suggest_float("x", 0.0, 10.0)
        y = ctx.suggest_int("y", 1, 100)
        return {"x": x, "y": y}


class SimpleRunSpecBuilder:
    def build(
        self,
        resolved_target: ResolvedTarget,
        params: dict[str, Any],
        execution_config: dict[str, Any],
    ) -> RunSpec:
        return RunSpec(
            kind="backtest",
            config=dict(params),
            resources=dict(execution_config.get("default_resources", {})),
            resolved_target=resolved_target,
        )


class DeterministicRunKeyBuilder:
    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str:
        payload = stable_json_serialize({
            "config": run_spec.config,
            "resolved_target": run_spec.resolved_target.to_dict(),
            "meta": spec.meta,
        })
        return "run:" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class TestRandomSamplerE2E:
    def test_full_flow(self, tmp_path: Any) -> None:
        db = os.path.join(str(tmp_path), "test.db")
        backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")

        exec_be = FakeExecutionBackend()
        exec_be.set_default_script(FakeRunScript(
            run_result=RunResult(
                metrics={"metric_1": 0.42},
                diagnostics={"runtime_sec": 1.0},
                artifact_refs=[],
            ),
        ))

        obj_def = ObjectiveDefinition(
            search_space=RandomSearchSpace(),
            run_spec_builder=SimpleRunSpecBuilder(),
            run_key_builder=DeterministicRunKeyBuilder(),
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

        spec_meta = {"dataset_version": "ds_v1", "engine_version": "e_v1"}
        obj_cfg = {
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "nop"},
        }
        exec_cfg = {"executor_kind": "backtest", "default_resources": {"cpu": 1}}
        target_spec = TargetSpec(
            target_id="target_backtest_v1",
            config={"market": "us_equity", "venue": "paper"},
        )
        spec = ExperimentSpec(
            spec_id="e2e_random",
            spec_hash=compute_spec_hash("e2e_random", spec_meta, target_spec, obj_cfg, exec_cfg),
            meta=spec_meta,
            target_spec=target_spec,
            objective_config=obj_cfg,
            execution_config=exec_cfg,
        )

        settings = make_settings(
            spec_id="e2e_random",
            meta=spec_meta,
            objective_config=obj_cfg,
            execution_config=exec_cfg,
            sampler={"type": "random", "seed": 42},
            stop={"max_trials": 10},
            parallelism={"max_in_flight_trials": 3},
        )

        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        assert m["trials_completed_total"] == 10
        assert m["trials_asked_total"] == 10
        submitted = exec_be.submitted_requests()
        assert submitted
        assert all(
            request.run_spec.resolved_target.target_id == "target_backtest_v1"
            for request in submitted
        )

        trial_result_files = list((Path(tmp_path) / "data" / "trial_results").glob("*.json"))
        assert len(trial_result_files) == 10
