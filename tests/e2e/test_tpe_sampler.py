"""E2E-2: TPESampler + ASYNC_FILL full flow."""
from __future__ import annotations

import hashlib
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
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    Job,
    RunResult,
    RunSpec,
    compute_spec_hash,
    stable_json_serialize,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext
from tests.conftest import (
    StubDatasetEnumerator,
    StubGroundTruthProvider,
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubTrialResultAggregator,
    make_settings,
)


class TPESearchSpace:
    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, Any]:
        x = ctx.suggest_float("x", 0.0, 10.0)
        y = ctx.suggest_int("y", 1, 100)
        return {"x": x, "y": y}


class SimpleRunSpecBuilder:
    def build(self, params: dict[str, Any], spec: ExperimentSpec, dataset_id: str) -> RunSpec:
        return RunSpec(
            job=Job(
                command=["python", "backtest.py"],
                args=[f"--{k}={params[k]}" for k in sorted(params)] + [f"--dataset={dataset_id}"],
            )
        )


class DeterministicRunKeyBuilder:
    def build(self, run_spec: RunSpec, spec: ExperimentSpec, dataset_id: str) -> str:
        payload = stable_json_serialize({
            "job": {
                "command": run_spec.job.command,
                "script_path": run_spec.job.script_path,
                "args": run_spec.job.args,
                "env": run_spec.job.env,
                "working_dir": run_spec.job.working_dir,
            },
            "meta": spec.meta,
            "dataset_id": dataset_id,
        })
        return "run:" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class TestTPESamplerE2E:
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
            search_space=TPESearchSpace(),
            dataset_enumerator=StubDatasetEnumerator(),
            run_spec_builder=SimpleRunSpecBuilder(),
            run_key_builder=DeterministicRunKeyBuilder(),
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
            run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
            objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
            result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
        )

        spec_meta = {"dataset_version": "ds_v1", "engine_version": "e_v1"}
        obj_cfg = {
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
            "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
            "sampler": {"type": "tpe", "n_startup_trials": 5, "constant_liar": True, "seed": 42},
            "pruner": {"type": "nop"},
        }
        exec_cfg = {"executor_kind": "backtest", "default_resources": {"cpu": 1}}
        spec = ExperimentSpec(
            spec_id="e2e_tpe",
            spec_hash=compute_spec_hash("e2e_tpe", spec_meta, obj_cfg, exec_cfg),
            meta=spec_meta,
            objective_config=obj_cfg,
            execution_config=exec_cfg,
        )

        settings = make_settings(
            spec_id="e2e_tpe",
            meta=spec_meta,
            objective_config=obj_cfg,
            execution_config=exec_cfg,
            sampler={"type": "tpe", "n_startup_trials": 5, "constant_liar": True, "seed": 42},
            stop={"max_trials": 15},
            parallelism={"max_in_flight_trials": 4},
        )

        orch.start(spec, settings)

        m = orch.metrics.snapshot()
        assert m["trials_completed_total"] == 15
        assert m["trials_asked_total"] == 15
