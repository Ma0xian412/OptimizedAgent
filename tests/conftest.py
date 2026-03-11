from __future__ import annotations

import hashlib
from typing import Any

from optimization_control_plane.domain.models import (
    Checkpoint,
    ExperimentSpec,
    GroundTruthData,
    Job,
    ObjectiveResult,
    ResourceRequest,
    RunResult,
    RunSpec,
    compute_spec_hash,
    stable_json_serialize,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext


def make_spec(**overrides: Any) -> ExperimentSpec:
    meta = overrides.pop("meta", {"dataset_version": "ds_v1", "engine_version": "e_v1"})
    obj_cfg = overrides.pop("objective_config", {
        "name": "test_loss",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
        "sampler": {"type": "random", "seed": 42},
        "pruner": {"type": "nop"},
    })
    exec_cfg = overrides.pop("execution_config", {
        "executor_kind": "backtest",
        "default_resources": {"cpu": 1},
    })
    spec_id = overrides.pop("spec_id", "test_spec")
    spec_hash = compute_spec_hash(spec_id, meta, obj_cfg, exec_cfg)
    return ExperimentSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        meta=meta,
        objective_config=obj_cfg,
        execution_config=exec_cfg,
    )


def make_settings(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "spec_id": "test_spec",
        "meta": {"dataset_version": "ds_v1", "engine_version": "e_v1"},
        "objective_config": {
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
            "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "nop"},
        },
        "execution_config": {
            "executor_kind": "backtest",
            "default_resources": {"cpu": 1},
        },
        "sampler": {"type": "random", "seed": 42},
        "pruner": {"type": "nop"},
        "parallelism": {"max_in_flight_trials": 2},
        "stop": {"max_trials": 10},
    }
    base.update(overrides)
    return base


class StubSearchSpace:
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {"x": 1.0}
        self.calls: list[dict[str, Any]] = []

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, Any]:
        self.calls.append(dict(self._params))
        return dict(self._params)


class StubRunSpecBuilder:
    def build(self, params: dict[str, Any], spec: ExperimentSpec) -> RunSpec:
        default_resources = spec.execution_config.get("default_resources", {})
        cpu = default_resources.get("cpu")
        memory_mb = default_resources.get("memory_mb")
        memory_gb = default_resources.get("memory_gb")
        if memory_mb is None and isinstance(memory_gb, int):
            memory_mb = memory_gb * 1024
        return RunSpec(
            job=Job(
                command=["python", "runner.py"],
                args=[f"--{k}={params[k]}" for k in sorted(params)],
            ),
            resource_request=ResourceRequest(
                cpu_cores=cpu if isinstance(cpu, int) else None,
                memory_mb=memory_mb if isinstance(memory_mb, int) else None,
            ),
        )


class StubRunKeyBuilder:
    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str:
        payload = stable_json_serialize({
            "job": {
                "command": run_spec.job.command,
                "script_path": run_spec.job.script_path,
                "args": run_spec.job.args,
                "env": run_spec.job.env,
                "working_dir": run_spec.job.working_dir,
            },
            "resource_request": {
                "cpu_cores": run_spec.resource_request.cpu_cores,
                "memory_mb": run_spec.resource_request.memory_mb,
                "gpu_count": run_spec.resource_request.gpu_count,
                "max_runtime_seconds": run_spec.resource_request.max_runtime_seconds,
            },
            "meta": spec.meta,
        })
        return "run:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class StubObjectiveKeyBuilder:
    def build(self, run_key: str, objective_config: dict[str, Any]) -> str:
        payload = stable_json_serialize({
            "run_key": run_key,
            "name": objective_config.get("name"),
            "version": objective_config.get("version"),
            "params": objective_config.get("params"),
        })
        return "obj:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


class StubProgressScorer:
    def __init__(self, metric_name: str = "loss") -> None:
        self._metric = metric_name

    def score(self, checkpoint: Checkpoint, spec: ExperimentSpec) -> float | None:
        val = checkpoint.metrics.get(self._metric)
        return float(val) if val is not None else None


class StubObjectiveEvaluator:
    def __init__(self, metric_name: str = "metric_1") -> None:
        self._metric = metric_name

    def evaluate(
        self,
        run_result: RunResult,
        spec: ExperimentSpec,
        groundtruth: GroundTruthData,
    ) -> ObjectiveResult:
        value = run_result.metrics.get(self._metric, 0.0)
        return ObjectiveResult(
            value=float(value),
            attrs={"metric": self._metric, "groundtruth_fingerprint": groundtruth.fingerprint},
            artifact_refs=list(run_result.artifact_refs),
        )


class StubGroundTruthProvider:
    def load(self, spec: ExperimentSpec) -> GroundTruthData:
        groundtruth = spec.objective_config.get("groundtruth")
        if not isinstance(groundtruth, dict):
            raise ValueError("spec.objective_config.groundtruth must be a dict")
        payload = stable_json_serialize(groundtruth)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return GroundTruthData(payload=groundtruth, fingerprint=f"sha256:{digest}")
