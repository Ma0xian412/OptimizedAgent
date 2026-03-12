from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    def build(
        self,
        params: dict[str, Any],
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> RunSpec:
        default_resources = spec.execution_config.get("default_resources", {})
        cpu = default_resources.get("cpu")
        memory_mb = default_resources.get("memory_mb")
        memory_gb = default_resources.get("memory_gb")
        if memory_mb is None and isinstance(memory_gb, int):
            memory_mb = memory_gb * 1024
        result_base_dir = spec.execution_config.get("result_base_dir", "/tmp/ocp_results")
        payload = stable_json_serialize(
            {"spec_id": spec.spec_id, "dataset_id": dataset_id, "params": params}
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return RunSpec(
            job=Job(
                command=["python", "runner.py"],
                args=[f"--{k}={params[k]}" for k in sorted(params)] + [f"--dataset={dataset_id}"],
            ),
            result_path=f"{result_base_dir}/{spec.spec_id}/{dataset_id}_{digest}.json",
            resource_request=ResourceRequest(
                cpu_cores=cpu if isinstance(cpu, int) else None,
                memory_mb=memory_mb if isinstance(memory_mb, int) else None,
            ),
        )


class StubRunKeyBuilder:
    def build(self, run_spec: RunSpec, spec: ExperimentSpec, dataset_id: str) -> str:
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
            "dataset_id": dataset_id,
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
        payload = run_result.payload
        if not isinstance(payload, dict):
            raise TypeError("run_result.payload must be a dict for StubObjectiveEvaluator")
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError("run_result.payload.metrics must be a dict for StubObjectiveEvaluator")
        artifact_refs = payload.get("artifact_refs", [])
        if not isinstance(artifact_refs, list):
            raise TypeError("run_result.payload.artifact_refs must be a list for StubObjectiveEvaluator")
        value = metrics.get(self._metric, 0.0)
        return ObjectiveResult(
            attrs={
                "value": float(value),
                "metric": self._metric,
                "groundtruth_fingerprint": groundtruth.fingerprint,
            },
            artifact_refs=list(artifact_refs),
        )


class StubGroundTruthProvider:
    def load(self, spec: ExperimentSpec, dataset_id: str) -> GroundTruthData:
        groundtruth = spec.objective_config.get("groundtruth")
        if not isinstance(groundtruth, dict):
            raise ValueError("spec.objective_config.groundtruth must be a dict")
        payload = stable_json_serialize(groundtruth)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return GroundTruthData(payload=groundtruth, fingerprint=f"sha256:{digest}")


class StubDatasetEnumerator:
    def __init__(self, dataset_ids: tuple[str, ...] | None = None) -> None:
        self._dataset_ids = dataset_ids

    def enumerate(self, spec: ExperimentSpec) -> tuple[str, ...]:
        if self._dataset_ids is not None:
            return self._dataset_ids
        dataset_ids = spec.meta.get("dataset_ids")
        if isinstance(dataset_ids, (list, tuple)):
            return tuple(str(dataset_id) for dataset_id in dataset_ids)
        dataset_version = spec.meta.get("dataset_version", "default")
        return (str(dataset_version),)


class StubTrialResultAggregator:
    def aggregate(
        self,
        results: list[tuple[str, ObjectiveResult]],
        spec: ExperimentSpec,
    ) -> ObjectiveResult:
        if not results:
            raise ValueError("cannot aggregate empty results")
        if len(results) == 1:
            return results[0][1]
        total = sum(result.value for _, result in results)
        attrs = {"value": total / len(results), "aggregated_dataset_count": len(results)}
        return ObjectiveResult(attrs=attrs, artifact_refs=[])


class StubRunResultLoader:
    def load(self, run_spec: RunSpec) -> RunResult:
        path = Path(run_spec.result_path)
        if not path.exists():
            raise FileNotFoundError(f"run result file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("run result file must contain a JSON object")
        return RunResult(payload=data["payload"])
