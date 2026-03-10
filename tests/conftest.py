from __future__ import annotations

import hashlib
from typing import Any

from optimization_control_plane.domain.models import (
    Checkpoint,
    ExperimentSpec,
    ObjectiveResult,
    ResolvedTarget,
    RunResult,
    RunSpec,
    TargetSpec,
    compute_spec_hash,
    stable_json_serialize,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext


def make_spec(**overrides: Any) -> ExperimentSpec:
    meta = overrides.pop("meta", {"dataset_version": "ds_v1", "engine_version": "e_v1"})
    target_spec = overrides.pop("target_spec", {
        "target_id": "target_backtest_v1",
        "config": {"market": "us_equity", "venue": "paper"},
    })
    obj_cfg = overrides.pop("objective_config", {
        "name": "test_loss",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "sampler": {"type": "random", "seed": 42},
        "pruner": {"type": "nop"},
    })
    exec_cfg = overrides.pop("execution_config", {
        "executor_kind": "backtest",
        "default_resources": {"cpu": 1},
    })
    spec_id = overrides.pop("spec_id", "test_spec")
    if isinstance(target_spec, TargetSpec):
        normalized_target = target_spec
    else:
        normalized_target = TargetSpec.from_dict(target_spec)
    spec_hash = compute_spec_hash(spec_id, meta, normalized_target, obj_cfg, exec_cfg)
    return ExperimentSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        meta=meta,
        target_spec=normalized_target,
        objective_config=obj_cfg,
        execution_config=exec_cfg,
    )


def make_settings(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "spec_id": "test_spec",
        "meta": {"dataset_version": "ds_v1", "engine_version": "e_v1"},
        "target_spec": {
            "target_id": "target_backtest_v1",
            "config": {"market": "us_equity", "venue": "paper"},
        },
        "objective_config": {
            "name": "test_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
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
        resolved_target: ResolvedTarget,
        params: dict[str, Any],
        execution_config: dict[str, Any],
    ) -> RunSpec:
        return RunSpec(
            kind="test",
            config=dict(params),
            resources=dict(execution_config.get("default_resources", {})),
            resolved_target=resolved_target,
        )


class StubRunKeyBuilder:
    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str:
        payload = stable_json_serialize({
            "kind": run_spec.kind,
            "config": run_spec.config,
            "resolved_target": run_spec.resolved_target.to_dict(),
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

    def evaluate(self, run_result: RunResult, spec: ExperimentSpec) -> ObjectiveResult:
        value = run_result.metrics.get(self._metric, 0.0)
        return ObjectiveResult(
            value=float(value),
            attrs={"metric": self._metric},
            artifact_refs=list(run_result.artifact_refs),
        )


class StubTargetResolver:
    def __init__(
        self,
        resolved_target: ResolvedTarget | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self._resolved_target = resolved_target
        self._fail_with = fail_with
        self.calls: list[tuple[TargetSpec, ExperimentSpec]] = []

    def resolve(self, target_spec: TargetSpec, spec: ExperimentSpec) -> ResolvedTarget:
        self.calls.append((target_spec, spec))
        if self._fail_with is not None:
            raise self._fail_with
        if self._resolved_target is not None:
            return self._resolved_target
        return ResolvedTarget.from_target_spec(target_spec)
