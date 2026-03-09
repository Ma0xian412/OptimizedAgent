from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
from optimization_control_plane.core.orchestration._start_spec import (
    spec_to_settings_payload,
)
from optimization_control_plane.domain.models import (
    Checkpoint,
    ExperimentSpec,
    ObjectiveResult,
    RunResult,
    RunSpec,
    compute_spec_hash,
    stable_json_serialize,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext

DEFAULT_ROOT_CONFIG_PATH = Path("config/config.json")
DEFAULT_METRIC_NAME = "metric_1"
START_MODE_BOTH = "both"
START_MODE_SPEC_ONLY = "spec_only"
START_MODE_SETTINGS_ONLY = "settings_only"


@dataclass(frozen=True)
class RootConfigPaths:
    framework_config_path: str
    iteratee_config_path: str
    hyperparams_config_path: str


class ConfigDrivenSearchSpace:
    def __init__(self, dimensions: list[dict[str, Any]]) -> None:
        self._dimensions = dimensions

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]:
        sampled: dict[str, object] = {}
        for dimension in self._dimensions:
            name, value = _sample_dimension(ctx, dimension)
            sampled[name] = value
        return sampled


class ConfigRunSpecBuilder:
    def __init__(
        self, run_kind: str, resources: dict[str, Any], fixed_params: dict[str, Any],
    ) -> None:
        self._run_kind = run_kind
        self._resources = dict(resources)
        self._fixed_params = dict(fixed_params)

    def build(self, params: dict[str, object], spec: ExperimentSpec) -> RunSpec:
        merged_params = {
            **self._fixed_params,
            **params,
        }
        return RunSpec(
            kind=self._run_kind,
            config=merged_params,
            resources=self._resources,
        )


class DeterministicRunKeyBuilder:
    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str:
        payload = stable_json_serialize({
            "kind": run_spec.kind,
            "config": run_spec.config,
            "meta": spec.meta,
        })
        return "run:" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class DeterministicObjectiveKeyBuilder:
    def build(self, run_key: str, objective_config: dict[str, object]) -> str:
        payload = stable_json_serialize({
            "run_key": run_key,
            "objective_config": objective_config,
        })
        return "obj:" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class MetricProgressScorer:
    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name

    def score(self, checkpoint: Checkpoint, spec: ExperimentSpec) -> float | None:
        value = checkpoint.metrics.get(self._metric_name)
        if value is None:
            return None
        return float(value)


class MetricObjectiveEvaluator:
    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name

    def evaluate(self, run_result: RunResult, spec: ExperimentSpec) -> ObjectiveResult:
        value = run_result.metrics.get(self._metric_name, 0.0)
        return ObjectiveResult(
            value=float(value),
            attrs={"metric": self._metric_name},
            artifact_refs=list(run_result.artifact_refs),
        )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return data


def parse_root_config(path: Path) -> RootConfigPaths:
    data = load_json(path)
    return RootConfigPaths(
        framework_config_path=_read_required_str(data, "framework_config_path", path),
        iteratee_config_path=_read_required_str(data, "iteratee_config_path", path),
        hyperparams_config_path=_read_required_str(data, "hyperparams_config_path", path),
    )


def build_experiment_spec(iteratee_config: dict[str, Any]) -> ExperimentSpec:
    spec_cfg = _read_required_dict(iteratee_config, "spec", "iteratee config")
    spec_id = _read_required_str(spec_cfg, "spec_id", "iteratee spec")
    meta = _read_required_dict(spec_cfg, "meta", "iteratee spec")
    objective_config = _read_required_dict(spec_cfg, "objective_config", "iteratee spec")
    execution_config = _read_required_dict(spec_cfg, "execution_config", "iteratee spec")
    spec_hash = compute_spec_hash(spec_id, meta, objective_config, execution_config)
    return ExperimentSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        meta=meta,
        objective_config=objective_config,
        execution_config=execution_config,
    )


def build_objective_definition(
    iteratee_config: dict[str, Any], hyperparams_config: dict[str, Any],
) -> ObjectiveDefinition:
    objective_cfg = _read_optional_dict(iteratee_config, "objective")
    spec_cfg = _read_required_dict(iteratee_config, "spec", "iteratee config")
    execution_cfg = _read_required_dict(spec_cfg, "execution_config", "iteratee spec")

    dimensions = _read_required_list_of_dict(hyperparams_config, "search_space", "hyperparams config")
    fixed_params = _read_optional_dict(hyperparams_config, "fixed_params")

    run_kind = str(
        objective_cfg.get(
            "run_kind",
            execution_cfg.get("executor_kind", "backtest"),
        )
    )
    resources = _read_optional_dict(execution_cfg, "default_resources")
    metric_name = str(objective_cfg.get("metric_name", DEFAULT_METRIC_NAME))
    progress_metric_name = objective_cfg.get("progress_metric_name")

    progress_scorer = None
    if progress_metric_name is not None:
        progress_scorer = MetricProgressScorer(str(progress_metric_name))

    return ObjectiveDefinition(
        search_space=ConfigDrivenSearchSpace(dimensions),
        run_spec_builder=ConfigRunSpecBuilder(
            run_kind=run_kind,
            resources=resources,
            fixed_params=fixed_params,
        ),
        run_key_builder=DeterministicRunKeyBuilder(),
        objective_key_builder=DeterministicObjectiveKeyBuilder(),
        progress_scorer=progress_scorer,
        objective_evaluator=MetricObjectiveEvaluator(metric_name),
    )


def build_orchestrator(
    framework_config: dict[str, Any], objective_definition: ObjectiveDefinition,
) -> TrialOrchestrator:
    study_cfg = _read_optional_dict(framework_config, "study")
    storage_cfg = _read_optional_dict(framework_config, "storage")
    execution_cfg = _read_optional_dict(framework_config, "execution")

    storage_dsn = str(study_cfg.get("storage_dsn", "sqlite:///study.db"))
    _ensure_sqlite_parent_directory(storage_dsn)
    study_name_prefix = str(study_cfg.get("study_name_prefix", ""))

    backend = OptunaBackendAdapter(
        storage_dsn=storage_dsn,
        study_name_prefix=study_name_prefix,
    )
    execution_backend = _build_execution_backend(execution_cfg)

    data_dir = storage_cfg.get("data_dir", "data")
    if not isinstance(data_dir, str):
        raise ValueError("framework.storage.data_dir must be a string")

    return TrialOrchestrator(
        backend=backend,
        objective_def=objective_definition,
        execution_backend=execution_backend,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(data_dir),
        objective_cache=FileObjectiveCache(data_dir),
        result_store=FileResultStore(data_dir),
    )


def run(root_config_path: Path) -> dict[str, float]:
    root_paths = parse_root_config(root_config_path)
    config_dir = root_config_path.parent

    framework_config = load_json(_resolve_config_path(config_dir, root_paths.framework_config_path))
    iteratee_config = load_json(_resolve_config_path(config_dir, root_paths.iteratee_config_path))
    hyperparams_config = load_json(_resolve_config_path(config_dir, root_paths.hyperparams_config_path))

    objective_definition = build_objective_definition(iteratee_config, hyperparams_config)
    spec = build_experiment_spec(iteratee_config)
    orchestrator = build_orchestrator(framework_config, objective_definition)

    settings = _read_required_dict(framework_config, "settings", "framework config")
    settings_with_spec = _merge_settings_spec(settings, spec)
    start_mode = str(framework_config.get("start_mode", START_MODE_BOTH))
    _start_with_mode(
        orchestrator=orchestrator,
        start_mode=start_mode,
        spec=spec,
        settings=settings_with_spec,
    )
    snapshot = orchestrator.metrics.snapshot()
    return {k: float(v) for k, v in snapshot.items()}


def _build_execution_backend(execution_cfg: dict[str, Any]) -> FakeExecutionBackend:
    run_result_cfg = _read_required_dict(execution_cfg, "default_run_result", "framework execution")
    metrics = _read_required_dict(run_result_cfg, "metrics", "framework execution result")
    diagnostics = _read_required_dict(run_result_cfg, "diagnostics", "framework execution result")
    artifact_refs_raw = run_result_cfg.get("artifact_refs", [])
    if not isinstance(artifact_refs_raw, list):
        raise ValueError("framework.execution.default_run_result.artifact_refs must be a list")

    run_result = RunResult(
        metrics=metrics,
        diagnostics=diagnostics,
        artifact_refs=[str(item) for item in artifact_refs_raw],
    )
    backend = FakeExecutionBackend()
    backend.set_default_script(FakeRunScript(run_result=run_result))
    return backend


def _sample_dimension(ctx: TrialContext, dimension: dict[str, Any]) -> tuple[str, object]:
    name = _read_required_str(dimension, "name", "search space dimension")
    dim_type = _read_required_str(dimension, "type", f"search space dimension '{name}'")
    if dim_type == "int":
        low = int(dimension["low"])
        high = int(dimension["high"])
        return name, ctx.suggest_int(name, low, high)
    if dim_type == "float":
        low = float(dimension["low"])
        high = float(dimension["high"])
        return name, ctx.suggest_float(name, low, high)
    if dim_type == "categorical":
        choices = dimension.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"categorical dimension '{name}' requires non-empty choices")
        return name, ctx.suggest_categorical(name, choices)
    raise ValueError(f"unsupported search space dimension type: {dim_type}")


def _merge_settings_spec(settings: dict[str, Any], spec: ExperimentSpec) -> dict[str, Any]:
    merged_settings = dict(settings)
    if "spec" not in merged_settings:
        merged_settings["spec"] = spec_to_settings_payload(spec)
    return merged_settings


def _start_with_mode(
    *,
    orchestrator: TrialOrchestrator,
    start_mode: str,
    spec: ExperimentSpec,
    settings: dict[str, Any],
) -> None:
    if start_mode == START_MODE_BOTH:
        orchestrator.start(spec=spec, settings=settings)
        return
    if start_mode == START_MODE_SPEC_ONLY:
        orchestrator.start(spec=spec)
        return
    if start_mode == START_MODE_SETTINGS_ONLY:
        orchestrator.start(settings=settings)
        return
    raise ValueError(f"unsupported framework.start_mode: {start_mode}")


def _resolve_config_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _ensure_sqlite_parent_directory(storage_dsn: str) -> None:
    sqlite_prefix = "sqlite:///"
    if not storage_dsn.startswith(sqlite_prefix):
        return
    db_path = storage_dsn.removeprefix(sqlite_prefix)
    if db_path == ":memory:":
        return
    parent = Path(db_path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _read_required_dict(data: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _read_optional_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _read_required_str(data: dict[str, Any], key: str, context: str | Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _read_required_list_of_dict(
    data: dict[str, Any], key: str, context: str,
) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{context}.{key} must only contain objects")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimization control-plane app entrypoint")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_ROOT_CONFIG_PATH),
        help="Root config path (contains paths of all sub-configs)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = _parse_args()
    metrics = run(Path(args.config).resolve())
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
