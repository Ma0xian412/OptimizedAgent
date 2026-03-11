from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optimization_control_plane.adapters.backtestsys import (
    BackTestSysCountDiffEvaluator,
    BackTestSysDatasetDiscoveryAdapter,
    BackTestSysExecutionBackend,
    BackTestSysGroundTruthAdapter,
    BackTestSysGroundTruthProvider,
    BackTestSysObjectiveKeyBuilder,
    BackTestSysRunKeyBuilder,
    BackTestSysRunSpecBuilder,
    BackTestSysRunSpecDefaults,
    BackTestSysSearchSpace,
    DatasetDiscoveryConfig,
    MeanTrialLossAggregator,
    SearchParam,
)
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

_DEFAULT_CONFIG_PATH = "config.xml"
_LOSS_COMPONENTS = ("curve", "terminal", "cancel", "post")
_DEFAULT_LOSS_WEIGHTS = {
    "curve": 1.0,
    "terminal": 1.0,
    "cancel": 1.0,
    "post": 1.0,
}
_DEFAULT_LOSS_EPS = {
    "curve": 1e-12,
    "terminal": 1e-12,
    "cancel": 1e-12,
    "post": 1e-12,
}


@dataclass(frozen=True)
class AppConfig:
    spec_id: str
    dataset_version: str
    engine_version: str
    storage_dsn: str
    max_trials: int
    max_failures: int
    max_in_flight: int
    max_workers: int
    sampler: dict[str, Any]
    pruner: dict[str, Any]
    data_dir: str
    repo_root: str
    base_config_path: str
    replay_order_file: str
    replay_cancel_file: str
    groundtruth_dir: str
    search_params: list[SearchParam]
    base_overrides: dict[str, Any]
    dataset_files: list[dict[str, str]]
    train_ratio: int
    test_ratio: int
    dataset_seed: int
    loss_weights: dict[str, float]
    loss_eps: dict[str, float]


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    settings = _build_settings(cfg)
    orchestrator, execution_backend = _build_orchestrator(cfg)
    try:
        orchestrator.start(settings=settings)
    finally:
        execution_backend.shutdown(wait_for_tasks=True)
    _print_summary(orchestrator.metrics.snapshot())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimization Control Plane for BackTestSys")
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help=f"Path to xml config file (default: {_DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args()


def _load_config(config_path: str) -> AppConfig:
    root = ET.parse(config_path).getroot()
    search_params = _parse_search_params(root)
    base_overrides = _parse_base_overrides(root)
    dataset_files = _parse_dataset_files(root)
    loss_weights, loss_eps = _parse_loss_config(root)
    fallback_order = _optional_text(root, "./paths/replay_order_file")
    fallback_cancel = _optional_text(root, "./paths/replay_cancel_file")
    replay_order_file = _resolve_replay_file(dataset_files, "order_file", fallback_order)
    replay_cancel_file = _resolve_replay_file(dataset_files, "cancel_file", fallback_cancel)
    return AppConfig(
        spec_id=_require_text(root, "./study/spec_id"),
        dataset_version=_require_text(root, "./study/dataset_version"),
        engine_version=_require_text(root, "./study/engine_version"),
        storage_dsn=_require_text(root, "./study/storage_dsn"),
        max_trials=_as_int(_require_text(root, "./study/max_trials")),
        max_failures=_as_int(_require_text(root, "./study/max_failures")),
        max_in_flight=_as_int(_require_text(root, "./study/max_in_flight_trials")),
        max_workers=_as_int(_require_text(root, "./study/max_workers")),
        sampler=_parse_sampler(root),
        pruner=_parse_pruner(root),
        data_dir=_require_text(root, "./paths/data_dir"),
        repo_root=_require_text(root, "./paths/backtestsys_repo_root"),
        base_config_path=_require_text(root, "./paths/backtestsys_base_config"),
        replay_order_file=replay_order_file,
        replay_cancel_file=replay_cancel_file,
        groundtruth_dir=_require_text(root, "./paths/groundtruth_dir"),
        search_params=search_params,
        base_overrides=base_overrides,
        dataset_files=dataset_files,
        train_ratio=_as_int(_require_text(root, "./dataset_plan/train_ratio")),
        test_ratio=_as_int(_require_text(root, "./dataset_plan/test_ratio")),
        dataset_seed=_as_int(_require_text(root, "./dataset_plan/seed")),
        loss_weights=loss_weights,
        loss_eps=loss_eps,
    )


def _parse_search_params(root: ET.Element) -> list[SearchParam]:
    params: list[SearchParam] = []
    for node in root.findall("./search_space/param"):
        name = _require_attr(node, "name")
        param_type = _require_attr(node, "type")
        if param_type == "categorical":
            choices_raw = _require_attr(node, "choices")
            choices = tuple(item.strip() for item in choices_raw.split(",") if item.strip())
            params.append(SearchParam(name=name, param_type=param_type, choices=choices))
            continue
        low = _as_number(_require_attr(node, "low"), param_type)
        high = _as_number(_require_attr(node, "high"), param_type)
        params.append(SearchParam(name=name, param_type=param_type, low=low, high=high))
    if not params:
        raise ValueError("config search_space must define at least one <param>")
    return params


def _parse_base_overrides(root: ET.Element) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for node in root.findall("./base_overrides/override"):
        key = _require_attr(node, "key")
        value_type = _require_attr(node, "type")
        text = (node.text or "").strip()
        overrides[key] = _cast_value(text, value_type)
    return overrides


def _parse_dataset_files(root: ET.Element) -> list[dict[str, str]]:
    explicit_files = root.findall("./dataset_plan/files/file")
    if explicit_files:
        return _parse_explicit_dataset_files(explicit_files)
    discovery = _parse_discovery_config(root)
    return BackTestSysDatasetDiscoveryAdapter().discover(discovery)


def _parse_explicit_dataset_files(nodes: list[ET.Element]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for idx, node in enumerate(nodes):
        file_id = _require_attr(node, "id")
        path = _require_attr(node, "path")
        parsed = {"id": file_id or f"file_{idx}", "path": path}
        date = node.attrib.get("date")
        order_file = node.attrib.get("order_file")
        cancel_file = node.attrib.get("cancel_file")
        if date:
            parsed["date"] = date
        if order_file:
            parsed["order_file"] = order_file
        if cancel_file:
            parsed["cancel_file"] = cancel_file
        files.append(parsed)
    if len(files) < 2:
        raise ValueError("dataset_plan.files must contain at least 2 file nodes")
    return files


def _parse_discovery_config(root: ET.Element) -> DatasetDiscoveryConfig:
    return DatasetDiscoveryConfig(
        data_dir=_require_text(root, "./dataset_plan/auto_discovery/data_dir"),
        data_glob=_require_text(root, "./dataset_plan/auto_discovery/data_glob"),
        data_date_regex=_require_text(root, "./dataset_plan/auto_discovery/data_date_regex"),
        replay_order_dir=_require_text(root, "./dataset_plan/auto_discovery/replay_order_dir"),
        replay_order_pattern=_require_text(root, "./dataset_plan/auto_discovery/replay_order_pattern"),
        replay_cancel_dir=_require_text(root, "./dataset_plan/auto_discovery/replay_cancel_dir"),
        replay_cancel_pattern=_require_text(root, "./dataset_plan/auto_discovery/replay_cancel_pattern"),
    )


def _resolve_replay_file(
    dataset_files: list[dict[str, str]],
    key: str,
    fallback: str | None,
) -> str:
    if fallback:
        return fallback
    for item in dataset_files:
        value = item.get(key)
        if value:
            return value
    raise ValueError(f"cannot resolve default replay file: {key}")


def _parse_sampler(root: ET.Element) -> dict[str, Any]:
    sampler_type = _require_text(root, "./sampler/type")
    seed = _as_int(_require_text(root, "./sampler/seed"))
    sampler: dict[str, Any] = {"type": sampler_type, "seed": seed}
    startup = _optional_text(root, "./sampler/n_startup_trials")
    if startup is not None:
        sampler["n_startup_trials"] = _as_int(startup)
    liar = _optional_text(root, "./sampler/constant_liar")
    if liar is not None:
        sampler["constant_liar"] = liar.lower() == "true"
    return sampler


def _parse_pruner(root: ET.Element) -> dict[str, Any]:
    pruner: dict[str, Any] = {"type": _require_text(root, "./pruner/type")}
    startup = _optional_text(root, "./pruner/n_startup_trials")
    warmup = _optional_text(root, "./pruner/n_warmup_steps")
    if startup is not None:
        pruner["n_startup_trials"] = _as_int(startup)
    if warmup is not None:
        pruner["n_warmup_steps"] = _as_int(warmup)
    return pruner


def _parse_loss_config(root: ET.Element) -> tuple[dict[str, float], dict[str, float]]:
    weights = _parse_loss_component_map(
        root=root,
        parent_path="./loss/weights",
        defaults=_DEFAULT_LOSS_WEIGHTS,
        field_name="weights",
    )
    eps = _parse_loss_component_map(
        root=root,
        parent_path="./loss/eps",
        defaults=_DEFAULT_LOSS_EPS,
        field_name="eps",
    )
    return weights, eps


def _parse_loss_component_map(
    *,
    root: ET.Element,
    parent_path: str,
    defaults: dict[str, float],
    field_name: str,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in _LOSS_COMPONENTS:
        text = _optional_text(root, f"{parent_path}/{name}")
        value = defaults[name] if text is None else float(text)
        if value < 0.0:
            raise ValueError(f"loss.{field_name}.{name} must be >= 0, got {value}")
        output[name] = value
    return output


def _build_settings(cfg: AppConfig) -> dict[str, Any]:
    meta = {"dataset_version": cfg.dataset_version, "engine_version": cfg.engine_version}
    objective_config = {
        "name": "backtestsys_count_diff",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": cfg.groundtruth_dir},
        "weights": dict(cfg.loss_weights),
        "eps": dict(cfg.loss_eps),
        "sampler": dict(cfg.sampler),
        "pruner": dict(cfg.pruner),
    }
    execution_config = {
        "executor_kind": "backtestsys",
        "default_resources": {"cpu": cfg.max_workers},
    }
    return {
        "spec_id": cfg.spec_id,
        "meta": meta,
        "objective_config": objective_config,
        "execution_config": execution_config,
        "sampler": dict(cfg.sampler),
        "pruner": dict(cfg.pruner),
        "parallelism": {"max_in_flight_trials": cfg.max_in_flight},
        "stop": {
            "max_trials": cfg.max_trials,
            "max_failures": cfg.max_failures,
        },
        "dataset_plan": {
            "files": list(cfg.dataset_files),
            "train_ratio": cfg.train_ratio,
            "test_ratio": cfg.test_ratio,
            "seed": cfg.dataset_seed,
        },
    }


def _build_orchestrator(cfg: AppConfig) -> tuple[TrialOrchestrator, BackTestSysExecutionBackend]:
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    objective_def = ObjectiveDefinition(
        search_space=BackTestSysSearchSpace(cfg.search_params),
        run_spec_builder=BackTestSysRunSpecBuilder(
            BackTestSysRunSpecDefaults(
                repo_root=cfg.repo_root,
                base_config_path=cfg.base_config_path,
                replay_order_file=cfg.replay_order_file,
                replay_cancel_file=cfg.replay_cancel_file,
                base_overrides=dict(cfg.base_overrides),
            )
        ),
        run_key_builder=BackTestSysRunKeyBuilder(),
        objective_key_builder=BackTestSysObjectiveKeyBuilder(),
        progress_scorer=None,
        objective_evaluator=BackTestSysCountDiffEvaluator(),
        trial_loss_aggregator=MeanTrialLossAggregator(),
    )
    execution_backend = BackTestSysExecutionBackend(max_workers=cfg.max_workers)
    groundtruth_adapter = BackTestSysGroundTruthAdapter()
    orchestrator = TrialOrchestrator(
        backend=OptunaBackendAdapter(storage_dsn=cfg.storage_dsn),
        objective_def=objective_def,
        groundtruth_provider=BackTestSysGroundTruthProvider(groundtruth_adapter),
        execution_backend=execution_backend,
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_cache=FileRunCache(str(data_dir)),
        objective_cache=FileObjectiveCache(str(data_dir)),
        result_store=FileResultStore(str(data_dir)),
    )
    return orchestrator, execution_backend


def _print_summary(metrics: dict[str, int]) -> None:
    print("Optimization finished.")
    print(f"trials_asked_total={metrics.get('trials_asked_total', 0)}")
    print(f"trials_completed_total={metrics.get('trials_completed_total', 0)}")
    print(f"trials_failed_total={metrics.get('trials_failed_total', 0)}")


def _require_attr(node: ET.Element, name: str) -> str:
    value = node.attrib.get(name, "").strip()
    if not value:
        raise ValueError(f"xml node missing attribute '{name}': tag={node.tag}")
    return value


def _require_text(root: ET.Element, path: str) -> str:
    text = _optional_text(root, path)
    if text is None:
        raise ValueError(f"xml missing required path: {path}")
    return text


def _optional_text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


def _cast_value(text: str, value_type: str) -> Any:
    if value_type == "int":
        return _as_int(text)
    if value_type == "float":
        return float(text)
    if value_type == "bool":
        return text.lower() == "true"
    if value_type == "str":
        return text
    raise ValueError(f"unsupported override type: {value_type}")


def _as_number(text: str, value_type: str) -> int | float:
    if value_type == "int":
        return _as_int(text)
    if value_type == "float":
        return float(text)
    raise ValueError(f"unsupported numeric param type: {value_type}")


def _as_int(text: str) -> int:
    return int(float(text))


if __name__ == "__main__":
    main()
