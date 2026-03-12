from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import optuna
from optimization_control_plane.adapters.backtestsys import (
    BackTestDatasetEnumeratorAdapter, BackTestGroundTruthProviderAdapter, BackTestObjectiveEvaluatorAdapter,
    BackTestObjectiveKeyBuilderAdapter, BackTestRunKeyBuilderAdapter, BackTestRunResultLoaderAdapter,
    BackTestRunSpecBuilderAdapter, BackTestTrialResultAggregatorAdapter,
)
from optimization_control_plane.adapters.execution import MultiprocessExecutionBackend
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter
from optimization_control_plane.adapters.policies import AsyncFillParallelismPolicy, SubmitNowDispatchPolicy
from optimization_control_plane.adapters.storage import FileObjectiveCache, FileResultStore, FileRunCache
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.models import ExperimentSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext

_BACKTEST_ATTR_KEY = "backtest_config_patch"
_RAW_COMPONENTS = ("curve", "terminal", "cancel", "post")


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: str
    market_data_file: str
    machine: str
    contract: str


@dataclass(frozen=True)
class BacktestDefaults:
    time_scale_lambda: float
    cancel_bias_k: float
    delay_in: int
    delay_out: int


@dataclass(frozen=True)
class StageResult:
    best_value: float
    best_params: dict[str, object]
    best_attrs: dict[str, object]


@dataclass(frozen=True)
class CalibrationConfig:
    workspace_root: Path
    backtestsys_root: Path
    base_config_path: Path
    mock_root: Path
    datasets: tuple[DatasetDefinition, ...]
    max_failures: int
    baseline_trials: int
    machine_delay_trials: int
    contract_core_trials: int
    verify_trials: int
    default_resources: dict[str, int]


class FixedBacktestSearchSpaceAdapter:
    def __init__(self, params: dict[str, object]) -> None:
        self._params = dict(params)

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]:
        del spec
        ctx.set_user_attr(_BACKTEST_ATTR_KEY, dict(self._params))
        return dict(self._params)


def validate_required_paths(config: CalibrationConfig) -> None:
    required = [config.backtestsys_root / "main.py", config.base_config_path, config.mock_root / "replay_orders.csv",
                config.mock_root / "replay_cancels.csv", config.mock_root / "contracts.xml"]
    required.extend(config.mock_root / item.market_data_file for item in config.datasets)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"required path not found: {path}")


def build_dataset_inputs(config: CalibrationConfig) -> dict[str, dict[str, str]]:
    return {item.dataset_id: {"market_data_path": str(config.mock_root / item.market_data_file),
                              "order_file": str(config.mock_root / "replay_orders.csv"),
                              "cancel_file": str(config.mock_root / "replay_cancels.csv"),
                              "machine": item.machine, "contract": item.contract}
            for item in config.datasets}


def build_settings(config: CalibrationConfig, runtime_root: Path, *, spec_id: str, dataset_inputs: dict[str, dict[str, str]],
                   dataset_ids: list[str], baseline_raw: dict[str, float] | None, max_trials: int,
                   backtest_search_space: dict[str, object] | None, backtest_fixed_params: dict[str, object] | None,
                   param_binding: dict[str, object] | None) -> dict[str, Any]:
    objective_config: dict[str, object] = {"name": "backtest_loss", "version": "v3_staged_calibration", "direction": "minimize",
                                           "params": _build_loss_params(baseline_raw),
                                           "groundtruth": {"doneinfo_path": str(config.mock_root / "groundtruth" / "PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv"),
                                                           "executiondetail_path": str(config.mock_root / "groundtruth" / "PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv")}}
    if backtest_search_space is not None:
        objective_config["backtest_search_space"] = backtest_search_space
    if backtest_fixed_params is not None:
        objective_config["backtest_fixed_params"] = backtest_fixed_params
    run_spec: dict[str, object] = {"backtestsys_root": str(config.backtestsys_root), "base_config_path": str(config.base_config_path),
                                   "output_root_dir": str(runtime_root / "artifacts"), "dataset_inputs": dataset_inputs,
                                   "python_executable": "python3"}
    if param_binding is not None:
        run_spec["param_binding"] = param_binding
    return {"spec_id": spec_id,
            "meta": {"dataset_version": "mock_v2", "engine_version": "backtestsys_main_py", "dataset_ids": dataset_ids},
            "objective_config": objective_config,
            "execution_config": {"executor_kind": "backtest", "default_resources": dict(config.default_resources), "backtest_run_spec": run_spec},
            "sampler": {"type": "random", "seed": 42}, "pruner": {"type": "nop"},
            "parallelism": {"max_in_flight_trials": 1},
            "stop": {"max_trials": max_trials, "max_failures": config.max_failures}}


def read_default_params(base_config_path: Path) -> BacktestDefaults:
    root = ET.parse(base_config_path).getroot()
    return BacktestDefaults(time_scale_lambda=float(_read_xml_text(root, ("tape", "time_scale_lambda"))),
                            cancel_bias_k=float(_read_xml_text(root, ("exchange", "cancel_bias_k"))),
                            delay_in=int(_read_xml_text(root, ("runner", "delay_in"))),
                            delay_out=int(_read_xml_text(root, ("runner", "delay_out"))))


def extract_baseline_raw(attrs: dict[str, object]) -> dict[str, float]:
    raw = attrs.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("baseline attrs.raw must be a dict")
    result: dict[str, float] = {}
    for key in _RAW_COMPONENTS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"baseline raw.{key} must be numeric")
        result[key] = float(value)
    return result


def group_dataset_ids(datasets: tuple[DatasetDefinition, ...], *, key: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in datasets:
        grouped.setdefault(str(getattr(item, key)), []).append(item.dataset_id)
    return grouped


def unique_machine_for_contract(datasets: tuple[DatasetDefinition, ...], contract: str) -> str:
    machines = {item.machine for item in datasets if item.contract == contract}
    if len(machines) != 1:
        raise ValueError(f"contract={contract} maps to {len(machines)} machines")
    return next(iter(machines))


def as_float(raw: object, contract: str, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} for contract={contract} must be numeric")
    return float(raw)


def run_stage(runtime_root: Path, stage_name: str, settings: dict[str, Any], search_space: object) -> StageResult:
    stage_root = runtime_root / "stages" / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    storage_dsn = f"sqlite:///{(stage_root / 'study.db').resolve()}"
    orchestrator = _build_orchestrator(storage_dsn=storage_dsn, data_root=stage_root / "ocp_data", search_space=search_space)
    orchestrator.start(settings=settings)
    best_trial = _load_best_trial(storage_dsn)
    if best_trial.value is None:
        raise ValueError("best trial value is missing")
    return StageResult(best_value=float(best_trial.value), best_params=dict(best_trial.params), best_attrs=dict(best_trial.user_attrs))


def _build_loss_params(baseline_raw: dict[str, float] | None) -> dict[str, object]:
    baseline = baseline_raw or {"curve": 1.0, "terminal": 1.0, "cancel": 1.0, "post": 1.0}
    return {"weights": {"curve": 0.5, "terminal": 0.5, "cancel": 0.0, "post": 0.0}, "baseline": dict(baseline),
            "eps": {"curve": 1e-12, "terminal": 1e-12, "cancel": 1e-12, "post": 1e-12}}


def _read_xml_text(root: ET.Element, path: tuple[str, ...]) -> str:
    node: ET.Element | None = root
    for tag in path:
        node = node.find(tag) if node is not None else None
    if node is None or node.text is None or node.text.strip() == "":
        raise ValueError(f"base config missing required field: {'.'.join(path)}")
    return node.text.strip()


def _build_orchestrator(*, storage_dsn: str, data_root: Path, search_space: object) -> TrialOrchestrator:
    objective_def = ObjectiveDefinition(search_space=search_space, dataset_enumerator=BackTestDatasetEnumeratorAdapter(),
                                        run_spec_builder=BackTestRunSpecBuilderAdapter(), run_key_builder=BackTestRunKeyBuilderAdapter(),
                                        objective_key_builder=BackTestObjectiveKeyBuilderAdapter(),
                                        trial_result_aggregator=BackTestTrialResultAggregatorAdapter(),
                                        progress_scorer=None, objective_evaluator=BackTestObjectiveEvaluatorAdapter())
    return TrialOrchestrator(backend=OptunaBackendAdapter(storage_dsn=storage_dsn), objective_def=objective_def,
                             groundtruth_provider=BackTestGroundTruthProviderAdapter(),
                             execution_backend=MultiprocessExecutionBackend(),
                             parallelism_policy=AsyncFillParallelismPolicy(), dispatch_policy=SubmitNowDispatchPolicy(),
                             run_result_loader=BackTestRunResultLoaderAdapter(), run_cache=FileRunCache(data_root),
                             objective_cache=FileObjectiveCache(data_root), result_store=FileResultStore(data_root))


def _load_best_trial(storage_dsn: str) -> optuna.trial.FrozenTrial:
    studies = optuna.study.get_all_study_summaries(storage=storage_dsn)
    if len(studies) != 1:
        raise ValueError(f"expected one study in stage storage, got {len(studies)}")
    study = optuna.load_study(study_name=studies[0].study_name, storage=storage_dsn)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise ValueError("stage has no COMPLETE trials")
    return study.best_trial
