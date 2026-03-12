from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
BACKTESTSYS_ROOT = WORKSPACE_ROOT / "BackTestSys"
BASE_CONFIG_PATH = WORKSPACE_ROOT / "config.xml"
MOCK_ROOT = WORKSPACE_ROOT / "mock_backtestsys"
MAX_TRIALS = 10
MAX_FAILURES = 2
FIXED_TIME_SCALE_LAMBDA = 0.0
FIXED_CANCEL_BIAS_K = 0.0

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optimization_control_plane.adapters.backtestsys import (  # noqa: E402
    BackTestDatasetEnumeratorAdapter,
    BackTestGroundTruthProviderAdapter,
    BackTestObjectiveEvaluatorAdapter,
    BackTestObjectiveKeyBuilderAdapter,
    BackTestRunKeyBuilderAdapter,
    BackTestRunResultLoaderAdapter,
    BackTestRunSpecBuilderAdapter,
    BackTestTrialResultAggregatorAdapter,
)
from optimization_control_plane.adapters.execution import MultiprocessExecutionBackend  # noqa: E402
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter  # noqa: E402
from optimization_control_plane.adapters.policies import (  # noqa: E402
    AsyncFillParallelismPolicy,
    SubmitNowDispatchPolicy,
)
from optimization_control_plane.adapters.storage import (  # noqa: E402
    FileObjectiveCache,
    FileResultStore,
    FileRunCache,
)
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator  # noqa: E402
from optimization_control_plane.domain.models import ExperimentSpec  # noqa: E402
from optimization_control_plane.ports.optimizer_backend import TrialContext  # noqa: E402


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required path not found: {path}")


def _read_delay_range(spec: ExperimentSpec) -> tuple[int, int]:
    space = spec.objective_config.get("backtest_search_space")
    if not isinstance(space, dict):
        raise ValueError("objective_config.backtest_search_space must be a dict")
    delay_cfg = space.get("delay")
    if not isinstance(delay_cfg, dict):
        raise ValueError("objective_config.backtest_search_space.delay must be a dict")
    low = delay_cfg.get("low")
    high = delay_cfg.get("high")
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError("objective_config.backtest_search_space.delay low/high must be int")
    if low > high:
        raise ValueError("objective_config.backtest_search_space.delay low must be <= high")
    return low, high


class DelayEqualitySearchSpaceAdapter:
    """Sample one delay and enforce delay_in == delay_out."""

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]:
        low, high = _read_delay_range(spec)
        sampled_delay = ctx.suggest_int("delay", low, high)
        sampled = {
            "time_scale_lambda": FIXED_TIME_SCALE_LAMBDA,
            "cancel_bias_k": FIXED_CANCEL_BIAS_K,
            "delay_in": sampled_delay,
            "delay_out": sampled_delay,
        }
        ctx.set_user_attr("backtest_config_patch", sampled)
        return sampled


def _build_objective_definition() -> ObjectiveDefinition:
    return ObjectiveDefinition(
        search_space=DelayEqualitySearchSpaceAdapter(),
        dataset_enumerator=BackTestDatasetEnumeratorAdapter(),
        run_spec_builder=BackTestRunSpecBuilderAdapter(),
        run_key_builder=BackTestRunKeyBuilderAdapter(),
        objective_key_builder=BackTestObjectiveKeyBuilderAdapter(),
        trial_result_aggregator=BackTestTrialResultAggregatorAdapter(),
        progress_scorer=None,
        objective_evaluator=BackTestObjectiveEvaluatorAdapter(),
    )


def _build_settings(runtime_root: Path) -> dict[str, Any]:
    doneinfo_gt = MOCK_ROOT / "groundtruth" / "PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv"
    executiondetail_gt = MOCK_ROOT / "groundtruth" / "PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv"
    dataset_paths = {
        "ds_01": str(MOCK_ROOT / "market_data_ds_01.csv"),
        "ds_02": str(MOCK_ROOT / "market_data_ds_02.csv"),
        "ds_03": str(MOCK_ROOT / "market_data_ds_03.csv"),
    }
    return {
        "spec_id": "iter_backtestsys_delay_equal_demo",
        "meta": {
            "dataset_version": "mock_v2",
            "engine_version": "backtestsys_main_py",
            "dataset_ids": ["ds_01", "ds_02", "ds_03"],
        },
        "objective_config": {
            "name": "backtest_loss",
            "version": "v2_delay_equal",
            "direction": "minimize",
            "params": {
                "weights": {"curve": 0.5, "terminal": 0.5, "cancel": 0.0, "post": 0.0},
                "baseline": {"curve": 1.0, "terminal": 1.0, "cancel": 1.0, "post": 1.0},
                "eps": {"curve": 1e-12, "terminal": 1e-12, "cancel": 1e-12, "post": 1e-12},
            },
            "groundtruth": {
                "doneinfo_path": str(doneinfo_gt),
                "executiondetail_path": str(executiondetail_gt),
            },
            "backtest_search_space": {
                "delay": {"low": 0, "high": 500000},
            },
        },
        "execution_config": {
            "executor_kind": "backtest",
            "default_resources": {"cpu": 1, "max_runtime_seconds": 60},
            "backtest_run_spec": {
                "backtestsys_root": str(BACKTESTSYS_ROOT),
                "base_config_path": str(BASE_CONFIG_PATH),
                "output_root_dir": str(runtime_root / "artifacts"),
                "dataset_paths": dataset_paths,
                "python_executable": sys.executable,
            },
        },
        "sampler": {"type": "random", "seed": 42},
        "pruner": {"type": "nop"},
        "parallelism": {"max_in_flight_trials": 1},
        "stop": {"max_trials": MAX_TRIALS, "max_failures": MAX_FAILURES},
    }


def _build_orchestrator(storage_dsn: str, data_root: Path) -> TrialOrchestrator:
    return TrialOrchestrator(
        backend=OptunaBackendAdapter(storage_dsn=storage_dsn),
        objective_def=_build_objective_definition(),
        groundtruth_provider=BackTestGroundTruthProviderAdapter(),
        execution_backend=MultiprocessExecutionBackend(),
        parallelism_policy=AsyncFillParallelismPolicy(),
        dispatch_policy=SubmitNowDispatchPolicy(),
        run_result_loader=BackTestRunResultLoaderAdapter(),
        run_cache=FileRunCache(data_root),
        objective_cache=FileObjectiveCache(data_root),
        result_store=FileResultStore(data_root),
    )


def main() -> None:
    _must_exist(BACKTESTSYS_ROOT / "main.py")
    _must_exist(BASE_CONFIG_PATH)
    _must_exist(MOCK_ROOT / "market_data_ds_01.csv")
    _must_exist(MOCK_ROOT / "market_data_ds_02.csv")
    _must_exist(MOCK_ROOT / "market_data_ds_03.csv")
    _must_exist(MOCK_ROOT / "replay_orders.csv")
    _must_exist(MOCK_ROOT / "contracts.xml")

    runtime_root = WORKSPACE_ROOT / "runtime" / "iter_backtestsys"
    runtime_root.mkdir(parents=True, exist_ok=True)
    storage_dsn = f"sqlite:///{(runtime_root / 'study.db').resolve()}"
    trial_ortrastrator = _build_orchestrator(
        storage_dsn=storage_dsn,
        data_root=runtime_root / "ocp_data",
    )
    trial_ortrastrator.start(settings=_build_settings(runtime_root))
    print(json.dumps(trial_ortrastrator.metrics.snapshot(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
