from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
BACKTESTSYS_ROOT = WORKSPACE_ROOT / "BackTestSys"
BASE_CONFIG_PATH = WORKSPACE_ROOT / "config.xml"
MOCK_ROOT = WORKSPACE_ROOT / "mock_backtestsys"
MAX_TRIALS = 2
MAX_FAILURES = 2

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
    BackTestSearchSpaceAdapter,
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


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required path not found: {path}")


def _build_objective_definition() -> ObjectiveDefinition:
    return ObjectiveDefinition(
        search_space=BackTestSearchSpaceAdapter(),
        dataset_enumerator=BackTestDatasetEnumeratorAdapter(),
        run_spec_builder=BackTestRunSpecBuilderAdapter(),
        run_key_builder=BackTestRunKeyBuilderAdapter(),
        objective_key_builder=BackTestObjectiveKeyBuilderAdapter(),
        trial_result_aggregator=BackTestTrialResultAggregatorAdapter(),
        progress_scorer=None,
        objective_evaluator=BackTestObjectiveEvaluatorAdapter(),
    )


def _build_settings(runtime_root: Path) -> dict[str, object]:
    doneinfo_gt = MOCK_ROOT / "groundtruth" / "PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv"
    executiondetail_gt = MOCK_ROOT / "groundtruth" / "PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv"
    dataset_path = MOCK_ROOT / "market_data.csv"
    return {
        "spec_id": "iter_backtestsys_demo",
        "meta": {
            "dataset_version": "mock_v1",
            "engine_version": "backtestsys_main_py",
            "dataset_ids": ["ds_01"],
        },
        "objective_config": {
            "name": "backtest_loss",
            "version": "v1",
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
                "time_scale_lambda": {"low": 0.0, "high": 0.0},
                "cancel_bias_k": {"low": 0.0, "high": 0.0},
                "delay_in": {"low": 0, "high": 0},
                "delay_out": {"low": 0, "high": 0},
            },
        },
        "execution_config": {
            "executor_kind": "backtest",
            "default_resources": {"cpu": 1, "max_runtime_seconds": 60},
            "backtest_run_spec": {
                "backtestsys_root": str(BACKTESTSYS_ROOT),
                "base_config_path": str(BASE_CONFIG_PATH),
                "output_root_dir": str(runtime_root / "artifacts"),
                "dataset_paths": {"ds_01": str(dataset_path)},
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
    _must_exist(MOCK_ROOT / "market_data.csv")
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
