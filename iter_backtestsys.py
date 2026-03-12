from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optimization_control_plane.adapters.backtestsys.staged_calibration import run_staged_calibration  # noqa: E402
from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (  # noqa: E402
    CalibrationConfig,
    DatasetDefinition,
)

BACKTESTSYS_ROOT = WORKSPACE_ROOT / "BackTestSys"
BASE_CONFIG_PATH = WORKSPACE_ROOT / "config.xml"
MOCK_ROOT = WORKSPACE_ROOT / "mock_backtestsys"
DATASETS: tuple[DatasetDefinition, ...] = (
    DatasetDefinition("ds_01", "market_data_ds_01.csv", "m1", "c1"),
    DatasetDefinition("ds_02", "market_data_ds_02.csv", "m2", "c2"),
    DatasetDefinition("ds_03", "market_data_ds_03.csv", "m3", "c3"),
)


def main() -> None:
    config = CalibrationConfig(
        workspace_root=WORKSPACE_ROOT,
        backtestsys_root=BACKTESTSYS_ROOT,
        base_config_path=BASE_CONFIG_PATH,
        mock_root=MOCK_ROOT,
        datasets=DATASETS,
        max_failures=2,
        baseline_trials=1,
        machine_delay_trials=12,
        contract_core_trials=12,
        verify_trials=1,
        default_resources={"cpu": 1, "max_runtime_seconds": 60},
    )
    print(json.dumps(run_staged_calibration(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
