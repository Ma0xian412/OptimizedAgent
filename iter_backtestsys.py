from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from optimization_control_plane.adapters.backtestsys.staged_calibration import run_staged_calibration  # noqa: E402
from optimization_control_plane.adapters.backtestsys.staged_calibration_config_loader import (  # noqa: E402
    calibration_config_summary,
    load_calibration_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Absolute path to staged calibration xml config")
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=2.0,
        help="Progress report interval in seconds (must be > 0)",
    )
    parser.add_argument(
        "--progress-format",
        choices=("text", "json"),
        default="text",
        help="Progress output format",
    )
    args = parser.parse_args()
    if args.progress_interval_seconds <= 0:
        raise ValueError("--progress-interval-seconds must be > 0")
    config_path = Path(args.config)
    config = load_calibration_config(config_path)
    summary = calibration_config_summary(config, source=config_path)
    print("[iter_backtestsys] effective config summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[iter_backtestsys] staged calibration output:")
    print(
        json.dumps(
            run_staged_calibration(
                config,
                progress_interval_seconds=args.progress_interval_seconds,
                progress_format=args.progress_format,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
