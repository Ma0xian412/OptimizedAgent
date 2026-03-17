from __future__ import annotations

from pathlib import Path
from typing import Any


def build_groundtruth_config(datasets: tuple[Any, ...]) -> dict[str, object]:
    first = datasets[0]
    datasets_cfg = {
        item.dataset_id: {
            "doneinfo_path": str(item.groundtruth_doneinfo_path),
            "executiondetail_path": str(item.groundtruth_executiondetail_path),
        }
        for item in datasets
    }
    return {
        "doneinfo_path": str(first.groundtruth_doneinfo_path),
        "executiondetail_path": str(first.groundtruth_executiondetail_path),
        "datasets": datasets_cfg,
    }


def collect_groundtruth_paths(groundtruth: dict[str, object]) -> tuple[Path, ...]:
    paths = [
        Path(str(groundtruth["doneinfo_path"])),
        Path(str(groundtruth["executiondetail_path"])),
    ]
    datasets_cfg = groundtruth.get("datasets")
    if not isinstance(datasets_cfg, dict):
        return tuple(paths)
    for dataset_cfg in datasets_cfg.values():
        if not isinstance(dataset_cfg, dict):
            continue
        paths.append(Path(str(dataset_cfg["doneinfo_path"])))
        paths.append(Path(str(dataset_cfg["executiondetail_path"])))
    return tuple(paths)
