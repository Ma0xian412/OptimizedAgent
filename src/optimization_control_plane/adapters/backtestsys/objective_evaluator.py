from __future__ import annotations

from typing import Any

from optimization_control_plane.adapters.backtestsys.groundtruth_adapter import (
    BackTestSysGroundTruthAdapter,
)
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ObjectiveResult,
    RunResult,
)

_DONEINFO_KEY = "doneinfo_count"
_EXECUTION_DETAIL_KEY = "executiondetail_count"


class BackTestSysCountDiffEvaluator:
    """Minimize absolute count gaps between BackTestSys result and GT."""

    def __init__(
        self,
        groundtruth_adapter: BackTestSysGroundTruthAdapter,
        groundtruth_dir: str | None = None,
    ) -> None:
        self._groundtruth_adapter = groundtruth_adapter
        self._groundtruth_dir = groundtruth_dir

    def evaluate(self, run_result: RunResult, spec: ExperimentSpec) -> ObjectiveResult:
        gt_dir = self._groundtruth_dir or self._resolve_groundtruth_dir(spec)
        gt = self._groundtruth_adapter.load(gt_dir)
        done_count = self._read_metric_count(run_result.metrics, _DONEINFO_KEY)
        execution_count = self._read_metric_count(run_result.metrics, _EXECUTION_DETAIL_KEY)
        done_gap = abs(done_count - gt.doneinfo_count)
        execution_gap = abs(execution_count - gt.executiondetail_count)
        loss = float(done_gap + execution_gap)
        attrs = {
            "objective_name": "backtestsys_count_diff",
            "groundtruth_dir": gt_dir,
            "doneinfo_count": done_count,
            "executiondetail_count": execution_count,
            "gt_doneinfo_count": gt.doneinfo_count,
            "gt_executiondetail_count": gt.executiondetail_count,
            "doneinfo_gap": done_gap,
            "executiondetail_gap": execution_gap,
        }
        return ObjectiveResult(
            value=loss,
            attrs=attrs,
            artifact_refs=list(run_result.artifact_refs),
        )

    @staticmethod
    def _read_metric_count(metrics: dict[str, Any], key: str) -> int:
        if key not in metrics:
            raise KeyError(f"run_result.metrics missing required key: {key}")
        value = metrics[key]
        if not isinstance(value, int):
            raise TypeError(f"run_result.metrics[{key}] must be int, got {type(value).__name__}")
        return value

    @staticmethod
    def _resolve_groundtruth_dir(spec: ExperimentSpec) -> str:
        groundtruth_cfg = spec.objective_config.get("groundtruth")
        if not isinstance(groundtruth_cfg, dict):
            raise ValueError("spec.objective_config.groundtruth must be a dict")
        gt_dir = groundtruth_cfg.get("dir")
        if not isinstance(gt_dir, str) or not gt_dir:
            raise ValueError("spec.objective_config.groundtruth.dir must be a non-empty string")
        return gt_dir
