from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import (
    BackTestSysCountDiffEvaluator,
    BackTestSysGroundTruthAdapter,
)
from optimization_control_plane.domain.models import RunResult
from tests.conftest import make_spec


def test_groundtruth_adapter_counts_rows() -> None:
    gt_dir = Path(__file__).resolve().parents[2] / "fixtures" / "backtestsys_gt"
    adapter = BackTestSysGroundTruthAdapter()
    gt = adapter.load(str(gt_dir))
    assert gt.doneinfo_count == 3
    assert gt.executiondetail_count == 2


def test_count_diff_evaluator_uses_gt_from_spec() -> None:
    gt_dir = Path(__file__).resolve().parents[2] / "fixtures" / "backtestsys_gt"
    spec = make_spec(objective_config={
        "name": "count_diff",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": str(gt_dir)},
    })
    run_result = RunResult(
        metrics={
            "doneinfo_count": 1,
            "executiondetail_count": 5,
        },
        diagnostics={},
        artifact_refs=[],
    )
    evaluator = BackTestSysCountDiffEvaluator(groundtruth_adapter=BackTestSysGroundTruthAdapter())
    objective = evaluator.evaluate(run_result, spec)
    assert objective.value == 5.0
    assert objective.attrs["doneinfo_gap"] == 2
    assert objective.attrs["executiondetail_gap"] == 3


def test_count_diff_evaluator_includes_base_loss_when_set() -> None:
    gt_dir = Path(__file__).resolve().parents[2] / "fixtures" / "backtestsys_gt"
    spec = make_spec(objective_config={
        "name": "count_diff",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": str(gt_dir)},
    })
    run_result = RunResult(
        metrics={"doneinfo_count": 3, "executiondetail_count": 2},
        diagnostics={},
        artifact_refs=[],
    )
    evaluator = BackTestSysCountDiffEvaluator(groundtruth_adapter=BackTestSysGroundTruthAdapter())
    evaluator.set_base_loss(7.5, attrs={"base_tag": "boot"})
    objective = evaluator.evaluate(run_result, spec)
    assert objective.value == 0.0
    assert objective.attrs["base_loss"] == 7.5
    assert objective.attrs["base_tag"] == "boot"


def test_count_diff_evaluator_requires_metrics(tmp_path: Path) -> None:
    (tmp_path / "doneinfo.csv").write_text("h1\nv1\n", encoding="utf-8")
    (tmp_path / "excutiondetail.csv").write_text("h1\nv1\n", encoding="utf-8")
    spec = make_spec(objective_config={
        "name": "count_diff",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": str(tmp_path)},
    })
    run_result = RunResult(metrics={"doneinfo_count": 1}, diagnostics={}, artifact_refs=[])
    evaluator = BackTestSysCountDiffEvaluator(groundtruth_adapter=BackTestSysGroundTruthAdapter())
    with pytest.raises(KeyError):
        evaluator.evaluate(run_result, spec)
