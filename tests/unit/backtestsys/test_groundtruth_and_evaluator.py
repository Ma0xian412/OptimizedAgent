from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import (
    BackTestSysCountDiffEvaluator,
    BackTestSysGroundTruthAdapter,
)
from optimization_control_plane.domain.models import RunResult
from tests.conftest import make_spec


def test_groundtruth_adapter_loads_rows_and_indices() -> None:
    gt_dir = Path(__file__).resolve().parents[2] / "fixtures" / "backtestsys_gt"
    adapter = BackTestSysGroundTruthAdapter()
    gt = adapter.load(str(gt_dir))
    assert gt.doneinfo_count == 3
    assert gt.executiondetail_count == 2
    assert len(gt.doneinfo.rows) == 3
    assert len(gt.executiondetail.rows) == 2
    assert len(gt.doneinfo.by_order_id[1001]) == 1
    assert len(gt.executiondetail.by_order_id[1002]) == 1


def test_evaluator_computes_four_components_and_total_loss(tmp_path: Path) -> None:
    gt_dir = _write_gt_files(tmp_path)
    spec = make_spec(objective_config={
        "name": "loss_v2",
        "version": "v2",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": str(gt_dir)},
        "weights": {"curve": 1.0, "terminal": 1.0, "cancel": 1.0, "post": 1.0},
        "eps": {"curve": 0.0, "terminal": 0.0, "cancel": 0.0, "post": 0.0},
    })
    run_result = _make_run_result()
    evaluator = BackTestSysCountDiffEvaluator(groundtruth_adapter=BackTestSysGroundTruthAdapter())
    objective = evaluator.evaluate(run_result, spec)
    raw = objective.attrs["raw_components"]
    assert raw["curve"] == pytest.approx(0.9)
    assert raw["terminal"] == pytest.approx(0.175)
    assert raw["cancel"] == pytest.approx(0.0)
    assert raw["post"] == pytest.approx(0.1)
    assert objective.value == pytest.approx(0.29375)
    assert objective.attrs["weights_used"]["curve"] == pytest.approx(0.25)
    assert objective.attrs["order_count"] == 2
    assert objective.attrs["cancel_order_count"] == 1


def test_evaluator_uses_baseline_components_for_normalization(tmp_path: Path) -> None:
    gt_dir = _write_gt_files(tmp_path)
    spec = make_spec(objective_config={
        "name": "loss_v2",
        "version": "v2",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": str(gt_dir)},
        "weights": {"curve": 1.0, "terminal": 1.0, "cancel": 1.0, "post": 1.0},
        "eps": {"curve": 0.0, "terminal": 0.0, "cancel": 0.0, "post": 0.0},
    })
    evaluator = BackTestSysCountDiffEvaluator(groundtruth_adapter=BackTestSysGroundTruthAdapter())
    evaluator.set_base_loss(
        7.5,
        attrs={
            "baseline_components": {
                "curve": 0.23,
                "terminal": 0.35,
                "cancel": 0.5,
                "post": 0.2,
            }
        },
    )
    objective = evaluator.evaluate(_make_run_result(), spec)
    normalized = objective.attrs["normalized_components"]
    assert normalized["curve"] == pytest.approx(0.9 / 0.23)
    assert normalized["terminal"] == pytest.approx(0.5)
    assert normalized["cancel"] == pytest.approx(0.0)
    assert normalized["post"] == pytest.approx(0.5)
    assert objective.value == pytest.approx((0.9 / 0.23 + 0.5 + 0.0 + 0.5) / 4.0)
    assert objective.attrs["base_loss"] == 7.5
    assert objective.attrs["baseline_components_used"] is True


def test_evaluator_raises_when_no_evaluable_orders(tmp_path: Path) -> None:
    gt_dir = _write_gt_files(tmp_path)
    spec = make_spec(objective_config={
        "name": "loss_v2",
        "version": "v2",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"dir": str(gt_dir)},
    })
    run_result = RunResult(
        metrics={},
        diagnostics={"result": {
            "orderinfo_rows": [],
            "doneinfo_rows": [],
            "executiondetail_rows": [],
            "cancelrequest_rows": [],
        }},
        artifact_refs=[],
    )
    evaluator = BackTestSysCountDiffEvaluator(groundtruth_adapter=BackTestSysGroundTruthAdapter())
    with pytest.raises(ValueError, match="no evaluable orders"):
        evaluator.evaluate(run_result, spec)


def _make_run_result() -> RunResult:
    return RunResult(
        metrics={},
        diagnostics={"result": {
            "orderinfo_rows": [
                {"OrderId": 1, "Volume": 10, "SentTime": 0},
                {"OrderId": 2, "Volume": 8, "SentTime": 0},
            ],
            "doneinfo_rows": [
                {"OrderId": 1, "DoneTime": 10},
                {"OrderId": 2, "DoneTime": 5},
            ],
            "executiondetail_rows": [
                {"OrderId": 1, "RecvTick": 2, "Volume": 3},
                {"OrderId": 1, "RecvTick": 6, "Volume": 2},
                {"OrderId": 2, "RecvTick": 3, "Volume": 8},
            ],
            "cancelrequest_rows": [
                {"OrderId": 1, "CancelSentTime": 5},
            ],
        }},
        artifact_refs=[],
    )


def _write_gt_files(base_dir: Path) -> Path:
    done_path = base_dir / "doneinfo.csv"
    execution_path = base_dir / "excutiondetail.csv"
    done_path.write_text(
        (
            "PartitionDay,ContractId,OrderId,DoneTime,OrderTradeState,MachineName\n"
            "20260310,1,1,10,CANCELED,mock\n"
            "20260310,1,2,5,PARTIAL,mock\n"
        ),
        encoding="utf-8",
    )
    execution_path.write_text(
        (
            "PartitionDay,RecvTick,ExchTick,OrderId,ContractId,Price,Volume,OrderDirection,MachineName\n"
            "20260310,1,1,1,1,1.0,4,BUY,mock\n"
            "20260310,4,4,1,1,1.0,1,BUY,mock\n"
            "20260310,7,7,1,1,1.0,1,BUY,mock\n"
            "20260310,3,3,2,1,1.0,6,BUY,mock\n"
        ),
        encoding="utf-8",
    )
    return base_dir
