from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestObjectiveEvaluatorAdapter
from optimization_control_plane.domain.models import GroundTruthData, RunResult
from tests.conftest import make_spec


def _key(
    partition_day: int,
    contract_id: int,
    order_id: int,
    machine_name: str,
) -> dict[str, str]:
    return {
        "PartitionDay": str(partition_day),
        "ContractId": str(contract_id),
        "OrderId": str(order_id),
        "MachineName": machine_name,
    }


class TestBackTestObjectiveEvaluatorAdapter:
    def test_evaluate_returns_expected_raw_losses(self, tmp_path: Path) -> None:
        del tmp_path
        run_result = RunResult(payload={
            "OrderInfo": [
                {
                    **_key(20260312, 1, 1, "m1"),
                    "LimitPrice": "100.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                    "SentTime": "0",
                },
                {
                    **_key(20260312, 1, 2, "m1"),
                    "LimitPrice": "101.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                    "SentTime": "0",
                },
            ],
            "DoneInfo": [
                {**_key(20260312, 1, 1, "m1"), "DoneTime": "10", "OrderTradeState": "A"},
                {**_key(20260312, 1, 2, "m1"), "DoneTime": "10", "OrderTradeState": "A"},
            ],
            "ExecutionDetail": [
                {
                    **_key(20260312, 1, 1, "m1"),
                    "RecvTick": "7",
                    "ExchTick": "7",
                    "Price": "100.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                },
                {
                    **_key(20260312, 1, 2, "m1"),
                    "RecvTick": "4",
                    "ExchTick": "4",
                    "Price": "100.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                },
            ],
            "CancelRequest": [
                {**_key(20260312, 1, 2, "m1"), "CancelSentTime": "2"},
            ],
        })
        groundtruth = GroundTruthData(
            payload={
                "DoneInfo": [
                    {**_key(20260312, 1, 1, "m1"), "DoneTime": "10", "OrderTradeState": "A"},
                    {**_key(20260312, 1, 2, "m1"), "DoneTime": "10", "OrderTradeState": "P"},
                    # extra row to ensure evaluator uses key intersection
                    {**_key(20260312, 1, 9, "m1"), "DoneTime": "10", "OrderTradeState": "A"},
                ],
                "ExecutionDetail": [
                    {
                        **_key(20260312, 1, 1, "m1"),
                        "RecvTick": "5",
                        "ExchTick": "5",
                        "Price": "100.0",
                        "Volume": "10",
                        "OrderDirection": "Buy",
                    },
                    {
                        **_key(20260312, 1, 2, "m1"),
                        "RecvTick": "3",
                        "ExchTick": "3",
                        "Price": "100.0",
                        "Volume": "4",
                        "OrderDirection": "Buy",
                    },
                    {
                        **_key(20260312, 1, 9, "m1"),
                        "RecvTick": "5",
                        "ExchTick": "5",
                        "Price": "100.0",
                        "Volume": "1",
                        "OrderDirection": "Buy",
                    },
                ],
            },
            fingerprint="sha256:gt",
        )

        result = BackTestObjectiveEvaluatorAdapter().evaluate(run_result, make_spec(), groundtruth)

        raw = result.attrs["raw"]
        counts = result.attrs["counts"]
        availability = result.attrs["availability"]
        assert counts["order_count"] == 2
        assert counts["cancel_order_count"] == 1
        assert raw["curve"] == pytest.approx(3.0)
        assert raw["terminal"] == pytest.approx(0.3)
        assert raw["cancel"] == pytest.approx(1.0)
        assert raw["post"] == pytest.approx(0.6)
        assert availability == {
            "curve": True,
            "terminal": True,
            "cancel": True,
            "post": True,
        }
        assert result.value == pytest.approx(1.225)

    def test_evaluate_without_cancel_orders_marks_cancel_components_unavailable(self) -> None:
        run_result = RunResult(payload={
            "OrderInfo": [
                {
                    **_key(20260312, 1, 1, "m1"),
                    "LimitPrice": "100.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                    "SentTime": "0",
                },
            ],
            "DoneInfo": [
                {**_key(20260312, 1, 1, "m1"), "DoneTime": "10", "OrderTradeState": "A"},
            ],
            "ExecutionDetail": [
                {
                    **_key(20260312, 1, 1, "m1"),
                    "RecvTick": "5",
                    "ExchTick": "5",
                    "Price": "100.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                },
            ],
            "CancelRequest": [],
        })
        groundtruth = GroundTruthData(
            payload={
                "DoneInfo": [
                    {**_key(20260312, 1, 1, "m1"), "DoneTime": "10", "OrderTradeState": "A"},
                ],
                "ExecutionDetail": [
                    {
                        **_key(20260312, 1, 1, "m1"),
                        "RecvTick": "5",
                        "ExchTick": "5",
                        "Price": "100.0",
                        "Volume": "10",
                        "OrderDirection": "Buy",
                    },
                ],
            },
            fingerprint="sha256:gt",
        )

        result = BackTestObjectiveEvaluatorAdapter().evaluate(run_result, make_spec(), groundtruth)

        assert result.attrs["raw"]["curve"] == 0.0
        assert result.attrs["raw"]["terminal"] == 0.0
        assert result.attrs["raw"]["cancel"] is None
        assert result.attrs["raw"]["post"] is None
        assert result.attrs["counts"]["cancel_order_count"] == 0
        assert result.attrs["availability"]["cancel"] is False
        assert result.attrs["availability"]["post"] is False
        assert result.value == 0.0

    def test_evaluate_raises_on_invalid_done_state(self) -> None:
        run_result = RunResult(payload={
            "OrderInfo": [
                {
                    **_key(20260312, 1, 1, "m1"),
                    "LimitPrice": "100.0",
                    "Volume": "10",
                    "OrderDirection": "Buy",
                    "SentTime": "0",
                },
            ],
            "DoneInfo": [
                {**_key(20260312, 1, 1, "m1"), "DoneTime": "10", "OrderTradeState": "X"},
            ],
            "ExecutionDetail": [],
            "CancelRequest": [],
        })
        groundtruth = GroundTruthData(
            payload={
                "DoneInfo": [{**_key(20260312, 1, 1, "m1"), "DoneTime": "10", "OrderTradeState": "A"}],
                "ExecutionDetail": [],
            },
            fingerprint="sha256:gt",
        )

        with pytest.raises(ValueError, match="must be one of A/P/N"):
            BackTestObjectiveEvaluatorAdapter().evaluate(run_result, make_spec(), groundtruth)
