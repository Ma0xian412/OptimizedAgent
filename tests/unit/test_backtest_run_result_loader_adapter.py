from __future__ import annotations

import csv
from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestRunResultLoaderAdapter
from optimization_control_plane.domain.models import Job, RunSpec


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(headers)
        writer.writerows(rows)


class TestBackTestRunResultLoaderAdapter:
    def test_load_reads_raw_rows_from_result_path(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "result_dir"
        _write_csv(
            run_dir / "DoneInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "DoneTime", "OrderTradeState", "MachineName"],
            [[20260311, 1, 1, 1000, "A", "m1"], [20260311, 1, 2, 2000, "P", "m1"]],
        )
        _write_csv(
            run_dir / "ExecutionDetail.csv",
            ["PartitionDay", "RecvTick", "ExchTick", "OrderId", "ContractId", "Price", "Volume", "OrderDirection", "MachineName"],
            [[20260311, 105, 100, 1, 1, 100.0, 2, "Buy", "m1"], [20260311, 220, 200, 2, 1, 101.0, 1, "Sell", "m1"]],
        )
        _write_csv(
            run_dir / "OrderInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "LimitPrice", "Volume", "OrderDirection", "SentTime", "MachineName"],
            [[20260311, 1, 1, 100.0, 3, "Buy", 10, "m1"], [20260311, 1, 2, 101.0, 2, "Sell", 20, "m1"]],
        )
        _write_csv(
            run_dir / "CancelRequest.csv",
            ["PartitionDay", "ContractId", "OrderId", "CancelSentTime", "MachineName"],
            [[20260311, 1, 2, 30, "m1"]],
        )

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(run_dir))
        result = BackTestRunResultLoaderAdapter().load(run_spec)
        payload = result.payload
        assert isinstance(payload, dict)

        assert payload["DoneInfo"][0]["OrderTradeState"] == "A"
        assert payload["DoneInfo"][1]["OrderTradeState"] == "P"
        assert len(payload["ExecutionDetail"]) == 2
        assert len(payload["OrderInfo"]) == 2
        assert len(payload["CancelRequest"]) == 1

    def test_load_reads_empty_cancel_request(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "result_dir"
        _write_csv(
            run_dir / "DoneInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "DoneTime", "OrderTradeState", "MachineName"],
            [[20260311, 1, 1, 1000, "A", "m1"]],
        )
        _write_csv(
            run_dir / "ExecutionDetail.csv",
            ["PartitionDay", "RecvTick", "ExchTick", "OrderId", "ContractId", "Price", "Volume", "OrderDirection", "MachineName"],
            [[20260311, 105, 100, 1, 1, 100.0, 1, "Buy", "m1"]],
        )
        _write_csv(
            run_dir / "OrderInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "LimitPrice", "Volume", "OrderDirection", "SentTime", "MachineName"],
            [[20260311, 1, 1, 100.0, 1, "Buy", 10, "m1"]],
        )
        _write_csv(
            run_dir / "CancelRequest.csv",
            ["PartitionDay", "ContractId", "OrderId", "CancelSentTime", "MachineName"],
            [],
        )

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(run_dir))
        result = BackTestRunResultLoaderAdapter().load(run_spec)
        payload = result.payload
        assert isinstance(payload, dict)

        assert len(payload["DoneInfo"]) == 1
        assert len(payload["ExecutionDetail"]) == 1
        assert len(payload["OrderInfo"]) == 1
        assert len(payload["CancelRequest"]) == 0

    def test_missing_required_tables_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "broken_result"
        _write_csv(
            run_dir / "DoneInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "DoneTime", "OrderTradeState", "MachineName"],
            [[20260311, 1, 1, 1000, "A", "m1"]],
        )
        _write_csv(
            run_dir / "ExecutionDetail.csv",
            ["PartitionDay", "RecvTick", "ExchTick", "OrderId", "ContractId", "Price", "Volume", "OrderDirection", "MachineName"],
            [[20260311, 105, 100, 1, 1, 100.0, 1, "Buy", "m1"]],
        )

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(run_dir))
        with pytest.raises(FileNotFoundError):
            BackTestRunResultLoaderAdapter().load(run_spec)

    def test_load_reads_tables_from_nested_result_subdir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "result_dir"
        nested = run_dir / "run_result_TEST_20260312_000001"
        _write_csv(
            nested / "DoneInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "DoneTime", "OrderTradeState", "MachineName"],
            [[20260311, 1, 1, 1000, "N", "m1"]],
        )
        _write_csv(
            nested / "ExecutionDetail.csv",
            ["PartitionDay", "RecvTick", "ExchTick", "OrderId", "ContractId", "Price", "Volume", "OrderDirection", "MachineName"],
            [],
        )
        _write_csv(
            nested / "OrderInfo.csv",
            ["PartitionDay", "ContractId", "OrderId", "LimitPrice", "Volume", "OrderDirection", "SentTime", "MachineName"],
            [[20260311, 1, 1, 100.0, 1, "Buy", 10, "m1"]],
        )
        _write_csv(
            nested / "CancelRequest.csv",
            ["PartitionDay", "ContractId", "OrderId", "CancelSentTime", "MachineName"],
            [],
        )

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(run_dir))
        result = BackTestRunResultLoaderAdapter().load(run_spec)
        payload = result.payload

        assert payload["DoneInfo"][0]["OrderTradeState"] == "N"
        assert len(payload["ExecutionDetail"]) == 0
