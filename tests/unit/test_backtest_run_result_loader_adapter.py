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
    def test_load_directory_layout_reads_only_four_tables(self, tmp_path: Path) -> None:
        result_root = tmp_path / "result_root"
        run_dir = result_root / "run_result_20260311_000000_000001"
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
        _write_csv(run_dir / "contract_info.csv", ["key", "value"], [["contract_id", "IF2401"]])
        _write_csv(
            run_dir / "receipts_20260311.csv",
            ["order_id", "exch_time", "recv_time", "receipt_type", "fill_qty", "fill_price", "remaining_qty"],
            [["1", "100", "105", "FILL", "2", "100.0", "1"], ["2", "200", "220", "PARTIAL", "1", "101.0", "1"]],
        )
        (run_dir / "backtest.log").write_text("line1\nline2\n", encoding="utf-8")

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(result_root))
        result = BackTestRunResultLoaderAdapter().load(run_spec)
        payload = result.payload
        assert isinstance(payload, dict)

        assert payload["layout"] == "directory"
        assert isinstance(payload["base_path"], str)
        assert len(payload["artifact_refs"]) == 4
        assert all(path.endswith(".csv") for path in payload["artifact_refs"])
        assert payload["table_rows"]["DoneInfo"][0]["OrderTradeState"] == "A"
        assert payload["table_rows"]["DoneInfo"][1]["OrderTradeState"] == "P"
        assert len(payload["table_rows"]["ExecutionDetail"]) == 2
        assert len(payload["table_rows"]["OrderInfo"]) == 2
        assert len(payload["table_rows"]["CancelRequest"]) == 1
        assert "metrics" not in payload
        assert "diagnostics" not in payload

    def test_load_prefix_layout(self, tmp_path: Path) -> None:
        prefix = tmp_path / "prefix" / "result"
        _write_csv(
            Path(f"{prefix}_DoneInfo.csv"),
            ["PartitionDay", "ContractId", "OrderId", "DoneTime", "OrderTradeState", "MachineName"],
            [[20260311, 1, 1, 1000, "A", "m1"]],
        )
        _write_csv(
            Path(f"{prefix}_ExecutionDetail.csv"),
            ["PartitionDay", "RecvTick", "ExchTick", "OrderId", "ContractId", "Price", "Volume", "OrderDirection", "MachineName"],
            [[20260311, 105, 100, 1, 1, 100.0, 1, "Buy", "m1"]],
        )
        _write_csv(
            Path(f"{prefix}_OrderInfo.csv"),
            ["PartitionDay", "ContractId", "OrderId", "LimitPrice", "Volume", "OrderDirection", "SentTime", "MachineName"],
            [[20260311, 1, 1, 100.0, 1, "Buy", 10, "m1"]],
        )
        _write_csv(
            Path(f"{prefix}_CancelRequest.csv"),
            ["PartitionDay", "ContractId", "OrderId", "CancelSentTime", "MachineName"],
            [],
        )

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(prefix))
        result = BackTestRunResultLoaderAdapter().load(run_spec)
        payload = result.payload
        assert isinstance(payload, dict)

        assert payload["layout"] == "prefix"
        assert len(payload["table_rows"]["DoneInfo"]) == 1
        assert len(payload["table_rows"]["ExecutionDetail"]) == 1
        assert len(payload["table_rows"]["OrderInfo"]) == 1
        assert len(payload["table_rows"]["CancelRequest"]) == 0

    def test_missing_required_tables_raises(self, tmp_path: Path) -> None:
        result_root = tmp_path / "broken_result"
        run_dir = result_root / "run_result_20260311_000000_000001"
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

        run_spec = RunSpec(job=Job(command=["python3", "main.py"]), result_path=str(result_root))
        with pytest.raises(FileNotFoundError):
            BackTestRunResultLoaderAdapter().load(run_spec)
