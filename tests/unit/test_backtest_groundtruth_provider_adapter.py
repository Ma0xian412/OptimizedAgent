from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import (
    BackTestGroundTruthProviderAdapter,
)
from optimization_control_plane.domain.models import stable_json_serialize
from tests.conftest import make_spec


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(headers)
        writer.writerows(rows)


def _expected_fingerprint(machine_name: str, day: str, contract_id: str) -> str:
    payload = stable_json_serialize(
        {
            "kind": "backtest_groundtruth_v1",
            "machine_name": machine_name,
            "time": day,
            "contract_id": contract_id,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


class TestBackTestGroundTruthProviderAdapter:
    def test_load_reads_two_tables_and_builds_fingerprint(self, tmp_path: Path) -> None:
        machine_name = "m1"
        day = "20260311"
        contract_id = "IF2401"
        doneinfo_path = (
            tmp_path
            / f"PubOrderDoneInfoLog_{machine_name}_{day}_{contract_id}.csv"
        )
        executiondetail_path = (
            tmp_path
            / f"PubExecutionDetailLog_{machine_name}_{day}_{contract_id}.csv"
        )
        _write_csv(
            doneinfo_path,
            ["PartitionDay", "ContractId", "OrderId", "DoneTime", "OrderTradeState", "MachineName"],
            [[20260311, 1, 11, 1000, "A", "m1"]],
        )
        _write_csv(
            executiondetail_path,
            [
                "PartitionDay",
                "RecvTick",
                "ExchTick",
                "OrderId",
                "ContractId",
                "Price",
                "Volume",
                "OrderDirection",
                "MachineName",
            ],
            [[20260311, 110, 100, 11, 1, 100.5, 2, "Buy", "m1"]],
        )
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {},
                "groundtruth": {
                    "doneinfo_path": str(doneinfo_path),
                    "executiondetail_path": str(executiondetail_path),
                },
                "sampler": {"type": "random", "seed": 42},
                "pruner": {"type": "nop"},
            }
        )

        result = BackTestGroundTruthProviderAdapter().load(spec, dataset_id="")

        assert isinstance(result.payload, dict)
        assert list(result.payload.keys()) == ["DoneInfo", "ExecutionDetail"]
        assert len(result.payload["DoneInfo"]) == 1
        assert len(result.payload["ExecutionDetail"]) == 1
        assert result.fingerprint == _expected_fingerprint(machine_name, day, contract_id)

    def test_load_raises_when_two_filenames_have_mismatched_identity(
        self,
        tmp_path: Path,
    ) -> None:
        doneinfo_path = tmp_path / "PubOrderDoneInfoLog_m1_20260311_IF2401.csv"
        executiondetail_path = tmp_path / "PubExecutionDetailLog_m2_20260311_IF2401.csv"
        _write_csv(doneinfo_path, ["PartitionDay"], [[20260311]])
        _write_csv(executiondetail_path, ["PartitionDay"], [[20260311]])
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {},
                "groundtruth": {
                    "doneinfo_path": str(doneinfo_path),
                    "executiondetail_path": str(executiondetail_path),
                },
                "sampler": {"type": "random", "seed": 42},
                "pruner": {"type": "nop"},
            }
        )

        with pytest.raises(ValueError, match="identity mismatch"):
            BackTestGroundTruthProviderAdapter().load(spec, dataset_id="")

    def test_load_uses_dataset_specific_config_when_present(self, tmp_path: Path) -> None:
        shared_doneinfo_path = tmp_path / "PubOrderDoneInfoLog_shared_20260311_IF2401.csv"
        shared_executiondetail_path = (
            tmp_path / "PubExecutionDetailLog_shared_20260311_IF2401.csv"
        )
        ds_doneinfo_path = tmp_path / "PubOrderDoneInfoLog_dsA_20260312_IF2402.csv"
        ds_executiondetail_path = (
            tmp_path / "PubExecutionDetailLog_dsA_20260312_IF2402.csv"
        )
        _write_csv(shared_doneinfo_path, ["PartitionDay"], [[20260311]])
        _write_csv(shared_executiondetail_path, ["PartitionDay"], [[20260311]])
        _write_csv(
            ds_doneinfo_path,
            ["PartitionDay", "MachineName"],
            [[20260312, "dsA"]],
        )
        _write_csv(
            ds_executiondetail_path,
            ["PartitionDay", "MachineName"],
            [[20260312, "dsA"]],
        )
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {},
                "groundtruth": {
                    "doneinfo_path": str(shared_doneinfo_path),
                    "executiondetail_path": str(shared_executiondetail_path),
                    "datasets": {
                        "ds_a": {
                            "doneinfo_path": str(ds_doneinfo_path),
                            "executiondetail_path": str(ds_executiondetail_path),
                        }
                    },
                },
                "sampler": {"type": "random", "seed": 42},
                "pruner": {"type": "nop"},
            }
        )

        result = BackTestGroundTruthProviderAdapter().load(spec, dataset_id="ds_a")

        assert result.payload["DoneInfo"][0]["MachineName"] == "dsA"
        assert result.payload["ExecutionDetail"][0]["MachineName"] == "dsA"
        assert result.fingerprint == _expected_fingerprint("dsA", "20260312", "IF2402")
