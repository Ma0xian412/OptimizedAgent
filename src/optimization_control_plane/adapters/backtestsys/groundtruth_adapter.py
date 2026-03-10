from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DONEINFO_FILE = "doneinfo.csv"
_EXECUTION_DETAIL_FILE = "excutiondetail.csv"


@dataclass(frozen=True)
class BackTestSysGroundTruthTable:
    rows: tuple[dict[str, Any], ...]
    by_order_id: dict[int, tuple[dict[str, Any], ...]]
    count: int


@dataclass(frozen=True)
class BackTestSysGroundTruth:
    doneinfo: BackTestSysGroundTruthTable
    executiondetail: BackTestSysGroundTruthTable

    @property
    def doneinfo_count(self) -> int:
        return self.doneinfo.count

    @property
    def executiondetail_count(self) -> int:
        return self.executiondetail.count


class BackTestSysGroundTruthAdapter:
    """Read BackTestSys ground-truth csv files and expose indexed rows."""

    def __init__(self) -> None:
        self._cache: dict[str, BackTestSysGroundTruth] = {}

    def load(self, groundtruth_dir: str) -> BackTestSysGroundTruth:
        cached = self._cache.get(groundtruth_dir)
        if cached is not None:
            return cached
        base = Path(groundtruth_dir)
        done_path = self._require_file(base / _DONEINFO_FILE)
        execution_path = self._require_file(base / _EXECUTION_DETAIL_FILE)
        result = BackTestSysGroundTruth(
            doneinfo=self._load_table(done_path),
            executiondetail=self._load_table(execution_path),
        )
        self._cache[groundtruth_dir] = result
        return result

    @staticmethod
    def _require_file(path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"groundtruth file not found: {path}")
        return path

    @staticmethod
    def _load_table(path: Path) -> BackTestSysGroundTruthTable:
        rows = BackTestSysGroundTruthAdapter._read_rows(path)
        index = BackTestSysGroundTruthAdapter._build_order_index(rows, path)
        return BackTestSysGroundTruthTable(
            rows=rows,
            by_order_id=index,
            count=len(rows),
        )

    @staticmethod
    def _read_rows(path: Path) -> tuple[dict[str, Any], ...]:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                return tuple()
            return tuple(dict(row) for row in reader)

    @staticmethod
    def _build_order_index(
        rows: tuple[dict[str, Any], ...],
        path: Path,
    ) -> dict[int, tuple[dict[str, Any], ...]]:
        order_rows: dict[int, list[dict[str, Any]]] = {}
        for idx, row in enumerate(rows):
            order_id_raw = row.get("OrderId")
            if order_id_raw is None:
                raise ValueError(f"missing OrderId in {path} row {idx}")
            try:
                order_id = int(order_id_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid OrderId in {path} row {idx}: {order_id_raw!r}") from exc
            order_rows.setdefault(order_id, []).append(row)
        return {key: tuple(value) for key, value in order_rows.items()}
