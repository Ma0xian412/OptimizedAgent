from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_DONEINFO_FILE = "doneinfo.csv"
_EXECUTION_DETAIL_FILE = "excutiondetail.csv"


@dataclass(frozen=True)
class BackTestSysGroundTruth:
    doneinfo_count: int
    executiondetail_count: int


class BackTestSysGroundTruthAdapter:
    """Read BackTestSys ground-truth csv files and expose row counts."""

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
            doneinfo_count=self._count_rows(done_path),
            executiondetail_count=self._count_rows(execution_path),
        )
        self._cache[groundtruth_dir] = result
        return result

    @staticmethod
    def _require_file(path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"groundtruth file not found: {path}")
        return path

    @staticmethod
    def _count_rows(path: Path) -> int:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for _ in reader)
