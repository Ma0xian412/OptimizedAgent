from __future__ import annotations

import csv
from pathlib import Path

from optimization_control_plane.domain.models import RunResult, RunSpec

_TABLE_NAMES = ("DoneInfo", "ExecutionDetail", "OrderInfo", "CancelRequest")


class BackTestRunResultLoaderAdapter:
    """Read four BackTestSys CSV tables as raw payload."""

    def load(self, run_spec: RunSpec) -> RunResult:
        result_dir = _resolve_result_dir(result_path=run_spec.result_path)
        return RunResult(payload=_read_table_rows(result_dir=result_dir))


def _resolve_result_dir(*, result_path: str) -> Path:
    if not result_path.strip():
        raise ValueError("run_spec.result_path must be non-empty")
    directory = Path(result_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"run_spec.result_path must be an existing directory: {directory}")
    if _contains_required_tables(directory):
        return directory
    nested = _find_latest_nested_result_dir(directory)
    if nested is not None:
        return nested
    raise FileNotFoundError(
        "missing required BackTestSys tables under run_spec.result_path: "
        f"{directory}"
    )


def _contains_required_tables(directory: Path) -> bool:
    return all((directory / f"{table_name}.csv").is_file() for table_name in _TABLE_NAMES)


def _find_latest_nested_result_dir(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_dir() and _contains_required_tables(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_table_rows(*, result_dir: Path) -> dict[str, list[dict[str, str]]]:
    payload: dict[str, list[dict[str, str]]] = {}
    for table_name in _TABLE_NAMES:
        csv_path = result_dir / f"{table_name}.csv"
        payload[table_name] = _read_csv_rows(csv_path=csv_path, table_name=table_name)
    return payload


def _read_csv_rows(*, csv_path: Path, table_name: str) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing required BackTestSys table: {table_name}.csv in {csv_path.parent}")
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [dict(row) for row in reader]
