from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from optimization_control_plane.domain.models import RunResult, RunSpec

_TABLE_NAMES = ("DoneInfo", "ExecutionDetail", "OrderInfo", "CancelRequest")


@dataclass(frozen=True)
class _ResolvedArtifacts:
    layout: str
    base_path: str
    done_path: str
    execution_path: str
    order_path: str
    cancel_path: str

    def table_paths(self) -> tuple[str, str, str, str]:
        return (
            self.done_path,
            self.execution_path,
            self.order_path,
            self.cancel_path,
        )


class BackTestRunResultLoaderAdapter:
    """Load only BackTestSys four result CSV tables."""

    def load(self, run_spec: RunSpec) -> RunResult:
        artifacts = _resolve_artifacts(run_spec.result_path)
        done_rows = _read_csv_rows(artifacts.done_path)
        execution_rows = _read_csv_rows(artifacts.execution_path)
        order_rows = _read_csv_rows(artifacts.order_path)
        cancel_rows = _read_csv_rows(artifacts.cancel_path)
        return RunResult(
            payload=_build_payload(
                artifacts=artifacts,
                done_rows=done_rows,
                execution_rows=execution_rows,
                order_rows=order_rows,
                cancel_rows=cancel_rows,
            )
        )


def _resolve_artifacts(result_path: str) -> _ResolvedArtifacts:
    if not result_path.strip():
        raise ValueError("run_spec.result_path must be non-empty")
    if os.path.isdir(result_path):
        return _resolve_directory_layout(result_path)
    return _resolve_prefix_layout(result_path)


def _resolve_directory_layout(result_dir: str) -> _ResolvedArtifacts:
    direct_paths = _table_paths_for_dir(result_dir)
    if _all_paths_exist(direct_paths):
        base_dir = result_dir
    else:
        base_dir = _find_latest_result_subdir(result_dir)
        if base_dir is None:
            raise FileNotFoundError(f"cannot locate BackTestSys result tables under: {result_dir}")
        direct_paths = _table_paths_for_dir(base_dir)
    return _ResolvedArtifacts(
        layout="directory",
        base_path=os.path.abspath(base_dir),
        done_path=os.path.abspath(direct_paths["DoneInfo"]),
        execution_path=os.path.abspath(direct_paths["ExecutionDetail"]),
        order_path=os.path.abspath(direct_paths["OrderInfo"]),
        cancel_path=os.path.abspath(direct_paths["CancelRequest"]),
    )


def _resolve_prefix_layout(prefix: str) -> _ResolvedArtifacts:
    table_paths = {name: _resolve_prefix_table(prefix, name) for name in _TABLE_NAMES}
    return _ResolvedArtifacts(
        layout="prefix",
        base_path=os.path.abspath(prefix),
        done_path=os.path.abspath(table_paths["DoneInfo"]),
        execution_path=os.path.abspath(table_paths["ExecutionDetail"]),
        order_path=os.path.abspath(table_paths["OrderInfo"]),
        cancel_path=os.path.abspath(table_paths["CancelRequest"]),
    )


def _table_paths_for_dir(base_dir: str) -> dict[str, str]:
    return {name: os.path.join(base_dir, f"{name}.csv") for name in _TABLE_NAMES}


def _all_paths_exist(paths: dict[str, str]) -> bool:
    return all(os.path.exists(path) for path in paths.values())


def _find_latest_result_subdir(parent_dir: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    for name in os.listdir(parent_dir):
        if not name.startswith("run_result"):
            continue
        path = os.path.join(parent_dir, name)
        if not os.path.isdir(path):
            continue
        if _all_paths_exist(_table_paths_for_dir(path)):
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _resolve_prefix_table(prefix: str, table_name: str) -> str:
    direct = f"{prefix}_{table_name}.csv"
    if os.path.exists(direct):
        return direct
    matched = _find_prefixed_match(prefix, f"_{table_name}.csv")
    if matched is None:
        raise FileNotFoundError(f"cannot locate table CSV for {table_name}, prefix={prefix}")
    return matched


def _find_prefixed_match(prefix: str, suffix: str) -> str | None:
    directory = os.path.dirname(prefix) or "."
    basename = os.path.basename(prefix)
    matches: list[tuple[float, str]] = []
    for name in os.listdir(directory):
        if not name.startswith(basename):
            continue
        if not name.endswith(suffix):
            continue
        full_path = os.path.join(directory, name)
        if os.path.isfile(full_path):
            matches.append((os.path.getmtime(full_path), full_path))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [dict(row) for row in reader]


def _build_payload(
    *,
    artifacts: _ResolvedArtifacts,
    done_rows: list[dict[str, str]],
    execution_rows: list[dict[str, str]],
    order_rows: list[dict[str, str]],
    cancel_rows: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "layout": artifacts.layout,
        "base_path": artifacts.base_path,
        "table_paths": {
            "DoneInfo": artifacts.done_path,
            "ExecutionDetail": artifacts.execution_path,
            "OrderInfo": artifacts.order_path,
            "CancelRequest": artifacts.cancel_path,
        },
        "artifact_refs": _collect_artifact_refs(artifacts),
        "table_rows": {
            "DoneInfo": done_rows,
            "ExecutionDetail": execution_rows,
            "OrderInfo": order_rows,
            "CancelRequest": cancel_rows,
        },
    }


def _collect_artifact_refs(artifacts: _ResolvedArtifacts) -> list[str]:
    return sorted(
        [
        artifacts.done_path,
        artifacts.execution_path,
        artifacts.order_path,
        artifacts.cancel_path,
        ]
    )
