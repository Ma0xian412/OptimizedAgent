from __future__ import annotations

from pathlib import Path

from optimization_control_plane.adapters.storage._file_helpers import (
    _atomic_write_json,
    _read_json,
    _safe_filename,
)
from optimization_control_plane.domain.models import RunResult

_SUBDIR = "run_cache"


class FileRunCache:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self._dir = Path(base_dir) / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, run_key: str) -> RunResult | None:
        data = _read_json(self._path(run_key))
        if data is None:
            return None
        return RunResult(
            metrics=data["metrics"],
            diagnostics=data["diagnostics"],
            artifact_refs=data.get("artifact_refs", []),
        )

    def put(self, run_key: str, run_result: RunResult) -> None:
        _atomic_write_json(self._path(run_key), {
            "run_key": run_key,
            "metrics": run_result.metrics,
            "diagnostics": run_result.diagnostics,
            "artifact_refs": run_result.artifact_refs,
        })

    def _path(self, run_key: str) -> Path:
        return self._dir / _safe_filename(run_key)
