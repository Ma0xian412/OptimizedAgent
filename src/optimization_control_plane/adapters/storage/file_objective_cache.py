from __future__ import annotations

from pathlib import Path

from optimization_control_plane.adapters.storage._file_helpers import (
    _atomic_write_json,
    _read_json,
    _safe_filename,
)
from optimization_control_plane.domain.models import ObjectiveResult

_SUBDIR = "objective_cache"


class FileObjectiveCache:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self._dir = Path(base_dir) / _SUBDIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, objective_key: str) -> ObjectiveResult | None:
        data = _read_json(self._path(objective_key))
        if data is None:
            return None
        return ObjectiveResult(
            value=data["value"],
            attrs=data["attrs"],
            artifact_refs=data.get("artifact_refs", []),
        )

    def put(self, objective_key: str, objective_result: ObjectiveResult) -> None:
        _atomic_write_json(self._path(objective_key), {
            "objective_key": objective_key,
            "value": objective_result.value,
            "attrs": objective_result.attrs,
            "artifact_refs": objective_result.artifact_refs,
        })

    def _path(self, objective_key: str) -> Path:
        return self._dir / _safe_filename(objective_key)
