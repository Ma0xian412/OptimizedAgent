from __future__ import annotations

from pathlib import Path
from typing import Any

from optimization_control_plane.adapters.storage._file_helpers import (
    _atomic_write_json,
    _safe_filename,
)
from optimization_control_plane.domain.models import ObjectiveResult, RunResult

_RUN_RECORDS_DIR = "run_records"
_TRIAL_RESULTS_DIR = "trial_results"
_TRIAL_FAILURES_DIR = "trial_failures"


class FileResultStore:
    def __init__(self, base_dir: str | Path = "data") -> None:
        self._base = Path(base_dir)
        (self._base / _RUN_RECORDS_DIR).mkdir(parents=True, exist_ok=True)
        (self._base / _TRIAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        (self._base / _TRIAL_FAILURES_DIR).mkdir(parents=True, exist_ok=True)

    def write_run_record(self, run_key: str, run_result: RunResult) -> None:
        path = self._base / _RUN_RECORDS_DIR / _safe_filename(run_key)
        _atomic_write_json(path, {
            "run_key": run_key,
            "metrics": run_result.metrics,
            "diagnostics": run_result.diagnostics,
            "artifact_refs": run_result.artifact_refs,
        })

    def write_trial_result(
        self, trial_id: str, objective_result: ObjectiveResult
    ) -> None:
        path = self._base / _TRIAL_RESULTS_DIR / _safe_filename(trial_id)
        _atomic_write_json(path, {
            "trial_id": trial_id,
            "value": objective_result.value,
            "attrs": objective_result.attrs,
            "artifact_refs": objective_result.artifact_refs,
        })

    def write_trial_failure(self, trial_id: str, error: Any) -> None:
        path = self._base / _TRIAL_FAILURES_DIR / _safe_filename(trial_id)
        _atomic_write_json(path, _failure_payload(trial_id, error))


def _failure_payload(trial_id: str, error: Any) -> dict[str, Any]:
    if isinstance(error, dict):
        return {
            "trial_id": trial_id,
            **error,
        }
    return {
        "trial_id": trial_id,
        "error": str(error),
    }
