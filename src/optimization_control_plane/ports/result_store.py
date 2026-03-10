from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from optimization_control_plane.domain.models import ObjectiveResult, RunResult


@runtime_checkable
class ResultStore(Protocol):
    def write_run_record(
        self,
        run_key: str,
        run_result: RunResult,
        *,
        target_id: str,
    ) -> None: ...
    def write_trial_result(self, trial_id: str, objective_result: ObjectiveResult) -> None: ...
    def write_trial_failure(self, trial_id: str, error: Any) -> None: ...
