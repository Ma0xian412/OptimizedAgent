from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StudyRuntimeState:
    active_executions: int = 0
    buffered_requests: int = 0
    attached_follower_trials: int = 0
    completed_trials: int = 0
    pruned_trials: int = 0
    failed_trials: int = 0
    active_cohort: str | None = None

    @property
    def total_finished(self) -> int:
        return self.completed_trials + self.pruned_trials + self.failed_trials


@dataclass
class ResourceState:
    configured_slots: int
    free_slots: int
    labels: dict[str, Any] = field(default_factory=dict)
