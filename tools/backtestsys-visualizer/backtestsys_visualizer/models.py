from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COMPONENT_NAMES = ("curve", "terminal", "cancel", "post")

_STAGE_PREFIX_RANK = {
    "baseline": 0,
    "contract_": 1,
    "verify": 2,
    "machine_delay_": 1,
    "contract_core_": 2,
    "final_verify": 3,
}


@dataclass(frozen=True)
class TrialPoint:
    run_tag: str
    stage_name: str
    trial_id: str
    stage_iteration: int | None
    global_iteration: int
    total_loss: float | None
    best_so_far: float | None
    state: str | None
    raw: dict[str, float | None]
    normalized: dict[str, float | None]
    params: dict[str, Any]
    from_trial_results: bool
    from_study_db: bool


def stage_sort_key(stage_name: str) -> tuple[int, str]:
    for prefix, rank in _STAGE_PREFIX_RANK.items():
        if stage_name.startswith(prefix):
            return (rank, stage_name)
    return (99, stage_name)

