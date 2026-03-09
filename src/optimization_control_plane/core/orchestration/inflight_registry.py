from __future__ import annotations

from dataclasses import dataclass, field

from optimization_control_plane.domain.models import RunHandle
from optimization_control_plane.ports.optimizer_backend import TrialContext


@dataclass
class TrialBinding:
    trial_id: str
    trial_number: int
    objective_key: str
    trial_ctx: TrialContext


@dataclass
class InflightEntry:
    run_key: str
    handle: RunHandle
    leader: TrialBinding
    followers: list[TrialBinding] = field(default_factory=list)


class InflightRegistry:
    def __init__(self) -> None:
        self._by_run_key: dict[str, InflightEntry] = {}
        self._by_handle_id: dict[str, InflightEntry] = {}

    def has(self, run_key: str) -> bool:
        return run_key in self._by_run_key

    def register_leader(
        self,
        run_key: str,
        handle: RunHandle,
        leader_binding: TrialBinding,
    ) -> None:
        if run_key in self._by_run_key:
            raise ValueError(f"run_key already registered as leader: {run_key}")
        entry = InflightEntry(
            run_key=run_key,
            handle=handle,
            leader=leader_binding,
        )
        self._by_run_key[run_key] = entry
        self._by_handle_id[handle.handle_id] = entry

    def attach_follower(
        self,
        run_key: str,
        follower_binding: TrialBinding,
    ) -> None:
        entry = self._by_run_key.get(run_key)
        if entry is None:
            raise ValueError(f"no leader for run_key: {run_key}")
        entry.followers.append(follower_binding)

    def get_by_handle(self, handle_id: str) -> InflightEntry:
        entry = self._by_handle_id.get(handle_id)
        if entry is None:
            raise KeyError(f"no inflight entry for handle_id: {handle_id}")
        return entry

    def pop_all_trials_for_run_key(self, run_key: str) -> list[TrialBinding]:
        entry = self._by_run_key.pop(run_key, None)
        if entry is None:
            raise KeyError(f"no inflight entry for run_key: {run_key}")
        self._by_handle_id.pop(entry.handle.handle_id, None)
        return [entry.leader, *entry.followers]

    def handles(self) -> list[RunHandle]:
        return [entry.handle for entry in self._by_run_key.values()]

    @property
    def active_leader_count(self) -> int:
        return len(self._by_run_key)

    @property
    def total_follower_count(self) -> int:
        return sum(len(e.followers) for e in self._by_run_key.values())
