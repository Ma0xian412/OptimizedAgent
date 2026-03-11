from __future__ import annotations

from dataclasses import dataclass, field

from optimization_control_plane.domain.enums import TrialState
from optimization_control_plane.domain.models import ObjectiveResult, RunHandle, RunSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext


@dataclass(frozen=True)
class RunBinding:
    trial_id: str
    trial_number: int
    dataset_id: str
    run_key: str
    per_run_objective_key: str
    trial_objective_key: str
    run_spec: RunSpec
    trial_ctx: TrialContext


@dataclass
class InflightEntry:
    run_key: str
    handle: RunHandle
    leader: RunBinding
    followers: list[RunBinding] = field(default_factory=list)


@dataclass(frozen=True)
class TrialRunFailure:
    dataset_id: str
    state: TrialState
    error: str


@dataclass
class TrialCohort:
    trial_id: str
    trial_number: int
    trial_ctx: TrialContext
    trial_objective_key: str
    run_bindings: tuple[RunBinding, ...]
    successful_results: list[tuple[str, ObjectiveResult]] = field(default_factory=list)
    failures: list[TrialRunFailure] = field(default_factory=list)
    shared_run_leader_trial_ids: set[str] = field(default_factory=set)
    _pending_run_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _dataset_by_run_key: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.run_bindings:
            raise ValueError("trial cohort must contain at least one run binding")
        dataset_by_run_key = {binding.run_key: binding.dataset_id for binding in self.run_bindings}
        if len(dataset_by_run_key) != len(self.run_bindings):
            raise ValueError("run bindings in a trial cohort must have unique run_key values")
        self._dataset_by_run_key = dataset_by_run_key
        self._pending_run_keys = set(dataset_by_run_key.keys())

    @property
    def is_complete(self) -> bool:
        return len(self._pending_run_keys) == 0

    def mark_success(
        self,
        run_key: str,
        dataset_id: str,
        objective_result: ObjectiveResult,
        leader_trial_id: str | None,
    ) -> None:
        self._assert_dataset(run_key, dataset_id)
        if run_key not in self._pending_run_keys:
            return
        self._pending_run_keys.remove(run_key)
        self.successful_results.append((dataset_id, objective_result))
        self._mark_shared_leader(leader_trial_id)

    def mark_failure(
        self,
        run_key: str,
        dataset_id: str,
        state: TrialState,
        error: str,
        leader_trial_id: str | None,
    ) -> None:
        self._assert_dataset(run_key, dataset_id)
        if run_key not in self._pending_run_keys:
            return
        self._pending_run_keys.remove(run_key)
        self.failures.append(TrialRunFailure(dataset_id=dataset_id, state=state, error=error))
        self._mark_shared_leader(leader_trial_id)

    def _assert_dataset(self, run_key: str, dataset_id: str) -> None:
        expected = self._dataset_by_run_key.get(run_key)
        if expected is None:
            raise KeyError(f"run_key not found in trial cohort: {run_key}")
        if expected != dataset_id:
            raise ValueError(
                f"dataset_id mismatch for run_key {run_key}: expected={expected}, got={dataset_id}"
            )

    def _mark_shared_leader(self, leader_trial_id: str | None) -> None:
        if leader_trial_id is None or leader_trial_id == self.trial_id:
            return
        self.shared_run_leader_trial_ids.add(leader_trial_id)


class InflightRegistry:
    def __init__(self) -> None:
        self._by_run_key: dict[str, InflightEntry] = {}
        self._by_handle_id: dict[str, InflightEntry] = {}
        self._by_trial_id: dict[str, TrialCohort] = {}

    def has(self, run_key: str) -> bool:
        return run_key in self._by_run_key

    def register_trial_cohort(self, cohort: TrialCohort) -> None:
        if cohort.trial_id in self._by_trial_id:
            raise ValueError(f"trial cohort already registered: {cohort.trial_id}")
        self._by_trial_id[cohort.trial_id] = cohort

    def register_leader(
        self,
        run_key: str,
        handle: RunHandle,
        leader_binding: RunBinding,
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
        follower_binding: RunBinding,
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

    def pop_all_trials_for_run_key(self, run_key: str) -> list[RunBinding]:
        entry = self._by_run_key.pop(run_key, None)
        if entry is None:
            raise KeyError(f"no inflight entry for run_key: {run_key}")
        self._by_handle_id.pop(entry.handle.handle_id, None)
        return [entry.leader, *entry.followers]

    def record_run_complete(
        self,
        trial_id: str,
        run_key: str,
        dataset_id: str,
        objective_result: ObjectiveResult,
        leader_trial_id: str | None = None,
    ) -> tuple[bool, list[tuple[str, ObjectiveResult]] | None]:
        cohort = self._get_trial_cohort(trial_id)
        cohort.mark_success(run_key, dataset_id, objective_result, leader_trial_id)
        if not cohort.is_complete:
            return False, None
        return True, list(cohort.successful_results)

    def record_run_failure(
        self,
        trial_id: str,
        run_key: str,
        dataset_id: str,
        state: TrialState,
        error: str,
        leader_trial_id: str | None = None,
    ) -> tuple[bool, list[tuple[str, ObjectiveResult]] | None]:
        cohort = self._get_trial_cohort(trial_id)
        cohort.mark_failure(run_key, dataset_id, state, error, leader_trial_id)
        if not cohort.is_complete:
            return False, None
        return True, list(cohort.successful_results)

    def pop_trial_cohort(self, trial_id: str) -> TrialCohort | None:
        return self._by_trial_id.pop(trial_id, None)

    def get_trial_cohort(self, trial_id: str) -> TrialCohort | None:
        return self._by_trial_id.get(trial_id)

    def _get_trial_cohort(self, trial_id: str) -> TrialCohort:
        cohort = self._by_trial_id.get(trial_id)
        if cohort is None:
            raise KeyError(f"no trial cohort for trial_id: {trial_id}")
        return cohort

    def handles(self) -> list[RunHandle]:
        return [entry.handle for entry in self._by_run_key.values()]

    @property
    def active_leader_count(self) -> int:
        return len(self._by_run_key)

    @property
    def total_follower_count(self) -> int:
        return sum(len(e.followers) for e in self._by_run_key.values())
