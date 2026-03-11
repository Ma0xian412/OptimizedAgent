"""UT-7: InflightRegistry leader/follower behavior."""
from __future__ import annotations

import pytest

from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
    RunBinding,
    TrialCohort,
)
from optimization_control_plane.domain.enums import JobStatus, TrialState
from optimization_control_plane.domain.models import Job, ObjectiveResult, RunHandle, RunSpec
from tests.unit.test_search_space import FakeCtx


def _binding(trial_id: str, run_key: str, dataset_id: str, number: int = 0) -> RunBinding:
    return RunBinding(
        trial_id=trial_id,
        trial_number=number,
        dataset_id=dataset_id,
        run_key=run_key,
        per_run_objective_key=f"obj:{trial_id}:{dataset_id}",
        trial_objective_key=f"trial:{trial_id}",
        run_spec=RunSpec(job=Job(command=["python"])),
        trial_ctx=FakeCtx(),
    )


def _cohort(trial_id: str, number: int = 0) -> TrialCohort:
    return TrialCohort(
        trial_id=trial_id,
        trial_number=number,
        trial_ctx=FakeCtx(),
        trial_objective_key=f"trial:{trial_id}",
        run_bindings=(
            _binding(trial_id, f"rk:{trial_id}:d1", "d1", number),
            _binding(trial_id, f"rk:{trial_id}:d2", "d2", number),
        ),
    )


def _handle(hid: str) -> RunHandle:
    return RunHandle(handle_id=hid, request_id="r1", state=JobStatus.RUNNING)


class TestInflightRegistry:
    def test_register_and_has(self) -> None:
        reg = InflightRegistry()
        assert not reg.has("rk1")
        reg.register_leader("rk1", _handle("h1"), _binding("t1", "rk1", "d1"))
        assert reg.has("rk1")

    def test_duplicate_leader_raises(self) -> None:
        reg = InflightRegistry()
        reg.register_leader("rk1", _handle("h1"), _binding("t1", "rk1", "d1"))
        with pytest.raises(ValueError):
            reg.register_leader("rk1", _handle("h2"), _binding("t2", "rk1", "d1"))

    def test_attach_follower(self) -> None:
        reg = InflightRegistry()
        reg.register_leader("rk1", _handle("h1"), _binding("t1", "rk1", "d1"))
        reg.attach_follower("rk1", _binding("t2", "rk1", "d1"))
        reg.attach_follower("rk1", _binding("t3", "rk1", "d1"))
        assert reg.total_follower_count == 2

    def test_attach_follower_no_leader_raises(self) -> None:
        reg = InflightRegistry()
        with pytest.raises(ValueError):
            reg.attach_follower("rk_missing", _binding("t1", "rk_missing", "d1"))

    def test_get_by_handle(self) -> None:
        reg = InflightRegistry()
        h = _handle("h1")
        reg.register_leader("rk1", h, _binding("t1", "rk1", "d1"))
        entry = reg.get_by_handle("h1")
        assert entry.run_key == "rk1"
        assert entry.leader.trial_id == "t1"

    def test_get_by_handle_missing_raises(self) -> None:
        reg = InflightRegistry()
        with pytest.raises(KeyError):
            reg.get_by_handle("missing")

    def test_pop_all_trials(self) -> None:
        reg = InflightRegistry()
        reg.register_leader("rk1", _handle("h1"), _binding("t1", "rk1", "d1"))
        reg.attach_follower("rk1", _binding("t2", "rk1", "d1"))
        reg.attach_follower("rk1", _binding("t3", "rk1", "d1"))

        bindings = reg.pop_all_trials_for_run_key("rk1")
        assert len(bindings) == 3
        assert bindings[0].trial_id == "t1"
        assert not reg.has("rk1")

    def test_pop_missing_raises(self) -> None:
        reg = InflightRegistry()
        with pytest.raises(KeyError):
            reg.pop_all_trials_for_run_key("missing")

    def test_handles(self) -> None:
        reg = InflightRegistry()
        h1 = _handle("h1")
        h2 = _handle("h2")
        reg.register_leader("rk1", h1, _binding("t1", "rk1", "d1"))
        reg.register_leader("rk2", h2, _binding("t2", "rk2", "d1"))
        handles = reg.handles()
        assert len(handles) == 2

    def test_active_leader_count(self) -> None:
        reg = InflightRegistry()
        assert reg.active_leader_count == 0
        reg.register_leader("rk1", _handle("h1"), _binding("t1", "rk1", "d1"))
        assert reg.active_leader_count == 1

    def test_record_run_completion_completes_cohort(self) -> None:
        reg = InflightRegistry()
        cohort = _cohort("t1")
        reg.register_trial_cohort(cohort)
        reg.record_run_complete(
            trial_id="t1",
            run_key="rk:t1:d1",
            dataset_id="d1",
            objective_result=ObjectiveResult(value=0.1, attrs={}, artifact_refs=[]),
        )
        complete, all_results = reg.record_run_complete(
            trial_id="t1",
            run_key="rk:t1:d2",
            dataset_id="d2",
            objective_result=ObjectiveResult(value=0.2, attrs={}, artifact_refs=[]),
        )
        assert complete is True
        assert all_results is not None
        assert len(all_results) == 2

    def test_record_run_failure_marks_cohort_complete(self) -> None:
        reg = InflightRegistry()
        reg.register_trial_cohort(_cohort("t1"))
        reg.record_run_complete(
            trial_id="t1",
            run_key="rk:t1:d1",
            dataset_id="d1",
            objective_result=ObjectiveResult(value=0.1, attrs={}, artifact_refs=[]),
        )
        complete, _ = reg.record_run_failure(
            trial_id="t1",
            run_key="rk:t1:d2",
            dataset_id="d2",
            state=TrialState.FAIL,
            error="OOM",
        )
        assert complete is True
