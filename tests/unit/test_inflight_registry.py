"""UT-7: InflightRegistry leader/follower behavior."""
from __future__ import annotations

import pytest

from optimization_control_plane.core.orchestration.inflight_registry import (
    InflightRegistry,
    TrialBinding,
)
from optimization_control_plane.domain.models import RunHandle
from tests.unit.test_search_space import FakeCtx


def _binding(trial_id: str, number: int = 0) -> TrialBinding:
    return TrialBinding(
        trial_id=trial_id,
        trial_number=number,
        objective_key=f"obj:{trial_id}",
        trial_ctx=FakeCtx(),
    )


def _handle(hid: str) -> RunHandle:
    return RunHandle(handle_id=hid, request_id="r1", state="RUNNING")


class TestInflightRegistry:
    def test_register_and_has(self) -> None:
        reg = InflightRegistry()
        assert not reg.has("rk1")
        reg.register_leader("rk1", _handle("h1"), _binding("t1"))
        assert reg.has("rk1")

    def test_duplicate_leader_raises(self) -> None:
        reg = InflightRegistry()
        reg.register_leader("rk1", _handle("h1"), _binding("t1"))
        with pytest.raises(ValueError):
            reg.register_leader("rk1", _handle("h2"), _binding("t2"))

    def test_attach_follower(self) -> None:
        reg = InflightRegistry()
        reg.register_leader("rk1", _handle("h1"), _binding("t1"))
        reg.attach_follower("rk1", _binding("t2"))
        reg.attach_follower("rk1", _binding("t3"))
        assert reg.total_follower_count == 2

    def test_attach_follower_no_leader_raises(self) -> None:
        reg = InflightRegistry()
        with pytest.raises(ValueError):
            reg.attach_follower("rk_missing", _binding("t1"))

    def test_get_by_handle(self) -> None:
        reg = InflightRegistry()
        h = _handle("h1")
        reg.register_leader("rk1", h, _binding("t1"))
        entry = reg.get_by_handle("h1")
        assert entry.run_key == "rk1"
        assert entry.leader.trial_id == "t1"

    def test_get_by_handle_missing_raises(self) -> None:
        reg = InflightRegistry()
        with pytest.raises(KeyError):
            reg.get_by_handle("missing")

    def test_pop_all_trials(self) -> None:
        reg = InflightRegistry()
        reg.register_leader("rk1", _handle("h1"), _binding("t1"))
        reg.attach_follower("rk1", _binding("t2"))
        reg.attach_follower("rk1", _binding("t3"))

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
        reg.register_leader("rk1", h1, _binding("t1"))
        reg.register_leader("rk2", h2, _binding("t2"))
        handles = reg.handles()
        assert len(handles) == 2

    def test_active_leader_count(self) -> None:
        reg = InflightRegistry()
        assert reg.active_leader_count == 0
        reg.register_leader("rk1", _handle("h1"), _binding("t1"))
        assert reg.active_leader_count == 1
