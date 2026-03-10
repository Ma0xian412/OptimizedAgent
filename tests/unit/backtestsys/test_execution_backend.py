from __future__ import annotations

from threading import Event, Lock
from typing import Any

from optimization_control_plane.adapters.backtestsys import BackTestSysExecutionBackend
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import ExecutionRequest, RunResult, RunSpec


def test_execution_backend_returns_completed_event() -> None:
    backend = BackTestSysExecutionBackend(max_workers=2, runner=_StaticRunner())
    request = _make_request("trial-1")
    handle = backend.submit(request)
    event = backend.wait_any([handle], timeout=1.0)
    backend.shutdown()
    assert event is not None
    assert event.kind == EventKind.COMPLETED
    assert event.run_result is not None
    assert event.run_result.metrics["doneinfo_count"] == 2


def test_execution_backend_surfaces_failed_event() -> None:
    backend = BackTestSysExecutionBackend(max_workers=1, runner=_ErrorRunner())
    handle = backend.submit(_make_request("trial-1"))
    event = backend.wait_any([handle], timeout=1.0)
    backend.shutdown()
    assert event is not None
    assert event.kind == EventKind.FAILED
    assert "empty result" in (event.error_code or "")


def test_execution_backend_supports_parallel_and_cancel() -> None:
    runner = _BlockingRunner()
    backend = BackTestSysExecutionBackend(max_workers=2, runner=runner)
    handle_1 = backend.submit(_make_request("trial-1"))
    handle_2 = backend.submit(_make_request("trial-2"))
    assert runner.two_started.wait(timeout=1.0)
    backend.cancel(handle_1, reason="pruned")
    runner.release.set()
    events = _wait_until_two_events(backend, [handle_1, handle_2])
    backend.shutdown()
    assert {event.kind for event in events} == {EventKind.CANCELLED, EventKind.COMPLETED}


class _StaticRunner:
    def run_request(self, request: ExecutionRequest) -> RunResult:
        return RunResult(
            metrics={"doneinfo_count": 2, "executiondetail_count": 1},
            diagnostics={"trial_id": request.trial_id},
            artifact_refs=[],
        )


class _ErrorRunner:
    def run_request(self, request: ExecutionRequest) -> RunResult:
        raise ValueError("empty result")


class _BlockingRunner:
    def __init__(self) -> None:
        self._started = 0
        self._lock = Lock()
        self.two_started = Event()
        self.release = Event()

    def run_request(self, request: ExecutionRequest) -> RunResult:
        with self._lock:
            self._started += 1
            if self._started == 2:
                self.two_started.set()
        self.release.wait(timeout=2.0)
        return RunResult(
            metrics={"doneinfo_count": 2, "executiondetail_count": 1},
            diagnostics={"trial_id": request.trial_id},
            artifact_refs=[],
        )


def _make_request(trial_id: str) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=f"req-{trial_id}",
        trial_id=trial_id,
        run_key=f"run-{trial_id}",
        objective_key=f"obj-{trial_id}",
        cohort_id=None,
        priority=0,
        run_spec=RunSpec(kind="backtestsys", config={"repo_root": "/tmp"}, resources={}),
    )


def _wait_until_two_events(
    backend: BackTestSysExecutionBackend,
    handles: list[Any],
) -> list[Any]:
    events: list[Any] = []
    for _ in range(10):
        event = backend.wait_any(handles, timeout=0.5)
        if event is None:
            continue
        events.append(event)
        if len(events) == 2:
            return events
    raise AssertionError("did not receive two terminal events in time")
