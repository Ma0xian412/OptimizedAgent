from __future__ import annotations

from dataclasses import dataclass

from optimization_control_plane.adapters.execution import LinearExecutionBackend
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import ExecutionRequest, RunResult, RunSpec


@dataclass
class _StubRuntime:
    should_raise: bool = False

    def run(self, run_spec: RunSpec) -> RunResult:
        if self.should_raise:
            raise ValueError("boom")
        return RunResult(
            metrics={"metric_1": float(run_spec.config.get("x", 0.0))},
            diagnostics={"ok": True},
            artifact_refs=[],
        )


def _make_request() -> ExecutionRequest:
    return ExecutionRequest(
        request_id="req_1",
        trial_id="t1",
        run_key="run:k1",
        objective_key="obj:k1",
        cohort_id=None,
        priority=0,
        run_spec=RunSpec(
            kind="python_blackbox",
            target_config={"kind": "python_callable", "ref": "tests.fixtures.blackboxes:echo_config"},
            config={"x": 0.5},
            resources={},
        ),
    )


class TestLinearExecutionBackend:
    def test_submit_emits_completed_event(self) -> None:
        backend = LinearExecutionBackend(_StubRuntime())
        handle = backend.submit(_make_request())
        event = backend.wait_any([handle])
        assert event is not None
        assert event.kind == EventKind.COMPLETED
        assert event.run_result is not None
        assert event.run_result.metrics["metric_1"] == 0.5

    def test_runtime_error_emits_failed_event(self) -> None:
        backend = LinearExecutionBackend(_StubRuntime(should_raise=True))
        handle = backend.submit(_make_request())
        event = backend.wait_any([handle])
        assert event is not None
        assert event.kind == EventKind.FAILED
        assert event.error_code == "ValueError"

    def test_cancel_replaces_pending_event(self) -> None:
        backend = LinearExecutionBackend(_StubRuntime())
        handle = backend.submit(_make_request())
        backend.cancel(handle, reason="user_stop")
        event = backend.wait_any([handle])
        assert event is not None
        assert event.kind == EventKind.CANCELLED
        assert event.reason == "user_stop"
