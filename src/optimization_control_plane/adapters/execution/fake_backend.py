"""Programmable fake ExecutionBackend for testing the control plane."""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import (
    Checkpoint,
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
    RunResult,
    validate_target_spec,
)


@dataclass
class FakeRunScript:
    """Describes the sequence of events a fake run will produce."""

    checkpoints: list[Checkpoint] = field(default_factory=list)
    final_event: EventKind = EventKind.COMPLETED
    run_result: RunResult | None = None
    fail_reason: str | None = None
    fail_error_code: str | None = None


class FakeExecutionBackend:
    """ExecutionBackend that replays scripted event sequences."""

    def __init__(self) -> None:
        self._scripts: dict[str, FakeRunScript] = {}
        self._default_script: FakeRunScript | None = None
        self._pending_events: deque[ExecutionEvent] = deque()
        self._handles: dict[str, RunHandle] = {}
        self._submitted_requests: dict[str, ExecutionRequest] = {}
        self._submitted_targets: dict[str, str] = {}
        self._cancelled: set[str] = set()

    def set_script(self, run_key: str, script: FakeRunScript) -> None:
        self._scripts[run_key] = script

    def set_default_script(self, script: FakeRunScript) -> None:
        self._default_script = script

    def submit(self, request: ExecutionRequest) -> RunHandle:
        if request.run_spec is None:
            raise ValueError("request.run_spec must be provided")
        target_spec = validate_target_spec(
            request.run_spec.target_spec,
            source="request.run_spec.target_spec",
        )
        handle_id = f"fh_{uuid.uuid4().hex[:12]}"
        handle = RunHandle(
            handle_id=handle_id,
            request_id=request.request_id,
            state="RUNNING",
        )
        self._handles[handle_id] = handle
        self._submitted_requests[handle_id] = request
        self._submitted_targets[handle_id] = target_spec.target_id

        script = self._scripts.get(request.run_key, self._default_script)
        if script is None:
            script = FakeRunScript()

        self._enqueue_events(handle_id, script)
        return handle

    def wait_any(
        self,
        handles: list[RunHandle],
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        active_ids = {h.handle_id for h in handles}
        for _ in range(len(self._pending_events)):
            event = self._pending_events.popleft()
            if event.handle_id in active_ids:
                return event
            self._pending_events.append(event)
        return None

    def cancel(self, handle: RunHandle, reason: str) -> None:
        self._cancelled.add(handle.handle_id)
        self._pending_events = deque(
            e for e in self._pending_events
            if e.handle_id != handle.handle_id
        )
        self._pending_events.append(ExecutionEvent(
            kind=EventKind.CANCELLED,
            handle_id=handle.handle_id,
            reason=reason,
        ))

    def get_submitted_request(self, handle_id: str) -> ExecutionRequest:
        return self._submitted_requests[handle_id]

    def submitted_requests(self) -> list[ExecutionRequest]:
        return list(self._submitted_requests.values())

    def get_submitted_target_id(self, handle_id: str) -> str:
        return self._submitted_targets[handle_id]

    def _enqueue_events(self, handle_id: str, script: FakeRunScript) -> None:
        for cp in script.checkpoints:
            self._pending_events.append(ExecutionEvent(
                kind=EventKind.CHECKPOINT,
                handle_id=handle_id,
                step=cp.step,
                checkpoint=cp,
            ))

        if script.final_event == EventKind.COMPLETED:
            result = script.run_result or RunResult(
                metrics={}, diagnostics={}, artifact_refs=[]
            )
            self._pending_events.append(ExecutionEvent(
                kind=EventKind.COMPLETED,
                handle_id=handle_id,
                run_result=result,
            ))
        elif script.final_event == EventKind.FAILED:
            self._pending_events.append(ExecutionEvent(
                kind=EventKind.FAILED,
                handle_id=handle_id,
                error_code=script.fail_error_code or "UNKNOWN",
            ))
        elif script.final_event == EventKind.CANCELLED:
            self._pending_events.append(ExecutionEvent(
                kind=EventKind.CANCELLED,
                handle_id=handle_id,
                reason=script.fail_reason or "user_stop",
            ))
