"""Programmable fake ExecutionBackend for testing the control plane."""
from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from optimization_control_plane.domain.enums import EventKind, JobStatus
from optimization_control_plane.domain.models import (
    Checkpoint,
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
    RunResult,
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
        self._cancelled: set[str] = set()

    def set_script(self, run_key: str, script: FakeRunScript) -> None:
        self._scripts[run_key] = script

    def set_default_script(self, script: FakeRunScript) -> None:
        self._default_script = script

    def submit(self, request: ExecutionRequest) -> RunHandle:
        handle_id = f"fh_{uuid.uuid4().hex[:12]}"
        handle = RunHandle(
            handle_id=handle_id,
            request_id=request.request_id,
            state=JobStatus.RUNNING,
        )
        self._handles[handle_id] = handle

        script = self._scripts.get(request.run_key, self._default_script)
        if script is None:
            script = FakeRunScript()

        self._write_result_file(request, script)
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

    def _enqueue_events(self, handle_id: str, script: FakeRunScript) -> None:
        for cp in script.checkpoints:
            self._pending_events.append(ExecutionEvent(
                kind=EventKind.CHECKPOINT,
                handle_id=handle_id,
                step=cp.step,
                checkpoint=cp,
            ))

        if script.final_event == EventKind.COMPLETED:
            self._pending_events.append(ExecutionEvent(
                kind=EventKind.COMPLETED,
                handle_id=handle_id,
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

    def _write_result_file(self, request: ExecutionRequest, script: FakeRunScript) -> None:
        if script.final_event != EventKind.COMPLETED:
            return
        result = script.run_result or RunResult(metrics={}, diagnostics={}, artifact_refs=[])
        path = Path(request.run_spec.result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": result.metrics,
            "diagnostics": result.diagnostics,
            "artifact_refs": result.artifact_refs,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
