"""Linear execution backend for deterministic local/mock execution."""
from __future__ import annotations

import uuid
from collections import deque

from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
)
from optimization_control_plane.ports.target_runtime import TargetRuntime


class LinearExecutionBackend:
    """ExecutionBackend that runs one submitted job immediately in-process."""

    def __init__(self, target_runtime: TargetRuntime) -> None:
        self._target_runtime = target_runtime
        self._pending_events: deque[ExecutionEvent] = deque()

    def submit(self, request: ExecutionRequest) -> RunHandle:
        handle = RunHandle(
            handle_id=f"lh_{uuid.uuid4().hex[:12]}",
            request_id=request.request_id,
            state="RUNNING",
        )
        self._enqueue_final_event(handle.handle_id, request)
        return handle

    def wait_any(
        self,
        handles: list[RunHandle],
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        _ = timeout
        active_ids = {h.handle_id for h in handles}
        for _ in range(len(self._pending_events)):
            event = self._pending_events.popleft()
            if event.handle_id in active_ids:
                return event
            self._pending_events.append(event)
        return None

    def cancel(self, handle: RunHandle, reason: str) -> None:
        self._pending_events = deque(
            event for event in self._pending_events if event.handle_id != handle.handle_id
        )
        self._pending_events.append(
            ExecutionEvent(
                kind=EventKind.CANCELLED,
                handle_id=handle.handle_id,
                reason=reason,
            )
        )

    def _enqueue_final_event(self, handle_id: str, request: ExecutionRequest) -> None:
        try:
            run_result = self._target_runtime.run(request.run_spec)
        except Exception as exc:
            self._pending_events.append(
                ExecutionEvent(
                    kind=EventKind.FAILED,
                    handle_id=handle_id,
                    error_code=type(exc).__name__,
                )
            )
            return
        self._pending_events.append(
            ExecutionEvent(
                kind=EventKind.COMPLETED,
                handle_id=handle_id,
                run_result=run_result,
            )
        )
