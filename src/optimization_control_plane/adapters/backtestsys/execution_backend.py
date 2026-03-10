from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from typing import Deque
from uuid import uuid4

from optimization_control_plane.adapters.backtestsys.runner import BackTestSysRunner
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
    RunResult,
)

_BACKTESTSYS_KIND = "backtestsys"


@dataclass
class _TaskRecord:
    request: ExecutionRequest
    future: Future[RunResult]
    cancel_reason: str | None = None


class BackTestSysExecutionBackend:
    """Multi-worker ExecutionBackend for BackTestSys."""

    def __init__(
        self,
        max_workers: int = 4,
        runner: BackTestSysRunner | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._runner = runner or BackTestSysRunner()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="backtestsys-worker",
        )
        self._tasks: dict[str, _TaskRecord] = {}
        self._event_queue: Deque[ExecutionEvent] = deque()
        self._lock = Lock()

    def submit(self, request: ExecutionRequest) -> RunHandle:
        if request.run_spec.kind != _BACKTESTSYS_KIND:
            raise ValueError(
                f"run_spec.kind must be '{_BACKTESTSYS_KIND}', got '{request.run_spec.kind}'"
            )
        handle_id = f"bh_{uuid4().hex[:12]}"
        handle = RunHandle(handle_id=handle_id, request_id=request.request_id, state="RUNNING")
        future = self._executor.submit(self._runner.run_request, request)
        with self._lock:
            self._tasks[handle_id] = _TaskRecord(request=request, future=future)
        return handle

    def wait_any(
        self,
        handles: list[RunHandle],
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        active_ids = {item.handle_id for item in handles}
        if not active_ids:
            return None
        queued = self._drain_queued_event(active_ids)
        if queued is not None:
            return queued
        wait_futures = self._collect_futures(active_ids)
        if not wait_futures:
            return None
        done, _ = wait(wait_futures, timeout=timeout, return_when=FIRST_COMPLETED)
        if not done:
            return None
        return self._collect_completed_event(active_ids)

    def cancel(self, handle: RunHandle, reason: str) -> None:
        with self._lock:
            task = self._tasks.get(handle.handle_id)
            if task is None:
                raise KeyError(f"unknown handle_id: {handle.handle_id}")
            task.cancel_reason = reason
            cancelled = task.future.cancel()
            if cancelled:
                self._tasks.pop(handle.handle_id, None)
                self._event_queue.append(
                    ExecutionEvent(
                        kind=EventKind.CANCELLED,
                        handle_id=handle.handle_id,
                        reason=reason,
                    )
                )

    def shutdown(self, wait_for_tasks: bool = True) -> None:
        self._executor.shutdown(wait=wait_for_tasks, cancel_futures=not wait_for_tasks)

    def _drain_queued_event(self, active_ids: set[str]) -> ExecutionEvent | None:
        with self._lock:
            for _ in range(len(self._event_queue)):
                event = self._event_queue.popleft()
                if event.handle_id in active_ids:
                    return event
                self._event_queue.append(event)
        return None

    def _collect_futures(self, active_ids: set[str]) -> list[Future[RunResult]]:
        with self._lock:
            return [
                task.future
                for handle_id, task in self._tasks.items()
                if handle_id in active_ids
            ]

    def _collect_completed_event(self, active_ids: set[str]) -> ExecutionEvent | None:
        with self._lock:
            completed = [
                (handle_id, task)
                for handle_id, task in self._tasks.items()
                if handle_id in active_ids and task.future.done()
            ]
            if not completed:
                return None
            first_event: ExecutionEvent | None = None
            for handle_id, task in completed:
                event = self._build_terminal_event(handle_id, task)
                self._tasks.pop(handle_id, None)
                if first_event is None:
                    first_event = event
                else:
                    self._event_queue.append(event)
            return first_event

    @staticmethod
    def _build_terminal_event(handle_id: str, task: _TaskRecord) -> ExecutionEvent:
        if task.cancel_reason is not None:
            return ExecutionEvent(
                kind=EventKind.CANCELLED,
                handle_id=handle_id,
                reason=task.cancel_reason,
            )
        exc = task.future.exception()
        if exc is not None:
            return ExecutionEvent(
                kind=EventKind.FAILED,
                handle_id=handle_id,
                error_code=str(exc),
            )
        run_result = task.future.result()
        return ExecutionEvent(
            kind=EventKind.COMPLETED,
            handle_id=handle_id,
            run_result=run_result,
        )
