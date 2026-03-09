from __future__ import annotations

import importlib
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from typing import Any

from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
    RunResult,
)

Runner = Callable[[ExecutionRequest], RunResult | dict[str, Any]]


@dataclass(frozen=True)
class _SubmittedRun:
    request: ExecutionRequest
    future: Future[RunResult | dict[str, Any]]


class PythonCallableExecutionBackend:
    """Execute real iteratee by calling a configured Python entrypoint."""

    def __init__(self, *, entrypoint: str, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._runner = _load_runner(entrypoint)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self._submitted: dict[str, _SubmittedRun] = {}
        self._future_to_handle: dict[Future[RunResult | dict[str, Any]], str] = {}
        self._queued_events: deque[ExecutionEvent] = deque()
        self._ignored_futures: set[Future[RunResult | dict[str, Any]]] = set()

    def submit(self, request: ExecutionRequest) -> RunHandle:
        handle_id = f"rh_{uuid.uuid4().hex[:12]}"
        future = self._executor.submit(self._runner, request)
        submitted = _SubmittedRun(request=request, future=future)
        with self._lock:
            self._submitted[handle_id] = submitted
            self._future_to_handle[future] = handle_id
        return RunHandle(handle_id=handle_id, request_id=request.request_id, state="RUNNING")

    def wait_any(
        self, handles: list[RunHandle], timeout: float | None = None,
    ) -> ExecutionEvent | None:
        active_ids = {handle.handle_id for handle in handles}
        queued = self._pop_queued_event(active_ids)
        if queued is not None:
            return queued
        futures = self._active_futures(active_ids)
        if not futures:
            return None
        done, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
        for future in done:
            event = self._event_for_completed_future(future, active_ids)
            if event is not None:
                return event
        return None

    def cancel(self, handle: RunHandle, reason: str) -> None:
        with self._lock:
            submitted = self._submitted.pop(handle.handle_id, None)
            if submitted is None:
                return
            self._future_to_handle.pop(submitted.future, None)
            self._ignored_futures.add(submitted.future)
            self._queued_events.append(ExecutionEvent(
                kind=EventKind.CANCELLED,
                handle_id=handle.handle_id,
                reason=reason,
            ))
        submitted.future.cancel()

    def _pop_queued_event(self, active_ids: set[str]) -> ExecutionEvent | None:
        with self._lock:
            for _ in range(len(self._queued_events)):
                event = self._queued_events.popleft()
                if event.handle_id in active_ids:
                    return event
                self._queued_events.append(event)
        return None

    def _active_futures(
        self, active_ids: set[str],
    ) -> list[Future[RunResult | dict[str, Any]]]:
        with self._lock:
            return [
                submitted.future
                for handle_id, submitted in self._submitted.items()
                if handle_id in active_ids
            ]

    def _event_for_completed_future(
        self,
        future: Future[RunResult | dict[str, Any]],
        active_ids: set[str],
    ) -> ExecutionEvent | None:
        with self._lock:
            if future in self._ignored_futures:
                self._ignored_futures.discard(future)
                return None
            handle_id = self._future_to_handle.pop(future, None)
            if handle_id is None:
                return None
            submitted = self._submitted.pop(handle_id, None)
        if submitted is None or handle_id not in active_ids:
            return None
        if future.cancelled():
            return ExecutionEvent(
                kind=EventKind.CANCELLED,
                handle_id=handle_id,
                reason="cancelled",
            )
        exc = future.exception()
        if exc is not None:
            return ExecutionEvent(
                kind=EventKind.FAILED,
                handle_id=handle_id,
                error_code=_format_error(exc),
            )
        run_result = _coerce_run_result(future.result())
        return ExecutionEvent(
            kind=EventKind.COMPLETED,
            handle_id=handle_id,
            run_result=run_result,
        )


def _load_runner(entrypoint: str) -> Runner:
    module_name, sep, attr_name = entrypoint.partition(":")
    if sep == "" or not module_name or not attr_name:
        raise ValueError("entrypoint must be in format 'module_path:function_name'")
    module = importlib.import_module(module_name)
    runner = getattr(module, attr_name, None)
    if runner is None or not callable(runner):
        raise ValueError(f"entrypoint not callable: {entrypoint}")
    return runner


def _coerce_run_result(value: RunResult | dict[str, Any]) -> RunResult:
    if isinstance(value, RunResult):
        return value
    if not isinstance(value, dict):
        raise TypeError("runner result must be RunResult or dict")
    metrics = value.get("metrics")
    diagnostics = value.get("diagnostics")
    artifact_refs_raw = value.get("artifact_refs", [])
    if not isinstance(metrics, dict) or not isinstance(diagnostics, dict):
        raise ValueError("runner dict result requires metrics and diagnostics objects")
    if not isinstance(artifact_refs_raw, list):
        raise ValueError("runner dict result artifact_refs must be a list")
    artifact_refs = [str(item) for item in artifact_refs_raw]
    return RunResult(
        metrics=metrics,
        diagnostics=diagnostics,
        artifact_refs=artifact_refs,
    )


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
