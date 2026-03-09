from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
)


@runtime_checkable
class ExecutionBackend(Protocol):
    def submit(self, request: ExecutionRequest) -> RunHandle: ...

    def wait_any(
        self, handles: list[RunHandle], timeout: float | None = None
    ) -> ExecutionEvent | None: ...

    def cancel(self, handle: RunHandle, reason: str) -> None: ...
