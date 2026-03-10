from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Execution boundary for dispatching concrete runs.

    submit() must consume request.run_spec.target_spec as an explicit target
    boundary. Execution adapters must not infer target identity from implicit
    fields in request.run_spec.config or other payload parts.
    """

    def submit(self, request: ExecutionRequest) -> RunHandle: ...

    def wait_any(
        self, handles: list[RunHandle], timeout: float | None = None
    ) -> ExecutionEvent | None: ...

    def cancel(self, handle: RunHandle, reason: str) -> None: ...
