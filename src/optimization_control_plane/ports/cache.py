from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import ObjectiveResult, RunResult


@runtime_checkable
class RunCache(Protocol):
    """Cache raw RunResult payloads keyed by run_key."""

    def get(self, run_key: str) -> RunResult | None: ...
    def put(self, run_key: str, run_result: RunResult) -> None: ...


@runtime_checkable
class ObjectiveCache(Protocol):
    def get(self, objective_key: str) -> ObjectiveResult | None: ...
    def put(self, objective_key: str, objective_result: ObjectiveResult) -> None: ...
