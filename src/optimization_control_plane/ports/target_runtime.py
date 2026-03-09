from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import RunResult, RunSpec


@runtime_checkable
class TargetRuntime(Protocol):
    """Executes a single resolved run_spec and returns run_result."""

    def run(self, run_spec: RunSpec) -> RunResult: ...
