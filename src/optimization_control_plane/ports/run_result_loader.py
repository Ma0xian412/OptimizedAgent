from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import RunResult, RunSpec


@runtime_checkable
class RunResultLoader(Protocol):
    """Load a raw run-result payload from the path declared in RunSpec."""

    def load(self, run_spec: RunSpec) -> RunResult: ...
