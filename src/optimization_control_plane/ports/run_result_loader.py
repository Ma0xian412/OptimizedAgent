"""Port for loading RunResult from a filesystem path."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import RunResult


@runtime_checkable
class RunResultLoader(Protocol):
    """Load RunResult from the given filesystem path."""

    def load(self, path: str) -> RunResult: ...
