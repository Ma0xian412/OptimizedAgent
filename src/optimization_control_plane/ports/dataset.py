from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import ExperimentSpec


@runtime_checkable
class DatasetEnumerator(Protocol):
    """Enumerate dataset IDs to evaluate for one trial."""

    def enumerate(self, spec: ExperimentSpec) -> tuple[str, ...]: ...
