from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ResolvedTarget,
    TargetSpec,
)


@runtime_checkable
class TargetResolver(Protocol):
    """Resolve experiment-level target binding for execution."""

    def resolve(
        self,
        target_spec: TargetSpec,
        spec: ExperimentSpec,
    ) -> ResolvedTarget: ...
