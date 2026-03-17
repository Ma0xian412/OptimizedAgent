from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import ExperimentSpec, GroundTruthData


@runtime_checkable
class GroundTruthProvider(Protocol):
    """Load ground truth data for evaluation.

    Use dataset_id="" for experiment-level (shared) ground truth.
    Use a concrete dataset_id to load per-dataset ground truth.
    """

    def load(self, spec: ExperimentSpec, dataset_id: str) -> GroundTruthData: ...
