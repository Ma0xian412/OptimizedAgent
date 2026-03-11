from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import ExperimentSpec, GroundTruthData


@runtime_checkable
class GroundTruthProvider(Protocol):
    def load(self, spec: ExperimentSpec) -> GroundTruthData: ...

    def load_for_dataset(
        self,
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> GroundTruthData: ...
