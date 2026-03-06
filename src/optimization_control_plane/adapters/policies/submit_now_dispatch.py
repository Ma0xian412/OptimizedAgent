from __future__ import annotations

from optimization_control_plane.domain.enums import DispatchDecision
from optimization_control_plane.domain.models import ExecutionRequest, SamplerProfile
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState


class SubmitNowDispatchPolicy:
    """V1 default: always SUBMIT_NOW, identity ordering."""

    def classify(
        self,
        request: ExecutionRequest,
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
        resource_state: ResourceState,
    ) -> DispatchDecision:
        return DispatchDecision.SUBMIT_NOW

    def order(
        self,
        requests: list[ExecutionRequest],
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
        resource_state: ResourceState,
    ) -> list[ExecutionRequest]:
        return list(requests)
