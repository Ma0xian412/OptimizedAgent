from __future__ import annotations

from optimization_control_plane.domain.models import SamplerProfile
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState


class AsyncFillParallelismPolicy:
    """V1 default: fill up to configured_slots, never release buffer."""

    def target_in_flight(
        self,
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
        resource_state: ResourceState,
    ) -> int:
        cap = resource_state.configured_slots
        if profile.recommended_max_inflight is not None:
            cap = min(cap, profile.recommended_max_inflight)
        return cap

    def should_release_buffer(
        self,
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
    ) -> bool:
        return False
