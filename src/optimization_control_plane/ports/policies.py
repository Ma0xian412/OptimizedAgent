from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.enums import DispatchDecision
from optimization_control_plane.domain.models import (
    ExecutionRequest,
    SamplerProfile,
)
from optimization_control_plane.domain.state import ResourceState, StudyRuntimeState


@runtime_checkable
class ParallelismPolicy(Protocol):
    def target_in_flight(
        self,
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
        resource_state: ResourceState,
    ) -> int: ...

    def should_release_buffer(
        self,
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
    ) -> bool: ...


@runtime_checkable
class DispatchPolicy(Protocol):
    def classify(
        self,
        request: ExecutionRequest,
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
        resource_state: ResourceState,
    ) -> DispatchDecision: ...

    def order(
        self,
        requests: list[ExecutionRequest],
        profile: SamplerProfile,
        study_state: StudyRuntimeState,
        resource_state: ResourceState,
    ) -> list[ExecutionRequest]: ...
