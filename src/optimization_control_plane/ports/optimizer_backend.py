from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    SamplerProfile,
    StudyHandle,
    TrialHandle,
)


@runtime_checkable
class TrialContext(Protocol):
    def suggest_int(self, name: str, low: int, high: int) -> int: ...
    def suggest_float(self, name: str, low: float, high: float) -> float: ...
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...
    def set_user_attr(self, key: str, val: Any) -> None: ...
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...


@runtime_checkable
class OptimizerBackend(Protocol):
    def open_or_resume_experiment(self, spec: ExperimentSpec) -> StudyHandle: ...

    def get_spec(self, study_id: str) -> ExperimentSpec: ...

    def get_sampler_profile(self, study_id: str) -> SamplerProfile: ...

    def ask(self, study_id: str) -> TrialHandle: ...

    def open_trial_context(
        self, study_id: str, trial_id: str
    ) -> TrialContext: ...

    def tell(
        self,
        study_id: str,
        trial_id: str,
        state: str,
        value: float | None,
        attrs: dict[str, Any] | None,
    ) -> None: ...
