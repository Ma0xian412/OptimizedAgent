from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import (
    Checkpoint,
    ExperimentSpec,
    ObjectiveResult,
    RunResult,
    RunSpec,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext


@runtime_checkable
class SearchSpace(Protocol):
    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]: ...


@runtime_checkable
class RunSpecBuilder(Protocol):
    def build(self, params: dict[str, object], spec: ExperimentSpec) -> RunSpec: ...


@runtime_checkable
class RunKeyBuilder(Protocol):
    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str: ...


@runtime_checkable
class ObjectiveKeyBuilder(Protocol):
    def build(self, run_key: str, objective_config: dict[str, object]) -> str: ...


@runtime_checkable
class ProgressScorer(Protocol):
    def score(self, checkpoint: Checkpoint, spec: ExperimentSpec) -> float | None: ...


@runtime_checkable
class ObjectiveEvaluator(Protocol):
    def evaluate(self, run_result: RunResult, spec: ExperimentSpec) -> ObjectiveResult: ...
