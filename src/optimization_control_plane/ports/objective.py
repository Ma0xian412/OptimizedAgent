from __future__ import annotations

from typing import Protocol, runtime_checkable

from optimization_control_plane.domain.models import (
    Checkpoint,
    ExperimentSpec,
    GroundTruthData,
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
    """Build an executable run specification from sampled parameters."""

    def build(
        self,
        params: dict[str, object],
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> RunSpec: ...


@runtime_checkable
class RunKeyBuilder(Protocol):
    def build(
        self,
        run_spec: RunSpec,
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> str: ...


@runtime_checkable
class ObjectiveKeyBuilder(Protocol):
    def build(self, run_key: str, objective_config: dict[str, object]) -> str: ...


@runtime_checkable
class ProgressScorer(Protocol):
    def score(self, checkpoint: Checkpoint, spec: ExperimentSpec) -> float | None: ...


@runtime_checkable
class ObjectiveEvaluator(Protocol):
    def evaluate(
        self,
        run_result: RunResult,
        spec: ExperimentSpec,
        groundtruth: GroundTruthData,
    ) -> ObjectiveResult: ...


@runtime_checkable
class TrialResultAggregator(Protocol):
    """Aggregate per-dataset objective results into one trial objective."""

    def aggregate(
        self,
        results: list[tuple[str, ObjectiveResult]],
        spec: ExperimentSpec,
    ) -> ObjectiveResult: ...
