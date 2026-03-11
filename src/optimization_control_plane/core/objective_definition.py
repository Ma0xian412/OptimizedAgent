from __future__ import annotations

from dataclasses import dataclass

from optimization_control_plane.ports.dataset import DatasetEnumerator
from optimization_control_plane.ports.objective import (
    ObjectiveEvaluator,
    ObjectiveKeyBuilder,
    ProgressScorer,
    RunKeyBuilder,
    RunSpecBuilder,
    SearchSpace,
    TrialResultAggregator,
)


@dataclass(frozen=True)
class ObjectiveDefinition:
    search_space: SearchSpace
    dataset_enumerator: DatasetEnumerator
    run_spec_builder: RunSpecBuilder
    run_key_builder: RunKeyBuilder
    objective_key_builder: ObjectiveKeyBuilder
    trial_result_aggregator: TrialResultAggregator
    progress_scorer: ProgressScorer | None
    objective_evaluator: ObjectiveEvaluator
