from __future__ import annotations

from dataclasses import dataclass

from optimization_control_plane.ports.objective import (
    ObjectiveEvaluator,
    ObjectiveKeyBuilder,
    ProgressScorer,
    RunKeyBuilder,
    RunSpecBuilder,
    SearchSpace,
    TrialLossAggregator,
)


@dataclass(frozen=True)
class ObjectiveDefinition:
    search_space: SearchSpace
    run_spec_builder: RunSpecBuilder
    run_key_builder: RunKeyBuilder
    objective_key_builder: ObjectiveKeyBuilder
    progress_scorer: ProgressScorer | None
    objective_evaluator: ObjectiveEvaluator
    trial_loss_aggregator: TrialLossAggregator | None = None
