from __future__ import annotations

from typing import Any

from optimization_control_plane.core.objective_definition import ObjectiveDefinition
from optimization_control_plane.core.orchestration._trial_utils import (
    scope_objective_key,
    validate_run_spec,
)
from optimization_control_plane.core.orchestration.inflight_registry import RunBinding
from optimization_control_plane.domain.models import ExperimentSpec, GroundTruthData


def enumerate_dataset_ids(objective_def: ObjectiveDefinition, spec: ExperimentSpec) -> tuple[str, ...]:
    dataset_ids = objective_def.dataset_enumerator.enumerate(spec)
    if not dataset_ids or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("dataset enumerator must return non-empty unique dataset_id tuple")
    return dataset_ids


def build_bindings(
    *,
    objective_def: ObjectiveDefinition,
    spec: ExperimentSpec,
    groundtruth_by_dataset: dict[str, GroundTruthData],
    params: dict[str, object],
    dataset_ids: tuple[str, ...],
    trial_id: str,
    trial_number: int,
    trial_objective_key: str,
    trial_ctx: Any,
) -> tuple[RunBinding, ...]:
    bindings: list[RunBinding] = []
    for dataset_id in dataset_ids:
        groundtruth = _groundtruth_for_dataset(
            groundtruth_by_dataset=groundtruth_by_dataset,
            dataset_id=dataset_id,
        )
        run_spec = objective_def.run_spec_builder.build(params, spec, dataset_id)
        validate_run_spec(run_spec)
        run_key = objective_def.run_key_builder.build(run_spec, spec, dataset_id)
        raw_obj_key = objective_def.objective_key_builder.build(run_key, spec.objective_config)
        bindings.append(
            RunBinding(
                trial_id=trial_id,
                trial_number=trial_number,
                dataset_id=dataset_id,
                run_key=run_key,
                per_run_objective_key=scope_objective_key(raw_obj_key, groundtruth.fingerprint),
                trial_objective_key=trial_objective_key,
                run_spec=run_spec,
                trial_ctx=trial_ctx,
            )
        )
    return tuple(bindings)


def _groundtruth_for_dataset(
    *,
    groundtruth_by_dataset: dict[str, GroundTruthData],
    dataset_id: str,
) -> GroundTruthData:
    data = groundtruth_by_dataset.get(dataset_id)
    if data is None:
        raise KeyError(f"missing groundtruth for dataset_id={dataset_id}")
    return data
