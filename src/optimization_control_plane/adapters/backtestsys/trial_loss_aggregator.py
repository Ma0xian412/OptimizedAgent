from __future__ import annotations

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ObjectiveResult,
)


class MeanTrialLossAggregator:
    """Aggregate per-run objectives with arithmetic mean."""

    def aggregate(
        self,
        run_objectives: list[ObjectiveResult],
        spec: ExperimentSpec,
        split: str,
    ) -> ObjectiveResult:
        if not run_objectives:
            raise ValueError("run_objectives cannot be empty")
        total = sum(item.value for item in run_objectives)
        value = float(total / len(run_objectives))
        attrs = {
            "aggregate": "mean",
            "split": split,
            "component_count": len(run_objectives),
            "component_values": [item.value for item in run_objectives],
        }
        artifact_refs = _merge_artifacts(run_objectives)
        return ObjectiveResult(value=value, attrs=attrs, artifact_refs=artifact_refs)


def _merge_artifacts(run_objectives: list[ObjectiveResult]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in run_objectives:
        for ref in item.artifact_refs:
            if ref in seen:
                continue
            seen.add(ref)
            merged.append(ref)
    return merged
