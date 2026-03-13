from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, ObjectiveResult

_COMPONENTS = ("curve", "terminal", "post")
_DEFAULT_EPS = {
    "curve": 1e-12,
    "terminal": 1e-12,
    "post": 1e-12,
}


@dataclass(frozen=True)
class _LossConfig:
    weights: dict[str, float]
    baseline: dict[str, float]
    eps: dict[str, float]


@dataclass(frozen=True)
class _DatasetObjective:
    dataset_id: str
    raw: dict[str, float | None]
    order_count: int
    cancel_order_count: int


class BackTestTrialResultAggregatorAdapter:
    """Aggregate per-dataset raw losses into trial-level objective value."""

    def aggregate(
        self,
        results: list[tuple[str, ObjectiveResult]],
        spec: ExperimentSpec,
    ) -> ObjectiveResult:
        if not results:
            raise ValueError("cannot aggregate empty objective results")
        config = _read_loss_config(spec)
        dataset_objectives = [_parse_dataset_objective(dataset_id, objective) for dataset_id, objective in results]
        raw_values = _aggregate_raw_values(dataset_objectives)
        normalized = _normalize_raw_values(raw_values=raw_values, config=config)
        available_components = _resolve_available_components(raw_values, dataset_objectives)
        effective_weights = _normalize_weights(config.weights, available_components)
        final_value = sum(effective_weights[name] * normalized[name] for name in available_components)
        attrs = {
            "value": final_value,
            "dataset_count": len(dataset_objectives),
            "order_total": sum(item.order_count for item in dataset_objectives),
            "cancel_order_total": sum(item.cancel_order_count for item in dataset_objectives),
            "raw": dict(raw_values),
            "baseline": dict(config.baseline),
            "normalized": dict(normalized),
            "effective_weights": dict(effective_weights),
            "available_components": list(available_components),
        }
        return ObjectiveResult(attrs=attrs, artifact_refs=[])


def _aggregate_raw_values(dataset_objectives: list[_DatasetObjective]) -> dict[str, float | None]:
    dataset_count = len(dataset_objectives)
    curve_raw = sum(_read_required_raw(item.raw, "curve") for item in dataset_objectives) / float(dataset_count)
    terminal_raw = (
        sum(_read_required_raw(item.raw, "terminal") for item in dataset_objectives) / float(dataset_count)
    )
    cancel_order_total = sum(item.cancel_order_count for item in dataset_objectives)
    if cancel_order_total > 0:
        post_weighted_sum = sum(
            item.cancel_order_count * _read_required_raw(item.raw, "post")
            for item in dataset_objectives
            if item.cancel_order_count > 0
        )
        post_raw = post_weighted_sum / float(cancel_order_total)
    else:
        post_raw = None
    return {
        "curve": curve_raw,
        "terminal": terminal_raw,
        "post": post_raw,
    }


def _normalize_raw_values(*, raw_values: dict[str, float | None], config: _LossConfig) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name in _COMPONENTS:
        raw = raw_values.get(name)
        if raw is None:
            normalized[name] = 0.0
            continue
        denominator = config.baseline[name] + config.eps[name]
        normalized[name] = raw / denominator
    return normalized


def _resolve_available_components(
    raw_values: dict[str, float | None],
    dataset_objectives: list[_DatasetObjective],
) -> tuple[str, ...]:
    order_total = sum(item.order_count for item in dataset_objectives)
    cancel_order_total = sum(item.cancel_order_count for item in dataset_objectives)
    available: list[str] = []
    if order_total > 0:
        available.extend(["curve", "terminal"])
    if cancel_order_total > 0:
        available.append("post")
    missing_raw = [name for name in available if raw_values.get(name) is None]
    if missing_raw:
        raise ValueError(f"missing raw values for available components: {missing_raw}")
    if not available:
        raise ValueError("no available components for trial aggregation")
    return tuple(available)


def _normalize_weights(weights: dict[str, float], available_components: tuple[str, ...]) -> dict[str, float]:
    total_weight = sum(weights[name] for name in available_components)
    if total_weight <= 0.0:
        raise ValueError(f"sum of available weights must be > 0, got: {total_weight}")
    return {name: weights[name] / total_weight for name in available_components}


def _parse_dataset_objective(dataset_id: str, objective: ObjectiveResult) -> _DatasetObjective:
    attrs = objective.attrs
    if not isinstance(attrs, dict):
        raise TypeError(f"objective attrs for dataset={dataset_id} must be a dict")
    raw = _read_raw_map(attrs, dataset_id)
    counts = attrs.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"objective attrs.counts for dataset={dataset_id} must be a dict")
    order_count = _read_non_negative_int(counts, "order_count", dataset_id)
    cancel_order_count = _read_non_negative_int(counts, "cancel_order_count", dataset_id)
    if order_count <= 0:
        raise ValueError(f"objective counts.order_count must be > 0 for dataset={dataset_id}")
    return _DatasetObjective(
        dataset_id=dataset_id,
        raw=raw,
        order_count=order_count,
        cancel_order_count=cancel_order_count,
    )


def _read_raw_map(attrs: dict[str, Any], dataset_id: str) -> dict[str, float | None]:
    raw = attrs.get("raw")
    if not isinstance(raw, dict):
        raise ValueError(f"objective attrs.raw for dataset={dataset_id} must be a dict")
    result: dict[str, float | None] = {}
    for name in _COMPONENTS:
        value = raw.get(name)
        if value is None:
            result[name] = None
            continue
        result[name] = _as_non_negative_float(value, f"raw.{name}", dataset_id)
    return result


def _read_required_raw(raw: dict[str, float | None], name: str) -> float:
    value = raw.get(name)
    if value is None:
        raise ValueError(f"raw.{name} must be present")
    return value


def _read_loss_config(spec: ExperimentSpec) -> _LossConfig:
    params = spec.objective_config.get("params")
    if not isinstance(params, dict):
        raise ValueError("spec.objective_config.params must be a dict")
    weights = _read_component_map(params, "weights", allow_zero=True, must_be_positive=False)
    baseline = _read_component_map(params, "baseline", allow_zero=True, must_be_positive=False)
    eps = _read_eps_map(params)
    return _LossConfig(weights=weights, baseline=baseline, eps=eps)


def _read_component_map(
    params: dict[str, Any],
    key: str,
    *,
    allow_zero: bool,
    must_be_positive: bool,
) -> dict[str, float]:
    raw = params.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"spec.objective_config.params.{key} must be a dict")
    _raise_on_unknown_component_keys(raw, f"spec.objective_config.params.{key}")
    result: dict[str, float] = {}
    for name in _COMPONENTS:
        if name not in raw:
            raise ValueError(f"spec.objective_config.params.{key}.{name} is required")
        value = _as_non_negative_float(raw[name], f"{key}.{name}", "spec")
        if must_be_positive and value <= 0.0:
            raise ValueError(f"spec.objective_config.params.{key}.{name} must be > 0")
        if not allow_zero and value == 0.0:
            raise ValueError(f"spec.objective_config.params.{key}.{name} must be non-zero")
        result[name] = value
    return result


def _read_eps_map(params: dict[str, Any]) -> dict[str, float]:
    raw_eps = params.get("eps")
    if raw_eps is None:
        return dict(_DEFAULT_EPS)
    if not isinstance(raw_eps, dict):
        raise ValueError("spec.objective_config.params.eps must be a dict")
    _raise_on_unknown_component_keys(raw_eps, "spec.objective_config.params.eps")
    result: dict[str, float] = {}
    for name in _COMPONENTS:
        if name not in raw_eps:
            result[name] = _DEFAULT_EPS[name]
            continue
        value = _as_non_negative_float(raw_eps[name], f"eps.{name}", "spec")
        if value <= 0.0:
            raise ValueError(f"spec.objective_config.params.eps.{name} must be > 0")
        result[name] = value
    return result


def _read_non_negative_int(source: dict[str, Any], key: str, dataset_id: str) -> int:
    raw = source.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"objective {key} for dataset={dataset_id} must be an int")
    if raw < 0:
        raise ValueError(f"objective {key} for dataset={dataset_id} must be >= 0")
    return raw


def _as_non_negative_float(value: Any, field_name: str, dataset_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} for dataset={dataset_id} must be numeric")
    cast_value = float(value)
    if cast_value < 0.0:
        raise ValueError(f"{field_name} for dataset={dataset_id} must be >= 0")
    return cast_value


def _raise_on_unknown_component_keys(raw: dict[str, Any], field_name: str) -> None:
    unknown = sorted(str(key) for key in raw.keys() if key not in _COMPONENTS)
    if unknown:
        raise ValueError(f"{field_name} has unsupported components: {unknown}")
