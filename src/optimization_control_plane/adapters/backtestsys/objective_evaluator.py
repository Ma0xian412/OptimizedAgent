from __future__ import annotations

from typing import Any

from optimization_control_plane.adapters.backtestsys.groundtruth_adapter import (
    BackTestSysGroundTruthAdapter,
)
from optimization_control_plane.adapters.backtestsys.loss_components import (
    ALL_COMPONENTS,
    DEFAULT_EPS,
    build_order_profiles,
    normalized_weights,
    read_component_value,
    read_result_tables,
    require_float,
)
from optimization_control_plane.adapters.backtestsys.loss_metrics import compute_raw_components
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ObjectiveResult,
    RunResult,
)


class BackTestSysCountDiffEvaluator:
    """BackTestSys loss evaluator with curve/terminal/cancel/post components."""

    def __init__(
        self,
        groundtruth_adapter: BackTestSysGroundTruthAdapter,
        groundtruth_dir: str | None = None,
    ) -> None:
        self._groundtruth_adapter = groundtruth_adapter
        self._groundtruth_dir = groundtruth_dir
        self._base_loss: float | None = None
        self._base_attrs: dict[str, Any] = {}

    def evaluate(self, run_result: RunResult, spec: ExperimentSpec) -> ObjectiveResult:
        gt_dir = self._groundtruth_dir or self._resolve_groundtruth_dir(spec)
        gt = self._groundtruth_adapter.load(gt_dir)
        tables = read_result_tables(run_result.diagnostics)
        order_profiles = build_order_profiles(
            order_rows=tables["orderinfo_rows"],
            run_done_rows=tables["doneinfo_rows"],
            run_execution_rows=tables["executiondetail_rows"],
            cancel_rows=tables["cancelrequest_rows"],
            gt_done_rows=gt.doneinfo.rows,
            gt_execution_rows=gt.executiondetail.rows,
        )
        if not order_profiles:
            raise ValueError("no evaluable orders: orderinfo_rows is empty")
        raw_components, available_components, order_stats = compute_raw_components(order_profiles)
        normalized_components, baseline_components_used = self._normalize_components(
            raw_components=raw_components,
            available_components=available_components,
            objective_config=spec.objective_config,
        )
        weights_used = normalized_weights(spec.objective_config, available_components)
        loss = sum(normalized_components[name] * weights_used[name] for name in available_components)
        attrs = {
            "objective_name": "backtestsys_curve_terminal_cancel_post",
            "groundtruth_dir": gt_dir,
            "raw_components": raw_components,
            "normalized_components": normalized_components,
            "weights_used": weights_used,
            "available_components": list(available_components),
            "order_count": order_stats["order_count"],
            "cancel_order_count": order_stats["cancel_order_count"],
        }
        if self._base_loss is not None:
            attrs["base_loss"] = self._base_loss
            attrs["baseline_components_used"] = baseline_components_used
            attrs["base_attrs"] = dict(self._base_attrs)
        return ObjectiveResult(
            value=loss,
            attrs=attrs,
            artifact_refs=list(run_result.artifact_refs),
        )

    @staticmethod
    def _resolve_groundtruth_dir(spec: ExperimentSpec) -> str:
        groundtruth_cfg = spec.objective_config.get("groundtruth")
        if not isinstance(groundtruth_cfg, dict):
            raise ValueError("spec.objective_config.groundtruth must be a dict")
        gt_dir = groundtruth_cfg.get("dir")
        if not isinstance(gt_dir, str) or not gt_dir:
            raise ValueError("spec.objective_config.groundtruth.dir must be a non-empty string")
        return gt_dir

    def set_base_loss(self, loss: float, attrs: dict[str, Any] | None = None) -> None:
        self._base_loss = float(loss)
        self._base_attrs = dict(attrs or {})

    def _normalize_components(
        self,
        raw_components: dict[str, float | None],
        available_components: tuple[str, ...],
        objective_config: dict[str, Any],
    ) -> tuple[dict[str, float | None], bool]:
        eps_cfg = objective_config.get("eps", {})
        if not isinstance(eps_cfg, dict):
            raise ValueError("spec.objective_config.eps must be a dict")
        normalized: dict[str, float | None] = {name: None for name in ALL_COMPONENTS}
        if self._base_loss is None:
            for name in available_components:
                normalized[name] = require_float(raw_components[name], f"raw_components.{name}")
            return normalized, False
        baseline_components = self._read_baseline_components_if_present()
        for name in available_components:
            eps = read_component_value(eps_cfg, name, DEFAULT_EPS, min_value=0.0)
            raw = require_float(raw_components[name], f"raw_components.{name}")
            denominator = self._base_loss
            if baseline_components is not None:
                denominator = read_component_value(baseline_components, name, None, min_value=0.0)
            normalized[name] = raw / (denominator + eps)
        return normalized, baseline_components is not None

    def _read_baseline_components_if_present(self) -> dict[str, Any] | None:
        value = self._base_attrs.get("baseline_components")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("base attrs baseline_components must be a dict")
        return value
