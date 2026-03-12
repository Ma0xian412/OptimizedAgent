from __future__ import annotations

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestTrialResultAggregatorAdapter
from optimization_control_plane.domain.models import ObjectiveResult
from tests.conftest import make_spec


def _build_objective_result(
    *,
    curve: float,
    terminal: float,
    cancel: float | None,
    post: float | None,
    order_count: int,
    cancel_order_count: int,
) -> ObjectiveResult:
    return ObjectiveResult(
        attrs={
            "value": 0.0,
            "raw": {
                "curve": curve,
                "terminal": terminal,
                "cancel": cancel,
                "post": post,
            },
            "counts": {
                "order_count": order_count,
                "cancel_order_count": cancel_order_count,
            },
        },
        artifact_refs=[],
    )


class TestBackTestTrialResultAggregatorAdapter:
    def test_aggregate_returns_weighted_normalized_value(self) -> None:
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {
                    "weights": {"curve": 0.5, "terminal": 0.25, "cancel": 0.1, "post": 0.15},
                    "baseline": {"curve": 3.0, "terminal": 4.0, "cancel": 2.0, "post": 1.0},
                    "eps": {"curve": 1e-12, "terminal": 1e-12, "cancel": 1e-12, "post": 1e-12},
                },
            }
        )
        results = [
            ("d1", _build_objective_result(
                curve=2.0,
                terminal=1.0,
                cancel=0.5,
                post=0.2,
                order_count=10,
                cancel_order_count=2,
            )),
            ("d2", _build_objective_result(
                curve=4.0,
                terminal=3.0,
                cancel=1.0,
                post=0.6,
                order_count=8,
                cancel_order_count=4,
            )),
        ]

        aggregated = BackTestTrialResultAggregatorAdapter().aggregate(results, spec)

        assert aggregated.attrs["raw"]["curve"] == pytest.approx(3.0)
        assert aggregated.attrs["raw"]["terminal"] == pytest.approx(2.0)
        assert aggregated.attrs["raw"]["cancel"] == pytest.approx(5.0 / 6.0)
        assert aggregated.attrs["raw"]["post"] == pytest.approx(14.0 / 30.0)
        assert aggregated.attrs["normalized"]["curve"] == pytest.approx(1.0)
        assert aggregated.attrs["normalized"]["terminal"] == pytest.approx(0.5)
        assert aggregated.attrs["normalized"]["cancel"] == pytest.approx(5.0 / 12.0)
        assert aggregated.attrs["normalized"]["post"] == pytest.approx(14.0 / 30.0)
        assert aggregated.value == pytest.approx(0.7366666666666667)

    def test_aggregate_reweights_when_cancel_components_unavailable(self) -> None:
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {
                    "weights": {"curve": 0.5, "terminal": 0.25, "cancel": 0.1, "post": 0.15},
                    "baseline": {"curve": 3.0, "terminal": 2.0, "cancel": 1.0, "post": 1.0},
                    "eps": {"curve": 1e-12, "terminal": 1e-12, "cancel": 1e-12, "post": 1e-12},
                },
            }
        )
        results = [
            ("d1", _build_objective_result(
                curve=2.0,
                terminal=1.0,
                cancel=None,
                post=None,
                order_count=10,
                cancel_order_count=0,
            )),
            ("d2", _build_objective_result(
                curve=4.0,
                terminal=3.0,
                cancel=None,
                post=None,
                order_count=10,
                cancel_order_count=0,
            )),
        ]

        aggregated = BackTestTrialResultAggregatorAdapter().aggregate(results, spec)

        assert aggregated.attrs["available_components"] == ["curve", "terminal"]
        assert aggregated.attrs["effective_weights"]["curve"] == pytest.approx(2.0 / 3.0)
        assert aggregated.attrs["effective_weights"]["terminal"] == pytest.approx(1.0 / 3.0)
        assert aggregated.value == pytest.approx(1.0)

    def test_aggregate_raises_when_baseline_missing(self) -> None:
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {
                    "weights": {"curve": 0.5, "terminal": 0.5, "cancel": 0.0, "post": 0.0},
                },
            }
        )
        results = [
            ("d1", _build_objective_result(
                curve=1.0,
                terminal=1.0,
                cancel=None,
                post=None,
                order_count=1,
                cancel_order_count=0,
            )),
        ]

        with pytest.raises(ValueError, match="params.baseline"):
            BackTestTrialResultAggregatorAdapter().aggregate(results, spec)
