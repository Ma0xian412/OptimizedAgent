from __future__ import annotations

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestSearchSpaceAdapter
from tests.conftest import make_spec


class RecordingCtx:
    def __init__(self) -> None:
        self.float_calls: list[tuple[str, float, float]] = []
        self.int_calls: list[tuple[str, int, int]] = []
        self.attrs: dict[str, object] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.int_calls.append((name, low, high))
        return high

    def suggest_float(self, name: str, low: float, high: float) -> float:
        self.float_calls.append((name, low, high))
        return (low + high) / 2.0

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        return choices[0]

    def set_user_attr(self, key: str, val: object) -> None:
        self.attrs[key] = val

    def report(self, value: float, step: int) -> None:
        return None

    def should_prune(self) -> bool:
        return False


class TestBackTestSearchSpaceAdapter:
    def test_sample_four_backtest_parameters(self) -> None:
        adapter = BackTestSearchSpaceAdapter()
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {},
                "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
                "sampler": {"type": "random", "seed": 42},
                "pruner": {"type": "nop"},
                "backtest_search_space": {
                    "time_scale_lambda": {"low": -0.5, "high": 0.5},
                    "cancel_bias_k": {"low": -1.0, "high": 1.0},
                    "delay_in": {"low": 0, "high": 1000000},
                    "delay_out": {"low": 1000, "high": 2000},
                },
            }
        )
        ctx = RecordingCtx()

        sampled = adapter.sample(ctx, spec)

        assert sampled == {
            "time_scale_lambda": 0.0,
            "cancel_bias_k": 0.0,
            "delay_in": 1000000,
            "delay_out": 2000,
        }
        assert ctx.float_calls == [
            ("time_scale_lambda", -0.5, 0.5),
            ("cancel_bias_k", -1.0, 1.0),
        ]
        assert ctx.int_calls == [
            ("delay_in", 0, 1000000),
            ("delay_out", 1000, 2000),
        ]
        assert ctx.attrs["backtest_config_patch"] == sampled

    def test_missing_search_space_config_raises(self) -> None:
        adapter = BackTestSearchSpaceAdapter()
        spec = make_spec()

        with pytest.raises(
            ValueError,
            match="spec.objective_config.backtest_search_space must be a dict",
        ):
            adapter.sample(RecordingCtx(), spec)

    def test_invalid_integer_range_raises(self) -> None:
        adapter = BackTestSearchSpaceAdapter()
        spec = make_spec(
            objective_config={
                "name": "loss",
                "version": "v1",
                "direction": "minimize",
                "params": {},
                "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
                "sampler": {"type": "random", "seed": 42},
                "pruner": {"type": "nop"},
                "backtest_search_space": {
                    "time_scale_lambda": {"low": -0.5, "high": 0.5},
                    "cancel_bias_k": {"low": -1.0, "high": 1.0},
                    "delay_in": {"low": 0.0, "high": 1000000},
                    "delay_out": {"low": 0, "high": 1000},
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="backtest_search_space.delay_in low/high must be int",
        ):
            adapter.sample(RecordingCtx(), spec)
