from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext

_SEARCH_SPACE_KEY = "backtest_search_space"
_ATTR_KEY = "backtest_config_patch"


class BackTestSearchSpaceAdapter:
    """Sample BackTestSys config parameters from trial context."""

    def sample(
        self,
        ctx: TrialContext,
        spec: ExperimentSpec,
    ) -> dict[str, object]:
        space = _read_search_space(spec)
        time_scale_lambda = ctx.suggest_float(
            "time_scale_lambda",
            *_read_float_range(space, "time_scale_lambda"),
        )
        cancel_bias_k = ctx.suggest_float(
            "cancel_bias_k",
            *_read_float_range(space, "cancel_bias_k"),
        )
        delay_in = ctx.suggest_int("delay_in", *_read_int_range(space, "delay_in"))
        delay_out = ctx.suggest_int("delay_out", *_read_int_range(space, "delay_out"))

        params: dict[str, object] = {
            "time_scale_lambda": time_scale_lambda,
            "cancel_bias_k": cancel_bias_k,
            "delay_in": delay_in,
            "delay_out": delay_out,
        }
        ctx.set_user_attr(_ATTR_KEY, params)
        return params


def _read_search_space(spec: ExperimentSpec) -> dict[str, Any]:
    raw = spec.objective_config.get(_SEARCH_SPACE_KEY)
    if not isinstance(raw, dict):
        raise ValueError(
            f"spec.objective_config.{_SEARCH_SPACE_KEY} must be a dict"
        )
    return raw


def _read_float_range(space: dict[str, Any], key: str) -> tuple[float, float]:
    item = space.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{_SEARCH_SPACE_KEY}.{key} must be a dict")
    low = item.get("low")
    high = item.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"{_SEARCH_SPACE_KEY}.{key} low/high must be numeric")
    low_val = float(low)
    high_val = float(high)
    if low_val > high_val:
        raise ValueError(f"{_SEARCH_SPACE_KEY}.{key} low must be <= high")
    return low_val, high_val


def _read_int_range(space: dict[str, Any], key: str) -> tuple[int, int]:
    item = space.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{_SEARCH_SPACE_KEY}.{key} must be a dict")
    low = item.get("low")
    high = item.get("high")
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError(f"{_SEARCH_SPACE_KEY}.{key} low/high must be int")
    if low > high:
        raise ValueError(f"{_SEARCH_SPACE_KEY}.{key} low must be <= high")
    return low, high
