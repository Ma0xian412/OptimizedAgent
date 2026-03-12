from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext

_SEARCH_SPACE_KEY = "backtest_search_space"
_FIXED_PARAMS_KEY = "backtest_fixed_params"
_ATTR_KEY = "backtest_config_patch"


class BackTestDelaySearchSpaceAdapter:
    """Sample one delay and keep two core params fixed."""

    def sample(
        self,
        ctx: TrialContext,
        spec: ExperimentSpec,
    ) -> dict[str, object]:
        space = _read_search_space(spec)
        fixed = _read_fixed_params(spec)
        delay = ctx.suggest_int("delay", *_read_int_range(space, "delay"))
        params: dict[str, object] = {
            "time_scale_lambda": _read_required_float(
                fixed,
                key="time_scale_lambda",
                owner=_FIXED_PARAMS_KEY,
            ),
            "cancel_bias_k": _read_required_float(
                fixed,
                key="cancel_bias_k",
                owner=_FIXED_PARAMS_KEY,
            ),
            "delay_in": delay,
            "delay_out": delay,
        }
        ctx.set_user_attr(_ATTR_KEY, params)
        return params


class BackTestCoreParamsSearchSpaceAdapter:
    """Sample two core params and keep one delay fixed."""

    def sample(
        self,
        ctx: TrialContext,
        spec: ExperimentSpec,
    ) -> dict[str, object]:
        space = _read_search_space(spec)
        fixed = _read_fixed_params(spec)
        time_scale_lambda = ctx.suggest_float(
            "time_scale_lambda",
            *_read_float_range(space, "time_scale_lambda"),
        )
        cancel_bias_k = ctx.suggest_float(
            "cancel_bias_k",
            *_read_float_range(space, "cancel_bias_k"),
        )
        delay = _read_required_int(fixed, key="delay", owner=_FIXED_PARAMS_KEY)
        params: dict[str, object] = {
            "time_scale_lambda": time_scale_lambda,
            "cancel_bias_k": cancel_bias_k,
            "delay_in": delay,
            "delay_out": delay,
        }
        ctx.set_user_attr(_ATTR_KEY, params)
        return params


def _read_search_space(spec: ExperimentSpec) -> dict[str, Any]:
    raw = spec.objective_config.get(_SEARCH_SPACE_KEY)
    if not isinstance(raw, dict):
        raise ValueError(f"spec.objective_config.{_SEARCH_SPACE_KEY} must be a dict")
    return raw


def _read_fixed_params(spec: ExperimentSpec) -> dict[str, Any]:
    raw = spec.objective_config.get(_FIXED_PARAMS_KEY)
    if not isinstance(raw, dict):
        raise ValueError(f"spec.objective_config.{_FIXED_PARAMS_KEY} must be a dict")
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


def _read_required_float(source: dict[str, Any], *, key: str, owner: str) -> float:
    raw = source.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{owner}.{key} must be numeric")
    return float(raw)


def _read_required_int(source: dict[str, Any], *, key: str, owner: str) -> int:
    raw = source.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{owner}.{key} must be int")
    return raw
