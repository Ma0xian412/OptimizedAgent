from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext

_TYPE_INT = "int"
_TYPE_FLOAT = "float"
_TYPE_CATEGORICAL = "categorical"


@dataclass(frozen=True)
class SearchParam:
    name: str
    param_type: str
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[str, ...] = ()


class BackTestSysSearchSpace:
    """Configured search space adapter for BackTestSys optimization."""

    def __init__(self, params: list[SearchParam]) -> None:
        if not params:
            raise ValueError("search space params cannot be empty")
        self._params = list(params)

    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]:
        sampled: dict[str, object] = {}
        for param in self._params:
            sampled[param.name] = self._sample_one(ctx, param)
        return sampled

    @staticmethod
    def _sample_one(ctx: TrialContext, param: SearchParam) -> object:
        if param.param_type == _TYPE_INT:
            low, high = _as_int_bounds(param)
            return ctx.suggest_int(param.name, low, high)
        if param.param_type == _TYPE_FLOAT:
            low, high = _as_float_bounds(param)
            return ctx.suggest_float(param.name, low, high)
        if param.param_type == _TYPE_CATEGORICAL:
            if not param.choices:
                raise ValueError(f"categorical param has no choices: {param.name}")
            return ctx.suggest_categorical(param.name, list(param.choices))
        raise ValueError(f"unsupported param type: {param.param_type}")


def _as_int_bounds(param: SearchParam) -> tuple[int, int]:
    if param.low is None or param.high is None:
        raise ValueError(f"int param missing bounds: {param.name}")
    return int(param.low), int(param.high)


def _as_float_bounds(param: SearchParam) -> tuple[float, float]:
    if param.low is None or param.high is None:
        raise ValueError(f"float param missing bounds: {param.name}")
    return float(param.low), float(param.high)
