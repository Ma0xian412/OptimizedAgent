from __future__ import annotations

from typing import Any

import optuna


class OptunaTrialContext:
    """Wraps a live Optuna Trial to satisfy the TrialContext protocol."""

    def __init__(self, trial: optuna.trial.Trial) -> None:
        self._trial = trial

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self._trial.suggest_int(name, low, high)

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return self._trial.suggest_float(name, low, high)

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self._trial.suggest_categorical(name, choices)

    def set_user_attr(self, key: str, val: Any) -> None:
        self._trial.set_user_attr(key, val)

    def report(self, value: float, step: int) -> None:
        self._trial.report(value, step)

    def should_prune(self) -> bool:
        return self._trial.should_prune()
