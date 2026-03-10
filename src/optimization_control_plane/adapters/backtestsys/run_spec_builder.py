from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, RunSpec

_BACKTESTSYS_KIND = "backtestsys"
_REPLAY_STRATEGY_NAME = "ReplayStrategy_Impl"


@dataclass(frozen=True)
class BackTestSysRunSpecDefaults:
    repo_root: str
    base_config_path: str
    replay_order_file: str
    replay_cancel_file: str


class BackTestSysRunSpecBuilder:
    """Build BackTestSys RunSpec with fixed replay strategy."""

    def __init__(self, defaults: BackTestSysRunSpecDefaults) -> None:
        self._defaults = defaults

    def build(self, params: dict[str, object], spec: ExperimentSpec) -> RunSpec:
        config: dict[str, Any] = {
            "repo_root": self._defaults.repo_root,
            "base_config_path": self._defaults.base_config_path,
            "strategy": {
                "name": _REPLAY_STRATEGY_NAME,
                "order_file": self._defaults.replay_order_file,
                "cancel_file": self._defaults.replay_cancel_file,
            },
            "overrides": dict(params),
        }
        resources = spec.execution_config.get("default_resources", {})
        return RunSpec(kind=_BACKTESTSYS_KIND, config=config, resources=dict(resources))
