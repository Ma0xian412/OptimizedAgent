from __future__ import annotations

from optimization_control_plane.adapters.backtestsys import (
    BackTestSysRunSpecBuilder,
    BackTestSysRunSpecDefaults,
)
from tests.conftest import make_spec


def test_run_spec_builder_emits_replay_strategy_payload() -> None:
    defaults = BackTestSysRunSpecDefaults(
        repo_root="/workspace/BackTestSys",
        base_config_path="/workspace/BackTestSys/config.xml",
        replay_order_file="/tmp/orders.csv",
        replay_cancel_file="/tmp/cancels.csv",
    )
    builder = BackTestSysRunSpecBuilder(defaults)
    spec = make_spec()
    run_spec = builder.build({"tape.epsilon": 1.5}, spec)
    assert run_spec.kind == "backtestsys"
    assert run_spec.config["strategy"]["name"] == "ReplayStrategy_Impl"
    assert run_spec.config["overrides"]["tape.epsilon"] == 1.5
