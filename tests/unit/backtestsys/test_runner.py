from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from optimization_control_plane.adapters.backtestsys.runner import BackTestSysRunner
from optimization_control_plane.domain.models import ExecutionRequest, RunSpec


def test_runner_outputs_result_details_and_metrics_match_row_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_result = SimpleNamespace(
        OrderInfo=(
            SimpleNamespace(OrderId=101, SentTime=10, Volume=3, ContractId=1),
            SimpleNamespace(OrderId=102, SentTime=20, Volume=5, ContractId=1),
        ),
        DoneInfo=(
            SimpleNamespace(OrderId=101, DoneTime=35),
            SimpleNamespace(OrderId=102, DoneTime=45),
        ),
        ExecutionDetail=(
            SimpleNamespace(OrderId=101, RecvTick=12, ExchTick=13, Volume=1),
            SimpleNamespace(OrderId=102, RecvTick=22, ExchTick=23, Volume=2),
        ),
        CancelRequest=(SimpleNamespace(OrderId=101, CancelSentTime=30),),
    )
    _patch_runtime(monkeypatch, raw_result)
    result = BackTestSysRunner().run_request(_make_request())

    diagnostics_result = result.diagnostics["result"]
    assert set(diagnostics_result.keys()) == {
        "orderinfo_rows",
        "doneinfo_rows",
        "executiondetail_rows",
        "cancelrequest_rows",
    }
    assert diagnostics_result["orderinfo_rows"][0]["OrderId"] == 101
    assert diagnostics_result["orderinfo_rows"][0]["SentTime"] == 10
    assert diagnostics_result["orderinfo_rows"][0]["Volume"] == 3
    assert diagnostics_result["doneinfo_rows"][0]["DoneTime"] == 35
    assert diagnostics_result["executiondetail_rows"][0]["RecvTick"] == 12
    assert diagnostics_result["executiondetail_rows"][0]["ExchTick"] == 13
    assert diagnostics_result["cancelrequest_rows"][0]["CancelSentTime"] == 30

    assert result.metrics["orderinfo_count"] == len(diagnostics_result["orderinfo_rows"])
    assert result.metrics["doneinfo_count"] == len(diagnostics_result["doneinfo_rows"])
    assert result.metrics["executiondetail_count"] == len(diagnostics_result["executiondetail_rows"])
    assert result.metrics["cancelrequest_count"] == len(diagnostics_result["cancelrequest_rows"])


def test_runner_raises_on_empty_done_and_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_result = SimpleNamespace(OrderInfo=(), DoneInfo=(), ExecutionDetail=(), CancelRequest=())
    _patch_runtime(monkeypatch, raw_result)
    with pytest.raises(ValueError, match="DoneInfo and ExecutionDetail are both zero"):
        BackTestSysRunner().run_request(_make_request())


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, raw_result: Any) -> None:
    runtime = _make_runtime(raw_result)
    monkeypatch.setattr(
        BackTestSysRunner,
        "_ensure_repo_on_sys_path",
        staticmethod(lambda repo_root: None),
    )
    monkeypatch.setattr(
        BackTestSysRunner,
        "_load_runtime_modules",
        staticmethod(lambda: runtime),
    )


def _make_runtime(raw_result: Any) -> dict[str, Any]:
    class _ConfigFactory:
        def create(self, config: object) -> object:
            return SimpleNamespace(strategy=None)

    class _ReplayStrategy:
        def __init__(self, name: str, order_file: str, cancel_file: str) -> None:
            self.name = name
            self.order_file = order_file
            self.cancel_file = cancel_file

    class _BacktestApp:
        def __init__(self, runtime_cfg: object) -> None:
            self.runtime_cfg = runtime_cfg
            self.last_context = None

        def run(self) -> object:
            return raw_result

    return {
        "load_config": lambda _: SimpleNamespace(),
        "BacktestConfigFactory": _ConfigFactory,
        "ReplayStrategy_Impl": _ReplayStrategy,
        "BacktestApp": _BacktestApp,
    }


def _make_request() -> ExecutionRequest:
    return ExecutionRequest(
        request_id="req-1",
        trial_id="trial-1",
        run_key="run-1",
        objective_key="obj-1",
        cohort_id=None,
        priority=0,
        run_spec=RunSpec(
            kind="backtestsys",
            config={
                "repo_root": "/workspace/BackTestSys",
                "base_config_path": "/workspace/BackTestSys/config.xml",
                "strategy": {
                    "name": "ReplayStrategy_Impl",
                    "order_file": "/tmp/orders.csv",
                    "cancel_file": "/tmp/cancels.csv",
                },
                "overrides": {},
            },
            resources={},
        ),
    )
