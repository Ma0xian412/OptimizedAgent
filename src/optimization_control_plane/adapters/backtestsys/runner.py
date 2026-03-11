from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from threading import Lock
from typing import Any

from optimization_control_plane.domain.models import ExecutionRequest, RunResult

_BACKTESTSYS_KIND = "backtestsys"
_REPLAY_STRATEGY_NAME = "ReplayStrategy_Impl"
_PATH_LOCK = Lock()
_RESULT_FIELDS = {
    "orderinfo_rows": ("OrderId", "SentTime", "Volume"),
    "doneinfo_rows": ("OrderId", "DoneTime"),
    "executiondetail_rows": ("OrderId", "RecvTick", "ExchTick", "Volume"),
    "cancelrequest_rows": ("OrderId", "CancelSentTime"),
}


@dataclass(frozen=True)
class BackTestSysRunConfig:
    repo_root: str
    base_config_path: str
    strategy_name: str
    replay_order_file: str
    replay_cancel_file: str
    overrides: dict[str, Any]


class BackTestSysRunner:
    """Bridge RunSpec to BackTestSys app.run() and convert to RunResult."""

    def run_request(self, request: ExecutionRequest) -> RunResult:
        run_cfg = self._parse_request(request)
        self._ensure_repo_on_sys_path(run_cfg.repo_root)
        runtime = self._load_runtime_modules()
        config = runtime["load_config"](run_cfg.base_config_path)
        self._apply_overrides(config, run_cfg.overrides)
        runtime_cfg = runtime["BacktestConfigFactory"]().create(config)
        runtime_cfg.strategy = runtime["ReplayStrategy_Impl"](
            name=run_cfg.strategy_name,
            order_file=run_cfg.replay_order_file,
            cancel_file=run_cfg.replay_cancel_file,
        )
        app = runtime["BacktestApp"](runtime_cfg)
        raw_result = app.run()
        done_count = len(raw_result.DoneInfo)
        execution_count = len(raw_result.ExecutionDetail)
        if done_count + execution_count == 0:
            raise ValueError("backtest result is empty: DoneInfo and ExecutionDetail are both zero")
        metrics: dict[str, Any] = {}
        self._merge_portfolio_metrics(metrics, app.last_context)
        diagnostics = {
            "run_key": request.run_key,
            "request_id": request.request_id,
            "strategy_name": run_cfg.strategy_name,
            "result": self._build_result_diagnostics(raw_result),
        }
        return RunResult(metrics=metrics, diagnostics=diagnostics, artifact_refs=[])

    @staticmethod
    def _parse_request(request: ExecutionRequest) -> BackTestSysRunConfig:
        if request.run_spec.kind != _BACKTESTSYS_KIND:
            raise ValueError(f"run_spec.kind must be '{_BACKTESTSYS_KIND}', got '{request.run_spec.kind}'")
        payload = request.run_spec.config
        repo_root = BackTestSysRunner._read_required_string(payload, "repo_root")
        base_config_path = BackTestSysRunner._read_required_string(payload, "base_config_path")
        strategy = payload.get("strategy")
        if not isinstance(strategy, dict):
            raise ValueError("run_spec.config.strategy must be a dict")
        strategy_name = BackTestSysRunner._read_required_string(strategy, "name")
        if strategy_name != _REPLAY_STRATEGY_NAME:
            raise ValueError(
                f"BackTestSys runner requires '{_REPLAY_STRATEGY_NAME}', got '{strategy_name}'"
            )
        replay_order_file = BackTestSysRunner._read_required_string(strategy, "order_file")
        replay_cancel_file = BackTestSysRunner._read_required_string(strategy, "cancel_file")
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("run_spec.config.overrides must be a dict")
        return BackTestSysRunConfig(
            repo_root=repo_root,
            base_config_path=base_config_path,
            strategy_name=strategy_name,
            replay_order_file=replay_order_file,
            replay_cancel_file=replay_cancel_file,
            overrides=dict(overrides),
        )

    @staticmethod
    def _read_required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"run_spec.config.{key} must be a non-empty string")
        return value

    @staticmethod
    def _ensure_repo_on_sys_path(repo_root: str) -> None:
        with _PATH_LOCK:
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)

    @staticmethod
    def _load_runtime_modules() -> dict[str, Any]:
        config_module = importlib.import_module("quant_framework.config")
        factory_module = importlib.import_module("quant_framework.adapters.factory")
        strategy_module = importlib.import_module("quant_framework.adapters.IStrategy")
        core_module = importlib.import_module("quant_framework.core")
        return {
            "load_config": getattr(config_module, "load_config"),
            "BacktestConfigFactory": getattr(factory_module, "BacktestConfigFactory"),
            "ReplayStrategy_Impl": getattr(strategy_module, "ReplayStrategy_Impl"),
            "BacktestApp": getattr(core_module, "BacktestApp"),
        }

    @staticmethod
    def _apply_overrides(config: object, overrides: dict[str, Any]) -> None:
        for key, value in overrides.items():
            BackTestSysRunner._set_path_value(config, key, value)

    @staticmethod
    def _set_path_value(target: object, key_path: str, value: Any) -> None:
        parts = key_path.split(".")
        cursor = target
        for part in parts[:-1]:
            if not hasattr(cursor, part):
                raise AttributeError(f"override path not found: {key_path}")
            cursor = getattr(cursor, part)
        leaf = parts[-1]
        if not hasattr(cursor, leaf):
            raise AttributeError(f"override path not found: {key_path}")
        setattr(cursor, leaf, value)

    @staticmethod
    def _merge_portfolio_metrics(metrics: dict[str, Any], context: object | None) -> None:
        if context is None:
            return
        oms = getattr(context, "oms", None)
        portfolio = getattr(oms, "portfolio", None)
        if portfolio is None:
            return
        metrics["final_cash"] = float(portfolio.cash)
        metrics["final_position"] = int(portfolio.position)
        metrics["final_realized_pnl"] = float(portfolio.realized_pnl)

    @staticmethod
    def _build_result_diagnostics(raw_result: object) -> dict[str, list[dict[str, Any]]]:
        return {
            "orderinfo_rows": BackTestSysRunner._rows_to_dicts(
                getattr(raw_result, "OrderInfo", ()),
                _RESULT_FIELDS["orderinfo_rows"],
            ),
            "doneinfo_rows": BackTestSysRunner._rows_to_dicts(
                getattr(raw_result, "DoneInfo", ()),
                _RESULT_FIELDS["doneinfo_rows"],
            ),
            "executiondetail_rows": BackTestSysRunner._rows_to_dicts(
                getattr(raw_result, "ExecutionDetail", ()),
                _RESULT_FIELDS["executiondetail_rows"],
            ),
            "cancelrequest_rows": BackTestSysRunner._rows_to_dicts(
                getattr(raw_result, "CancelRequest", ()),
                _RESULT_FIELDS["cancelrequest_rows"],
            ),
        }

    @staticmethod
    def _rows_to_dicts(rows: Any, prioritized_keys: tuple[str, ...]) -> list[dict[str, Any]]:
        return [BackTestSysRunner._row_to_dict(row, prioritized_keys) for row in rows]

    @staticmethod
    def _row_to_dict(row: Any, prioritized_keys: tuple[str, ...]) -> dict[str, Any]:
        if isinstance(row, dict):
            values = dict(row)
        elif hasattr(row, "__dict__"):
            values = dict(vars(row))
        else:
            raise TypeError(f"backtest result row must be dict-like, got {type(row).__name__}")
        output: dict[str, Any] = {}
        for key in prioritized_keys:
            if key in values:
                output[key] = values[key]
        for key, value in values.items():
            if key not in output:
                output[key] = value
        return output
