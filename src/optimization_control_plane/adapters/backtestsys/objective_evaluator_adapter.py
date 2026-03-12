from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_control_plane.domain.models import GroundTruthData, ObjectiveResult, RunResult

_LOSS_SCHEMA_VERSION = "backtest_loss_v1"
_DONE_INFO_TABLE = "DoneInfo"
_EXECUTION_DETAIL_TABLE = "ExecutionDetail"
_ORDER_INFO_TABLE = "OrderInfo"
_CANCEL_REQUEST_TABLE = "CancelRequest"
_STATE_FULL = "A"
_STATE_PARTIAL = "P"
_STATE_NONE = "N"
_CANCEL_SUCCESS_STATES = frozenset({_STATE_PARTIAL, _STATE_NONE})
_COMPONENT_NAMES = ("curve", "terminal", "cancel", "post")


@dataclass(frozen=True)
class _OrderKey:
    partition_day: int
    contract_id: int
    order_id: int
    machine_name: str


@dataclass(frozen=True)
class _OrderInfoRow:
    sent_time: int
    quantity: int


@dataclass(frozen=True)
class _DoneInfoRow:
    done_time: int
    state: str


@dataclass(frozen=True)
class _ExecutionRow:
    recv_tick: int
    volume: int


class BackTestObjectiveEvaluatorAdapter:
    """Evaluate one BackTestSys run into single-dataset raw objective metrics."""

    def evaluate(
        self,
        run_result: RunResult,
        spec: object,
        groundtruth: GroundTruthData,
    ) -> ObjectiveResult:
        del spec  # Evaluator is data-driven; config-level weighting is done by trial aggregator.
        sim_tables = _read_tables(
            payload=run_result.payload,
            required_tables=(
                _DONE_INFO_TABLE,
                _EXECUTION_DETAIL_TABLE,
                _ORDER_INFO_TABLE,
                _CANCEL_REQUEST_TABLE,
            ),
            payload_name="run_result.payload",
        )
        gt_tables = _read_tables(
            payload=groundtruth.payload,
            required_tables=(_DONE_INFO_TABLE, _EXECUTION_DETAIL_TABLE),
            payload_name="groundtruth.payload",
        )
        order_info_by_key = _index_order_info(sim_tables[_ORDER_INFO_TABLE])
        sim_done_by_key = _index_done_info(sim_tables[_DONE_INFO_TABLE], source_name="Sim")
        gt_done_by_key = _index_done_info(gt_tables[_DONE_INFO_TABLE], source_name="GT")
        evaluation_keys = tuple(sorted(
            set(order_info_by_key) & set(sim_done_by_key) & set(gt_done_by_key),
            key=_order_key_sort_key,
        ))
        if not evaluation_keys:
            raise ValueError("no evaluable orders found in intersection(OrderInfo, GT.DoneInfo, Sim.DoneInfo)")
        sim_exec_by_key = _index_execution_detail(sim_tables[_EXECUTION_DETAIL_TABLE], source_name="Sim")
        gt_exec_by_key = _index_execution_detail(gt_tables[_EXECUTION_DETAIL_TABLE], source_name="GT")
        cancel_time_by_key = _index_earliest_cancel_time(sim_tables[_CANCEL_REQUEST_TABLE])
        raw_metrics = _calculate_daily_raw_metrics(
            order_info_by_key=order_info_by_key,
            sim_done_by_key=sim_done_by_key,
            gt_done_by_key=gt_done_by_key,
            sim_exec_by_key=sim_exec_by_key,
            gt_exec_by_key=gt_exec_by_key,
            cancel_time_by_key=cancel_time_by_key,
            evaluation_keys=evaluation_keys,
        )
        attrs = {
            "value": _daily_intermediate_value(raw_metrics),
            "loss_schema_version": _LOSS_SCHEMA_VERSION,
            "raw": {
                "curve": raw_metrics.curve,
                "terminal": raw_metrics.terminal,
                "cancel": raw_metrics.cancel,
                "post": raw_metrics.post,
            },
            "counts": {
                "order_count": raw_metrics.order_count,
                "cancel_order_count": raw_metrics.cancel_order_count,
            },
            "availability": {
                "curve": True,
                "terminal": True,
                "cancel": raw_metrics.cancel is not None,
                "post": raw_metrics.post is not None,
            },
        }
        return ObjectiveResult(attrs=attrs, artifact_refs=_read_artifact_refs(run_result.payload))


@dataclass(frozen=True)
class _DailyRawMetrics:
    curve: float
    terminal: float
    cancel: float | None
    post: float | None
    order_count: int
    cancel_order_count: int


def _calculate_daily_raw_metrics(
    *,
    order_info_by_key: dict[_OrderKey, _OrderInfoRow],
    sim_done_by_key: dict[_OrderKey, _DoneInfoRow],
    gt_done_by_key: dict[_OrderKey, _DoneInfoRow],
    sim_exec_by_key: dict[_OrderKey, tuple[_ExecutionRow, ...]],
    gt_exec_by_key: dict[_OrderKey, tuple[_ExecutionRow, ...]],
    cancel_time_by_key: dict[_OrderKey, int],
    evaluation_keys: tuple[_OrderKey, ...],
) -> _DailyRawMetrics:
    curve_losses: list[float] = []
    terminal_losses: list[float] = []
    cancel_losses: list[float] = []
    post_losses: list[float] = []
    for key in evaluation_keys:
        order_info = order_info_by_key[key]
        sim_done = sim_done_by_key[key]
        gt_done = gt_done_by_key[key]
        sim_execs = sim_exec_by_key.get(key, ())
        gt_execs = gt_exec_by_key.get(key, ())
        curve_losses.append(_curve_loss(
            sent_time=order_info.sent_time,
            quantity=order_info.quantity,
            real_done_time=gt_done.done_time,
            sim_done_time=sim_done.done_time,
            real_execs=gt_execs,
            sim_execs=sim_execs,
        ))
        terminal_losses.append(_terminal_loss(
            quantity=order_info.quantity,
            real_execs=gt_execs,
            sim_execs=sim_execs,
        ))
        cancel_time = cancel_time_by_key.get(key)
        if cancel_time is None:
            continue
        cancel_losses.append(_cancel_loss(real_state=gt_done.state, sim_state=sim_done.state))
        post_losses.append(_post_loss(
            quantity=order_info.quantity,
            cancel_time=cancel_time,
            real_done_time=gt_done.done_time,
            sim_done_time=sim_done.done_time,
            real_execs=gt_execs,
            sim_execs=sim_execs,
        ))
    return _DailyRawMetrics(
        curve=_mean(curve_losses),
        terminal=_mean(terminal_losses),
        cancel=_mean(cancel_losses) if cancel_losses else None,
        post=_mean(post_losses) if post_losses else None,
        order_count=len(evaluation_keys),
        cancel_order_count=len(cancel_losses),
    )


def _curve_loss(
    *,
    sent_time: int,
    quantity: int,
    real_done_time: int,
    sim_done_time: int,
    real_execs: tuple[_ExecutionRow, ...],
    sim_execs: tuple[_ExecutionRow, ...],
) -> float:
    end_time = max(real_done_time, sim_done_time)
    if end_time <= sent_time:
        return 0.0
    real_deltas = _build_execution_delta_map(real_execs)
    sim_deltas = _build_execution_delta_map(sim_execs)
    breakpoints = _build_curve_breakpoints(sent_time, end_time, real_execs, sim_execs)
    cumulative_real = _cumulative_at_or_before(real_deltas, sent_time)
    cumulative_sim = _cumulative_at_or_before(sim_deltas, sent_time)
    area = 0.0
    for index in range(len(breakpoints) - 1):
        left = breakpoints[index]
        right = breakpoints[index + 1]
        area += abs(cumulative_real - cumulative_sim) * (right - left)
        cumulative_real += real_deltas.get(right, 0)
        cumulative_sim += sim_deltas.get(right, 0)
    return area / float(quantity)


def _terminal_loss(
    *,
    quantity: int,
    real_execs: tuple[_ExecutionRow, ...],
    sim_execs: tuple[_ExecutionRow, ...],
) -> float:
    real_total = _sum_execution_volume(real_execs)
    sim_total = _sum_execution_volume(sim_execs)
    return abs(real_total - sim_total) / float(quantity)


def _cancel_loss(*, real_state: str, sim_state: str) -> float:
    real_cancel_success = real_state in _CANCEL_SUCCESS_STATES
    sim_cancel_success = sim_state in _CANCEL_SUCCESS_STATES
    return 0.0 if real_cancel_success == sim_cancel_success else 1.0


def _post_loss(
    *,
    quantity: int,
    cancel_time: int,
    real_done_time: int,
    sim_done_time: int,
    real_execs: tuple[_ExecutionRow, ...],
    sim_execs: tuple[_ExecutionRow, ...],
) -> float:
    real_post = _sum_volume_in_window(real_execs, lower_exclusive=cancel_time, upper_inclusive=real_done_time)
    sim_post = _sum_volume_in_window(sim_execs, lower_exclusive=cancel_time, upper_inclusive=sim_done_time)
    return abs(real_post - sim_post) / float(quantity)


def _build_curve_breakpoints(
    sent_time: int,
    end_time: int,
    real_execs: tuple[_ExecutionRow, ...],
    sim_execs: tuple[_ExecutionRow, ...],
) -> tuple[int, ...]:
    points = {sent_time, end_time}
    for execution in real_execs:
        if sent_time < execution.recv_tick < end_time:
            points.add(execution.recv_tick)
    for execution in sim_execs:
        if sent_time < execution.recv_tick < end_time:
            points.add(execution.recv_tick)
    return tuple(sorted(points))


def _build_execution_delta_map(executions: tuple[_ExecutionRow, ...]) -> dict[int, int]:
    deltas: dict[int, int] = {}
    for execution in executions:
        deltas[execution.recv_tick] = deltas.get(execution.recv_tick, 0) + execution.volume
    return deltas


def _cumulative_at_or_before(deltas: dict[int, int], tick: int) -> int:
    cumulative = 0
    for recv_tick, volume in deltas.items():
        if recv_tick <= tick:
            cumulative += volume
    return cumulative


def _sum_execution_volume(executions: tuple[_ExecutionRow, ...]) -> int:
    return sum(execution.volume for execution in executions)


def _sum_volume_in_window(
    executions: tuple[_ExecutionRow, ...],
    *,
    lower_exclusive: int,
    upper_inclusive: int,
) -> int:
    return sum(
        execution.volume
        for execution in executions
        if lower_exclusive < execution.recv_tick <= upper_inclusive
    )


def _read_tables(
    *,
    payload: Any,
    required_tables: tuple[str, ...],
    payload_name: str,
) -> dict[str, list[dict[str, str]]]:
    if not isinstance(payload, dict):
        raise TypeError(f"{payload_name} must be a dict")
    result: dict[str, list[dict[str, str]]] = {}
    for table_name in required_tables:
        table_rows = payload.get(table_name)
        if not isinstance(table_rows, list):
            raise ValueError(f"{payload_name}.{table_name} must be a list of rows")
        validated_rows = [row for row in table_rows if isinstance(row, dict)]
        if len(validated_rows) != len(table_rows):
            raise ValueError(f"{payload_name}.{table_name} must only contain dict rows")
        result[table_name] = validated_rows
    return result


def _index_order_info(rows: list[dict[str, str]]) -> dict[_OrderKey, _OrderInfoRow]:
    indexed: dict[_OrderKey, _OrderInfoRow] = {}
    for row in rows:
        key = _build_order_key(row, table_name=_ORDER_INFO_TABLE)
        quantity = _read_positive_int(row, "Volume", _ORDER_INFO_TABLE)
        sent_time = _read_int(row, "SentTime", _ORDER_INFO_TABLE)
        _ensure_unique_order(indexed, key, _ORDER_INFO_TABLE)
        indexed[key] = _OrderInfoRow(sent_time=sent_time, quantity=quantity)
    return indexed


def _index_done_info(rows: list[dict[str, str]], *, source_name: str) -> dict[_OrderKey, _DoneInfoRow]:
    indexed: dict[_OrderKey, _DoneInfoRow] = {}
    table_name = f"{source_name}.{_DONE_INFO_TABLE}"
    for row in rows:
        key = _build_order_key(row, table_name=table_name)
        done_time = _read_int(row, "DoneTime", table_name)
        state = _read_done_state(row, table_name)
        _ensure_unique_order(indexed, key, table_name)
        indexed[key] = _DoneInfoRow(done_time=done_time, state=state)
    return indexed


def _index_execution_detail(
    rows: list[dict[str, str]],
    *,
    source_name: str,
) -> dict[_OrderKey, tuple[_ExecutionRow, ...]]:
    grouped: dict[_OrderKey, list[_ExecutionRow]] = {}
    table_name = f"{source_name}.{_EXECUTION_DETAIL_TABLE}"
    for row in rows:
        key = _build_order_key(row, table_name=table_name)
        execution = _ExecutionRow(
            recv_tick=_read_int(row, "RecvTick", table_name),
            volume=_read_positive_int(row, "Volume", table_name),
        )
        grouped.setdefault(key, []).append(execution)
    return {key: tuple(sorted(executions, key=lambda item: item.recv_tick)) for key, executions in grouped.items()}


def _index_earliest_cancel_time(rows: list[dict[str, str]]) -> dict[_OrderKey, int]:
    indexed: dict[_OrderKey, int] = {}
    for row in rows:
        key = _build_order_key(row, table_name=_CANCEL_REQUEST_TABLE)
        cancel_sent_time = _read_int(row, "CancelSentTime", _CANCEL_REQUEST_TABLE)
        previous_time = indexed.get(key)
        indexed[key] = cancel_sent_time if previous_time is None else min(previous_time, cancel_sent_time)
    return indexed


def _build_order_key(row: dict[str, str], *, table_name: str) -> _OrderKey:
    machine_name = _read_non_empty_str(row, "MachineName", table_name)
    return _OrderKey(
        partition_day=_read_int(row, "PartitionDay", table_name),
        contract_id=_read_int(row, "ContractId", table_name),
        order_id=_read_int(row, "OrderId", table_name),
        machine_name=machine_name,
    )


def _read_done_state(row: dict[str, str], table_name: str) -> str:
    state = _read_non_empty_str(row, "OrderTradeState", table_name)
    if state not in {_STATE_FULL, _STATE_PARTIAL, _STATE_NONE}:
        raise ValueError(f"{table_name}.OrderTradeState must be one of A/P/N, got: {state}")
    return state


def _read_non_empty_str(row: dict[str, str], key: str, table_name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{table_name}.{key} must be a non-empty string")
    return value.strip()


def _read_int(row: dict[str, str], key: str, table_name: str) -> int:
    raw_value = _read_non_empty_str(row, key, table_name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{table_name}.{key} must be an int, got: {raw_value}") from exc


def _read_positive_int(row: dict[str, str], key: str, table_name: str) -> int:
    value = _read_int(row, key, table_name)
    if value <= 0:
        raise ValueError(f"{table_name}.{key} must be > 0, got: {value}")
    return value


def _ensure_unique_order(indexed: dict[_OrderKey, Any], key: _OrderKey, table_name: str) -> None:
    if key in indexed:
        raise ValueError(f"{table_name} has duplicate order key: {key}")


def _daily_intermediate_value(metrics: _DailyRawMetrics) -> float:
    components: list[float] = [metrics.curve, metrics.terminal]
    if metrics.cancel is not None:
        components.append(metrics.cancel)
    if metrics.post is not None:
        components.append(metrics.post)
    return _mean(components)


def _read_artifact_refs(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw_refs = payload.get("artifact_refs")
    if raw_refs is None:
        return []
    if not isinstance(raw_refs, list):
        raise TypeError("run_result.payload.artifact_refs must be a list when provided")
    refs = [ref for ref in raw_refs if isinstance(ref, str) and ref]
    if len(refs) != len(raw_refs):
        raise TypeError("run_result.payload.artifact_refs must contain non-empty strings")
    return refs


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty list")
    return sum(values) / float(len(values))


def _order_key_sort_key(key: _OrderKey) -> tuple[int, int, int, str]:
    return (key.partition_day, key.contract_id, key.order_id, key.machine_name)
