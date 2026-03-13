from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_LOSS_SCHEMA_VERSION = "backtest_loss_v2"
_DONE_INFO_TABLE = "DoneInfo"
_EXECUTION_DETAIL_TABLE = "ExecutionDetail"
_ORDER_INFO_TABLE = "OrderInfo"
_CANCEL_REQUEST_TABLE = "CancelRequest"
_STATE_FULL = "A"
_STATE_PARTIAL = "P"
_STATE_NONE = "N"


@dataclass(frozen=True)
class OrderKey:
    partition_day: int
    contract_id: int
    order_id: int
    machine_name: str


@dataclass(frozen=True)
class OrderInfoRow:
    sent_time: int
    quantity: int


@dataclass(frozen=True)
class DoneInfoRow:
    done_time: int
    state: str


@dataclass(frozen=True)
class ExecutionRow:
    recv_tick: int
    volume: int


def read_tables(
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


def index_order_info(rows: list[dict[str, str]]) -> dict[OrderKey, OrderInfoRow]:
    indexed: dict[OrderKey, OrderInfoRow] = {}
    for row in rows:
        key = build_order_key(row, table_name=_ORDER_INFO_TABLE)
        quantity = read_positive_int(row, "Volume", _ORDER_INFO_TABLE)
        sent_time = read_int(row, "SentTime", _ORDER_INFO_TABLE)
        ensure_unique_order(indexed, key, _ORDER_INFO_TABLE)
        indexed[key] = OrderInfoRow(sent_time=sent_time, quantity=quantity)
    return indexed


def index_done_info(rows: list[dict[str, str]], *, source_name: str) -> dict[OrderKey, DoneInfoRow]:
    indexed: dict[OrderKey, DoneInfoRow] = {}
    table_name = f"{source_name}.{_DONE_INFO_TABLE}"
    for row in rows:
        key = build_order_key(row, table_name=table_name)
        done_time = read_int(row, "DoneTime", table_name)
        state = read_done_state(row, table_name)
        ensure_unique_order(indexed, key, table_name)
        indexed[key] = DoneInfoRow(done_time=done_time, state=state)
    return indexed


def index_execution_detail(
    rows: list[dict[str, str]],
    *,
    source_name: str,
) -> dict[OrderKey, tuple[ExecutionRow, ...]]:
    grouped: dict[OrderKey, list[ExecutionRow]] = {}
    table_name = f"{source_name}.{_EXECUTION_DETAIL_TABLE}"
    for row in rows:
        key = build_order_key(row, table_name=table_name)
        execution = ExecutionRow(
            recv_tick=read_int(row, "RecvTick", table_name),
            volume=read_positive_int(row, "Volume", table_name),
        )
        grouped.setdefault(key, []).append(execution)
    return {key: tuple(sorted(executions, key=lambda item: item.recv_tick)) for key, executions in grouped.items()}


def index_earliest_cancel_time(rows: list[dict[str, str]]) -> dict[OrderKey, int]:
    indexed: dict[OrderKey, int] = {}
    for row in rows:
        key = build_order_key(row, table_name=_CANCEL_REQUEST_TABLE)
        cancel_sent_time = read_int(row, "CancelSentTime", _CANCEL_REQUEST_TABLE)
        previous_time = indexed.get(key)
        indexed[key] = cancel_sent_time if previous_time is None else min(previous_time, cancel_sent_time)
    return indexed


def build_order_key(row: dict[str, str], *, table_name: str) -> OrderKey:
    machine_name = read_non_empty_str(row, "MachineName", table_name)
    return OrderKey(
        partition_day=read_int(row, "PartitionDay", table_name),
        contract_id=read_int(row, "ContractId", table_name),
        order_id=read_int(row, "OrderId", table_name),
        machine_name=machine_name,
    )


def read_done_state(row: dict[str, str], table_name: str) -> str:
    state = read_non_empty_str(row, "OrderTradeState", table_name)
    if state not in {_STATE_FULL, _STATE_PARTIAL, _STATE_NONE}:
        raise ValueError(f"{table_name}.OrderTradeState must be one of A/P/N, got: {state}")
    return state


def read_non_empty_str(row: dict[str, str], key: str, table_name: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{table_name}.{key} must be a non-empty string")
    return value.strip()


def read_int(row: dict[str, str], key: str, table_name: str) -> int:
    raw_value = read_non_empty_str(row, key, table_name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{table_name}.{key} must be an int, got: {raw_value}") from exc


def read_positive_int(row: dict[str, str], key: str, table_name: str) -> int:
    value = read_int(row, key, table_name)
    if value <= 0:
        raise ValueError(f"{table_name}.{key} must be > 0, got: {value}")
    return value


def ensure_unique_order(indexed: dict[OrderKey, Any], key: OrderKey, table_name: str) -> None:
    if key in indexed:
        raise ValueError(f"{table_name} has duplicate order key: {key}")


def read_artifact_refs(payload: Any) -> list[str]:
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


def order_key_sort_key(key: OrderKey) -> tuple[int, int, int, str]:
    return (key.partition_day, key.contract_id, key.order_id, key.machine_name)
