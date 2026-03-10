from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization_control_plane.adapters.backtestsys.loss_metrics import ALL_COMPONENTS, OrderProfile

DEFAULT_WEIGHT = 1.0
DEFAULT_EPS = 1e-12
REQUIRED_RESULT_KEYS = (
    "orderinfo_rows",
    "doneinfo_rows",
    "executiondetail_rows",
    "cancelrequest_rows",
)
ORDER_ID_KEYS = ("OrderId", "order_id")
VOLUME_KEYS = ("Volume", "volume", "qty", "Qty")
SENT_TIME_KEYS = ("SentTime", "sent_time")
DONE_TIME_KEYS = ("DoneTime", "done_time")
CXL_TIME_KEYS = ("CancelSentTime", "cancel_sent_time")
EXEC_TIME_KEYS = ("RecvTick", "recv_tick", "ExchTick", "exch_tick")


@dataclass(frozen=True)
class OrderDefinition:
    quantity: float
    sent_time: int


def read_result_tables(diagnostics: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    result = diagnostics.get("result")
    if not isinstance(result, dict):
        raise ValueError("run_result.diagnostics.result must be a dict")
    tables: dict[str, tuple[Any, ...]] = {}
    for key in REQUIRED_RESULT_KEYS:
        rows = result.get(key)
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"run_result.diagnostics.result.{key} must be a list")
        tables[key] = tuple(rows)
    return tables


def build_order_profiles(
    order_rows: tuple[Any, ...],
    run_done_rows: tuple[Any, ...],
    run_execution_rows: tuple[Any, ...],
    cancel_rows: tuple[Any, ...],
    gt_done_rows: tuple[dict[str, Any], ...],
    gt_execution_rows: tuple[dict[str, Any], ...],
) -> dict[int, OrderProfile]:
    order_defs = _read_order_definitions(order_rows)
    run_done_times = _read_done_times(run_done_rows)
    gt_done_times = _read_done_times(gt_done_rows)
    run_events = _read_execution_events(run_execution_rows)
    gt_events = _read_execution_events(gt_execution_rows)
    cancel_times = _read_cancel_times(cancel_rows)
    profiles: dict[int, OrderProfile] = {}
    for order_id, order_def in order_defs.items():
        current_run_events = run_events.get(order_id, tuple())
        current_gt_events = gt_events.get(order_id, tuple())
        run_done_time = run_done_times.get(order_id, _last_event_time(current_run_events, order_def.sent_time))
        gt_done_time = gt_done_times.get(order_id, _last_event_time(current_gt_events, order_def.sent_time))
        profiles[order_id] = OrderProfile(
            quantity=order_def.quantity,
            sent_time=order_def.sent_time,
            run_done_time=run_done_time,
            gt_done_time=gt_done_time,
            run_events=current_run_events,
            gt_events=current_gt_events,
            cancel_time=cancel_times.get(order_id),
        )
    return profiles


def normalized_weights(objective_config: dict[str, Any], available: tuple[str, ...]) -> dict[str, float]:
    weights_cfg = objective_config.get("weights", {})
    if not isinstance(weights_cfg, dict):
        raise ValueError("spec.objective_config.weights must be a dict")
    raw_weights = {
        name: read_component_value(weights_cfg, name, DEFAULT_WEIGHT, min_value=0.0)
        for name in available
    }
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("sum of available component weights must be > 0")
    output = {name: 0.0 for name in ALL_COMPONENTS}
    for name in available:
        output[name] = raw_weights[name] / total
    return output


def read_component_value(
    values: dict[str, Any],
    component: str,
    default: float | None,
    min_value: float,
) -> float:
    raw = values.get(component, default)
    if raw is None:
        raise KeyError(f"missing required component value: {component}")
    value = to_float(raw, component)
    if value < min_value:
        raise ValueError(f"{component} must be >= {min_value}, got {value}")
    return value


def require_float(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is unavailable")
    return value


def _read_order_definitions(order_rows: tuple[Any, ...]) -> dict[int, OrderDefinition]:
    output: dict[int, OrderDefinition] = {}
    for row in order_rows:
        row_map = _row_to_dict(row)
        order_id = _read_row_int(row_map, ORDER_ID_KEYS, "orderinfo.order_id")
        quantity = _read_row_float(row_map, VOLUME_KEYS, "orderinfo.volume")
        if quantity <= 0:
            raise ValueError(f"orderinfo.volume must be > 0, got {quantity} for order_id={order_id}")
        sent_time = _read_row_int(row_map, SENT_TIME_KEYS, "orderinfo.sent_time")
        output[order_id] = OrderDefinition(quantity=quantity, sent_time=sent_time)
    return output


def _read_done_times(done_rows: tuple[Any, ...]) -> dict[int, int]:
    output: dict[int, int] = {}
    for row in done_rows:
        row_map = _row_to_dict(row)
        order_id = _read_row_int(row_map, ORDER_ID_KEYS, "doneinfo.order_id")
        done_time = _read_row_int(row_map, DONE_TIME_KEYS, "doneinfo.done_time")
        output[order_id] = max(done_time, output.get(order_id, done_time))
    return output


def _read_execution_events(execution_rows: tuple[Any, ...]) -> dict[int, tuple[tuple[int, float], ...]]:
    output: dict[int, list[tuple[int, float]]] = {}
    for row in execution_rows:
        row_map = _row_to_dict(row)
        order_id = _read_row_int(row_map, ORDER_ID_KEYS, "executiondetail.order_id")
        event_time = _read_row_int(row_map, EXEC_TIME_KEYS, "executiondetail.time")
        volume = _read_row_float(row_map, VOLUME_KEYS, "executiondetail.volume")
        if volume < 0:
            raise ValueError(f"executiondetail.volume must be >= 0, got {volume}")
        output.setdefault(order_id, []).append((event_time, volume))
    return {order_id: tuple(sorted(events, key=lambda item: item[0])) for order_id, events in output.items()}


def _read_cancel_times(cancel_rows: tuple[Any, ...]) -> dict[int, int]:
    output: dict[int, int] = {}
    for row in cancel_rows:
        row_map = _row_to_dict(row)
        order_id = _read_row_int(row_map, ORDER_ID_KEYS, "cancelrequest.order_id")
        cancel_time = _read_row_int(row_map, CXL_TIME_KEYS, "cancelrequest.cancel_sent_time")
        output[order_id] = min(cancel_time, output.get(order_id, cancel_time))
    return output


def _last_event_time(events: tuple[tuple[int, float], ...], fallback: int) -> int:
    return fallback if not events else events[-1][0]


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "__dict__"):
        return vars(row)
    raise TypeError(f"row must be dict-like, got {type(row).__name__}")


def _read_row_int(row: dict[str, Any], keys: tuple[str, ...], field_name: str) -> int:
    return to_int(_read_row_value(row, keys, field_name), field_name)


def _read_row_float(row: dict[str, Any], keys: tuple[str, ...], field_name: str) -> float:
    return to_float(_read_row_value(row, keys, field_name), field_name)


def _read_row_value(row: dict[str, Any], keys: tuple[str, ...], field_name: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    raise KeyError(f"missing required field {field_name}; tried keys={keys}")


def to_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be numeric, got {value!r}") from exc


def to_int(value: Any, field_name: str) -> int:
    as_float = to_float(value, field_name)
    if not as_float.is_integer():
        raise ValueError(f"{field_name} must be integer-like, got {value!r}")
    return int(as_float)
