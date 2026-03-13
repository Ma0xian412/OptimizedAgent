from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_CANCEL_SUCCESS_STATES = frozenset({"P", "N"})


@dataclass(frozen=True)
class OrderEval:
    state_match: float
    done_time_abs_diff: float
    sim_fill_ratio: float
    gt_fill_ratio: float
    fill_gap_ratio: float
    terminal_raw: float
    post_cancel_gap: float | None
    cancel_state_match: float | None


def evaluate_dataset_metrics(
    *,
    sim_payload: dict[str, Any],
    gt_tables: dict[str, list[dict[str, str]]],
) -> dict[str, float] | None:
    orders = _evaluate_orders(sim_payload=sim_payload, gt_tables=gt_tables)
    if not orders:
        return None
    cancel_orders = [item for item in orders if item.post_cancel_gap is not None]
    cancel_match_values = [item.cancel_state_match for item in cancel_orders if item.cancel_state_match is not None]
    return {
        "order_count": float(len(orders)),
        "cancel_order_count": float(len(cancel_orders)),
        "state_match_rate": _mean([item.state_match for item in orders]),
        "done_time_mae": _mean([item.done_time_abs_diff for item in orders]),
        "sim_fill_ratio": _mean([item.sim_fill_ratio for item in orders]),
        "gt_fill_ratio": _mean([item.gt_fill_ratio for item in orders]),
        "fill_gap_ratio": _mean([item.fill_gap_ratio for item in orders]),
        "terminal_raw": _mean([item.terminal_raw for item in orders]),
        "post_cancel_gap": _mean([item.post_cancel_gap for item in cancel_orders]) if cancel_orders else 0.0,
        "cancel_state_match_rate": _mean(cancel_match_values) if cancel_match_values else 0.0,
    }


def build_trial_metric_row(
    *,
    stage_name: str,
    stage_order: int,
    trial_number: int,
    expected_dataset_count: int,
    dataset_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    dataset_count = len(dataset_metrics)
    order_total = sum(item["order_count"] for item in dataset_metrics)
    cancel_total = sum(item["cancel_order_count"] for item in dataset_metrics)
    return {
        "stage": stage_name,
        "stage_order": stage_order,
        "trial_id": str(trial_number),
        "stage_iteration": int(trial_number),
        "dataset_count": dataset_count,
        "expected_dataset_count": expected_dataset_count,
        "missing_dataset_count": max(expected_dataset_count - dataset_count, 0),
        "order_count": int(order_total),
        "cancel_order_count": int(cancel_total),
        "state_match_rate": _weighted_mean(dataset_metrics, "state_match_rate", "order_count"),
        "done_time_mae": _weighted_mean(dataset_metrics, "done_time_mae", "order_count"),
        "sim_fill_ratio": _weighted_mean(dataset_metrics, "sim_fill_ratio", "order_count"),
        "gt_fill_ratio": _weighted_mean(dataset_metrics, "gt_fill_ratio", "order_count"),
        "fill_gap_ratio": _weighted_mean(dataset_metrics, "fill_gap_ratio", "order_count"),
        "terminal_raw_from_tables": _weighted_mean(dataset_metrics, "terminal_raw", "order_count"),
        "post_cancel_gap": _weighted_mean(dataset_metrics, "post_cancel_gap", "cancel_order_count"),
        "cancel_state_match_rate": _weighted_mean(dataset_metrics, "cancel_state_match_rate", "cancel_order_count"),
    }


def _evaluate_orders(sim_payload: dict[str, Any], gt_tables: dict[str, list[dict[str, str]]]) -> list[OrderEval]:
    sim_order = _index_order_info(_read_table(sim_payload, "OrderInfo"))
    sim_done = _index_done_info(_read_table(sim_payload, "DoneInfo"))
    sim_exec = _index_execution(_read_table(sim_payload, "ExecutionDetail"))
    sim_cancel = _index_cancel(_read_table(sim_payload, "CancelRequest"))
    gt_done = _index_done_info(gt_tables.get("DoneInfo", []))
    gt_exec = _index_execution(gt_tables.get("ExecutionDetail", []))
    keys = sorted(set(sim_order) & set(sim_done) & set(gt_done))
    evaluated: list[OrderEval] = []
    for key in keys:
        quantity = sim_order[key]["quantity"]
        sim_total = sum(item["volume"] for item in sim_exec.get(key, []))
        gt_total = sum(item["volume"] for item in gt_exec.get(key, []))
        sim_fill_ratio = sim_total / float(quantity)
        gt_fill_ratio = gt_total / float(quantity)
        terminal_raw = abs(sim_total - gt_total) / float(quantity)
        done_diff = abs(sim_done[key]["done_time"] - gt_done[key]["done_time"])
        cancel_time = sim_cancel.get(key)
        post_gap = None
        cancel_state_match = None
        if cancel_time is not None:
            sim_post = _sum_volume_window(sim_exec.get(key, []), cancel_time, sim_done[key]["done_time"])
            gt_post = _sum_volume_window(gt_exec.get(key, []), cancel_time, gt_done[key]["done_time"])
            post_gap = abs(sim_post - gt_post) / float(quantity)
            sim_success = sim_done[key]["state"] in _CANCEL_SUCCESS_STATES
            gt_success = gt_done[key]["state"] in _CANCEL_SUCCESS_STATES
            cancel_state_match = 1.0 if sim_success == gt_success else 0.0
        evaluated.append(
            OrderEval(
                state_match=1.0 if sim_done[key]["state"] == gt_done[key]["state"] else 0.0,
                done_time_abs_diff=float(done_diff),
                sim_fill_ratio=sim_fill_ratio,
                gt_fill_ratio=gt_fill_ratio,
                fill_gap_ratio=abs(sim_fill_ratio - gt_fill_ratio),
                terminal_raw=terminal_raw,
                post_cancel_gap=post_gap,
                cancel_state_match=cancel_state_match,
            )
        )
    return evaluated


def _read_table(payload: dict[str, Any], name: str) -> list[dict[str, str]]:
    raw = payload.get(name)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _key(row: dict[str, str]) -> tuple[int, int, int, str]:
    return (
        int(row["PartitionDay"]),
        int(row["ContractId"]),
        int(row["OrderId"]),
        str(row["MachineName"]).strip(),
    )


def _index_order_info(rows: list[dict[str, str]]) -> dict[tuple[int, int, int, str], dict[str, int]]:
    result: dict[tuple[int, int, int, str], dict[str, int]] = {}
    for row in rows:
        quantity = int(row["Volume"])
        if quantity > 0:
            result[_key(row)] = {"quantity": quantity}
    return result


def _index_done_info(rows: list[dict[str, str]]) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    return {_key(row): {"done_time": int(row["DoneTime"]), "state": str(row["OrderTradeState"]).strip()} for row in rows}


def _index_execution(rows: list[dict[str, str]]) -> dict[tuple[int, int, int, str], list[dict[str, int]]]:
    grouped: dict[tuple[int, int, int, str], list[dict[str, int]]] = {}
    for row in rows:
        grouped.setdefault(_key(row), []).append({"recv_tick": int(row["RecvTick"]), "volume": int(row["Volume"])})
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda item: item["recv_tick"])
    return grouped


def _index_cancel(rows: list[dict[str, str]]) -> dict[tuple[int, int, int, str], int]:
    result: dict[tuple[int, int, int, str], int] = {}
    for row in rows:
        key = _key(row)
        current = int(row["CancelSentTime"])
        previous = result.get(key)
        result[key] = current if previous is None else min(previous, current)
    return result


def _sum_volume_window(execs: list[dict[str, int]], lower_exclusive: int, upper_inclusive: int) -> int:
    return sum(item["volume"] for item in execs if lower_exclusive < item["recv_tick"] <= upper_inclusive)


def _weighted_mean(items: list[dict[str, float]], value_key: str, weight_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for item in items:
        weight = float(item[weight_key])
        if weight <= 0.0:
            continue
        numerator += weight * float(item[value_key])
        denominator += weight
    return numerator / denominator if denominator > 0.0 else 0.0


def _mean(values: list[float | None]) -> float:
    filtered = [float(item) for item in values if item is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / float(len(filtered))

