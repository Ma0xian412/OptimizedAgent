from __future__ import annotations

from dataclasses import dataclass

from optimization_control_plane.adapters.backtestsys.backtest_loss_parsing import (
    DoneInfoRow,
    ExecutionRow,
    OrderInfoRow,
    OrderKey,
)


@dataclass(frozen=True)
class DailyRawMetrics:
    curve: float
    terminal: float
    post: float | None
    order_count: int
    cancel_order_count: int


def calculate_daily_raw_metrics(
    *,
    order_info_by_key: dict[OrderKey, OrderInfoRow],
    sim_done_by_key: dict[OrderKey, DoneInfoRow],
    gt_done_by_key: dict[OrderKey, DoneInfoRow],
    sim_exec_by_key: dict[OrderKey, tuple[ExecutionRow, ...]],
    gt_exec_by_key: dict[OrderKey, tuple[ExecutionRow, ...]],
    cancel_time_by_key: dict[OrderKey, int],
    evaluation_keys: tuple[OrderKey, ...],
) -> DailyRawMetrics:
    curve_losses: list[float] = []
    terminal_losses: list[float] = []
    post_losses: list[float] = []
    for key in evaluation_keys:
        order_info = order_info_by_key[key]
        sim_done = sim_done_by_key[key]
        gt_done = gt_done_by_key[key]
        sim_execs = sim_exec_by_key.get(key, ())
        gt_execs = gt_exec_by_key.get(key, ())
        curve_losses.append(curve_loss(
            sent_time=order_info.sent_time,
            quantity=order_info.quantity,
            real_done_time=gt_done.done_time,
            sim_done_time=sim_done.done_time,
            real_execs=gt_execs,
            sim_execs=sim_execs,
        ))
        terminal_losses.append(terminal_loss(
            quantity=order_info.quantity,
            real_execs=gt_execs,
            sim_execs=sim_execs,
        ))
        append_post_cancel_loss(
            key=key,
            cancel_time_by_key=cancel_time_by_key,
            post_losses=post_losses,
            quantity=order_info.quantity,
            gt_done=gt_done,
            sim_done=sim_done,
            gt_execs=gt_execs,
            sim_execs=sim_execs,
        )
    return DailyRawMetrics(
        curve=mean(curve_losses),
        terminal=mean(terminal_losses),
        post=mean(post_losses) if post_losses else None,
        order_count=len(evaluation_keys),
        cancel_order_count=len(post_losses),
    )


def append_post_cancel_loss(
    *,
    key: OrderKey,
    cancel_time_by_key: dict[OrderKey, int],
    post_losses: list[float],
    quantity: int,
    gt_done: DoneInfoRow,
    sim_done: DoneInfoRow,
    gt_execs: tuple[ExecutionRow, ...],
    sim_execs: tuple[ExecutionRow, ...],
) -> None:
    cancel_time = cancel_time_by_key.get(key)
    if cancel_time is None:
        return
    post_losses.append(post_loss(
        quantity=quantity,
        cancel_time=cancel_time,
        real_done_time=gt_done.done_time,
        sim_done_time=sim_done.done_time,
        real_execs=gt_execs,
        sim_execs=sim_execs,
    ))


def curve_loss(
    *,
    sent_time: int,
    quantity: int,
    real_done_time: int,
    sim_done_time: int,
    real_execs: tuple[ExecutionRow, ...],
    sim_execs: tuple[ExecutionRow, ...],
) -> float:
    end_time = max(real_done_time, sim_done_time)
    if end_time <= sent_time:
        return 0.0
    real_deltas = build_execution_delta_map(real_execs)
    sim_deltas = build_execution_delta_map(sim_execs)
    breakpoints = build_curve_breakpoints(sent_time, end_time, real_execs, sim_execs)
    cumulative_real = cumulative_at_or_before(real_deltas, sent_time)
    cumulative_sim = cumulative_at_or_before(sim_deltas, sent_time)
    area = 0.0
    for index in range(len(breakpoints) - 1):
        left = breakpoints[index]
        right = breakpoints[index + 1]
        area += abs(cumulative_real - cumulative_sim) * (right - left)
        cumulative_real += real_deltas.get(right, 0)
        cumulative_sim += sim_deltas.get(right, 0)
    return area / float(quantity)


def terminal_loss(
    *,
    quantity: int,
    real_execs: tuple[ExecutionRow, ...],
    sim_execs: tuple[ExecutionRow, ...],
) -> float:
    real_total = sum_execution_volume(real_execs)
    sim_total = sum_execution_volume(sim_execs)
    return abs(real_total - sim_total) / float(quantity)


def post_loss(
    *,
    quantity: int,
    cancel_time: int,
    real_done_time: int,
    sim_done_time: int,
    real_execs: tuple[ExecutionRow, ...],
    sim_execs: tuple[ExecutionRow, ...],
) -> float:
    real_post = sum_volume_in_window(real_execs, lower_exclusive=cancel_time, upper_inclusive=real_done_time)
    sim_post = sum_volume_in_window(sim_execs, lower_exclusive=cancel_time, upper_inclusive=sim_done_time)
    return abs(real_post - sim_post) / float(quantity)


def build_curve_breakpoints(
    sent_time: int,
    end_time: int,
    real_execs: tuple[ExecutionRow, ...],
    sim_execs: tuple[ExecutionRow, ...],
) -> tuple[int, ...]:
    points = {sent_time, end_time}
    for execution in real_execs:
        if sent_time < execution.recv_tick < end_time:
            points.add(execution.recv_tick)
    for execution in sim_execs:
        if sent_time < execution.recv_tick < end_time:
            points.add(execution.recv_tick)
    return tuple(sorted(points))


def build_execution_delta_map(executions: tuple[ExecutionRow, ...]) -> dict[int, int]:
    deltas: dict[int, int] = {}
    for execution in executions:
        deltas[execution.recv_tick] = deltas.get(execution.recv_tick, 0) + execution.volume
    return deltas


def cumulative_at_or_before(deltas: dict[int, int], tick: int) -> int:
    cumulative = 0
    for recv_tick, volume in deltas.items():
        if recv_tick <= tick:
            cumulative += volume
    return cumulative


def sum_execution_volume(executions: tuple[ExecutionRow, ...]) -> int:
    return sum(execution.volume for execution in executions)


def sum_volume_in_window(
    executions: tuple[ExecutionRow, ...],
    *,
    lower_exclusive: int,
    upper_inclusive: int,
) -> int:
    return sum(
        execution.volume
        for execution in executions
        if lower_exclusive < execution.recv_tick <= upper_inclusive
    )


def daily_intermediate_value(metrics: DailyRawMetrics) -> float:
    components: list[float] = [metrics.curve, metrics.terminal]
    if metrics.post is not None:
        components.append(metrics.post)
    return mean(components)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty list")
    return sum(values) / float(len(values))
