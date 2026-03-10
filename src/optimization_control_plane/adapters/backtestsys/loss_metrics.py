from __future__ import annotations

from dataclasses import dataclass

COMPONENT_CURVE = "curve"
COMPONENT_TERMINAL = "terminal"
COMPONENT_CANCEL = "cancel"
COMPONENT_POST = "post"
ALL_COMPONENTS = (
    COMPONENT_CURVE,
    COMPONENT_TERMINAL,
    COMPONENT_CANCEL,
    COMPONENT_POST,
)


@dataclass(frozen=True)
class OrderProfile:
    quantity: float
    sent_time: int
    run_done_time: int
    gt_done_time: int
    run_events: tuple[tuple[int, float], ...]
    gt_events: tuple[tuple[int, float], ...]
    cancel_time: int | None


def compute_raw_components(
    order_profiles: dict[int, OrderProfile],
) -> tuple[dict[str, float | None], tuple[str, ...], dict[str, int]]:
    curve_values: list[float] = []
    terminal_values: list[float] = []
    cancel_values: list[float] = []
    post_values: list[float] = []
    cancel_order_count = 0
    for profile in order_profiles.values():
        curve_values.append(_curve_loss(profile))
        run_ratio = _terminal_fill_ratio(profile.run_events, profile.quantity)
        gt_ratio = _terminal_fill_ratio(profile.gt_events, profile.quantity)
        terminal_values.append(abs(run_ratio - gt_ratio))
        if profile.cancel_time is not None:
            cancel_order_count += 1
            cancel_values.append(_cancel_classification_error(profile))
            post_values.append(_post_cancel_error(profile))
    raw = {
        COMPONENT_CURVE: _mean(curve_values),
        COMPONENT_TERMINAL: _mean(terminal_values),
        COMPONENT_CANCEL: _mean(cancel_values) if cancel_values else None,
        COMPONENT_POST: _mean(post_values) if post_values else None,
    }
    available = (COMPONENT_CURVE, COMPONENT_TERMINAL) if not cancel_values else ALL_COMPONENTS
    stats = {"order_count": len(order_profiles), "cancel_order_count": cancel_order_count}
    return raw, available, stats


def _curve_loss(profile: OrderProfile) -> float:
    end_time = max(profile.run_done_time, profile.gt_done_time, profile.sent_time)
    if end_time <= profile.sent_time:
        run_ratio = _terminal_fill_ratio(profile.run_events, profile.quantity)
        gt_ratio = _terminal_fill_ratio(profile.gt_events, profile.quantity)
        return abs(run_ratio - gt_ratio)
    time_points = {profile.sent_time, end_time}
    time_points.update(event_time for event_time, _ in profile.run_events)
    time_points.update(event_time for event_time, _ in profile.gt_events)
    sorted_times = sorted(time_points)
    run_cum = 0.0
    gt_cum = 0.0
    run_idx = 0
    gt_idx = 0
    area = 0.0
    for idx in range(len(sorted_times) - 1):
        left = sorted_times[idx]
        right = sorted_times[idx + 1]
        run_cum, run_idx = _advance_cumulative(profile.run_events, run_idx, left, profile.quantity, run_cum)
        gt_cum, gt_idx = _advance_cumulative(profile.gt_events, gt_idx, left, profile.quantity, gt_cum)
        area += abs(run_cum - gt_cum) / profile.quantity * float(right - left)
    return area / float(end_time - profile.sent_time)


def _advance_cumulative(
    events: tuple[tuple[int, float], ...],
    start_idx: int,
    threshold_time: int,
    quantity: float,
    cumulative: float,
) -> tuple[float, int]:
    idx = start_idx
    current = cumulative
    while idx < len(events) and events[idx][0] <= threshold_time:
        current = min(quantity, current + events[idx][1])
        idx += 1
    return current, idx


def _cancel_classification_error(profile: OrderProfile) -> float:
    run_full = _terminal_filled_volume(profile.run_events, profile.quantity)
    gt_full = _terminal_filled_volume(profile.gt_events, profile.quantity)
    run_label = 1.0 if run_full < profile.quantity else 0.0
    gt_label = 1.0 if gt_full < profile.quantity else 0.0
    return abs(run_label - gt_label)


def _post_cancel_error(profile: OrderProfile) -> float:
    if profile.cancel_time is None:
        raise ValueError("cancel_time is required for post-cancel loss")
    run_ratio = _post_cancel_fill_ratio(profile.run_events, profile.quantity, profile.cancel_time)
    gt_ratio = _post_cancel_fill_ratio(profile.gt_events, profile.quantity, profile.cancel_time)
    return abs(run_ratio - gt_ratio)


def _post_cancel_fill_ratio(
    events: tuple[tuple[int, float], ...],
    quantity: float,
    cancel_time: int,
) -> float:
    before_cancel = _filled_volume_before(events, quantity, cancel_time)
    total_filled = _terminal_filled_volume(events, quantity)
    return max(0.0, total_filled - before_cancel) / quantity


def _filled_volume_before(
    events: tuple[tuple[int, float], ...],
    quantity: float,
    threshold_time: int,
) -> float:
    cumulative = 0.0
    for event_time, volume in events:
        if event_time >= threshold_time:
            break
        cumulative = min(quantity, cumulative + volume)
    return cumulative


def _terminal_fill_ratio(events: tuple[tuple[int, float], ...], quantity: float) -> float:
    return _terminal_filled_volume(events, quantity) / quantity


def _terminal_filled_volume(events: tuple[tuple[int, float], ...], quantity: float) -> float:
    cumulative = 0.0
    for _, volume in events:
        cumulative = min(quantity, cumulative + volume)
    return cumulative


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty values")
    return sum(values) / float(len(values))
