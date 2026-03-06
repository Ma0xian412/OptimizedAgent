from __future__ import annotations

import threading


class Metrics:
    """Simple thread-safe counter-based metrics for V1 observability."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "trials_asked_total": 0,
            "trials_completed_total": 0,
            "trials_pruned_total": 0,
            "trials_failed_total": 0,
            "run_cache_hit_total": 0,
            "objective_cache_hit_total": 0,
            "execution_submitted_total": 0,
        }
        self._gauges: dict[str, int] = {
            "inflight_leader_executions_gauge": 0,
            "attached_follower_trials_gauge": 0,
            "buffered_requests_gauge": 0,
        }

    def inc(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta

    def set_gauge(self, name: str, value: int) -> None:
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {**self._counters, **self._gauges}
