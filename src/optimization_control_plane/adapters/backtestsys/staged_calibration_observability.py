from __future__ import annotations

import datetime as dt
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_SUPPORTED_OUTPUT_FORMATS = {"text", "json"}
_EVENT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass(frozen=True)
class StageProgressContext:
    stage_name: str
    unit_index: int
    unit_total: int
    max_trials: int | None


class StagedCalibrationProgressReporter:
    def __init__(self, runtime_root: Path, run_tag: str, *, output_format: str) -> None:
        if output_format not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"unsupported output_format={output_format}")
        self._run_tag = run_tag
        self._output_format = output_format
        self._output_path = runtime_root / "progress.jsonl"
        self._lock = threading.Lock()

    def run_started(self, *, unit_total: int, dataset_count: int) -> None:
        payload = {
            "unit_total": unit_total,
            "dataset_count": dataset_count,
            "overall_progress": 0.0,
        }
        self._emit("run_started", payload, f"RUN START total_units={unit_total} datasets={dataset_count}")

    def run_finished(self, *, final_best_value: float, runtime_root: Path) -> None:
        payload = {
            "final_best_value": final_best_value,
            "runtime_root": str(runtime_root),
            "overall_progress": 1.0,
        }
        self._emit("run_finished", payload, f"RUN DONE best={final_best_value:.6f}")

    def stage_started(self, ctx: StageProgressContext, metadata: Mapping[str, object] | None = None) -> float:
        started_at = time.monotonic()
        payload = self._with_stage_payload(
            ctx=ctx,
            stage_progress=0.0,
            payload={"metadata": dict(metadata or {}), "elapsed_seconds": 0.0},
        )
        self._emit("stage_started", payload, f"START {ctx.stage_name} ({ctx.unit_index}/{ctx.unit_total})")
        return started_at

    def stage_finished(
        self,
        ctx: StageProgressContext,
        *,
        started_at: float,
        best_value: float | None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        elapsed_seconds = self._elapsed_seconds(started_at)
        payload = {"metadata": dict(metadata or {}), "elapsed_seconds": elapsed_seconds}
        if best_value is not None:
            payload["best_value"] = best_value
        event_payload = self._with_stage_payload(ctx=ctx, stage_progress=1.0, payload=payload)
        text = f"DONE {ctx.stage_name} ({ctx.unit_index}/{ctx.unit_total}) elapsed={elapsed_seconds:.2f}s"
        self._emit("stage_finished", event_payload, text)

    def stage_failed(self, ctx: StageProgressContext, *, started_at: float, error: Exception) -> None:
        elapsed_seconds = self._elapsed_seconds(started_at)
        event_payload = self._with_stage_payload(
            ctx=ctx,
            stage_progress=None,
            payload={"elapsed_seconds": elapsed_seconds, "error": repr(error)},
        )
        text = f"FAILED {ctx.stage_name} ({ctx.unit_index}/{ctx.unit_total}) elapsed={elapsed_seconds:.2f}s"
        self._emit("stage_failed", event_payload, text)

    def stage_progress(self, ctx: StageProgressContext, metrics: Mapping[str, int]) -> None:
        stage_progress = _trial_progress(metrics, ctx.max_trials)
        payload = self._with_stage_payload(ctx=ctx, stage_progress=stage_progress, payload={"metrics": dict(metrics)})
        text = (
            f"PROGRESS {ctx.stage_name} completed={metrics.get('trials_completed_total', 0)} "
            f"failed={metrics.get('trials_failed_total', 0)} "
            f"inflight={metrics.get('inflight_leader_executions_gauge', 0)}"
        )
        self._emit("stage_progress", payload, text)

    def baseline_cache_hit(self, ctx: StageProgressContext, *, cache_path: Path) -> None:
        payload = self._with_stage_payload(
            ctx=ctx,
            stage_progress=1.0,
            payload={"cache_path": str(cache_path)},
        )
        text = f"CACHE HIT {ctx.stage_name} path={cache_path}"
        self._emit("baseline_cache_hit", payload, text)

    def _with_stage_payload(
        self,
        *,
        ctx: StageProgressContext,
        stage_progress: float | None,
        payload: dict[str, object],
    ) -> dict[str, object]:
        result = {
            "stage_name": ctx.stage_name,
            "unit_index": ctx.unit_index,
            "unit_total": ctx.unit_total,
            "max_trials": ctx.max_trials,
        }
        result.update(payload)
        if stage_progress is not None:
            result["stage_progress"] = stage_progress
            result["overall_progress"] = _overall_progress(ctx, stage_progress)
        return result

    def _emit(self, event: str, payload: dict[str, object], text_message: str) -> None:
        record = {
            "ts": dt.datetime.now(dt.UTC).strftime(_EVENT_TIME_FORMAT),
            "run_tag": self._run_tag,
            "event": event,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with self._output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self._output_format == "json":
            print(line)
            return
        print(f"[iter_backtestsys] {text_message}")

    @staticmethod
    def _elapsed_seconds(started_at: float) -> float:
        return max(0.0, time.monotonic() - started_at)


def _trial_progress(metrics: Mapping[str, int], max_trials: int | None) -> float:
    if max_trials is None or max_trials <= 0:
        return 0.0
    completed = int(metrics.get("trials_completed_total", 0))
    ratio = completed / max_trials
    return min(max(ratio, 0.0), 1.0)


def _overall_progress(ctx: StageProgressContext, stage_progress: float) -> float:
    numerator = max(ctx.unit_index - 1, 0) + stage_progress
    ratio = numerator / ctx.unit_total
    return min(max(ratio, 0.0), 1.0)
