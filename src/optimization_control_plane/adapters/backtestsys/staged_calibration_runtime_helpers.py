from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from optimization_control_plane.adapters.backtestsys.staged_calibration_observability import (
    StageProgressContext,
    StagedCalibrationProgressReporter,
)
from optimization_control_plane.adapters.backtestsys.staged_calibration_support import (
    StageResult,
    _build_orchestrator,
    _load_best_trial,
    extract_baseline_raw,
    run_stage,
)

_BASELINE_CACHE_VERSION = 1
_BASELINE_CACHE_SPEC_ID = "__baseline_cache_key__"
_BASELINE_CACHE_OUTPUT_ROOT = "__baseline_cache_output_root__"
T = TypeVar("T")


def run_observed_block(
    reporter: StagedCalibrationProgressReporter,
    ctx: StageProgressContext,
    fn: Callable[[], T],
    *,
    resolve_best_value: Callable[[T], float | None] | None = None,
) -> T:
    started_at = reporter.stage_started(ctx)
    try:
        result = fn()
    except Exception as exc:
        reporter.stage_failed(ctx, started_at=started_at, error=exc)
        raise
    best_value = resolve_best_value(result) if resolve_best_value is not None else None
    reporter.stage_finished(ctx, started_at=started_at, best_value=best_value)
    return result


def run_stage_for_progress(
    *,
    runtime_root: Path,
    stage_name: str,
    settings: dict[str, object],
    search_space: object,
    progress_reporter: StagedCalibrationProgressReporter | None,
    progress_context: StageProgressContext,
    progress_interval_seconds: float,
) -> StageResult:
    if progress_reporter is None:
        return run_stage(runtime_root, stage_name, settings, search_space)
    return run_stage_with_progress(
        runtime_root,
        stage_name,
        settings,
        search_space,
        progress_reporter=progress_reporter,
        progress_context=progress_context,
        progress_interval_seconds=progress_interval_seconds,
    )


def run_stage_with_progress(
    runtime_root: Path,
    stage_name: str,
    settings: dict[str, object],
    search_space: object,
    *,
    progress_reporter: StagedCalibrationProgressReporter,
    progress_context: StageProgressContext,
    progress_interval_seconds: float,
) -> StageResult:
    stage_root = runtime_root / "stages" / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    storage_dsn = f"sqlite:///{(stage_root / 'study.db').resolve()}"
    orchestrator = _build_orchestrator(storage_dsn=storage_dsn, data_root=stage_root / "ocp_data", search_space=search_space)
    _run_orchestrator_with_progress(
        orchestrator=orchestrator,
        settings=settings,
        progress_reporter=progress_reporter,
        progress_context=progress_context,
        progress_interval_seconds=progress_interval_seconds,
    )
    best_trial = _load_best_trial(storage_dsn)
    if best_trial.value is None:
        raise ValueError("best trial value is missing")
    return StageResult(
        best_value=float(best_trial.value),
        best_params=dict(best_trial.params),
        best_attrs=dict(best_trial.user_attrs),
    )


def _run_orchestrator_with_progress(
    *,
    orchestrator,
    settings: dict[str, Any],
    progress_reporter: StagedCalibrationProgressReporter,
    progress_context: StageProgressContext,
    progress_interval_seconds: float,
) -> None:
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(orchestrator.start, settings=settings)
        while not future.done():
            progress_reporter.stage_progress(progress_context, orchestrator.metrics.snapshot())
            time.sleep(progress_interval_seconds)
        progress_reporter.stage_progress(progress_context, orchestrator.metrics.snapshot())
        future.result()


def build_baseline_cache_path(workspace_root: Path, settings: dict[str, object], fixed_params: dict[str, object]) -> Path:
    cache_root = workspace_root / "runtime" / "cache" / "iter_backtestsys"
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {"version": _BASELINE_CACHE_VERSION, "settings": normalize_baseline_settings_for_cache(settings), "fixed_params": dict(fixed_params)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return cache_root / f"baseline_{digest[:24]}.json"


def normalize_baseline_settings_for_cache(settings: dict[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(settings))
    normalized["spec_id"] = _BASELINE_CACHE_SPEC_ID
    execution = normalized.get("execution_config")
    if not isinstance(execution, dict):
        raise ValueError("settings.execution_config must be a dict")
    run_spec = execution.get("backtest_run_spec")
    if not isinstance(run_spec, dict):
        raise ValueError("settings.execution_config.backtest_run_spec must be a dict")
    run_spec["output_root_dir"] = _BASELINE_CACHE_OUTPUT_ROOT
    return normalized


def read_cached_baseline(cache_path: Path) -> dict[str, float] | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"baseline cache payload must be dict: {cache_path}")
    baseline_raw = payload.get("baseline_raw")
    if not isinstance(baseline_raw, dict):
        raise ValueError(f"baseline cache payload missing baseline_raw dict: {cache_path}")
    return extract_baseline_raw({"raw": baseline_raw})


def write_cached_baseline(cache_path: Path, baseline_raw: dict[str, float]) -> None:
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temp_path.write_text(json.dumps({"baseline_raw": dict(baseline_raw)}, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(cache_path)
