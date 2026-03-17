from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from backtestsys_visualizer.models import COMPONENT_NAMES, TrialPoint, stage_sort_key


@dataclass(frozen=True)
class _MergedTrialRecord:
    stage_name: str
    trial_id: str
    stage_iteration: int | None
    total_loss: float | None
    state: str | None
    raw: dict[str, float | None]
    normalized: dict[str, float | None]
    params: dict[str, Any]
    from_trial_results: bool
    from_study_db: bool


def discover_run_dirs(*, runtime_root: Path) -> list[Path]:
    if not runtime_root.exists():
        raise FileNotFoundError(f"runtime root not found: {runtime_root}")
    if not runtime_root.is_dir():
        raise ValueError(f"runtime root must be a directory: {runtime_root}")
    run_dirs = [
        path for path in runtime_root.iterdir() if path.is_dir() and path.name.startswith("iter_backtestsys_")
    ]
    return sorted(run_dirs, key=lambda item: item.name, reverse=True)


def load_trial_points(
    *,
    run_root: Path,
    selected_stages: set[str] | None = None,
) -> list[TrialPoint]:
    stage_dirs = _discover_stage_dirs(run_root=run_root, selected_stages=selected_stages)
    merged = _merge_all_stages(stage_dirs)
    ranked = _sort_records(merged)
    return _with_global_iterations(run_tag=run_root.name, ranked=ranked)


def to_dataframe(points: list[TrialPoint]) -> pd.DataFrame:
    rows = [_trial_point_to_row(point) for point in points]
    return pd.DataFrame(rows)


def _discover_stage_dirs(*, run_root: Path, selected_stages: set[str] | None) -> list[Path]:
    stages_root = run_root / "stages"
    if not stages_root.is_dir():
        raise FileNotFoundError(f"stages directory not found: {stages_root}")
    all_stage_dirs = [path for path in stages_root.iterdir() if path.is_dir()]
    if selected_stages is None:
        return sorted(all_stage_dirs, key=lambda item: stage_sort_key(item.name))
    selected = [path for path in all_stage_dirs if path.name in selected_stages]
    return sorted(selected, key=lambda item: stage_sort_key(item.name))


def _merge_all_stages(stage_dirs: list[Path]) -> list[_MergedTrialRecord]:
    merged: list[_MergedTrialRecord] = []
    for stage_dir in stage_dirs:
        merged.extend(_load_stage_records(stage_dir=stage_dir))
    return merged


def _load_stage_records(*, stage_dir: Path) -> list[_MergedTrialRecord]:
    stage_name = stage_dir.name
    trial_result_map = _load_trial_results(stage_dir=stage_dir)
    study_trial_map = _load_study_trials(stage_dir=stage_dir)
    trial_ids = sorted(
        set(trial_result_map) | set(study_trial_map),
        key=_trial_id_sort_key,
    )
    records = [
        _build_record(
            stage_name=stage_name,
            trial_id=trial_id,
            result_payload=trial_result_map.get(trial_id),
            study_payload=study_trial_map.get(trial_id),
        )
        for trial_id in trial_ids
    ]
    return records


def _build_record(
    *,
    stage_name: str,
    trial_id: str,
    result_payload: dict[str, Any] | None,
    study_payload: dict[str, Any] | None,
) -> _MergedTrialRecord:
    attrs = _pick_attrs(result_payload=result_payload, study_payload=study_payload)
    raw = _read_component_map(attrs.get("raw"))
    normalized = _read_component_map(attrs.get("normalized"))
    stage_iteration = _pick_stage_iteration(trial_id=trial_id, study_payload=study_payload)
    total_loss = _pick_total_loss(attrs=attrs, study_payload=study_payload)
    params = _pick_params(study_payload=study_payload)
    state = _pick_state(study_payload=study_payload)
    return _MergedTrialRecord(
        stage_name=stage_name,
        trial_id=trial_id,
        stage_iteration=stage_iteration,
        total_loss=total_loss,
        state=state,
        raw=raw,
        normalized=normalized,
        params=params,
        from_trial_results=result_payload is not None,
        from_study_db=study_payload is not None,
    )


def _pick_attrs(
    *,
    result_payload: dict[str, Any] | None,
    study_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if result_payload is not None:
        attrs = result_payload.get("attrs")
        if isinstance(attrs, dict):
            return attrs
    if study_payload is not None:
        attrs = study_payload.get("user_attrs")
        if isinstance(attrs, dict):
            return attrs
    return {}


def _pick_stage_iteration(*, trial_id: str, study_payload: dict[str, Any] | None) -> int | None:
    if study_payload is not None and isinstance(study_payload.get("number"), int):
        return int(study_payload["number"])
    parsed = _try_parse_int(trial_id)
    return parsed


def _pick_total_loss(*, attrs: dict[str, Any], study_payload: dict[str, Any] | None) -> float | None:
    attrs_value = _as_float_or_none(attrs.get("value"))
    if attrs_value is not None:
        return attrs_value
    if study_payload is None:
        return None
    return _as_float_or_none(study_payload.get("value"))


def _pick_state(*, study_payload: dict[str, Any] | None) -> str | None:
    if study_payload is None:
        return None
    state = study_payload.get("state")
    return state if isinstance(state, str) else None


def _pick_params(*, study_payload: dict[str, Any] | None) -> dict[str, Any]:
    if study_payload is None:
        return {}
    params = study_payload.get("params")
    if isinstance(params, dict):
        return dict(params)
    return {}


def _read_component_map(raw_map: Any) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if not isinstance(raw_map, dict):
        for component in COMPONENT_NAMES:
            result[component] = None
        return result
    for component in COMPONENT_NAMES:
        result[component] = _as_float_or_none(raw_map.get(component))
    return result


def _sort_records(records: list[_MergedTrialRecord]) -> list[_MergedTrialRecord]:
    return sorted(
        records,
        key=lambda item: (
            stage_sort_key(item.stage_name),
            item.stage_iteration if item.stage_iteration is not None else 10**9,
            _trial_id_sort_key(item.trial_id),
        ),
    )


def _with_global_iterations(*, run_tag: str, ranked: list[_MergedTrialRecord]) -> list[TrialPoint]:
    points: list[TrialPoint] = []
    best_so_far: float | None = None
    for index, record in enumerate(ranked, start=1):
        if record.total_loss is not None:
            best_so_far = record.total_loss if best_so_far is None else min(best_so_far, record.total_loss)
        points.append(
            TrialPoint(
                run_tag=run_tag,
                stage_name=record.stage_name,
                trial_id=record.trial_id,
                stage_iteration=record.stage_iteration,
                global_iteration=index,
                total_loss=record.total_loss,
                best_so_far=best_so_far,
                state=record.state,
                raw=dict(record.raw),
                normalized=dict(record.normalized),
                params=dict(record.params),
                from_trial_results=record.from_trial_results,
                from_study_db=record.from_study_db,
            )
        )
    return points


def _load_trial_results(*, stage_dir: Path) -> dict[str, dict[str, Any]]:
    trial_results_dir = stage_dir / "ocp_data" / "trial_results"
    if not trial_results_dir.is_dir():
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    for json_path in sorted(trial_results_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"trial result payload must be dict: {json_path}")
        trial_id = payload.get("trial_id")
        if not isinstance(trial_id, str):
            raise ValueError(f"trial result payload missing trial_id string: {json_path}")
        payloads[trial_id] = payload
    return payloads


def _load_study_trials(*, stage_dir: Path) -> dict[str, dict[str, Any]]:
    study_db_path = stage_dir / "study.db"
    if not study_db_path.is_file():
        return {}
    storage_dsn = f"sqlite:///{study_db_path.resolve()}"
    summaries = optuna.study.get_all_study_summaries(storage=storage_dsn)
    if not summaries:
        return {}
    if len(summaries) != 1:
        raise ValueError(f"expected exactly one study in {study_db_path}, got {len(summaries)}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_dsn)
    mapped: dict[str, dict[str, Any]] = {}
    for trial in study.get_trials(deepcopy=False):
        mapped[str(trial.number)] = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
        }
    return mapped


def _trial_point_to_row(point: TrialPoint) -> dict[str, Any]:
    row = {
        "run_tag": point.run_tag,
        "stage": point.stage_name,
        "global_iteration": point.global_iteration,
        "stage_iteration": point.stage_iteration,
        "trial_id": point.trial_id,
        "total_loss": point.total_loss,
        "best_so_far": point.best_so_far,
        "state": point.state,
        "from_trial_results": point.from_trial_results,
        "from_study_db": point.from_study_db,
        "params_json": json.dumps(point.params, ensure_ascii=False, sort_keys=True),
    }
    for component in COMPONENT_NAMES:
        row[f"raw_{component}"] = point.raw.get(component)
        row[f"normalized_{component}"] = point.normalized.get(component)
    return row


def _trial_id_sort_key(trial_id: str) -> tuple[int, int | str]:
    parsed = _try_parse_int(trial_id)
    if parsed is not None:
        return (0, parsed)
    return (1, trial_id)


def _try_parse_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _as_float_or_none(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)

