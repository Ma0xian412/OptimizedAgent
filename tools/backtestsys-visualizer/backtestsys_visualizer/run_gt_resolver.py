from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from backtestsys_visualizer.models import stage_sort_key
from backtestsys_visualizer.run_gt_param_binding import resolve_effective_params

_SPEC_HASH_ATTR = "spec_hash"
_SPEC_JSON_ATTR = "spec_json"
_PARAM_PATCH_ATTR = "backtest_config_patch"


@dataclass(frozen=True)
class DatasetTrialPayload:
    dataset_id: str
    machine: str | None
    contract: str | None
    sim_payload: dict[str, Any]
    gt_tables: dict[str, list[dict[str, str]]]


@dataclass(frozen=True)
class StageTrialPayload:
    stage_name: str
    trial_number: int
    expected_dataset_count: int
    dataset_payloads: tuple[DatasetTrialPayload, ...]


def discover_stage_dirs(*, run_root: Path, selected_stages: set[str] | None) -> list[Path]:
    stages_root = run_root / "stages"
    if not stages_root.is_dir():
        return []
    candidates = [item for item in stages_root.iterdir() if item.is_dir()]
    if selected_stages is not None:
        candidates = [item for item in candidates if item.name in selected_stages]
    return sorted(candidates, key=lambda item: stage_sort_key(item.name))


def load_stage_trial_payloads(stage_dir: Path) -> list[StageTrialPayload]:
    study_db = stage_dir / "study.db"
    if not study_db.is_file():
        return []
    storage = f"sqlite:///{study_db.resolve()}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if len(summaries) != 1:
        return []
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    spec_hash = study.user_attrs.get(_SPEC_HASH_ATTR)
    spec_json = study.user_attrs.get(_SPEC_JSON_ATTR)
    if not isinstance(spec_hash, str) or not isinstance(spec_json, str):
        return []
    spec = json.loads(spec_json)
    if not isinstance(spec, dict):
        return []
    dataset_ids = _resolve_dataset_ids(spec)
    if not dataset_ids:
        return []
    gt_tables_by_dataset = _load_groundtruth_tables_by_dataset(spec=spec, dataset_ids=dataset_ids)
    payloads: list[StageTrialPayload] = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        trial_params = _resolve_trial_param_patch(trial)
        if trial_params is None:
            continue
        dataset_payloads = _resolve_trial_dataset_payloads(
            stage_dir=stage_dir,
            spec=spec,
            spec_hash=spec_hash,
            dataset_ids=dataset_ids,
            trial_params=trial_params,
            gt_tables_by_dataset=gt_tables_by_dataset,
        )
        if not dataset_payloads:
            continue
        payloads.append(
            StageTrialPayload(
                stage_name=stage_dir.name,
                trial_number=trial.number,
                expected_dataset_count=len(dataset_ids),
                dataset_payloads=tuple(dataset_payloads),
            )
        )
    return payloads


def _resolve_trial_dataset_payloads(
    *,
    stage_dir: Path,
    spec: dict[str, Any],
    spec_hash: str,
    dataset_ids: tuple[str, ...],
    trial_params: dict[str, object],
    gt_tables_by_dataset: dict[str, dict[str, list[dict[str, str]]]],
) -> list[DatasetTrialPayload]:
    run_cfg = _read_run_cfg(spec)
    payloads: list[DatasetTrialPayload] = []
    for dataset_id in dataset_ids:
        gt_tables = gt_tables_by_dataset.get(dataset_id)
        if gt_tables is None:
            continue
        dataset_input = _read_dataset_input(run_cfg, dataset_id)
        if dataset_input is None:
            continue
        effective_params = resolve_effective_params(run_cfg, trial_params, dataset_input)
        config_path = _resolve_config_path(run_cfg, dataset_id, effective_params, spec_hash)
        if not config_path.is_file():
            continue
        run_key = _compute_run_key(
            config_path=config_path,
            spec_hash=spec_hash,
            dataset_id=dataset_id,
            spec_meta=spec.get("meta"),
        )
        run_payload = _read_run_payload(stage_dir=stage_dir, run_key=run_key)
        if run_payload is not None:
            payloads.append(
                DatasetTrialPayload(
                    dataset_id=dataset_id,
                    machine=_read_optional_label(dataset_input, "machine"),
                    contract=_read_optional_label(dataset_input, "contract"),
                    sim_payload=run_payload,
                    gt_tables=gt_tables,
                )
            )
    return payloads


def _load_groundtruth_tables_by_dataset(
    *,
    spec: dict[str, Any],
    dataset_ids: tuple[str, ...],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    obj_cfg = spec.get("objective_config")
    if not isinstance(obj_cfg, dict):
        return {}
    gt_cfg = obj_cfg.get("groundtruth")
    if not isinstance(gt_cfg, dict):
        return {}
    dataset_cfg_map = gt_cfg.get("datasets")
    if not isinstance(dataset_cfg_map, dict):
        dataset_cfg_map = {}
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    for dataset_id in dataset_ids:
        gt_paths = _resolve_groundtruth_paths(
            base_cfg=gt_cfg,
            dataset_cfg=dataset_cfg_map.get(dataset_id),
        )
        if gt_paths is None:
            continue
        done_rows = _read_csv_rows(Path(gt_paths["doneinfo_path"]))
        exec_rows = _read_csv_rows(Path(gt_paths["executiondetail_path"]))
        if not done_rows or not exec_rows:
            continue
        result[dataset_id] = {
            "DoneInfo": done_rows,
            "ExecutionDetail": exec_rows,
        }
    return result


def _resolve_groundtruth_paths(
    *,
    base_cfg: dict[str, Any],
    dataset_cfg: object,
) -> dict[str, str] | None:
    source = dataset_cfg if isinstance(dataset_cfg, dict) else base_cfg
    done_path = source.get("doneinfo_path") if isinstance(source, dict) else None
    exec_path = source.get("executiondetail_path") if isinstance(source, dict) else None
    if not isinstance(done_path, str) or not isinstance(exec_path, str):
        return None
    if not done_path or not exec_path:
        return None
    return {
        "doneinfo_path": done_path,
        "executiondetail_path": exec_path,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(item) for item in csv.DictReader(fp)]


def _resolve_dataset_ids(spec: dict[str, Any]) -> tuple[str, ...]:
    meta = spec.get("meta")
    if isinstance(meta, dict):
        raw = meta.get("dataset_ids")
        if isinstance(raw, list) and all(isinstance(item, str) and item for item in raw):
            return tuple(raw)
    run_cfg = _read_run_cfg(spec)
    dataset_inputs = run_cfg.get("dataset_inputs")
    if not isinstance(dataset_inputs, dict):
        return tuple()
    keys = [key for key in dataset_inputs if isinstance(key, str) and key]
    return tuple(sorted(keys))


def _resolve_trial_param_patch(trial: optuna.trial.FrozenTrial) -> dict[str, object] | None:
    patched = trial.user_attrs.get(_PARAM_PATCH_ATTR)
    if isinstance(patched, dict):
        return dict(patched)
    if trial.params:
        return dict(trial.params)
    return None


def _read_run_cfg(spec: dict[str, Any]) -> dict[str, Any]:
    execution = spec.get("execution_config")
    if not isinstance(execution, dict):
        return {}
    run_cfg = execution.get("backtest_run_spec")
    return run_cfg if isinstance(run_cfg, dict) else {}


def _read_dataset_input(run_cfg: dict[str, Any], dataset_id: str) -> dict[str, Any] | None:
    dataset_inputs = run_cfg.get("dataset_inputs")
    if not isinstance(dataset_inputs, dict):
        return None
    raw = dataset_inputs.get(dataset_id)
    return raw if isinstance(raw, dict) else None


def _read_optional_label(dataset_input: dict[str, Any], key: str) -> str | None:
    value = dataset_input.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _resolve_config_path(run_cfg: dict[str, Any], dataset_id: str, params: dict[str, object], spec_hash: str) -> Path:
    output_root = run_cfg.get("output_root_dir")
    if not isinstance(output_root, str) or not output_root:
        return Path("__missing__")
    digest_payload = {"spec_hash": spec_hash, "dataset_id": dataset_id, "params": params}
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]
    return Path(output_root) / "configs" / f"{dataset_id}_{digest}.xml"


def _compute_run_key(*, config_path: Path, spec_hash: str, dataset_id: str, spec_meta: Any) -> str:
    root = ET.parse(config_path).getroot()
    payload = {
        "kind": "backtest_run_key_v1",
        "spec_hash": spec_hash,
        "dataset_id": dataset_id,
        "meta": spec_meta if isinstance(spec_meta, dict) else {},
        "config": _element_to_payload(root),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"run:{digest[:24]}"


def _element_to_payload(element: ET.Element) -> object:
    children = [child for child in list(element) if not callable(child.tag)]
    if not children:
        return (element.text or "").strip()
    grouped: dict[str, list[object]] = {}
    for child in children:
        grouped.setdefault(str(child.tag), []).append(_element_to_payload(child))
    return {key: grouped[key][0] if len(grouped[key]) == 1 else grouped[key] for key in sorted(grouped)}


def _read_run_payload(*, stage_dir: Path, run_key: str) -> dict[str, Any] | None:
    safe_name = _safe_filename(run_key)
    path = stage_dir / "ocp_data" / "run_records" / safe_name
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    run_payload = payload.get("payload")
    return run_payload if isinstance(run_payload, dict) else None


def _safe_filename(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    short = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key[:60])
    return f"{short}__{digest}.json"

