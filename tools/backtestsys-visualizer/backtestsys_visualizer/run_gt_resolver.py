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

_SPEC_HASH_ATTR = "spec_hash"
_SPEC_JSON_ATTR = "spec_json"
_PARAM_PATCH_ATTR = "backtest_config_patch"


@dataclass(frozen=True)
class StageTrialPayload:
    stage_name: str
    trial_number: int
    expected_dataset_count: int
    dataset_payloads: tuple[dict[str, Any], ...]
    gt_tables: dict[str, list[dict[str, str]]]


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
    gt_tables = _load_groundtruth_tables(spec)
    if not gt_tables["DoneInfo"] or not gt_tables["ExecutionDetail"]:
        return []
    dataset_ids = _resolve_dataset_ids(spec)
    if not dataset_ids:
        return []
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
        )
        if not dataset_payloads:
            continue
        payloads.append(
            StageTrialPayload(
                stage_name=stage_dir.name,
                trial_number=trial.number,
                expected_dataset_count=len(dataset_ids),
                dataset_payloads=tuple(dataset_payloads),
                gt_tables=gt_tables,
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
) -> list[dict[str, Any]]:
    run_cfg = _read_run_cfg(spec)
    payloads: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset_input = _read_dataset_input(run_cfg, dataset_id)
        if dataset_input is None:
            continue
        effective_params = _resolve_effective_params(run_cfg, trial_params, dataset_input)
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
            payloads.append(run_payload)
    return payloads


def _load_groundtruth_tables(spec: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    obj_cfg = spec.get("objective_config")
    if not isinstance(obj_cfg, dict):
        return {"DoneInfo": [], "ExecutionDetail": []}
    gt_cfg = obj_cfg.get("groundtruth")
    if not isinstance(gt_cfg, dict):
        return {"DoneInfo": [], "ExecutionDetail": []}
    done_path = gt_cfg.get("doneinfo_path")
    exec_path = gt_cfg.get("executiondetail_path")
    if not isinstance(done_path, str) or not isinstance(exec_path, str):
        return {"DoneInfo": [], "ExecutionDetail": []}
    return {
        "DoneInfo": _read_csv_rows(Path(done_path)),
        "ExecutionDetail": _read_csv_rows(Path(exec_path)),
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


def _resolve_effective_params(
    run_cfg: dict[str, Any],
    trial_params: dict[str, object],
    dataset_input: dict[str, Any],
) -> dict[str, object]:
    binding = run_cfg.get("param_binding")
    if not isinstance(binding, dict) or binding.get("mode", "trial_global") == "trial_global":
        return dict(trial_params)
    machine = dataset_input.get("machine")
    contract = dataset_input.get("contract")
    machine_map = binding.get("machine_delay_map")
    core_map = binding.get("contract_core_map")
    if not isinstance(machine, str) or not isinstance(contract, str):
        return dict(trial_params)
    if not isinstance(machine_map, dict) or not isinstance(core_map, dict):
        return dict(trial_params)
    delay = machine_map.get(machine)
    core = core_map.get(contract)
    if not isinstance(delay, int) or isinstance(delay, bool):
        return dict(trial_params)
    if not isinstance(core, dict):
        return dict(trial_params)
    lam = core.get("time_scale_lambda")
    bias = core.get("cancel_bias_k")
    if not isinstance(lam, (int, float)) or not isinstance(bias, (int, float)):
        return dict(trial_params)
    return {
        "time_scale_lambda": float(lam),
        "cancel_bias_k": float(bias),
        "delay_in": int(delay),
        "delay_out": int(delay),
    }


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

