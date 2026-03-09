from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, compute_spec_hash

_SPEC_SETTINGS_KEY = "spec"


def build_spec_from_settings(settings: dict[str, Any]) -> ExperimentSpec | None:
    payload = settings.get(_SPEC_SETTINGS_KEY)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("settings.spec must be an object")
    return _build_spec_from_payload(payload)


def assert_spec_matches_settings(
    spec: ExperimentSpec, spec_from_settings: ExperimentSpec | None,
) -> None:
    if spec_from_settings is None:
        raise ValueError("when spec and settings are both provided, settings.spec is required")
    if spec != spec_from_settings:
        raise ValueError("spec mismatch between explicit spec and settings.spec")


def spec_to_settings_payload(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "spec_hash": spec.spec_hash,
        "meta": dict(spec.meta),
        "objective_config": dict(spec.objective_config),
        "execution_config": dict(spec.execution_config),
    }


def _build_spec_from_payload(payload: dict[str, Any]) -> ExperimentSpec:
    spec_id = _read_required_str(payload, "spec_id")
    meta = _read_required_dict(payload, "meta")
    objective_config = _read_required_dict(payload, "objective_config")
    execution_config = _read_required_dict(payload, "execution_config")
    computed_hash = compute_spec_hash(spec_id, meta, objective_config, execution_config)
    declared_hash = payload.get("spec_hash")
    if declared_hash is None:
        spec_hash = computed_hash
    else:
        if not isinstance(declared_hash, str):
            raise ValueError("settings.spec.spec_hash must be a string")
        if declared_hash != computed_hash:
            raise ValueError("settings.spec.spec_hash does not match computed hash")
        spec_hash = declared_hash
    return ExperimentSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        meta=meta,
        objective_config=objective_config,
        execution_config=execution_config,
    )


def _read_required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"settings.spec.{key} must be a non-empty string")
    return value


def _read_required_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"settings.spec.{key} must be an object")
    return value
