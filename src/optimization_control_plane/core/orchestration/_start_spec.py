from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, compute_spec_hash

_SPEC_REQUIRED_KEYS = (
    "spec_id",
    "meta",
    "objective_config",
    "execution_config",
)
_SPEC_HASH_KEY = "spec_hash"


def build_spec_from_settings(settings: dict[str, Any]) -> ExperimentSpec | None:
    if not _contains_any_spec_field(settings):
        return None
    _assert_all_required_spec_fields(settings)
    return _build_spec_from_payload(settings)


def assert_spec_matches_settings(
    spec: ExperimentSpec, spec_from_settings: ExperimentSpec | None,
) -> None:
    if spec_from_settings is None:
        raise ValueError(
            "when spec and settings are both provided, settings must include spec construction fields",
        )
    if spec != spec_from_settings:
        raise ValueError("spec mismatch between explicit spec and settings-derived spec")


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
    declared_hash = payload.get(_SPEC_HASH_KEY)
    if declared_hash is None:
        spec_hash = computed_hash
    else:
        if not isinstance(declared_hash, str):
            raise ValueError("settings.spec_hash must be a string")
        if declared_hash != computed_hash:
            raise ValueError("settings.spec_hash does not match computed hash")
        spec_hash = declared_hash
    return ExperimentSpec(
        spec_id=spec_id,
        spec_hash=spec_hash,
        meta=meta,
        objective_config=objective_config,
        execution_config=execution_config,
    )


def _contains_any_spec_field(settings: dict[str, Any]) -> bool:
    keys = set(_SPEC_REQUIRED_KEYS) | {_SPEC_HASH_KEY}
    return any(key in settings for key in keys)


def _assert_all_required_spec_fields(settings: dict[str, Any]) -> None:
    missing_keys = [key for key in _SPEC_REQUIRED_KEYS if key not in settings]
    if missing_keys:
        raise ValueError(
            "settings missing fields for spec construction: "
            + ",".join(missing_keys),
        )


def _read_required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"settings.{key} must be a non-empty string")
    return value


def _read_required_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"settings.{key} must be an object")
    return value
