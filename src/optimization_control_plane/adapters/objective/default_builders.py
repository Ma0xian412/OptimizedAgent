from __future__ import annotations

import hashlib
from typing import Any

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    RunSpec,
    stable_json_serialize,
)

_DEFAULT_EXECUTOR_KIND = "python_blackbox"
_RUN_KEY_PREFIX = "run:"
_OBJECTIVE_KEY_PREFIX = "obj:"


class TargetConfigRunSpecBuilder:
    """Builds RunSpec from target default_config and sampled params."""

    def build(self, params: dict[str, object], spec: ExperimentSpec) -> RunSpec:
        target_config = _read_target_config(spec)
        default_config = _read_default_config(target_config)
        _validate_sampled_params(params, target_config, default_config)
        merged_config = _merge_dict(default_config, params)
        resources = _read_default_resources(spec.execution_config)
        kind = _read_executor_kind(spec.execution_config)
        return RunSpec(
            kind=kind,
            target_config=target_config,
            config=merged_config,
            resources=resources,
        )


class TargetAwareRunKeyBuilder:
    """Builds run key from inputs that affect run results."""

    def build(self, run_spec: RunSpec, spec: ExperimentSpec) -> str:
        payload = stable_json_serialize({
            "kind": run_spec.kind,
            "target_config": run_spec.target_config,
            "config": run_spec.config,
            "meta": spec.meta,
            "execution_static": {
                "executor_kind": spec.execution_config.get("executor_kind"),
            },
        })
        return _RUN_KEY_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DefaultObjectiveKeyBuilder:
    """Builds objective key as run_key + objective_config."""

    def build(self, run_key: str, objective_config: dict[str, object]) -> str:
        payload = stable_json_serialize({
            "run_key": run_key,
            "objective_config": objective_config,
        })
        return _OBJECTIVE_KEY_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_target_config(spec: ExperimentSpec) -> dict[str, Any]:
    target_config = dict(spec.target_config)
    if not target_config:
        raise ValueError("spec.target_config must be a non-empty dict")
    return target_config


def _read_default_config(target_config: dict[str, Any]) -> dict[str, Any]:
    default_config = target_config.get("default_config", {})
    if not isinstance(default_config, dict):
        raise ValueError("target_config.default_config must be a dict")
    return dict(default_config)


def _validate_sampled_params(
    params: dict[str, object],
    target_config: dict[str, Any],
    default_config: dict[str, Any],
) -> None:
    allowed_param_keys = target_config.get("allowed_param_keys")
    if allowed_param_keys is None:
        return
    if not isinstance(allowed_param_keys, list):
        raise ValueError("target_config.allowed_param_keys must be a list when provided")
    allowed = set(str(key) for key in allowed_param_keys)
    unknown = sorted(key for key in params if key not in allowed and key not in default_config)
    if unknown:
        raise ValueError(f"sampled params not allowed by target_config: {unknown}")


def _merge_dict(base: dict[str, Any], override: dict[str, object]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _read_default_resources(execution_config: dict[str, Any]) -> dict[str, Any]:
    resources = execution_config.get("default_resources", {})
    if not isinstance(resources, dict):
        raise ValueError("execution_config.default_resources must be a dict when provided")
    return dict(resources)


def _read_executor_kind(execution_config: dict[str, Any]) -> str:
    kind = execution_config.get("executor_kind", _DEFAULT_EXECUTOR_KIND)
    if not isinstance(kind, str) or not kind:
        raise ValueError("execution_config.executor_kind must be a non-empty string")
    return kind
