from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import (
    ExperimentSpec,
    ResolvedTarget,
    TargetSpec,
)

_ENVELOPE_KEY = "envelope"
_KIND_KEY = "kind"
_REF_KEY = "ref"
_CONFIG_KEY = "config"
_TARGET_PREFIX_BY_KIND = {
    "package": "pkg",
    "project": "proj",
}
_SUPPORTED_KINDS = tuple(sorted(_TARGET_PREFIX_BY_KIND.keys()))


class SimpleTargetResolver:
    """Resolve package/project target envelope into a normalized target id."""

    def resolve(
        self,
        target_spec: TargetSpec,
        spec: ExperimentSpec,
    ) -> ResolvedTarget:
        envelope = self._read_envelope(target_spec)
        kind = self._read_kind(envelope)
        ref = self._read_ref(envelope)
        resolved_config = self._read_config(envelope)
        prefix = _TARGET_PREFIX_BY_KIND[kind]
        return ResolvedTarget(target_id=f"{prefix}::{ref}", config=resolved_config)

    @staticmethod
    def _read_envelope(target_spec: TargetSpec) -> dict[str, Any]:
        envelope = target_spec.config.get(_ENVELOPE_KEY)
        if not isinstance(envelope, dict):
            raise ValueError("target_spec.config.envelope must be a dict")
        return envelope

    @staticmethod
    def _read_kind(envelope: dict[str, Any]) -> str:
        kind = envelope.get(_KIND_KEY)
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("target_spec.config.envelope.kind must be a non-empty string")
        normalized = kind.strip()
        if normalized not in _TARGET_PREFIX_BY_KIND:
            raise ValueError(
                "target_spec.config.envelope.kind must be one of "
                f"{list(_SUPPORTED_KINDS)}"
            )
        return normalized

    @staticmethod
    def _read_ref(envelope: dict[str, Any]) -> str:
        ref = envelope.get(_REF_KEY)
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("target_spec.config.envelope.ref must be a non-empty string")
        return ref.strip()

    @staticmethod
    def _read_config(envelope: dict[str, Any]) -> dict[str, Any]:
        target_config = envelope.get(_CONFIG_KEY)
        if not isinstance(target_config, dict):
            raise ValueError("target_spec.config.envelope.config must be a dict")
        return dict(target_config)
