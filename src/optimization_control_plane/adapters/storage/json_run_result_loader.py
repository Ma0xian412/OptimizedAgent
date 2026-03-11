"""JSON-based RunResultLoader implementation."""
from __future__ import annotations

import json
from pathlib import Path

from optimization_control_plane.domain.models import RunResult


class JsonRunResultLoader:
    """Load RunResult from a JSON file at the given path."""

    def load(self, path: str) -> RunResult:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"run result file not found: {path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        diagnostics = data.get("diagnostics", {})
        artifact_refs = data.get("artifact_refs", [])
        if not isinstance(metrics, dict):
            metrics = {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        if not isinstance(artifact_refs, list):
            artifact_refs = []
        return RunResult(
            metrics=metrics,
            diagnostics=diagnostics,
            artifact_refs=artifact_refs,
        )
