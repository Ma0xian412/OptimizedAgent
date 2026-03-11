from __future__ import annotations

from pathlib import Path
from typing import Any

from optimization_control_plane.domain.models import RunResult, RunSpec


class FileRunResultLoader:
    def load(self, run_spec: RunSpec) -> RunResult:
        path = Path(run_spec.result_path)
        payload = _read_required_json(path)
        metrics = payload.get("metrics")
        diagnostics = payload.get("diagnostics")
        artifact_refs = payload.get("artifact_refs", [])
        if not isinstance(metrics, dict):
            raise ValueError(f"result file metrics must be dict: {path}")
        if not isinstance(diagnostics, dict):
            raise ValueError(f"result file diagnostics must be dict: {path}")
        if not isinstance(artifact_refs, list):
            raise ValueError(f"result file artifact_refs must be list: {path}")
        return RunResult(
            metrics=metrics,
            diagnostics=diagnostics,
            artifact_refs=artifact_refs,
        )


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"result file does not exist: {path}")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"result file payload must be object: {path}")
    return data
