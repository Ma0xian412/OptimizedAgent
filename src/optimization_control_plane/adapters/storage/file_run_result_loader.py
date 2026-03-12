from __future__ import annotations

import json
from pathlib import Path

from optimization_control_plane.domain.models import RunResult, RunSpec


class FileRunResultLoader:
    def load(self, run_spec: RunSpec) -> RunResult:
        path = Path(run_spec.result_path)
        if not path.exists():
            raise FileNotFoundError(f"run result file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("run result file must contain a JSON object")
        return RunResult(payload=data["payload"])
