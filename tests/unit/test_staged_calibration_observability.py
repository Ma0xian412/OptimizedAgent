from __future__ import annotations

import json
from pathlib import Path

from optimization_control_plane.adapters.backtestsys.staged_calibration_observability import (
    StageProgressContext,
    StagedCalibrationProgressReporter,
)


def _read_progress_lines(progress_path: Path) -> list[dict[str, object]]:
    lines = progress_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_stage_progress_only_emits_when_metrics_change(tmp_path: Path, capsys) -> None:
    reporter = StagedCalibrationProgressReporter(tmp_path, "run_tag", output_format="text")
    ctx = StageProgressContext(stage_name="baseline", unit_index=1, unit_total=4, max_trials=10)
    stable_metrics = {"trials_completed_total": 0, "trials_failed_total": 0, "inflight_leader_executions_gauge": 1}

    reporter.stage_progress(ctx, stable_metrics)
    reporter.stage_progress(ctx, dict(stable_metrics))
    reporter.stage_progress(ctx, {**stable_metrics, "trials_completed_total": 1})

    stdout = capsys.readouterr().out
    progress_lines = [line for line in stdout.splitlines() if "PROGRESS baseline" in line]
    assert len(progress_lines) == 2

    records = _read_progress_lines(tmp_path / "progress.jsonl")
    stage_progress_records = [record for record in records if record.get("event") == "stage_progress"]
    assert len(stage_progress_records) == 2
