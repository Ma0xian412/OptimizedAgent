"""Unit tests for MultiprocessExecutionBackend."""
from __future__ import annotations

import json
import os

from optimization_control_plane.adapters.execution import MultiprocessExecutionBackend
from optimization_control_plane.domain.enums import EventKind, JobStatus
from optimization_control_plane.domain.models import (
    ExecutionRequest,
    Job,
    RunSpec,
)


def _request(run_spec: RunSpec, request_id: str = "req_1") -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        trial_id="t1",
        run_key="rk1",
        objective_key="ok1",
        cohort_id=None,
        priority=0,
        run_spec=run_spec,
    )


class TestMultiprocessBackend:
    def test_submit_returns_handle(self) -> None:
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(
            job=Job(command=[os.environ.get("SHELL", "sh"), "-c", "exit 0"]),
        )
        handle = backend.submit(_request(spec))
        assert handle.handle_id.startswith("mp_")
        assert handle.state == JobStatus.RUNNING
        assert handle.request_id == "req_1"

    def test_wait_any_completed_exit_zero(self, tmp_path: object) -> None:
        script = tmp_path / "ok.py"  # type: ignore[union-attr]
        script.write_text("import sys\nsys.exit(0)\n")
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(job=Job(script_path=str(script)))
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=5.0)
        assert event is not None
        assert event.kind == EventKind.COMPLETED
        assert event.run_result is not None
        assert event.run_result.metrics.get("exit_code") == 0

    def test_wait_any_failed_exit_nonzero(self, tmp_path: object) -> None:
        script = tmp_path / "fail.py"  # type: ignore[union-attr]
        script.write_text("import sys\nsys.exit(42)\n")
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(job=Job(script_path=str(script)))
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=5.0)
        assert event is not None
        assert event.kind == EventKind.FAILED
        assert event.error_code is not None

    def test_wait_any_stdout_result_protocol(self, tmp_path: object) -> None:
        payload = {"metrics": {"loss": 0.5}, "diagnostics": {"steps": 10}, "artifact_refs": []}
        script = tmp_path / "result.py"  # type: ignore[union-attr]
        line = "__OCP_RESULT__" + json.dumps(payload)
        script.write_text(
            f"print({repr(line)})\n"
            "import sys\n"
            "sys.exit(0)\n",
        )
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(job=Job(script_path=str(script)))
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=5.0)
        assert event is not None
        assert event.kind == EventKind.COMPLETED
        assert event.run_result is not None
        assert event.run_result.metrics == {"loss": 0.5}
        assert event.run_result.diagnostics == {"steps": 10}

    def test_wait_any_checkpoint_protocol(self, tmp_path: object) -> None:
        script = tmp_path / "checkpoint.py"  # type: ignore[union-attr]
        script.write_text(
            'print("__OCP_CHECKPOINT__" + \'{"step":1,"metrics":{"loss":0.8}}\')\n'
            'print("__OCP_CHECKPOINT__" + \'{"step":2,"metrics":{"loss":0.3}}\')\n'
            'print("__OCP_RESULT__" + \'{"metrics":{"loss":0.3},"diagnostics":{},"artifact_refs":[]}\')\n'
            "import sys\n"
            "sys.exit(0)\n",
        )
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(job=Job(script_path=str(script)))
        handle = backend.submit(_request(spec))
        events: list = []
        while len(events) < 3:
            ev = backend.wait_any([handle], timeout=5.0)
            if ev is None:
                continue
            events.append(ev)
            if ev.kind == EventKind.COMPLETED:
                break
        assert len(events) >= 2
        checkpoints = [e for e in events if e.kind == EventKind.CHECKPOINT]
        assert len(checkpoints) == 2
        assert checkpoints[0].step == 1
        assert checkpoints[0].checkpoint is not None
        assert checkpoints[0].checkpoint.metrics["loss"] == 0.8
        assert checkpoints[1].step == 2
        assert events[-1].kind == EventKind.COMPLETED

    def test_cancel(self, tmp_path: object) -> None:
        script = tmp_path / "sleep.py"  # type: ignore[union-attr]
        script.write_text("import time\ntime.sleep(60)\n")
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(job=Job(script_path=str(script)))
        handle = backend.submit(_request(spec))
        backend.cancel(handle, "pruned")
        event = backend.wait_any([handle], timeout=3.0)
        assert event is not None
        assert event.kind == EventKind.CANCELLED
        assert event.reason == "pruned"

    def test_wait_any_timeout_returns_none(self) -> None:
        backend = MultiprocessExecutionBackend()
        spec = RunSpec(
            job=Job(command=[os.environ.get("SHELL", "sh"), "-c", "sleep 10"]),
        )
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=0.1)
        assert event is None
        backend.cancel(handle, "cleanup")
