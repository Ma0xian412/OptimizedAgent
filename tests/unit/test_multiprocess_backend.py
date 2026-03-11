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


def _write_script(tmp_path: object, body: str) -> str:
    p = tmp_path / "script.py"  # type: ignore[union-attr]
    p.write_text(body)
    return str(p)


class TestMultiprocessBackend:
    def test_submit_returns_handle(self, tmp_path: object) -> None:
        body = (
            "import os,json\n"
            "p=os.environ.get('OCP_RESULT_OUTPUT_PATH','/tmp/out.json')\n"
            "open(p,'w').write(json.dumps({'metrics':{},'diagnostics':{},'artifact_refs':[]}))\n"
        )
        script = _write_script(tmp_path, body)
        backend = MultiprocessExecutionBackend()
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(job=Job(script_path=script), result_output_path=str(result_path))
        handle = backend.submit(_request(spec))
        assert handle.handle_id.startswith("mp_")
        assert handle.state == JobStatus.RUNNING
        assert handle.request_id == "req_1"

    def test_wait_any_completed_exit_zero(self, tmp_path: object) -> None:
        body = (
            "import os,json,sys\n"
            "p=os.environ.get('OCP_RESULT_OUTPUT_PATH')\n"
            "open(p,'w').write(json.dumps({'metrics':{'exit_code':0},'diagnostics':{},'artifact_refs':[]}))\n"
            "sys.exit(0)\n"
        )
        backend = MultiprocessExecutionBackend()
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(job=Job(script_path=_write_script(tmp_path, body)), result_output_path=str(result_path))
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=5.0)
        assert event is not None
        assert event.kind == EventKind.COMPLETED
        assert event.run_result is None

    def test_wait_any_failed_exit_nonzero(self, tmp_path: object) -> None:
        script = tmp_path / "fail.py"  # type: ignore[union-attr]
        script.write_text("import sys\nsys.exit(42)\n")
        backend = MultiprocessExecutionBackend()
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(job=Job(script_path=str(script)), result_output_path=str(result_path))
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=5.0)
        assert event is not None
        assert event.kind == EventKind.FAILED
        assert event.error_code is not None

    def test_wait_any_file_result(self, tmp_path: object) -> None:
        body = (
            "import os,json,sys\n"
            "p=os.environ.get('OCP_RESULT_OUTPUT_PATH')\n"
            'open(p,"w").write(json.dumps({"metrics":{"loss":0.5},"diagnostics":{"steps":10},"artifact_refs":[]}))\n'
            "sys.exit(0)\n"
        )
        backend = MultiprocessExecutionBackend()
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(job=Job(script_path=_write_script(tmp_path, body)), result_output_path=str(result_path))
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=5.0)
        assert event is not None
        assert event.kind == EventKind.COMPLETED
        assert event.run_result is None
        data = json.loads((tmp_path / "r.json").read_text())  # type: ignore[union-attr]
        assert data["metrics"]["loss"] == 0.5
        assert data["diagnostics"]["steps"] == 10

    def test_wait_any_checkpoint_protocol(self, tmp_path: object) -> None:
        body = (
            'print("__OCP_CHECKPOINT__" + \'{"step":1,"metrics":{"loss":0.8}}\')\n'
            'print("__OCP_CHECKPOINT__" + \'{"step":2,"metrics":{"loss":0.3}}\')\n'
            "import os,json,sys\n"
            "p=os.environ.get('OCP_RESULT_OUTPUT_PATH')\n"
            'open(p,"w").write(json.dumps({"metrics":{"loss":0.3},"diagnostics":{},"artifact_refs":[]}))\n'
            "sys.exit(0)\n"
        )
        backend = MultiprocessExecutionBackend()
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(job=Job(script_path=_write_script(tmp_path, body)), result_output_path=str(result_path))
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
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(job=Job(script_path=str(script)), result_output_path=str(result_path))
        handle = backend.submit(_request(spec))
        backend.cancel(handle, "pruned")
        event = backend.wait_any([handle], timeout=3.0)
        assert event is not None
        assert event.kind == EventKind.CANCELLED
        assert event.reason == "pruned"

    def test_wait_any_timeout_returns_none(self, tmp_path: object) -> None:
        backend = MultiprocessExecutionBackend()
        result_path = tmp_path / "r.json"  # type: ignore[union-attr]
        spec = RunSpec(
            job=Job(command=[os.environ.get("SHELL", "sh"), "-c", "sleep 10"]),
            result_output_path=str(result_path),
        )
        handle = backend.submit(_request(spec))
        event = backend.wait_any([handle], timeout=0.1)
        assert event is None
        backend.cancel(handle, "cleanup")
