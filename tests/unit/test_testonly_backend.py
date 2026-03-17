from __future__ import annotations

import json
from pathlib import Path

from optimization_control_plane.adapters.execution import FakeExecutionBackend, FakeRunScript
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import ExecutionRequest, Job, RunResult, RunSpec


def _request(result_path: Path, request_id: str = "req_1") -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        trial_id="trial_1",
        run_key="run_1",
        objective_key="obj_1",
        cohort_id=None,
        priority=0,
        run_spec=RunSpec(job=Job(command=["echo", "ok"]), result_path=str(result_path)),
    )


def test_fake_backend_completed_writes_payload(tmp_path: Path) -> None:
    backend = FakeExecutionBackend()
    result_path = tmp_path / "result.json"
    backend.set_default_script(FakeRunScript(run_result=RunResult(payload={"metrics": {"m1": 1.23}})))

    handle = backend.submit(_request(result_path))
    event = backend.wait_any([handle], timeout=1.0)

    assert event is not None
    assert event.kind == EventKind.COMPLETED
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written == {"payload": {"metrics": {"m1": 1.23}}}
