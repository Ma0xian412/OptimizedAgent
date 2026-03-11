"""Single-machine multi-process ExecutionBackend implementation.

Job stdout protocol (optional):
- __OCP_CHECKPOINT__{"step":N,"metrics":{...}}  → CHECKPOINT event
- __OCP_RESULT__{"metrics":{...},"diagnostics":{...},"artifact_refs":[]}  → COMPLETED event

If neither prefix is used: exit 0 → COMPLETED with default RunResult; exit !=0 → FAILED.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from multiprocessing import Process, Queue
from typing import Any

from optimization_control_plane.domain.enums import EventKind, JobStatus
from optimization_control_plane.domain.models import (
    Checkpoint,
    ExecutionEvent,
    ExecutionRequest,
    Job,
    RunHandle,
    RunResult,
)

CHECKPOINT_PREFIX = "__OCP_CHECKPOINT__"
RESULT_PREFIX = "__OCP_RESULT__"


def _build_argv(job: Job) -> list[str]:
    if job.command is not None:
        return list(job.command) + list(job.args)
    if job.script_path is not None:
        return [sys.executable, job.script_path] + list(job.args)
    raise ValueError("job must set command or script_path")


def _run_worker(handle_id: str, request: ExecutionRequest, event_queue: Queue[tuple[str, ExecutionEvent]]) -> None:
    """Worker process: runs job subprocess and puts events to queue."""
    run_spec = request.run_spec
    job = run_spec.job
    env = dict(os.environ)
    env.update(job.env)
    cwd = job.working_dir
    argv = _build_argv(job)

    proc: subprocess.Popen[bytes] | None = None

    def on_term(signum: int, frame: Any) -> None:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        event_queue.put((handle_id, ExecutionEvent(
            kind=EventKind.CANCELLED,
            handle_id=handle_id,
            reason="cancelled",
        )))
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=False,
        )
    except Exception as e:
        event_queue.put((handle_id, ExecutionEvent(
            kind=EventKind.FAILED,
            handle_id=handle_id,
            error_code=f"LAUNCH_ERROR:{type(e).__name__}",
        )))
        return

    last_step = -1
    result_emitted = False

    try:
        assert proc.stdout is not None
        for line_bytes in proc.stdout:
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if line.startswith(CHECKPOINT_PREFIX):
                try:
                    payload = json.loads(line[len(CHECKPOINT_PREFIX):])
                    step = int(payload.get("step", 0))
                    metrics = payload.get("metrics", {})
                    if not isinstance(metrics, dict):
                        metrics = {}
                    if step > last_step:
                        last_step = step
                        event_queue.put((handle_id, ExecutionEvent(
                            kind=EventKind.CHECKPOINT,
                            handle_id=handle_id,
                            step=step,
                            checkpoint=Checkpoint(step=step, metrics=metrics),
                        )))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            elif line.startswith(RESULT_PREFIX):
                try:
                    payload = json.loads(line[len(RESULT_PREFIX):])
                    metrics = payload.get("metrics", {})
                    diagnostics = payload.get("diagnostics", {})
                    artifact_refs = payload.get("artifact_refs", [])
                    if not isinstance(metrics, dict):
                        metrics = {}
                    if not isinstance(diagnostics, dict):
                        diagnostics = {}
                    if not isinstance(artifact_refs, list):
                        artifact_refs = []
                    result_emitted = True
                    event_queue.put((handle_id, ExecutionEvent(
                        kind=EventKind.COMPLETED,
                        handle_id=handle_id,
                        run_result=RunResult(
                            metrics=metrics,
                            diagnostics=diagnostics,
                            artifact_refs=artifact_refs,
                        ),
                    )))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        proc.wait()
    except Exception as e:
        if proc is not None and proc.poll() is None:
            proc.kill()
        event_queue.put((handle_id, ExecutionEvent(
            kind=EventKind.FAILED,
            handle_id=handle_id,
            error_code=f"RUN_ERROR:{type(e).__name__}",
        )))
        return

    if result_emitted:
        return

    exit_code = proc.returncode if proc else -1
    if exit_code == 0:
        event_queue.put((handle_id, ExecutionEvent(
            kind=EventKind.COMPLETED,
            handle_id=handle_id,
            run_result=RunResult(
                metrics={"exit_code": exit_code},
                diagnostics={},
                artifact_refs=[],
            ),
        )))
    else:
        stderr = b""
        if proc and proc.stderr:
            stderr = proc.stderr.read()[:512]
        err_msg = stderr.decode("utf-8", errors="replace").strip() or f"exit_{exit_code}"
        event_queue.put((handle_id, ExecutionEvent(
            kind=EventKind.FAILED,
            handle_id=handle_id,
            error_code=err_msg[:200] if err_msg else f"EXIT_{exit_code}",
        )))


class MultiprocessExecutionBackend:
    """ExecutionBackend that runs each job in a separate process on the same machine."""

    def __init__(self) -> None:
        self._event_queue: Queue[tuple[str, ExecutionEvent]] = Queue()
        self._processes: dict[str, Process] = {}
        self._cancelled_events: deque[ExecutionEvent] = deque()
        self._lock = threading.Lock()

    def submit(self, request: ExecutionRequest) -> RunHandle:
        handle_id = f"mp_{uuid.uuid4().hex[:12]}"
        handle = RunHandle(
            handle_id=handle_id,
            request_id=request.request_id,
            state=JobStatus.RUNNING,
        )
        proc = Process(
            target=_run_worker,
            args=(handle_id, request, self._event_queue),
        )
        proc.start()
        with self._lock:
            self._processes[handle_id] = proc
        return handle

    def wait_any(
        self,
        handles: list[RunHandle],
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        active_ids = {h.handle_id for h in handles}
        with self._lock:
            for _ in range(len(self._cancelled_events)):
                event = self._cancelled_events.popleft()
                if event.handle_id in active_ids:
                    self._processes.pop(event.handle_id, None)
                    return event

        try:
            event_pair = self._event_queue.get(timeout=timeout if timeout and timeout > 0 else 1.0)
        except Exception:
            return None

        handle_id, event = event_pair
        with self._lock:
            self._processes.pop(handle_id, None)
        if handle_id in active_ids:
            return event
        return None

    def cancel(self, handle: RunHandle, reason: str) -> None:
        cancelled_event = ExecutionEvent(
            kind=EventKind.CANCELLED,
            handle_id=handle.handle_id,
            reason=reason,
        )
        with self._lock:
            self._cancelled_events.append(cancelled_event)
            proc = self._processes.pop(handle.handle_id, None)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.kill()
