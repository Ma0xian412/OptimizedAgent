"""IT-5: Graceful stop."""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

from optimization_control_plane.adapters.optuna import OptunaBackendAdapter
from optimization_control_plane.adapters.policies import (
    AsyncFillParallelismPolicy,
    SubmitNowDispatchPolicy,
)
from optimization_control_plane.adapters.storage import (
    FileObjectiveCache,
    FileResultStore,
    FileRunCache,
)
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import ExecutionEvent, ExecutionRequest, RunHandle, RunResult
from tests.conftest import (
    StubGroundTruthProvider,
    StubObjectiveEvaluator,
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    StubRunSpecBuilder,
    StubSearchSpace,
    make_settings,
    make_spec,
)


class DelayedCompletionBackend:
    def __init__(self) -> None:
        self._handle: RunHandle | None = None
        self._wait_count = 0
        self.submitted = threading.Event()

    def submit(self, request: ExecutionRequest) -> RunHandle:
        self._handle = RunHandle(
            handle_id=f"fh_{uuid.uuid4().hex[:12]}",
            request_id=request.request_id,
            state="RUNNING",
        )
        self.submitted.set()
        return self._handle

    def wait_any(
        self,
        handles: list[RunHandle],
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        if self._handle is None or self._handle not in handles:
            return None
        self._wait_count += 1
        time.sleep(0.05)
        if self._wait_count == 1:
            return None
        return ExecutionEvent(
            kind=EventKind.COMPLETED,
            handle_id=self._handle.handle_id,
            run_result=RunResult(metrics={"metric_1": 0.5}, diagnostics={}, artifact_refs=[]),
        )

    def cancel(self, handle: RunHandle, reason: str) -> None:
        return None


class TestGracefulStop:
    def test_stop_prevents_new_trials(self, tmp_path: Any) -> None:
        db = os.path.join(str(tmp_path), "test.db")
        backend = OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")

        exec_be = DelayedCompletionBackend()

        obj_def = ObjectiveDefinition(
            search_space=StubSearchSpace({"x": 1.0}),
            run_spec_builder=StubRunSpecBuilder(),
            run_key_builder=StubRunKeyBuilder(),
            objective_key_builder=StubObjectiveKeyBuilder(),
            progress_scorer=None,
            objective_evaluator=StubObjectiveEvaluator(),
        )

        orch = TrialOrchestrator(
            backend=backend,
            objective_def=obj_def,
            groundtruth_provider=StubGroundTruthProvider(),
            execution_backend=exec_be,
            parallelism_policy=AsyncFillParallelismPolicy(),
            dispatch_policy=SubmitNowDispatchPolicy(),
            run_cache=FileRunCache(os.path.join(str(tmp_path), "data")),
            objective_cache=FileObjectiveCache(os.path.join(str(tmp_path), "data")),
            result_store=FileResultStore(os.path.join(str(tmp_path), "data")),
        )

        spec = make_spec()
        settings = make_settings(
            stop={"max_trials": 1000},
            parallelism={"max_in_flight_trials": 1},
        )
        runner = threading.Thread(target=orch.start, args=(spec, settings))
        runner.start()
        assert exec_be.submitted.wait(timeout=1)
        orch.stop()
        runner.join(timeout=5)

        m = orch.metrics.snapshot()
        assert not runner.is_alive()
        assert m["trials_asked_total"] == 1
