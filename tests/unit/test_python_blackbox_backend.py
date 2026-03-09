from __future__ import annotations

from optimization_control_plane.adapters.execution import PythonBlackBoxExecutionBackend
from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import ExecutionRequest, RunSpec
from tests.fixtures.blackboxes import reset_instance_counter


def _make_request(target_config: dict[str, object], config: dict[str, object]) -> ExecutionRequest:
    run_spec = RunSpec(
        kind="python_blackbox",
        target_config=target_config,
        config=config,
        resources={},
    )
    return ExecutionRequest(
        request_id="req_1",
        trial_id="t1",
        run_key="run:k1",
        objective_key="obj:k1",
        cohort_id=None,
        priority=0,
        run_spec=run_spec,
    )


class TestPythonBlackBoxExecutionBackend:
    def test_callable_target_executes(self) -> None:
        backend = PythonBlackBoxExecutionBackend()
        request = _make_request(
            {
                "kind": "python_callable",
                "ref": "tests.fixtures.blackboxes:callable_target",
            },
            {"score": 0.7},
        )
        handle = backend.submit(request)

        event = backend.wait_any([handle])
        assert event is not None
        assert event.kind == EventKind.COMPLETED
        assert event.run_result is not None
        assert event.run_result.metrics["metric_1"] == 0.7

    def test_class_target_is_fresh_per_trial(self) -> None:
        reset_instance_counter()
        backend = PythonBlackBoxExecutionBackend()
        target_config: dict[str, object] = {
            "kind": "python_class",
            "ref": "tests.fixtures.blackboxes:StatefulClassTarget",
            "invoke_method": "run",
        }

        handle1 = backend.submit(_make_request(target_config, {"x": 1}))
        handle2 = backend.submit(_make_request(target_config, {"x": 1}))
        event1 = backend.wait_any([handle1, handle2])
        event2 = backend.wait_any([handle1, handle2])

        assert event1 is not None and event2 is not None
        assert event1.run_result is not None and event2.run_result is not None
        ids = {
            event1.run_result.diagnostics["instance_id"],
            event2.run_result.diagnostics["instance_id"],
        }
        call_counts = {
            event1.run_result.diagnostics["call_count"],
            event2.run_result.diagnostics["call_count"],
        }
        assert ids == {1, 2}
        assert call_counts == {1}

    def test_invalid_ref_emits_failed_event(self) -> None:
        backend = PythonBlackBoxExecutionBackend()
        request = _make_request(
            {
                "kind": "python_callable",
                "ref": "tests.fixtures.blackboxes:not_exists",
            },
            {},
        )
        handle = backend.submit(request)

        event = backend.wait_any([handle])
        assert event is not None
        assert event.kind == EventKind.FAILED
        assert event.error_code == "AttributeError"
