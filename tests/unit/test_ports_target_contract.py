"""UT: target explicit boundary contracts for objective/execution ports."""
from __future__ import annotations

from optimization_control_plane.adapters.execution import FakeExecutionBackend
from optimization_control_plane.domain.models import ExecutionRequest, RunSpec, TargetSpec
from tests.conftest import StubRunSpecBuilder


def test_run_spec_builder_output_contains_explicit_target_spec() -> None:
    builder = StubRunSpecBuilder()
    target_spec = TargetSpec(target_id="target_alpha", config={"market": "us_equity"})
    run_spec = builder.build(
        target_spec=target_spec,
        params={"x": 1.5},
        execution_config={"default_resources": {"cpu": 2}},
    )

    assert run_spec.target_spec == target_spec
    assert run_spec.config == {"x": 1.5}
    assert run_spec.resources == {"cpu": 2}


def test_fake_execution_backend_reads_explicit_target_from_run_spec() -> None:
    backend = FakeExecutionBackend()
    target_spec = TargetSpec(target_id="target_beta", config={"region": "us"})
    request = ExecutionRequest(
        request_id="req_1",
        trial_id="trial_1",
        run_key="run_1",
        objective_key="obj_1",
        cohort_id=None,
        priority=0,
        run_spec=RunSpec(
            kind="backtest",
            config={"x": 1.0},
            resources={"cpu": 1},
            target_spec=target_spec,
        ),
    )

    handle = backend.submit(request)
    submitted = backend.get_submitted_request(handle.handle_id)

    assert submitted.run_spec.target_spec.target_id == "target_beta"
    assert "target_id" not in submitted.run_spec.config
