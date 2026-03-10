"""UT: target explicit boundary contracts for objective/execution ports."""
from __future__ import annotations

import pytest

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
    request = _build_request("req_1", "trial_1", "run_1", "obj_1", "target_beta")

    handle = backend.submit(request)
    submitted = backend.get_submitted_request(handle.handle_id)

    assert submitted.run_spec.target_spec.target_id == "target_beta"
    assert backend.get_submitted_target_id(handle.handle_id) == "target_beta"
    assert "target_id" not in submitted.run_spec.config


def test_fake_execution_backend_distinguishes_two_explicit_targets() -> None:
    backend = FakeExecutionBackend()
    handle_a = backend.submit(
        _build_request("req_a", "trial_a", "run_a", "obj_a", "target_a")
    )
    handle_b = backend.submit(
        _build_request("req_b", "trial_b", "run_b", "obj_b", "target_b")
    )

    assert backend.get_submitted_target_id(handle_a.handle_id) == "target_a"
    assert backend.get_submitted_target_id(handle_b.handle_id) == "target_b"


def test_fake_execution_backend_submit_missing_target_fails_fast() -> None:
    backend = FakeExecutionBackend()
    run_spec = RunSpec(
        kind="backtest",
        config={"x": 1.0},
        resources={"cpu": 1},
        target_spec=TargetSpec(target_id="target_ok", config={}),
    )
    object.__setattr__(run_spec, "target_spec", None)
    request = _build_request(
        "req_missing",
        "trial_missing",
        "run_missing",
        "obj_missing",
        "target_ignored",
        run_spec=run_spec,
    )

    with pytest.raises(ValueError, match="request.run_spec.target_spec must be a TargetSpec"):
        backend.submit(request)


def _build_request(
    request_id: str,
    trial_id: str,
    run_key: str,
    objective_key: str,
    target_id: str,
    *,
    run_spec: RunSpec | None = None,
) -> ExecutionRequest:
    payload = run_spec or RunSpec(
        kind="backtest",
        config={"x": 1.0},
        resources={"cpu": 1},
        target_spec=TargetSpec(target_id=target_id, config={"region": "us"}),
    )
    return ExecutionRequest(
        request_id=request_id,
        trial_id=trial_id,
        run_key=run_key,
        objective_key=objective_key,
        cohort_id=None,
        priority=0,
        run_spec=payload,
    )
