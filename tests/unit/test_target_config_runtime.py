from __future__ import annotations

import pytest

from optimization_control_plane.adapters.objective import (
    DefaultObjectiveKeyBuilder,
    TargetAwareRunKeyBuilder,
    TargetConfigRunSpecBuilder,
)
from tests.conftest import make_spec


class TestTargetConfigRunSpecBuilder:
    def test_merge_default_config_and_params(self) -> None:
        builder = TargetConfigRunSpecBuilder()
        spec = make_spec(
            target_config={
                "kind": "python_callable",
                "ref": "tests.fixtures.blackboxes:echo_config",
                "default_config": {"x": 1, "y": 2},
            },
            execution_config={"executor_kind": "python_blackbox", "default_resources": {"cpu": 2}},
        )

        run_spec = builder.build({"y": 9, "z": 10}, spec)

        assert run_spec.kind == "python_blackbox"
        assert run_spec.target_config["ref"] == "tests.fixtures.blackboxes:echo_config"
        assert run_spec.config == {"x": 1, "y": 9, "z": 10}
        assert run_spec.resources == {"cpu": 2}

    def test_reject_unknown_params_when_allowlist_present(self) -> None:
        builder = TargetConfigRunSpecBuilder()
        spec = make_spec(
            target_config={
                "kind": "python_callable",
                "ref": "tests.fixtures.blackboxes:echo_config",
                "default_config": {"x": 1},
                "allowed_param_keys": ["x"],
            }
        )

        with pytest.raises(ValueError, match="not allowed"):
            builder.build({"z": 2}, spec)


class TestKeyBuilders:
    def test_run_key_ignores_objective_config(self) -> None:
        run_key_builder = TargetAwareRunKeyBuilder()
        run_spec_builder = TargetConfigRunSpecBuilder()
        base = make_spec()
        changed_objective = make_spec(
            objective_config={
                "name": "other_loss",
                "version": "v2",
                "direction": "minimize",
                "params": {"alpha": 1},
                "sampler": {"type": "random", "seed": 42},
                "pruner": {"type": "nop"},
            }
        )
        run_spec = run_spec_builder.build({"x": 1.0}, base)

        k1 = run_key_builder.build(run_spec, base)
        k2 = run_key_builder.build(run_spec, changed_objective)
        assert k1 == k2

    def test_run_key_changes_with_target_config(self) -> None:
        run_key_builder = TargetAwareRunKeyBuilder()
        run_spec_builder = TargetConfigRunSpecBuilder()
        spec1 = make_spec(
            target_config={
                "kind": "python_callable",
                "ref": "tests.fixtures.blackboxes:echo_config",
                "default_config": {"x": 1},
            }
        )
        spec2 = make_spec(
            target_config={
                "kind": "python_callable",
                "ref": "tests.fixtures.blackboxes:callable_target",
                "default_config": {"x": 1},
            }
        )
        run_spec1 = run_spec_builder.build({"x": 1.0}, spec1)
        run_spec2 = run_spec_builder.build({"x": 1.0}, spec2)

        k1 = run_key_builder.build(run_spec1, spec1)
        k2 = run_key_builder.build(run_spec2, spec2)
        assert k1 != k2

    def test_objective_key_depends_on_objective_config(self) -> None:
        objective_key_builder = DefaultObjectiveKeyBuilder()
        run_key = "run:abc"
        k1 = objective_key_builder.build(run_key, {"name": "loss", "version": "v1"})
        k2 = objective_key_builder.build(run_key, {"name": "loss", "version": "v2"})
        assert k1 != k2
