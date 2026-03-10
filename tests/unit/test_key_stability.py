"""UT-1/2/3: spec_hash, run_key, objective_key stability."""
from __future__ import annotations

from optimization_control_plane.domain.models import RunSpec, TargetSpec, compute_spec_hash
from tests.conftest import (
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    make_spec,
)


class TestSpecHash:
    def test_deterministic(self) -> None:
        target = TargetSpec(target_id="target_a", config={"exchange": "A"})
        h1 = compute_spec_hash("s1", {"a": 1}, target, {"b": 2}, {"c": 3})
        h2 = compute_spec_hash("s1", {"a": 1}, target, {"b": 2}, {"c": 3})
        assert h1 == h2

    def test_changes_when_target_spec_changes(self) -> None:
        h1 = compute_spec_hash(
            "s1",
            {"a": 1},
            TargetSpec(target_id="target_a", config={"exchange": "A"}),
            {"b": 2},
            {"c": 3},
        )
        h2 = compute_spec_hash(
            "s1",
            {"a": 1},
            TargetSpec(target_id="target_b", config={"exchange": "A"}),
            {"b": 2},
            {"c": 3},
        )
        assert h1 != h2

    def test_stable_when_target_spec_is_equivalent(self) -> None:
        h1 = compute_spec_hash(
            "s1",
            {"a": 1},
            TargetSpec(target_id="target_a", config={"b": 2, "a": 1}),
            {"x": 1},
            {"y": 1},
        )
        h2 = compute_spec_hash(
            "s1",
            {"a": 1},
            TargetSpec(target_id="target_a", config={"a": 1, "b": 2}),
            {"x": 1},
            {"y": 1},
        )
        assert h1 == h2

    def test_changes_when_objective_or_execution_changes(self) -> None:
        target = TargetSpec(target_id="target_a", config={"exchange": "A"})
        baseline = compute_spec_hash("s1", {"a": 1}, target, {"b": 2}, {"c": 3})
        changed_objective = compute_spec_hash("s1", {"a": 1}, target, {"b": 3}, {"c": 3})
        changed_execution = compute_spec_hash("s1", {"a": 1}, target, {"b": 2}, {"c": 4})
        assert baseline != changed_objective
        assert baseline != changed_execution


class TestRunKeyStability:
    def test_deterministic(self) -> None:
        builder = StubRunKeyBuilder()
        spec = make_spec()
        rs = RunSpec(
            kind="test",
            config={"x": 1.0},
            resources={},
            target_spec=spec.target_spec,
        )
        k1 = builder.build(rs, spec)
        k2 = builder.build(rs, spec)
        assert k1 == k2

    def test_different_config_different_key(self) -> None:
        builder = StubRunKeyBuilder()
        spec = make_spec()
        k1 = builder.build(
            RunSpec(
                kind="test",
                config={"x": 1.0},
                resources={},
                target_spec=spec.target_spec,
            ),
            spec,
        )
        k2 = builder.build(
            RunSpec(
                kind="test",
                config={"x": 2.0},
                resources={},
                target_spec=spec.target_spec,
            ),
            spec,
        )
        assert k1 != k2

    def test_same_params_different_target_different_key(self) -> None:
        builder = StubRunKeyBuilder()
        spec_a = make_spec(target_spec={"target_id": "target_a", "config": {"market": "us"}})
        spec_b = make_spec(target_spec={"target_id": "target_b", "config": {"market": "us"}})
        run_spec_a = RunSpec(
            kind="test",
            config={"x": 1.0},
            resources={},
            target_spec=spec_a.target_spec,
        )
        run_spec_b = RunSpec(
            kind="test",
            config={"x": 1.0},
            resources={},
            target_spec=spec_b.target_spec,
        )
        assert builder.build(run_spec_a, spec_a) != builder.build(run_spec_b, spec_b)

    def test_same_params_same_target_same_key(self) -> None:
        builder = StubRunKeyBuilder()
        spec = make_spec(target_spec={"target_id": "target_a", "config": {"market": "us"}})
        run_spec_1 = RunSpec(
            kind="test",
            config={"x": 1.0},
            resources={},
            target_spec=spec.target_spec,
        )
        run_spec_2 = RunSpec(
            kind="test",
            config={"x": 1.0},
            resources={},
            target_spec=spec.target_spec,
        )
        assert builder.build(run_spec_1, spec) == builder.build(run_spec_2, spec)


class TestObjectiveKeyStability:
    def test_deterministic(self) -> None:
        builder = StubObjectiveKeyBuilder()
        cfg = {"name": "loss", "version": "v1", "params": {}}
        k1 = builder.build("run:abc", cfg)
        k2 = builder.build("run:abc", cfg)
        assert k1 == k2

    def test_different_run_key_different_objective_key(self) -> None:
        builder = StubObjectiveKeyBuilder()
        cfg = {"name": "loss", "version": "v1", "params": {}}
        k1 = builder.build("run:abc", cfg)
        k2 = builder.build("run:def", cfg)
        assert k1 != k2

    def test_different_version_different_key(self) -> None:
        builder = StubObjectiveKeyBuilder()
        k1 = builder.build("run:abc", {"name": "loss", "version": "v1", "params": {}})
        k2 = builder.build("run:abc", {"name": "loss", "version": "v2", "params": {}})
        assert k1 != k2
