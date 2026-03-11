"""UT-1/2/3: spec_hash, run_key, objective_key stability."""
from __future__ import annotations

from optimization_control_plane.domain.models import Job, RunSpec, compute_spec_hash
from tests.conftest import (
    StubObjectiveKeyBuilder,
    StubRunKeyBuilder,
    make_spec,
)


class TestSpecHash:
    def test_deterministic(self) -> None:
        h1 = compute_spec_hash("s1", {"a": 1}, {"b": 2}, {"c": 3})
        h2 = compute_spec_hash("s1", {"a": 1}, {"b": 2}, {"c": 3})
        assert h1 == h2

    def test_different_input_different_hash(self) -> None:
        h1 = compute_spec_hash("s1", {"a": 1}, {"b": 2}, {"c": 3})
        h2 = compute_spec_hash("s2", {"a": 1}, {"b": 2}, {"c": 3})
        assert h1 != h2

    def test_key_order_irrelevant(self) -> None:
        h1 = compute_spec_hash("s1", {"b": 2, "a": 1}, {}, {})
        h2 = compute_spec_hash("s1", {"a": 1, "b": 2}, {}, {})
        assert h1 == h2


class TestRunKeyStability:
    def test_deterministic(self) -> None:
        builder = StubRunKeyBuilder()
        spec = make_spec()
        rs = RunSpec(job=Job(command=["python"], args=["--x=1.0"]))
        k1 = builder.build(rs, spec, "ds_v1")
        k2 = builder.build(rs, spec, "ds_v1")
        assert k1 == k2

    def test_different_config_different_key(self) -> None:
        builder = StubRunKeyBuilder()
        spec = make_spec()
        k1 = builder.build(RunSpec(job=Job(command=["python"], args=["--x=1.0"])), spec, "ds_v1")
        k2 = builder.build(RunSpec(job=Job(command=["python"], args=["--x=2.0"])), spec, "ds_v1")
        assert k1 != k2


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
