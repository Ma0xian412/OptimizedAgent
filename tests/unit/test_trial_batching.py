from __future__ import annotations

from optimization_control_plane.core.orchestration.trial_batching import (
    build_dataset_plan,
    with_dataset_path,
)
from optimization_control_plane.domain.models import RunSpec


def test_build_dataset_plan_split_train_test() -> None:
    settings = {
        "dataset_plan": {
            "files": [f"/tmp/file_{idx}.csv" for idx in range(10)],
            "train_ratio": 9,
            "test_ratio": 1,
            "seed": 7,
        }
    }
    plan = build_dataset_plan(settings)
    assert plan is not None
    assert len(plan.train_shards) == 9
    assert len(plan.test_shards) == 1
    assert all(shard.split == "train" for shard in plan.train_shards)
    assert all(shard.split == "test" for shard in plan.test_shards)


def test_with_dataset_path_sets_override_and_metadata() -> None:
    settings = {
        "dataset_plan": {
            "files": [{"id": "a", "path": "/tmp/a.csv"}, {"id": "b", "path": "/tmp/b.csv"}],
        }
    }
    plan = build_dataset_plan(settings)
    assert plan is not None
    base = RunSpec(kind="backtest", config={"overrides": {"x": 1}}, resources={})
    run_spec = with_dataset_path(base, plan.train_shards[0])
    assert run_spec.config["overrides"]["data.path"] == plan.train_shards[0].data_path
    assert run_spec.config["dataset"]["split"] == "train"
