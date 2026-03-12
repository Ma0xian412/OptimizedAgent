from __future__ import annotations

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestObjectiveKeyBuilderAdapter


class TestBackTestObjectiveKeyBuilderAdapter:
    def test_same_input_has_stable_key(self) -> None:
        builder = BackTestObjectiveKeyBuilderAdapter()
        objective_config: dict[str, object] = {
            "name": "pnl_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {"alpha": 1.0},
            "sampler": {"type": "tpe"},
        }

        key1 = builder.build("run:abc", objective_config)
        key2 = builder.build("run:abc", objective_config)

        assert key1 == key2
        assert key1.startswith("obj:")

    def test_run_key_change_changes_objective_key(self) -> None:
        builder = BackTestObjectiveKeyBuilderAdapter()
        objective_config: dict[str, object] = {
            "name": "pnl_loss",
            "version": "v1",
            "direction": "minimize",
            "params": {"alpha": 1.0},
        }

        key1 = builder.build("run:abc", objective_config)
        key2 = builder.build("run:def", objective_config)

        assert key1 != key2

    def test_objective_semantic_change_changes_key(self) -> None:
        builder = BackTestObjectiveKeyBuilderAdapter()
        key1 = builder.build(
            "run:abc",
            {
                "name": "pnl_loss",
                "version": "v1",
                "direction": "minimize",
                "params": {"alpha": 1.0},
            },
        )
        key2 = builder.build(
            "run:abc",
            {
                "name": "pnl_loss",
                "version": "v2",
                "direction": "minimize",
                "params": {"alpha": 1.0},
            },
        )

        assert key1 != key2

    def test_missing_name_raises(self) -> None:
        builder = BackTestObjectiveKeyBuilderAdapter()

        with pytest.raises(
            ValueError,
            match="objective_config.name must be a non-empty string",
        ):
            builder.build(
                "run:abc",
                {
                    "version": "v1",
                    "direction": "minimize",
                    "params": {},
                },
            )

    def test_non_dict_params_raises(self) -> None:
        builder = BackTestObjectiveKeyBuilderAdapter()

        with pytest.raises(
            ValueError,
            match="objective_config.params must be a dict",
        ):
            builder.build(
                "run:abc",
                {
                    "name": "pnl_loss",
                    "version": "v1",
                    "direction": "minimize",
                    "params": "alpha=1",
                },
            )
