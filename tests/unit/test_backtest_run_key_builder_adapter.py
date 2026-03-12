from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import (
    BackTestRunKeyBuilderAdapter,
    BackTestRunSpecBuilderAdapter,
)
from optimization_control_plane.domain.models import Job, RunSpec
from tests.conftest import make_spec


def _write_base_config(base_config_path: str) -> None:
    Path(base_config_path).parent.mkdir(parents=True, exist_ok=True)
    Path(base_config_path).write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <data><path>data/default.csv</path></data>
  <tape><time_scale_lambda>0.0</time_scale_lambda></tape>
  <exchange><cancel_bias_k>0.0</cancel_bias_k></exchange>
  <runner><delay_in>0</delay_in><delay_out>0</delay_out></runner>
  <strategy>
    <params>
      <order_file>orders/default.csv</order_file>
      <cancel_file>cancels/default.csv</cancel_file>
    </params>
  </strategy>
</config>
""",
        encoding="utf-8",
    )


def _make_spec(tmp_root: str) -> object:
    backtest_root = f"{tmp_root}/BackTestSys"
    base_config_path = f"{backtest_root}/config.xml"
    _write_base_config(base_config_path)
    return make_spec(
        objective_config={
            "name": "loss",
            "version": "v1",
            "direction": "minimize",
            "params": {},
            "groundtruth": {"version": "gt_v1", "path": "/tmp/gt.json"},
            "sampler": {"type": "random", "seed": 42},
            "pruner": {"type": "nop"},
        },
        execution_config={
            "executor_kind": "backtest",
            "default_resources": {"cpu": 1},
            "backtest_run_spec": {
                "backtestsys_root": backtest_root,
                "base_config_path": base_config_path,
                "output_root_dir": f"{tmp_root}/artifacts",
                "dataset_inputs": {
                    "ds_a": {
                        "market_data_path": "data/ds_a.csv",
                        "order_file": "orders/ds_a.csv",
                        "cancel_file": "cancels/ds_a.csv",
                    },
                    "ds_b": {
                        "market_data_path": "data/ds_b.csv",
                        "order_file": "orders/ds_b.csv",
                        "cancel_file": "cancels/ds_b.csv",
                    },
                },
            },
        },
    )


class TestBackTestRunKeyBuilderAdapter:
    def test_same_input_has_stable_key(self, tmp_path: object) -> None:
        spec = _make_spec(str(tmp_path))
        run_spec_builder = BackTestRunSpecBuilderAdapter()
        run_key_builder = BackTestRunKeyBuilderAdapter()
        params = {
            "time_scale_lambda": 0.1,
            "cancel_bias_k": -0.3,
            "delay_in": 50,
            "delay_out": 100,
        }
        run_spec = run_spec_builder.build(params, spec, "ds_a")

        key1 = run_key_builder.build(run_spec, spec, "ds_a")
        key2 = run_key_builder.build(run_spec, spec, "ds_a")

        assert key1 == key2
        assert key1.startswith("run:")

    def test_parameter_change_changes_key(self, tmp_path: object) -> None:
        spec = _make_spec(str(tmp_path))
        run_spec_builder = BackTestRunSpecBuilderAdapter()
        run_key_builder = BackTestRunKeyBuilderAdapter()
        run_spec_1 = run_spec_builder.build(
            {
                "time_scale_lambda": 0.1,
                "cancel_bias_k": -0.3,
                "delay_in": 50,
                "delay_out": 100,
            },
            spec,
            "ds_a",
        )
        run_spec_2 = run_spec_builder.build(
            {
                "time_scale_lambda": 0.2,
                "cancel_bias_k": -0.3,
                "delay_in": 50,
                "delay_out": 100,
            },
            spec,
            "ds_a",
        )

        key1 = run_key_builder.build(run_spec_1, spec, "ds_a")
        key2 = run_key_builder.build(run_spec_2, spec, "ds_a")

        assert key1 != key2

    def test_dataset_id_change_changes_key(self, tmp_path: object) -> None:
        spec = _make_spec(str(tmp_path))
        run_spec_builder = BackTestRunSpecBuilderAdapter()
        run_key_builder = BackTestRunKeyBuilderAdapter()
        params = {
            "time_scale_lambda": 0.1,
            "cancel_bias_k": -0.3,
            "delay_in": 50,
            "delay_out": 100,
        }
        run_spec_a = run_spec_builder.build(params, spec, "ds_a")
        run_spec_b = run_spec_builder.build(params, spec, "ds_b")

        key_a = run_key_builder.build(run_spec_a, spec, "ds_a")
        key_b = run_key_builder.build(run_spec_b, spec, "ds_b")

        assert key_a != key_b

    def test_missing_config_flag_raises(self) -> None:
        run_key_builder = BackTestRunKeyBuilderAdapter()
        spec = make_spec()
        run_spec = RunSpec(
            job=Job(command=["python3", "main.py"], args=["--save-result", "/tmp/out"]),
            result_path="/tmp/out",
        )

        with pytest.raises(ValueError, match="run_spec.job.args must include --config <path>"):
            run_key_builder.build(run_spec, spec, "ds_a")
