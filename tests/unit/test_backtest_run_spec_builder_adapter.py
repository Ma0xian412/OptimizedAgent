from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from optimization_control_plane.adapters.backtestsys import BackTestRunSpecBuilderAdapter
from tests.conftest import make_spec


def _make_spec_for_builder(tmp_path: str) -> object:
    backtest_root = f"{tmp_path}/BackTestSys"
    base_config_path = f"{backtest_root}/config.xml"
    output_root_dir = f"{tmp_path}/ocp_artifacts"
    Path(backtest_root).mkdir(parents=True, exist_ok=True)
    Path(base_config_path).write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <data><path>data/default.csv</path></data>
  <tape><time_scale_lambda>0.0</time_scale_lambda></tape>
  <exchange><cancel_bias_k>0.0</cancel_bias_k></exchange>
  <runner><delay_in>0</delay_in><delay_out>0</delay_out></runner>
</config>
""",
        encoding="utf-8",
    )
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
            "default_resources": {
                "cpu": 2,
                "memory_gb": 1,
                "gpu": 1,
                "max_runtime_seconds": 120,
            },
            "backtest_run_spec": {
                "backtestsys_root": backtest_root,
                "base_config_path": base_config_path,
                "output_root_dir": output_root_dir,
                "dataset_paths": {"ds_a": "data/ds_a.csv"},
            },
        },
    )


class TestBackTestRunSpecBuilderAdapter:
    def test_build_run_spec_and_materialize_xml(self, tmp_path: object) -> None:
        base_root = str(tmp_path)
        backtest_root = f"{base_root}/BackTestSys"
        adapter = BackTestRunSpecBuilderAdapter()
        spec = _make_spec_for_builder(base_root)
        params = {
            "time_scale_lambda": 0.25,
            "cancel_bias_k": -0.2,
            "delay_in": 100,
            "delay_out": 200,
        }

        run_spec = adapter.build(params=params, spec=spec, dataset_id="ds_a")

        assert run_spec.job.command is not None
        assert run_spec.job.command[0] == "python3"
        assert run_spec.job.command[1] == f"{backtest_root}/main.py"
        assert run_spec.job.working_dir == backtest_root
        assert run_spec.job.args[:2] == ["--config", run_spec.job.args[1]]
        assert run_spec.job.args[2:] == ["--save-result", run_spec.result_path]
        assert run_spec.resource_request.cpu_cores == 2
        assert run_spec.resource_request.memory_mb == 1024
        assert run_spec.resource_request.gpu_count == 1
        assert run_spec.resource_request.max_runtime_seconds == 120

        trial_config_path = run_spec.job.args[1]
        tree = ET.parse(trial_config_path)
        root = tree.getroot()
        assert root.find("tape/time_scale_lambda") is not None
        assert root.find("tape/time_scale_lambda").text == "0.25"  # type: ignore[union-attr]
        assert root.find("exchange/cancel_bias_k") is not None
        assert root.find("exchange/cancel_bias_k").text == "-0.2"  # type: ignore[union-attr]
        assert root.find("runner/delay_in") is not None
        assert root.find("runner/delay_in").text == "100"  # type: ignore[union-attr]
        assert root.find("runner/delay_out") is not None
        assert root.find("runner/delay_out").text == "200"  # type: ignore[union-attr]
        assert root.find("data/path") is not None
        assert root.find("data/path").text == "data/ds_a.csv"  # type: ignore[union-attr]

    def test_missing_backtest_run_spec_raises(self, tmp_path: object) -> None:
        adapter = BackTestRunSpecBuilderAdapter()
        spec = make_spec(
            execution_config={
                "executor_kind": "backtest",
                "default_resources": {"cpu": 1},
            }
        )
        with pytest.raises(
            ValueError,
            match="spec.execution_config.backtest_run_spec must be a dict",
        ):
            adapter.build(
                params={
                    "time_scale_lambda": 0.1,
                    "cancel_bias_k": 0.1,
                    "delay_in": 1,
                    "delay_out": 1,
                },
                spec=spec,
                dataset_id="ds_a",
            )

    def test_invalid_delay_in_type_raises(self, tmp_path: object) -> None:
        adapter = BackTestRunSpecBuilderAdapter()
        spec = _make_spec_for_builder(str(tmp_path))
        with pytest.raises(ValueError, match="param delay_in must be int"):
            adapter.build(
                params={
                    "time_scale_lambda": 0.1,
                    "cancel_bias_k": 0.1,
                    "delay_in": 1.0,
                    "delay_out": 1,
                },
                spec=spec,
                dataset_id="ds_a",
            )
