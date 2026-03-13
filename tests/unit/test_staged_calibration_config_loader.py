from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys.staged_calibration_config_loader import (
    calibration_config_summary,
    load_calibration_config,
)


def _write_config(path: Path, xml_body: str) -> None:
    path.write_text(xml_body, encoding="utf-8")


def _base_xml(tmp_path: Path, market_path: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<staged_calibration>
  <workspace_root>{tmp_path}</workspace_root>
  <runtime_root>{tmp_path / "runtime"}</runtime_root>
  <backtestsys_root>{tmp_path / "BackTestSys"}</backtestsys_root>
  <base_config_path>{tmp_path / "config.xml"}</base_config_path>
  <python_executable>/usr/bin/python3</python_executable>
  <max_failures>2</max_failures>
  <baseline_trials>1</baseline_trials>
  <machine_delay_trials>5</machine_delay_trials>
  <contract_core_trials>6</contract_core_trials>
  <verify_trials>1</verify_trials>
  <max_in_flight_trials>3</max_in_flight_trials>
  <default_resources>
    <cpu>1</cpu>
    <max_runtime_seconds>60</max_runtime_seconds>
  </default_resources>
  <search_ranges>
    <delay>
      <low>0</low>
      <high>500000</high>
    </delay>
    <time_scale_lambda>
      <low>-0.5</low>
      <high>0.5</high>
    </time_scale_lambda>
    <cancel_bias_k>
      <low>-1.0</low>
      <high>1.0</high>
    </cancel_bias_k>
  </search_ranges>
  <datasets>
    <dataset>
      <dataset_id>ds_01</dataset_id>
      <market_data_path>{market_path}</market_data_path>
      <order_file>{tmp_path / "inputs" / "orders.csv"}</order_file>
      <cancel_file>{tmp_path / "inputs" / "cancels.csv"}</cancel_file>
      <machine>m1</machine>
      <contract>c1</contract>
      <groundtruth_doneinfo_path>{tmp_path / "inputs" / "gt_done.csv"}</groundtruth_doneinfo_path>
      <groundtruth_executiondetail_path>{tmp_path / "inputs" / "gt_exec.csv"}</groundtruth_executiondetail_path>
    </dataset>
  </datasets>
</staged_calibration>
"""


def test_load_calibration_config_reads_xml_and_builds_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "staged_config.xml"
    _write_config(config_path, _base_xml(tmp_path, str(tmp_path / "inputs" / "market.csv")))

    config = load_calibration_config(config_path)
    summary = calibration_config_summary(config, source=config_path)

    assert config.datasets[0].dataset_id == "ds_01"
    assert config.delay_range.low == 0
    assert config.delay_range.high == 500000
    assert config.max_in_flight_trials == 3
    assert summary["dataset_count"] == 1
    assert summary["parallelism"]["max_in_flight_trials"] == 3
    assert summary["search_ranges"]["cancel_bias_k"]["high"] == 1.0


def test_load_calibration_config_requires_absolute_dataset_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "staged_config.xml"
    _write_config(config_path, _base_xml(tmp_path, "relative/market.csv"))

    with pytest.raises(ValueError, match="config.datasets.dataset\\[0\\]\\.market_data_path"):
        load_calibration_config(config_path)


def test_load_calibration_config_requires_absolute_config_path(tmp_path: Path, monkeypatch: object) -> None:
    config_path = tmp_path / "staged_config.xml"
    _write_config(config_path, _base_xml(tmp_path, str(tmp_path / "inputs" / "market.csv")))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="--config must be an absolute path"):
        load_calibration_config(Path("staged_config.xml"))


def test_load_calibration_config_parallelism_defaults_to_one(tmp_path: Path) -> None:
    config_path = tmp_path / "staged_config.xml"
    xml = _base_xml(tmp_path, str(tmp_path / "inputs" / "market.csv")).replace(
        "  <max_in_flight_trials>3</max_in_flight_trials>\n", ""
    )
    _write_config(config_path, xml)

    config = load_calibration_config(config_path)
    assert config.max_in_flight_trials == 1
