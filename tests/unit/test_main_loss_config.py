from __future__ import annotations

from pathlib import Path

from main import _build_spec, _load_config


def test_build_spec_includes_loss_config_from_xml(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        loss_block="""
    <loss>
        <weights>
            <curve>2.0</curve>
            <terminal>1.0</terminal>
            <cancel>3.0</cancel>
            <post>4.0</post>
        </weights>
        <eps>
            <curve>0.1</curve>
            <terminal>0.2</terminal>
            <cancel>0.3</cancel>
            <post>0.4</post>
        </eps>
    </loss>
""",
    )
    spec = _build_spec(_load_config(str(config_path)))
    assert spec.objective_config["weights"] == {
        "curve": 2.0,
        "terminal": 1.0,
        "cancel": 3.0,
        "post": 4.0,
    }
    assert spec.objective_config["eps"] == {
        "curve": 0.1,
        "terminal": 0.2,
        "cancel": 0.3,
        "post": 0.4,
    }


def test_build_spec_writes_default_loss_config_when_loss_missing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, loss_block="")
    spec = _build_spec(_load_config(str(config_path)))
    assert spec.objective_config["weights"] == {
        "curve": 1.0,
        "terminal": 1.0,
        "cancel": 1.0,
        "post": 1.0,
    }
    assert spec.objective_config["eps"] == {
        "curve": 1e-12,
        "terminal": 1e-12,
        "cancel": 1e-12,
        "post": 1e-12,
    }


def _write_config(tmp_path: Path, *, loss_block: str) -> Path:
    config = tmp_path / "config.xml"
    config.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<optimization>
    <study>
        <spec_id>test_spec</spec_id>
        <dataset_version>v1</dataset_version>
        <engine_version>engine_v1</engine_version>
        <storage_dsn>sqlite:///{tmp_path / "study.db"}</storage_dsn>
        <max_trials>2</max_trials>
        <max_failures>2</max_failures>
        <max_in_flight_trials>1</max_in_flight_trials>
        <max_workers>1</max_workers>
    </study>
    <paths>
        <data_dir>{tmp_path / "data"}</data_dir>
        <backtestsys_repo_root>/workspace/BackTestSys</backtestsys_repo_root>
        <backtestsys_base_config>/workspace/BackTestSys/config.xml</backtestsys_base_config>
        <groundtruth_dir>{tmp_path}</groundtruth_dir>
    </paths>
    <dataset_plan>
        <train_ratio>1</train_ratio>
        <test_ratio>1</test_ratio>
        <seed>42</seed>
        <files>
            <file id="d1" path="/tmp/data_1.csv" order_file="/tmp/orders_1.csv" cancel_file="/tmp/cancels_1.csv" />
            <file id="d2" path="/tmp/data_2.csv" order_file="/tmp/orders_2.csv" cancel_file="/tmp/cancels_2.csv" />
        </files>
    </dataset_plan>
    <sampler>
        <type>random</type>
        <seed>42</seed>
    </sampler>
    <pruner>
        <type>nop</type>
    </pruner>
{loss_block}
    <search_space>
        <param name="runner.delay_in" type="int" low="0" high="1" />
    </search_space>
</optimization>
""",
        encoding="utf-8",
    )
    return config
