from __future__ import annotations

from pathlib import Path

import pytest

from optimization_control_plane.adapters.backtestsys import (
    BackTestSysDatasetDiscoveryAdapter,
    DatasetDiscoveryConfig,
)


def test_discover_data_and_replay_files_by_date() -> None:
    base = Path(__file__).resolve().parents[2] / "fixtures" / "backtestsys_gt"
    cfg = DatasetDiscoveryConfig(
        data_dir=str(base / "data_auto"),
        data_glob="market_*.csv",
        data_date_regex=r"market_(?P<date>\d{8})\.csv",
        replay_order_dir=str(base / "replay_auto"),
        replay_order_pattern="orders_{date}.csv",
        replay_cancel_dir=str(base / "replay_auto"),
        replay_cancel_pattern="cancels_{date}.csv",
    )
    discovered = BackTestSysDatasetDiscoveryAdapter().discover(cfg)
    assert len(discovered) == 2
    assert discovered[0]["date"] == "20240101"
    assert discovered[0]["order_file"].endswith("orders_20240101.csv")
    assert discovered[0]["cancel_file"].endswith("cancels_20240101.csv")


def test_discover_raises_when_replay_missing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    replay_dir = tmp_path / "replay"
    data_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    (data_dir / "market_20240101.csv").write_text("h\n", encoding="utf-8")
    (data_dir / "market_20240102.csv").write_text("h\n", encoding="utf-8")
    (replay_dir / "orders_20240101.csv").write_text("h\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        BackTestSysDatasetDiscoveryAdapter().discover(
            DatasetDiscoveryConfig(
                data_dir=str(data_dir),
                data_glob="market_*.csv",
                data_date_regex=r"market_(\d{8})\.csv",
                replay_order_dir=str(replay_dir),
                replay_order_pattern="orders_{date}.csv",
                replay_cancel_dir=str(replay_dir),
                replay_cancel_pattern="cancels_{date}.csv",
            )
        )
