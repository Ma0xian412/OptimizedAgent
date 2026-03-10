from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetDiscoveryConfig:
    data_dir: str
    data_glob: str
    data_date_regex: str
    replay_order_dir: str
    replay_order_pattern: str
    replay_cancel_dir: str
    replay_cancel_pattern: str


class BackTestSysDatasetDiscoveryAdapter:
    """Discover data files and resolve replay order/cancel files by date."""

    def discover(self, cfg: DatasetDiscoveryConfig) -> list[dict[str, str]]:
        data_root = Path(cfg.data_dir)
        order_root = Path(cfg.replay_order_dir)
        cancel_root = Path(cfg.replay_cancel_dir)
        if not data_root.is_dir():
            raise FileNotFoundError(f"data_dir not found: {cfg.data_dir}")
        if not order_root.is_dir():
            raise FileNotFoundError(f"replay_order_dir not found: {cfg.replay_order_dir}")
        if not cancel_root.is_dir():
            raise FileNotFoundError(f"replay_cancel_dir not found: {cfg.replay_cancel_dir}")
        regex = re.compile(cfg.data_date_regex)
        discovered: list[dict[str, str]] = []
        for data_file in sorted(data_root.glob(cfg.data_glob)):
            if not data_file.is_file():
                continue
            date = _extract_date(regex, data_file.name)
            order_file = order_root / cfg.replay_order_pattern.format(date=date)
            cancel_file = cancel_root / cfg.replay_cancel_pattern.format(date=date)
            if not order_file.is_file():
                raise FileNotFoundError(f"missing replay order file for {data_file.name}: {order_file}")
            if not cancel_file.is_file():
                raise FileNotFoundError(f"missing replay cancel file for {data_file.name}: {cancel_file}")
            discovered.append(
                {
                    "id": f"ds_{date}",
                    "path": str(data_file),
                    "date": date,
                    "order_file": str(order_file),
                    "cancel_file": str(cancel_file),
                }
            )
        if len(discovered) < 2:
            raise ValueError("auto discovery requires at least 2 data files")
        return discovered


def _extract_date(regex: re.Pattern[str], filename: str) -> str:
    matched = regex.fullmatch(filename)
    if matched is None:
        raise ValueError(f"filename does not match data_date_regex: {filename}")
    if "date" in matched.groupdict():
        date = matched.group("date")
    elif matched.groups():
        date = matched.group(1)
    else:
        raise ValueError("data_date_regex must have a named group 'date' or first capture group")
    if not date:
        raise ValueError(f"invalid date extracted from filename: {filename}")
    return date
