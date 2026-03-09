from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _safe_filename(key: str) -> str:
    """Deterministic, filesystem-safe filename from an arbitrary key string."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    short = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key[:60])
    return f"{short}__{digest}.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp-rename to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
