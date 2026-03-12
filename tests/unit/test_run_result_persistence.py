from __future__ import annotations

import json
from pathlib import Path

from optimization_control_plane.adapters.storage.file_result_store import FileResultStore
from optimization_control_plane.adapters.storage.file_run_cache import FileRunCache
from optimization_control_plane.domain.models import RunResult


def test_file_run_cache_roundtrip_payload(tmp_path: Path) -> None:
    cache = FileRunCache(base_dir=tmp_path)
    expected = RunResult(payload={"raw": {"x": 1}, "artifact_refs": ["a.csv"]})

    cache.put("run_key_1", expected)
    loaded = cache.get("run_key_1")

    assert loaded is not None
    assert loaded.payload == expected.payload


def test_file_result_store_write_run_record_payload(tmp_path: Path) -> None:
    store = FileResultStore(base_dir=tmp_path)
    run_key = "rk_payload"
    result = RunResult(payload={"table_rows": {"DoneInfo": [{"OrderId": "1"}]}})

    store.write_run_record(run_key, result)

    records_dir = tmp_path / "run_records"
    record_files = sorted(records_dir.glob("*.json"))
    assert len(record_files) == 1
    written = json.loads(record_files[0].read_text(encoding="utf-8"))
    assert written["run_key"] == run_key
    assert written["payload"] == result.payload
