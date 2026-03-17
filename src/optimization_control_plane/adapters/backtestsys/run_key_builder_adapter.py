from __future__ import annotations

import hashlib
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec, RunSpec, stable_json_serialize

_RUN_KEY_KIND = "backtest_run_key_v2"
_CONFIG_FLAG = "--config"
_INPUT_XML_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("market_data_path", ("data", "path")),
    ("order_file", ("strategy", "params", "order_file")),
    ("cancel_file", ("strategy", "params", "cancel_file")),
)


class BackTestRunKeyBuilderAdapter:
    """Build stable run key for BackTestSys execution payload."""

    def build(
        self,
        run_spec: RunSpec,
        spec: ExperimentSpec,
        dataset_id: str,
    ) -> str:
        del spec
        config_path = _extract_config_path(run_spec)
        root = _parse_config_root(config_path)
        config_payload = _element_to_payload(root)
        if not isinstance(config_payload, dict):
            raise ValueError("config root must contain nested elements")
        working_dir = _read_working_dir(run_spec)
        input_paths = _collect_input_paths(root)
        input_file_hashes = _hash_input_files(config_path, working_dir, input_paths)
        engine_commit = _read_git_commit(working_dir)
        payload = {
            "kind": _RUN_KEY_KIND,
            "dataset_id": dataset_id,
            "config": config_payload,
            "input_file_hashes": input_file_hashes,
            "engine_commit": engine_commit,
        }
        digest = hashlib.sha256(stable_json_serialize(payload).encode("utf-8")).hexdigest()
        return f"run:{digest[:24]}"


def _extract_config_path(run_spec: RunSpec) -> str:
    args = run_spec.job.args
    for idx, arg in enumerate(args):
        if arg != _CONFIG_FLAG:
            continue
        if idx + 1 >= len(args):
            raise ValueError("run_spec.job.args missing value for --config")
        config_path = args[idx + 1]
        if not config_path:
            raise ValueError("run_spec.job.args config path must be non-empty")
        return config_path
    raise ValueError("run_spec.job.args must include --config <path>")


def _parse_config_root(config_path: str) -> ET.Element:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    return ET.parse(config_path).getroot()


def _read_working_dir(run_spec: RunSpec) -> str:
    working_dir = run_spec.job.working_dir
    if not isinstance(working_dir, str) or not working_dir:
        raise ValueError("run_spec.job.working_dir must be a non-empty string")
    if not os.path.isdir(working_dir):
        raise FileNotFoundError(f"working_dir not found: {working_dir}")
    return working_dir


def _collect_input_paths(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in _INPUT_XML_PATHS:
        result[name] = _read_xml_text(root, path, name=name)
    return result


def _read_xml_text(root: ET.Element, path: tuple[str, ...], *, name: str) -> str:
    current: ET.Element | None = root
    for tag in path:
        current = current.find(tag) if current is not None else None
    if current is None or current.text is None or current.text.strip() == "":
        joined = ".".join(path)
        raise ValueError(f"config xml missing required field: {joined} for {name}")
    return current.text.strip()


def _hash_input_files(
    config_path: str,
    working_dir: str,
    input_paths: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    config_dir = Path(config_path).resolve().parent
    for name in sorted(input_paths):
        resolved_path = _resolve_input_path(
            raw_path=input_paths[name],
            working_dir=Path(working_dir).resolve(),
            config_dir=config_dir,
        )
        result[name] = _sha256_file(resolved_path)
    return result


def _resolve_input_path(raw_path: str, working_dir: Path, config_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"input file not found: {candidate}")
        return candidate.resolve()
    work_relative = (working_dir / candidate).resolve()
    if work_relative.exists():
        return work_relative
    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    raise FileNotFoundError(
        "relative input file not found under working_dir or config_dir: "
        f"{raw_path} (working_dir={working_dir}, config_dir={config_dir})"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _read_git_commit(working_dir: str) -> str:
    completed = subprocess.run(
        ["git", "-C", working_dir, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        error_msg = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise ValueError(f"failed to read git commit from {working_dir}: {error_msg}")
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"invalid git commit hash from {working_dir}: {commit}")
    return commit


def _element_to_payload(element: ET.Element) -> object:
    children = [child for child in list(element) if not callable(child.tag)]
    if not children:
        return (element.text or "").strip()
    grouped: dict[str, list[object]] = {}
    for child in children:
        grouped.setdefault(str(child.tag), []).append(_element_to_payload(child))
    result: dict[str, object] = {}
    for key in sorted(grouped):
        values = grouped[key]
        result[key] = values[0] if len(values) == 1 else values
    return result
