"""DatasetEnumerator adapter for BackTestSys.

Enumerates dataset IDs from execution_config.backtest_run_spec.dataset_paths.
Supports meta.dataset_ids to restrict to a subset.
"""

from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec

_RUN_SPEC_KEY = "backtest_run_spec"
_DATASET_PATHS_KEY = "dataset_paths"


class BackTestDatasetEnumeratorAdapter:
    """Enumerate dataset IDs for BackTestSys trials from pre-generated dataset_paths."""

    def enumerate(self, spec: ExperimentSpec) -> tuple[str, ...]:
        run_cfg = _read_run_spec_config(spec)
        dataset_paths = _read_dataset_paths(run_cfg)

        meta_ids = spec.meta.get("dataset_ids")
        if isinstance(meta_ids, (list, tuple)) and meta_ids:
            ids = tuple(str(x) for x in meta_ids)
            _validate_ids_in_paths(ids, dataset_paths)
            return ids

        return tuple(sorted(dataset_paths.keys()))


def _read_run_spec_config(spec: ExperimentSpec) -> dict[str, Any]:
    value = spec.execution_config.get(_RUN_SPEC_KEY)
    if not isinstance(value, dict):
        raise ValueError(f"spec.execution_config.{_RUN_SPEC_KEY} must be a dict")
    return value


def _read_dataset_paths(run_cfg: dict[str, Any]) -> dict[str, str]:
    dataset_paths = run_cfg.get(_DATASET_PATHS_KEY)
    if not isinstance(dataset_paths, dict) or not dataset_paths:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_DATASET_PATHS_KEY} must be a non-empty dict"
        )
    for k, v in dataset_paths.items():
        if not isinstance(k, str) or not k:
            raise ValueError(f"{_DATASET_PATHS_KEY} keys must be non-empty strings")
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"{_DATASET_PATHS_KEY}[{k!r}] must be a non-empty string path"
            )
    return dataset_paths


def _validate_ids_in_paths(ids: tuple[str, ...], dataset_paths: dict[str, str]) -> None:
    missing = [i for i in ids if i not in dataset_paths]
    if missing:
        raise ValueError(
            f"meta.dataset_ids {missing} not found in {_DATASET_PATHS_KEY}"
        )
