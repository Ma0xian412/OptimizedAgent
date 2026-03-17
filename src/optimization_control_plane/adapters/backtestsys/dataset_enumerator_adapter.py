"""DatasetEnumerator adapter for BackTestSys.

Enumerates dataset IDs from execution_config.backtest_run_spec.dataset_inputs.
Supports meta.dataset_ids to restrict to a subset.
"""

from __future__ import annotations

from typing import Any

from optimization_control_plane.domain.models import ExperimentSpec

_RUN_SPEC_KEY = "backtest_run_spec"
_DATASET_INPUTS_KEY = "dataset_inputs"
_MARKET_DATA_PATH_KEY = "market_data_path"
_ORDER_FILE_KEY = "order_file"
_CANCEL_FILE_KEY = "cancel_file"
_REQUIRED_INPUT_KEYS = (
    _MARKET_DATA_PATH_KEY,
    _ORDER_FILE_KEY,
    _CANCEL_FILE_KEY,
)


class BackTestDatasetEnumeratorAdapter:
    """Enumerate dataset IDs for BackTestSys trials from pre-generated dataset_inputs."""

    def enumerate(self, spec: ExperimentSpec) -> tuple[str, ...]:
        run_cfg = _read_run_spec_config(spec)
        dataset_inputs = _read_dataset_inputs(run_cfg)

        meta_ids = spec.meta.get("dataset_ids")
        if isinstance(meta_ids, (list, tuple)) and meta_ids:
            ids = tuple(str(x) for x in meta_ids)
            _validate_ids_in_inputs(ids, dataset_inputs)
            return ids

        return tuple(sorted(dataset_inputs.keys()))


def _read_run_spec_config(spec: ExperimentSpec) -> dict[str, Any]:
    value = spec.execution_config.get(_RUN_SPEC_KEY)
    if not isinstance(value, dict):
        raise ValueError(f"spec.execution_config.{_RUN_SPEC_KEY} must be a dict")
    return value


def _read_dataset_inputs(run_cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    dataset_inputs = run_cfg.get(_DATASET_INPUTS_KEY)
    if not isinstance(dataset_inputs, dict) or not dataset_inputs:
        raise ValueError(
            f"{_RUN_SPEC_KEY}.{_DATASET_INPUTS_KEY} must be a non-empty dict"
        )
    normalized: dict[str, dict[str, str]] = {}
    for dataset_id, raw_input in dataset_inputs.items():
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{_DATASET_INPUTS_KEY} keys must be non-empty strings")
        if not isinstance(raw_input, dict):
            raise ValueError(f"{_DATASET_INPUTS_KEY}[{dataset_id!r}] must be a dict")
        normalized[dataset_id] = _normalize_dataset_input(dataset_id, raw_input)
    return normalized


def _normalize_dataset_input(dataset_id: str, source: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in _REQUIRED_INPUT_KEYS:
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{_DATASET_INPUTS_KEY}[{dataset_id!r}].{key} must be a non-empty string"
            )
        normalized[key] = value
    return normalized


def _validate_ids_in_inputs(
    ids: tuple[str, ...],
    dataset_inputs: dict[str, dict[str, str]],
) -> None:
    missing = [i for i in ids if i not in dataset_inputs]
    if missing:
        raise ValueError(
            f"meta.dataset_ids {missing} not found in {_DATASET_INPUTS_KEY}"
        )
