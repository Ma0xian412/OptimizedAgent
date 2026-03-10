from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from optimization_control_plane.domain.models import ObjectiveResult, RunSpec
from optimization_control_plane.ports.optimizer_backend import TrialContext

_TRAIN_SPLIT = "train"
_TEST_SPLIT = "test"


@dataclass(frozen=True)
class DatasetShard:
    shard_id: str
    data_path: str
    split: str


@dataclass(frozen=True)
class DatasetPlan:
    train_shards: tuple[DatasetShard, ...]
    test_shards: tuple[DatasetShard, ...]

    def all_shards(self) -> tuple[DatasetShard, ...]:
        return (*self.train_shards, *self.test_shards)


@dataclass
class TrialBatchState:
    trial_id: str
    trial_number: int
    trial_ctx: TrialContext
    split: str
    expected_runs: int
    base_run_spec: RunSpec
    objectives: list[ObjectiveResult] = field(default_factory=list)
    terminal: bool = False
    terminal_state: str | None = None
    terminal_reason: str | None = None


class TrialBatchRegistry:
    def __init__(self) -> None:
        self._states: dict[str, TrialBatchState] = {}
        self._best_train_trial_id: str | None = None
        self._best_train_value: float | None = None
        self._best_train_run_spec: RunSpec | None = None

    def register(
        self,
        *,
        trial_id: str,
        trial_number: int,
        trial_ctx: TrialContext,
        split: str,
        expected_runs: int,
        base_run_spec: RunSpec,
    ) -> None:
        self._states[trial_id] = TrialBatchState(
            trial_id=trial_id,
            trial_number=trial_number,
            trial_ctx=trial_ctx,
            split=split,
            expected_runs=expected_runs,
            base_run_spec=base_run_spec,
        )

    def get(self, trial_id: str) -> TrialBatchState:
        state = self._states.get(trial_id)
        if state is None:
            raise KeyError(f"trial batch state not found: {trial_id}")
        return state

    def add_objective(self, trial_id: str, objective: ObjectiveResult) -> None:
        state = self.get(trial_id)
        if state.terminal:
            return
        state.objectives.append(objective)

    def mark_terminal(self, trial_id: str, state: str, reason: str | None = None) -> bool:
        trial_state = self.get(trial_id)
        if trial_state.terminal:
            return False
        trial_state.terminal = True
        trial_state.terminal_state = state
        trial_state.terminal_reason = reason
        return True

    def is_ready(self, trial_id: str) -> bool:
        state = self.get(trial_id)
        return (not state.terminal) and len(state.objectives) >= state.expected_runs

    def track_best_train(self, trial_id: str, value: float) -> None:
        state = self.get(trial_id)
        if state.split != _TRAIN_SPLIT:
            return
        if self._best_train_value is not None and value >= self._best_train_value:
            return
        self._best_train_value = value
        self._best_train_trial_id = trial_id
        self._best_train_run_spec = state.base_run_spec

    @property
    def best_train_run_spec(self) -> RunSpec | None:
        return self._best_train_run_spec

    @property
    def best_train_trial_id(self) -> str | None:
        return self._best_train_trial_id


def build_dataset_plan(settings: dict[str, Any]) -> DatasetPlan | None:
    plan_cfg = settings.get("dataset_plan")
    if not isinstance(plan_cfg, dict):
        return None
    files = _parse_files(plan_cfg.get("files"))
    train_ratio = _read_positive_int(plan_cfg, "train_ratio", default=9)
    test_ratio = _read_positive_int(plan_cfg, "test_ratio", default=1)
    seed = int(plan_cfg.get("seed", 42))
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    train_count = _calc_train_count(
        total=len(shuffled),
        train_ratio=train_ratio,
        test_ratio=test_ratio,
    )
    train_items = shuffled[:train_count]
    test_items = shuffled[train_count:]
    return DatasetPlan(
        train_shards=tuple(_to_shard(item, _TRAIN_SPLIT) for item in train_items),
        test_shards=tuple(_to_shard(item, _TEST_SPLIT) for item in test_items),
    )


def with_dataset_path(base_run_spec: RunSpec, shard: DatasetShard) -> RunSpec:
    config = dict(base_run_spec.config)
    overrides = dict(config.get("overrides", {}))
    overrides["data.path"] = shard.data_path
    config["overrides"] = overrides
    config["dataset"] = {
        "split": shard.split,
        "shard_id": shard.shard_id,
        "data_path": shard.data_path,
    }
    return RunSpec(
        kind=base_run_spec.kind,
        config=config,
        resources=dict(base_run_spec.resources),
    )


def _parse_files(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("dataset_plan.files must be a non-empty list")
    parsed: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            path = item.strip()
            if not path:
                raise ValueError(f"dataset_plan.files[{idx}] cannot be empty")
            parsed.append({"id": f"ds_{idx}", "path": path})
            continue
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            shard_id = str(item.get("id", f"ds_{idx}")).strip()
            if not path:
                raise ValueError(f"dataset_plan.files[{idx}].path cannot be empty")
            if not shard_id:
                raise ValueError(f"dataset_plan.files[{idx}].id cannot be empty")
            parsed.append({"id": shard_id, "path": path})
            continue
        raise ValueError(f"dataset_plan.files[{idx}] must be str or dict")
    if len(parsed) < 2:
        raise ValueError("dataset_plan.files must contain at least 2 files")
    return parsed


def _read_positive_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = int(cfg.get(key, default))
    if value <= 0:
        raise ValueError(f"dataset_plan.{key} must be > 0")
    return value


def _calc_train_count(total: int, train_ratio: int, test_ratio: int) -> int:
    ratio_sum = train_ratio + test_ratio
    raw_count = int((total * train_ratio) / ratio_sum)
    if raw_count <= 0:
        return 1
    if raw_count >= total:
        return total - 1
    return raw_count


def _to_shard(item: dict[str, str], split: str) -> DatasetShard:
    return DatasetShard(
        shard_id=item["id"],
        data_path=item["path"],
        split=split,
    )
