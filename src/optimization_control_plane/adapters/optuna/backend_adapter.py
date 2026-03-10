from __future__ import annotations

import json
import logging
from typing import Any

import optuna

from optimization_control_plane.adapters.optuna.sampler_profile import (
    build_sampler_profile,
)
from optimization_control_plane.adapters.optuna.trial_context import OptunaTrialContext
from optimization_control_plane.domain.enums import TrialState
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    SamplerProfile,
    StudyHandle,
    TrialHandle,
)
from optimization_control_plane.ports.optimizer_backend import TrialContext

logger = logging.getLogger(__name__)

_SPEC_HASH_ATTR = "spec_hash"
_SPEC_JSON_ATTR = "spec_json"

_STATE_MAP: dict[str, optuna.trial.TrialState] = {
    TrialState.COMPLETE: optuna.trial.TrialState.COMPLETE,
    TrialState.PRUNED: optuna.trial.TrialState.PRUNED,
    TrialState.FAIL: optuna.trial.TrialState.FAIL,
}


class OptunaBackendAdapter:
    """OptimizerBackend implementation backed by Optuna ask/tell API."""

    def __init__(
        self,
        storage_dsn: str = "sqlite:///study.db",
        study_name_prefix: str = "",
    ) -> None:
        self._storage_dsn = storage_dsn
        self._study_name_prefix = study_name_prefix
        self._studies: dict[str, optuna.Study] = {}
        self._specs: dict[str, ExperimentSpec] = {}
        self._live_trials: dict[str, optuna.trial.Trial] = {}

    def open_or_resume_experiment(self, spec: ExperimentSpec) -> StudyHandle:
        study_name = self._resolve_study_name(spec)
        direction_str: str = spec.objective_config.get("direction", "minimize")
        direction = (
            optuna.study.StudyDirection.MAXIMIZE
            if direction_str == "maximize"
            else optuna.study.StudyDirection.MINIMIZE
        )

        sampler = self._build_sampler(spec)
        pruner = self._build_pruner(spec)

        study = optuna.create_study(
            study_name=study_name,
            storage=self._storage_dsn,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        stored_hash = study.user_attrs.get(_SPEC_HASH_ATTR)
        if stored_hash is not None and stored_hash != spec.spec_hash:
            raise ValueError(
                f"spec_hash mismatch for study '{study_name}': "
                f"stored={stored_hash}, given={spec.spec_hash}"
            )

        if stored_hash is None:
            study.set_user_attr(_SPEC_HASH_ATTR, spec.spec_hash)
            study.set_user_attr(
                _SPEC_JSON_ATTR,
                json.dumps({
                    "spec_id": spec.spec_id,
                    "meta": spec.meta,
                    "target_spec": spec.target_spec.to_dict(),
                    "objective_config": spec.objective_config,
                    "execution_config": spec.execution_config,
                }, sort_keys=True),
            )

        study_id = study_name
        self._studies[study_id] = study
        self._specs[study_id] = spec

        return StudyHandle(
            study_id=study_id,
            name=study_name,
            spec_hash=spec.spec_hash,
            direction=direction_str,
            settings=self._extract_study_settings(spec, study_name),
        )

    def get_spec(self, study_id: str) -> ExperimentSpec:
        return self._specs[study_id]

    def get_sampler_profile(self, study_id: str) -> SamplerProfile:
        study = self._get_study(study_id)
        return build_sampler_profile(study.sampler)

    def ask(self, study_id: str) -> TrialHandle:
        study = self._get_study(study_id)
        trial = study.ask()
        trial_id = str(trial.number)
        self._live_trials[self._trial_key(study_id, trial_id)] = trial
        return TrialHandle(
            study_id=study_id,
            trial_id=trial_id,
            number=trial.number,
            state="RUNNING",
        )

    def open_trial_context(
        self, study_id: str, trial_id: str
    ) -> TrialContext:
        key = self._trial_key(study_id, trial_id)
        trial = self._live_trials.get(key)
        if trial is None:
            raise KeyError(
                f"no live trial for study_id={study_id}, trial_id={trial_id}"
            )
        return OptunaTrialContext(trial)

    def tell(
        self,
        study_id: str,
        trial_id: str,
        state: str,
        value: float | None,
        attrs: dict[str, Any] | None,
    ) -> None:
        study = self._get_study(study_id)
        trial_number = self._parse_trial_id(trial_id)
        incoming_attrs = dict(attrs or {})
        if self._ensure_idempotent_terminal_payload(
            study=study,
            study_id=study_id,
            trial_number=trial_number,
            trial_id=trial_id,
            state=state,
            value=value,
            attrs=incoming_attrs,
        ):
            return

        key = self._trial_key(study_id, trial_id)
        trial = self._live_trials.get(key)
        if trial is None:
            raise KeyError(
                f"no live trial for study_id={study_id}, trial_id={trial_id}"
            )

        optuna_state = _STATE_MAP.get(state)
        if optuna_state is None:
            raise ValueError(f"unknown trial state: {state}")

        if incoming_attrs:
            for k, v in incoming_attrs.items():
                trial.set_user_attr(k, v)

        values: list[float] | None = [value] if value is not None else None
        study.tell(trial, state=optuna_state, values=values)
        self._live_trials.pop(key, None)

    # -- private helpers --

    def _get_study(self, study_id: str) -> optuna.Study:
        study = self._studies.get(study_id)
        if study is None:
            raise KeyError(f"study not opened: {study_id}")
        return study

    @staticmethod
    def _trial_key(study_id: str, trial_id: str) -> str:
        return f"{study_id}::{trial_id}"

    @staticmethod
    def _parse_trial_id(trial_id: str) -> int:
        try:
            return int(trial_id)
        except ValueError as exc:
            raise ValueError(f"trial_id must be an integer string: {trial_id}") from exc

    def _ensure_idempotent_terminal_payload(
        self,
        *,
        study: optuna.Study,
        study_id: str,
        trial_number: int,
        trial_id: str,
        state: str,
        value: float | None,
        attrs: dict[str, Any],
    ) -> bool:
        frozen_trial = self._find_trial(study, trial_number)
        if frozen_trial is None or frozen_trial.state not in _STATE_MAP.values():
            return False

        self._assert_matching_terminal_payload(
            study_id=study_id,
            trial_id=trial_id,
            frozen_trial=frozen_trial,
            state=state,
            value=value,
            attrs=attrs,
        )
        logger.info(
            "idempotent tell skipped",
            extra={"study_id": study_id, "trial_id": trial_id, "state": state},
        )
        return True

    def _assert_matching_terminal_payload(
        self,
        *,
        study_id: str,
        trial_id: str,
        frozen_trial: optuna.trial.FrozenTrial,
        state: str,
        value: float | None,
        attrs: dict[str, Any],
    ) -> None:
        expected_state = _STATE_MAP.get(state)
        if expected_state is None:
            raise ValueError(f"unknown trial state: {state}")
        if frozen_trial.state != expected_state:
            raise ValueError(
                "conflicting tell state for "
                f"study_id={study_id}, trial_id={trial_id}: "
                f"stored={frozen_trial.state.name}, incoming={state}"
            )

        if expected_state == optuna.trial.TrialState.COMPLETE:
            if frozen_trial.value != value:
                raise ValueError(
                    "conflicting tell value for "
                    f"study_id={study_id}, trial_id={trial_id}: "
                    f"stored={frozen_trial.value}, incoming={value}"
                )
        elif value is not None:
            raise ValueError(
                "non-complete tell must not include value for "
                f"study_id={study_id}, trial_id={trial_id}"
            )

        for key, expected_value in attrs.items():
            stored_value = frozen_trial.user_attrs.get(key)
            if stored_value != expected_value:
                raise ValueError(
                    "conflicting tell attrs for "
                    f"study_id={study_id}, trial_id={trial_id}, attr={key}: "
                    f"stored={stored_value}, incoming={expected_value}"
                )

    @staticmethod
    def _find_trial(
        study: optuna.Study,
        trial_number: int,
    ) -> optuna.trial.FrozenTrial | None:
        for trial in study.get_trials(deepcopy=False):
            if trial.number == trial_number:
                return trial
        return None

    def _resolve_study_name(self, spec: ExperimentSpec) -> str:
        short_hash = spec.spec_hash.replace("sha256:", "")[:16]
        return f"{self._study_name_prefix}{spec.spec_id}-{short_hash}"

    @staticmethod
    def _build_sampler(spec: ExperimentSpec) -> optuna.samplers.BaseSampler:
        sampler_cfg = spec.objective_config.get("sampler", {})
        sampler_type = sampler_cfg.get("type", "tpe")
        seed = sampler_cfg.get("seed")

        if sampler_type == "random":
            return optuna.samplers.RandomSampler(seed=seed)

        n_startup = sampler_cfg.get("n_startup_trials", 10)
        constant_liar = sampler_cfg.get("constant_liar", False)
        return optuna.samplers.TPESampler(
            n_startup_trials=n_startup,
            constant_liar=constant_liar,
            seed=seed,
        )

    @staticmethod
    def _build_pruner(spec: ExperimentSpec) -> optuna.pruners.BasePruner:
        pruner_cfg = spec.objective_config.get("pruner", {})
        pruner_type = pruner_cfg.get("type", "median")

        if pruner_type == "median":
            return optuna.pruners.MedianPruner(
                n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
                n_warmup_steps=pruner_cfg.get("n_warmup_steps", 0),
            )

        return optuna.pruners.NopPruner()

    @staticmethod
    def _extract_study_settings(
        spec: ExperimentSpec,
        study_name: str,
    ) -> dict[str, Any]:
        settings: dict[str, Any] = {"study_name": study_name}
        sampler_cfg = spec.objective_config.get("sampler")
        pruner_cfg = spec.objective_config.get("pruner")
        if isinstance(sampler_cfg, dict):
            settings["sampler"] = dict(sampler_cfg)
        if isinstance(pruner_cfg, dict):
            settings["pruner"] = dict(pruner_cfg)
        return settings
