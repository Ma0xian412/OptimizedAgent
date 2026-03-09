"""IT-1: OptunaBackendAdapter ask/tell closed loop."""
from __future__ import annotations

import os

import pytest

from optimization_control_plane.adapters.optuna.backend_adapter import (
    OptunaBackendAdapter,
)
from optimization_control_plane.domain.enums import TrialState
from tests.conftest import make_settings, make_spec


@pytest.fixture()
def adapter(tmp_path: object) -> OptunaBackendAdapter:
    db = os.path.join(str(tmp_path), "test.db")
    return OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")


class TestOptunaAskTell:
    def test_create_and_ask(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        settings = make_settings()
        handle = adapter.open_or_resume_experiment(spec, settings)
        assert handle.study_id

        trial = adapter.ask(handle.study_id)
        assert trial.trial_id is not None
        assert trial.state == "RUNNING"

    def test_tell_complete(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, {"k": "v"})

    def test_tell_pruned(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.PRUNED, None, None)

    def test_tell_fail(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.FAIL, None, None)

    def test_resume_study(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        settings = make_settings()
        h1 = adapter.open_or_resume_experiment(spec, settings)
        h2 = adapter.open_or_resume_experiment(spec, settings)
        assert h1.study_id == h2.study_id

    def test_spec_hash_mismatch_raises(self, adapter: OptunaBackendAdapter) -> None:
        spec1 = make_spec()
        settings = make_settings()
        adapter.open_or_resume_experiment(spec1, settings)

        spec2 = make_spec(meta={"dataset_version": "DIFFERENT"})
        with pytest.raises(ValueError, match="spec_hash mismatch"):
            adapter.open_or_resume_experiment(spec2, settings)

    def test_get_sampler_profile(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        profile = adapter.get_sampler_profile(handle.study_id)
        assert profile.mode.value == "ASYNC_FILL"

    def test_trial_context(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        trial = adapter.ask(handle.study_id)
        ctx = adapter.open_trial_context(handle.study_id, trial.trial_id)
        assert ctx is not None

    def test_tell_idempotent_after_told(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, None)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, None)

    def test_tell_conflict_after_told_raises(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec, make_settings())
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, {"k": "v"})

        with pytest.raises(ValueError, match="conflicting tell state"):
            adapter.tell(handle.study_id, trial.trial_id, TrialState.PRUNED, None, {"k": "v"})
