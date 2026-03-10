"""IT-1: OptunaBackendAdapter ask/tell closed loop."""
from __future__ import annotations

import json
import os

import pytest

from optimization_control_plane.adapters.optuna.backend_adapter import (
    OptunaBackendAdapter,
)
from optimization_control_plane.domain.enums import TrialState
from tests.conftest import make_spec


@pytest.fixture()
def adapter(tmp_path: object) -> OptunaBackendAdapter:
    db = os.path.join(str(tmp_path), "test.db")
    return OptunaBackendAdapter(storage_dsn=f"sqlite:///{db}")


class TestOptunaAskTell:
    def test_create_and_ask(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        assert handle.study_id

        trial = adapter.ask(handle.study_id)
        assert trial.trial_id is not None
        assert trial.state == "RUNNING"

    def test_tell_complete(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, {"k": "v"})

    def test_tell_pruned(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.PRUNED, None, None)

    def test_tell_fail(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.FAIL, None, None)

    def test_resume_study(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        h1 = adapter.open_or_resume_experiment(spec)
        h2 = adapter.open_or_resume_experiment(spec)
        assert h1.study_id == h2.study_id

    def test_different_spec_creates_different_study(self, adapter: OptunaBackendAdapter) -> None:
        spec1 = make_spec()
        h1 = adapter.open_or_resume_experiment(spec1)

        spec2 = make_spec(meta={"dataset_version": "DIFFERENT"})
        h2 = adapter.open_or_resume_experiment(spec2)
        assert h1.study_id != h2.study_id

    def test_get_sampler_profile(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        profile = adapter.get_sampler_profile(handle.study_id)
        assert profile.mode.value == "ASYNC_FILL"

    def test_trial_context(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        trial = adapter.ask(handle.study_id)
        ctx = adapter.open_trial_context(handle.study_id, trial.trial_id)
        assert ctx is not None

    def test_tell_idempotent_after_told(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, None)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, None)

    def test_tell_conflict_after_told_raises(self, adapter: OptunaBackendAdapter) -> None:
        spec = make_spec()
        handle = adapter.open_or_resume_experiment(spec)
        trial = adapter.ask(handle.study_id)
        adapter.tell(handle.study_id, trial.trial_id, TrialState.COMPLETE, 0.5, {"k": "v"})

        with pytest.raises(ValueError, match="conflicting tell state"):
            adapter.tell(handle.study_id, trial.trial_id, TrialState.PRUNED, None, {"k": "v"})

    def test_study_user_attrs_spec_json_contains_target_spec(
        self,
        adapter: OptunaBackendAdapter,
    ) -> None:
        spec = make_spec(
            target_spec={"target_id": "target_resume_a", "config": {"market": "us"}}
        )
        handle = adapter.open_or_resume_experiment(spec)
        study = adapter._studies[handle.study_id]
        spec_json = study.user_attrs["spec_json"]
        payload = json.loads(spec_json)

        assert payload["target_spec"] == spec.target_spec.to_dict()

    def test_resume_keeps_target_spec_consistent_via_persisted_spec_json(
        self,
        tmp_path: object,
    ) -> None:
        db = os.path.join(str(tmp_path), "resume.db")
        storage_dsn = f"sqlite:///{db}"

        first = OptunaBackendAdapter(storage_dsn=storage_dsn)
        spec = make_spec(
            target_spec={"target_id": "target_resume_b", "config": {"market": "crypto"}}
        )
        first.open_or_resume_experiment(spec)

        second = OptunaBackendAdapter(storage_dsn=storage_dsn)
        resumed_handle = second.open_or_resume_experiment(spec)
        resumed_spec = second.get_spec(resumed_handle.study_id)

        assert resumed_spec == spec
        assert resumed_spec.target_spec.to_dict() == {
            "target_id": "target_resume_b",
            "config": {"market": "crypto"},
        }

    def test_resume_rejects_persisted_spec_json_without_target_spec(
        self,
        tmp_path: object,
    ) -> None:
        db = os.path.join(str(tmp_path), "resume_invalid.db")
        storage_dsn = f"sqlite:///{db}"
        spec = make_spec(
            target_spec={"target_id": "target_resume_invalid", "config": {"market": "fx"}}
        )
        first = OptunaBackendAdapter(storage_dsn=storage_dsn)
        handle = first.open_or_resume_experiment(spec)
        study = first._studies[handle.study_id]
        payload = json.loads(study.user_attrs["spec_json"])
        payload.pop("target_spec", None)
        study.set_user_attr("spec_json", json.dumps(payload, sort_keys=True))

        second = OptunaBackendAdapter(storage_dsn=storage_dsn)
        with pytest.raises(ValueError, match="persisted target_spec is invalid"):
            second.open_or_resume_experiment(spec)
