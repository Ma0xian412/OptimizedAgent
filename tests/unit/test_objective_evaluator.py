"""UT-6: ObjectiveEvaluator replacability."""
from __future__ import annotations

from optimization_control_plane.domain.models import GroundTruthData, ObjectiveResult, RunResult
from tests.conftest import StubGroundTruthProvider, StubObjectiveEvaluator, make_spec


class AlternativeEvaluator:
    def evaluate(
        self,
        run_result: RunResult,
        spec: object,
        groundtruth: GroundTruthData,
    ) -> ObjectiveResult:
        return ObjectiveResult(value=-999.0, attrs={"custom": True}, artifact_refs=[])


class TestObjectiveEvaluator:
    def test_default_evaluator(self) -> None:
        ev = StubObjectiveEvaluator(metric_name="m1")
        gt = StubGroundTruthProvider().load(make_spec())
        rr = RunResult(metrics={"m1": 0.42}, diagnostics={}, artifact_refs=[])
        result = ev.evaluate(rr, make_spec(), gt)
        assert result.value == 0.42

    def test_replaceable(self) -> None:
        ev = AlternativeEvaluator()
        gt = StubGroundTruthProvider().load(make_spec())
        rr = RunResult(metrics={"m1": 0.42}, diagnostics={}, artifact_refs=[])
        result = ev.evaluate(rr, make_spec(), gt)
        assert result.value == -999.0
        assert result.attrs["custom"] is True
