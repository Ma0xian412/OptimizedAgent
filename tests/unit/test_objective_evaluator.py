"""UT-6: ObjectiveEvaluator replacability."""
from __future__ import annotations

from optimization_control_plane.domain.models import ObjectiveResult, RunResult
from tests.conftest import StubObjectiveEvaluator, make_spec


class AlternativeEvaluator:
    def evaluate(self, run_result: RunResult, spec: object) -> ObjectiveResult:
        return ObjectiveResult(value=-999.0, attrs={"custom": True}, artifact_refs=[])


class TestObjectiveEvaluator:
    def test_default_evaluator(self) -> None:
        ev = StubObjectiveEvaluator(metric_name="m1")
        rr = RunResult(metrics={"m1": 0.42}, diagnostics={}, artifact_refs=[])
        result = ev.evaluate(rr, make_spec())
        assert result.value == 0.42

    def test_replaceable(self) -> None:
        ev = AlternativeEvaluator()
        rr = RunResult(metrics={"m1": 0.42}, diagnostics={}, artifact_refs=[])
        result = ev.evaluate(rr, make_spec())
        assert result.value == -999.0
        assert result.attrs["custom"] is True
