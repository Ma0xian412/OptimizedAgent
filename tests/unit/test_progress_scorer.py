"""UT-5: ProgressScorer.score() skip semantics when returning None."""
from __future__ import annotations

from optimization_control_plane.domain.models import Checkpoint
from tests.conftest import StubProgressScorer, make_spec


class TestProgressScorer:
    def test_returns_value_when_metric_present(self) -> None:
        scorer = StubProgressScorer(metric_name="loss")
        cp = Checkpoint(step=1, metrics={"loss": 0.5})
        assert scorer.score(cp, make_spec()) == 0.5

    def test_returns_none_when_metric_absent(self) -> None:
        scorer = StubProgressScorer(metric_name="loss")
        cp = Checkpoint(step=1, metrics={"other": 0.5})
        assert scorer.score(cp, make_spec()) is None

    def test_none_skip_semantics(self) -> None:
        scorer = StubProgressScorer(metric_name="absent")
        cp = Checkpoint(step=1, metrics={})
        result = scorer.score(cp, make_spec())
        assert result is None
