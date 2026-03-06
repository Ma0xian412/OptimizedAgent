"""UT-4: SearchSpace.sample() parameter recording."""
from __future__ import annotations

from tests.conftest import StubSearchSpace, make_spec


class FakeCtx:
    def suggest_int(self, name: str, low: int, high: int) -> int:
        return low

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return low

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        return choices[0]

    def set_user_attr(self, key: str, val: object) -> None:
        pass

    def report(self, value: float, step: int) -> None:
        pass

    def should_prune(self) -> bool:
        return False


class TestSearchSpace:
    def test_sample_records_params(self) -> None:
        ss = StubSearchSpace({"a": 10, "b": 20})
        spec = make_spec()
        result = ss.sample(FakeCtx(), spec)
        assert result == {"a": 10, "b": 20}
        assert len(ss.calls) == 1
        assert ss.calls[0] == {"a": 10, "b": 20}

    def test_multiple_samples(self) -> None:
        ss = StubSearchSpace({"x": 1})
        spec = make_spec()
        ctx = FakeCtx()
        ss.sample(ctx, spec)
        ss.sample(ctx, spec)
        assert len(ss.calls) == 2
