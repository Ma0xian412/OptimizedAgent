from __future__ import annotations

import pytest

from optimization_control_plane.adapters.target_runtime import PythonTargetRuntime
from optimization_control_plane.domain.models import RunSpec
from tests.fixtures.blackboxes import reset_instance_counter


def _run_spec(target_config: dict[str, object], config: dict[str, object]) -> RunSpec:
    return RunSpec(
        kind="python_blackbox",
        target_config=target_config,
        config=config,
        resources={},
    )


class TestPythonTargetRuntime:
    def test_callable_target_executes(self) -> None:
        runtime = PythonTargetRuntime()
        result = runtime.run(
            _run_spec(
                {
                    "kind": "python_callable",
                    "ref": "tests.fixtures.blackboxes:callable_target",
                },
                {"score": 0.7},
            )
        )
        assert result.metrics["metric_1"] == 0.7

    def test_class_target_is_fresh_per_run(self) -> None:
        reset_instance_counter()
        runtime = PythonTargetRuntime()
        target_config: dict[str, object] = {
            "kind": "python_class",
            "ref": "tests.fixtures.blackboxes:StatefulClassTarget",
            "invoke_method": "run",
        }
        result1 = runtime.run(_run_spec(target_config, {"x": 1}))
        result2 = runtime.run(_run_spec(target_config, {"x": 1}))

        ids = {
            result1.diagnostics["instance_id"],
            result2.diagnostics["instance_id"],
        }
        call_counts = {
            result1.diagnostics["call_count"],
            result2.diagnostics["call_count"],
        }
        assert ids == {1, 2}
        assert call_counts == {1}

    def test_invalid_ref_raises(self) -> None:
        runtime = PythonTargetRuntime()
        with pytest.raises(AttributeError, match="symbol not found in module"):
            runtime.run(
                _run_spec(
                    {
                        "kind": "python_callable",
                        "ref": "tests.fixtures.blackboxes:not_exists",
                    },
                    {},
                )
            )

    def test_invalid_result_shape_raises(self) -> None:
        runtime = PythonTargetRuntime()
        with pytest.raises(TypeError, match="dict 'metrics'"):
            runtime.run(
                _run_spec(
                    {
                        "kind": "python_callable",
                        "ref": "tests.fixtures.blackboxes:return_invalid_result",
                    },
                    {},
                )
            )
