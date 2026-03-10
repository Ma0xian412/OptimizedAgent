# optimization-control-plane

优化控制面（Optimization Control Plane）负责试验编排，而不负责具体执行引擎实现。

## 核心边界（三分）

- `target_spec`：**执行目标边界**（执行对象是谁）。例如市场、交易所、环境、数据域等。
- `objective_config`：**目标函数边界**（怎么评估好坏）。例如 name/version/params/direction/sampler/pruner。
- `execution_config`：**执行资源边界**（怎么跑）。例如 executor_kind、default_resources。

三者必须显式输入，不允许把 target 隐式塞进 `params` 或其他字段。

## Fail-fast 约束

- `TrialOrchestrator.start()` 会校验 `spec.target_spec`。
- `FakeExecutionBackend.submit()` 会显式校验 `request.run_spec.resolved_target`，缺失或非法直接报错。
- 旧格式中把 target 藏在 `execution_config.target` / `execution_config.target_spec` 会直接报错，不做 fallback。
- `OptunaBackendAdapter.open_or_resume_experiment()` 在 open/resume 时校验 target，并从持久化 `spec_json` 恢复 canonical spec；`target_spec` 缺失会报错而不是静默降级。

## 最小示例

```python
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.adapters.execution import FakeExecutionBackend, FakeRunScript
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter
from optimization_control_plane.adapters.policies import (
    AsyncFillParallelismPolicy,
    SubmitNowDispatchPolicy,
)
from optimization_control_plane.adapters.storage import (
    FileObjectiveCache,
    FileResultStore,
    FileRunCache,
)
from optimization_control_plane.adapters.target_resolution import SimpleTargetResolver
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    RunResult,
    TargetSpec,
    compute_spec_hash,
)

# 1) 明确 target/objective/execution 三分边界
target_spec = TargetSpec(target_id="target_backtest_v1", config={"market": "us_equity"})
objective_config = {"name": "loss", "version": "v1", "direction": "minimize", "params": {}}
execution_config = {"executor_kind": "backtest", "default_resources": {"cpu": 1}}
meta = {"dataset_version": "ds_v1", "engine_version": "e_v1"}

spec = ExperimentSpec(
    spec_id="demo",
    spec_hash=compute_spec_hash("demo", meta, target_spec, objective_config, execution_config),
    meta=meta,
    target_spec=target_spec,
    objective_config=objective_config,
    execution_config=execution_config,
)

# 2) 组装 orchestrator
optimizer_backend = OptunaBackendAdapter(storage_dsn="sqlite:///demo.db")
execution_backend = FakeExecutionBackend()
execution_backend.set_default_script(
    FakeRunScript(run_result=RunResult(metrics={"metric_1": 0.42}, diagnostics={}, artifact_refs=[]))
)

orchestrator = TrialOrchestrator(
    backend=optimizer_backend,
    objective_def=ObjectiveDefinition(...),  # SearchSpace/RunSpecBuilder/RunKeyBuilder 等
    execution_backend=execution_backend,
    parallelism_policy=AsyncFillParallelismPolicy(),
    dispatch_policy=SubmitNowDispatchPolicy(),
    run_cache=FileRunCache("data"),
    objective_cache=FileObjectiveCache("data"),
    result_store=FileResultStore("data"),
    target_resolver=SimpleTargetResolver(),
)

settings = {
    "spec_id": spec.spec_id,
    "meta": spec.meta,
    "target_spec": spec.target_spec.to_dict(),
    "objective_config": spec.objective_config,
    "execution_config": spec.execution_config,
    "parallelism": {"max_in_flight_trials": 2},
    "stop": {"max_trials": 10},
}
orchestrator.start(spec, settings)
```
