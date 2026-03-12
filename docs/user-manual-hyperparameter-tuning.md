# 超参数调优使用手册

本文档说明：当用户希望使用本框架进行超参数调优时，需要实现哪些接口。

---

## 1. 概述

本框架采用 **Ports & Adapters（六边形架构）**。用户通过实现若干 **Port 接口** 来接入自己的业务逻辑（搜索空间、执行任务、目标计算等），框架则负责采样、调度、缓存和结果持久化。

- **必须实现**：与业务强相关的接口（搜索空间、执行、目标评估等）
- **可选实现**：有内置适配器，可按需替换

---

## 2. 必须实现的接口

以下接口与具体业务强相关，用户必须自行实现并注入到 `ObjectiveDefinition` 或 `TrialOrchestrator` 中。

### 2.1 `SearchSpace` — 搜索空间

**用途**：定义超参数空间，并通过 `TrialContext` 从优化后端采样超参。

**接口**：

```python
def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]: ...
```

**说明**：

- `ctx` 由优化后端提供，用于建议超参值，支持 `suggest_int`、`suggest_float`、`suggest_categorical`
- 返回 `dict[str, object]`，键为超参名，值为本次 trial 的采样值
- 若需 early pruning，可在 `sample` 后利用 `ctx.report()` 与 `ctx.should_prune()`，但通常这部分在 `ProgressScorer` 中处理

**示例**：

```python
from optimization_control_plane.ports.optimizer_backend import TrialContext
from optimization_control_plane.domain.models import ExperimentSpec

class MySearchSpace:
    def sample(self, ctx: TrialContext, spec: ExperimentSpec) -> dict[str, object]:
        lr = ctx.suggest_float("lr", 1e-5, 1e-1)
        batch = ctx.suggest_categorical("batch_size", [16, 32, 64])
        return {"lr": lr, "batch_size": batch}
```

---

### 2.2 `RunSpecBuilder` — 可执行规格构造器

**用途**：将采样得到的超参和 dataset 组合，构建可执行的 `RunSpec`（包含 `Job`、`ResourceRequest` 等）。

**接口**：

```python
def build(
    self,
    params: dict[str, object],
    spec: ExperimentSpec,
    dataset_id: str,
) -> RunSpec: ...
```

**说明**：

- `params` 即 `SearchSpace.sample()` 的返回值
- `RunSpec` 包含 `job: Job`（命令、参数、工作目录等）和 `resource_request: ResourceRequest`

**示例**：

```python
from optimization_control_plane.domain.models import ExperimentSpec, Job, RunSpec

class MyRunSpecBuilder:
    def build(self, params: dict[str, object], spec: ExperimentSpec, dataset_id: str) -> RunSpec:
        args = [f"--{k}={v}" for k, v in sorted(params.items())] + [f"--dataset={dataset_id}"]
        return RunSpec(job=Job(command=["python", "train.py"], args=args))
```

---

### 2.3 `RunKeyBuilder` — Run 缓存键构造器

**用途**：为 `RunSpec` 生成稳定、可复用的缓存键，用于 `RunCache` 去重和复用。

**接口**：

```python
def build(
    self,
    run_spec: RunSpec,
    spec: ExperimentSpec,
    dataset_id: str,
) -> str: ...
```

**要求**：相同输入必须产生相同 key；不同输入应尽量不同，以避免误命中。

---

### 2.4 `ObjectiveKeyBuilder` — 目标缓存键构造器

**用途**：为一次 objective 计算生成缓存键（通常依赖 `run_key` + `objective_config`）。

**接口**：

```python
def build(self, run_key: str, objective_config: dict[str, object]) -> str: ...
```

---

### 2.5 `ObjectiveEvaluator` — 目标评估器

**用途**：根据 `RunResult`、`ExperimentSpec` 和 `GroundTruthData` 计算目标值（loss/metric）。

**接口**：

```python
def evaluate(
    self,
    run_result: RunResult,
    spec: ExperimentSpec,
    groundtruth: GroundTruthData,
) -> ObjectiveResult: ...
```

**说明**：

- `RunResult` 仅作为原始结果承载对象，核心字段是 `payload`
- `ObjectiveEvaluator` 负责从 `run_result.payload` 中自行解析出业务指标（如 `metrics`、`artifact_refs`）
- `ObjectiveResult` 包含 `value: float`、`attrs: dict[str, Any]`、`artifact_refs: list[str]`
- 若业务不涉及 ground truth，可返回占位 `GroundTruthData`，在评估逻辑中忽略

---

### 2.6 `TrialResultAggregator` — Trial 结果聚合器

**用途**：当一次 trial 对应多个 dataset 时，将多个 `ObjectiveResult` 聚合成一个 trial 级 `ObjectiveResult`。

**接口**：

```python
def aggregate(
    self,
    results: list[tuple[str, ObjectiveResult]],
    spec: ExperimentSpec,
) -> ObjectiveResult: ...
```

**说明**：`results` 中每个元素为 `(dataset_id, objective_result)`。单 dataset 时可直接返回唯一结果。

---

### 2.7 `DatasetEnumerator` — 数据集枚举器

**用途**：为一次 trial 枚举需要评估的 dataset ID 列表。

**接口**：

```python
def enumerate(self, spec: ExperimentSpec) -> tuple[str, ...]: ...
```

**说明**：可从 `spec.meta` 或配置文件读取 dataset 列表。

---

### 2.8 `GroundTruthProvider` — 真值提供者

**用途**：加载 ground truth 数据，供 `ObjectiveEvaluator` 使用。

**接口**：

```python
def load(self, spec: ExperimentSpec) -> GroundTruthData: ...
def load_for_dataset(
    self,
    spec: ExperimentSpec,
    dataset_id: str,
) -> GroundTruthData: ...
```

**说明**：

- `GroundTruthData` 需包含非空的 `fingerprint`
- 若无 ground truth，可返回固定指纹的占位数据，在 `ObjectiveEvaluator` 中忽略

---

### 2.9 `ExecutionBackend` — 执行后端

**用途**：提交 `RunSpec` 对应的执行任务，并等待完成或失败事件。

**接口**：

```python
def submit(self, request: ExecutionRequest) -> RunHandle: ...
def wait_any(
    self, handles: list[RunHandle], timeout: float | None = None
) -> ExecutionEvent | None: ...
def cancel(self, handle: RunHandle, reason: str) -> None: ...
```

**说明**：

- `ExecutionRequest` 包含 `run_spec`、`run_key`、`objective_key`、`trial_id` 等
- `ExecutionEvent` 的 `kind` 包括 `COMPLETED`、`FAILED`、`CHECKPOINT`、`CANCELLED` 等
- 测试阶段可使用 `FakeExecutionBackend`，生产环境需对接实际执行系统（如任务队列、K8s Job 等）

---

## 3. 可选实现（有内置适配器）

以下接口有默认实现，通常可直接使用；若有特殊需求，可自行实现并替换。

### 3.1 `OptimizerBackend` — 优化后端

**用途**：管理 study、ask/tell、pruning 等。

**默认实现**：`OptunaBackendAdapter`（基于 Optuna ask/tell API）

**使用**：

```python
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter

backend = OptunaBackendAdapter(storage_dsn="sqlite:///optimization.db")
```

---

### 3.2 `ParallelismPolicy` — 并行策略

**用途**：决定同时在飞的 trial 数量等。

**默认实现**：`AsyncFillParallelismPolicy`

---

### 3.3 `DispatchPolicy` — 调度策略

**用途**：分类与排序执行请求。

**默认实现**：`SubmitNowDispatchPolicy`

---

### 3.4 `RunCache` / `ObjectiveCache` — 缓存

**用途**：缓存 `RunResult` 和 `ObjectiveResult`，避免重复计算。

**默认实现**：`FileRunCache`、`FileObjectiveCache`

---

### 3.5 `ResultStore` — 结果存储

**用途**：持久化 run 与 trial 结果。

**默认实现**：`FileResultStore`

---

## 4. 可选接口

### 4.1 `ProgressScorer` — 进度打分（Early Pruning）

**用途**：根据 `Checkpoint` 计算中间分数，用于 early pruning；若为 `None`，则禁用 pruning。

**接口**：

```python
def score(self, checkpoint: Checkpoint, spec: ExperimentSpec) -> float | None: ...
```

**说明**：返回 `None` 表示不参与 pruning；返回分数时由优化后端决定是否 prune。

---

## 5. Settings 配置示例

`settings` 是传递给 `orchestrator.start(spec=..., settings=...)` 的配置字典，用于控制实验规格与运行参数。

**完整示例**：

```python
my_settings = {
    "spec_id": "my_experiment_v1",
    "meta": {
        "dataset_version": "ds_v1",
        "engine_version": "e_v1",
    },
    "objective_config": {
        "name": "test_loss",
        "version": "v1",
        "direction": "minimize",
        "params": {},
        "groundtruth": {"version": "gt_v1", "path": "/path/to/gt.json"},
        "sampler": {"type": "tpe", "n_startup_trials": 5, "constant_liar": True, "seed": 42},
        "pruner": {"type": "median", "n_startup_trials": 5, "n_warmup_steps": 0},
    },
    "execution_config": {
        "executor_kind": "backtest",
        "default_resources": {"cpu": 1},
    },
    "sampler": {"type": "tpe", "n_startup_trials": 5, "constant_liar": True, "seed": 42},
    "pruner": {"type": "median", "n_startup_trials": 5, "n_warmup_steps": 0},
    "parallelism": {"max_in_flight_trials": 4},
    "stop": {"max_trials": 100, "max_failures": 10},
}

orchestrator.start(spec=my_spec, settings=my_settings)
```

**主要字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| `spec_id` | ✓ | 实验唯一标识 |
| `meta` | ✓ | 元信息（如 `dataset_version`、`engine_version`） |
| `objective_config` | ✓ | 目标配置（含 `sampler`、`pruner` 等） |
| `execution_config` | ✓ | 执行配置（如 `executor_kind`、`default_resources`） |
| `sampler` | 可选 | 采样器：`random`（`seed`）、`tpe`（`n_startup_trials`、`constant_liar`、`seed`） |
| `pruner` | 可选 | 剪枝器：`nop`（无剪枝）、`median`（`n_startup_trials`、`n_warmup_steps`） |
| `parallelism` | 可选 | `max_in_flight_trials`：最大并行 trial 数，默认 1 |
| `stop` | 可选 | `max_trials`：最大 trial 数；`max_failures`：最大失败次数 |

**仅用 settings 启动（不传 spec）**：

```python
orchestrator.start(settings=my_settings)
```

此时 `settings` 中必须包含 `spec_id`、`meta`、`objective_config`、`execution_config`，框架会据此构建 `ExperimentSpec`。

---

## 6. 组装与启动

将所有实现组装成 `ObjectiveDefinition` 和 `TrialOrchestrator`：

```python
from optimization_control_plane.core import ObjectiveDefinition, TrialOrchestrator
from optimization_control_plane.adapters.optuna import OptunaBackendAdapter
from optimization_control_plane.adapters.policies import (
    AsyncFillParallelismPolicy,
    SubmitNowDispatchPolicy,
)
from optimization_control_plane.adapters.storage import (
    FileRunCache,
    FileObjectiveCache,
    FileResultStore,
)

obj_def = ObjectiveDefinition(
    search_space=MySearchSpace(),           # 必须实现
    dataset_enumerator=MyDatasetEnumerator(),  # 必须实现
    run_spec_builder=MyRunSpecBuilder(),    # 必须实现
    run_key_builder=MyRunKeyBuilder(),      # 必须实现
    objective_key_builder=MyObjectiveKeyBuilder(),  # 必须实现
    trial_result_aggregator=MyTrialResultAggregator(),  # 必须实现
    progress_scorer=MyProgressScorer() or None,  # 可选
    objective_evaluator=MyObjectiveEvaluator(),  # 必须实现
)

orchestrator = TrialOrchestrator(
    backend=OptunaBackendAdapter(storage_dsn="..."),
    objective_def=obj_def,
    groundtruth_provider=MyGroundTruthProvider(),  # 必须实现
    execution_backend=MyExecutionBackend(),       # 必须实现
    parallelism_policy=AsyncFillParallelismPolicy(),
    dispatch_policy=SubmitNowDispatchPolicy(),
    run_cache=FileRunCache(path),
    objective_cache=FileObjectiveCache(path),
    result_store=FileResultStore(path),
)

orchestrator.start(spec=my_spec, settings=my_settings)
```

---

## 7. 接口清单速查

| 接口 | 必须实现 | 说明 |
|------|----------|------|
| `SearchSpace` | ✓ | 定义超参空间，通过 `TrialContext` 采样 |
| `RunSpecBuilder` | ✓ | 从超参构建可执行 RunSpec |
| `RunKeyBuilder` | ✓ | Run 缓存键 |
| `ObjectiveKeyBuilder` | ✓ | 目标缓存键 |
| `ObjectiveEvaluator` | ✓ | 从 RunResult 计算目标值 |
| `TrialResultAggregator` | ✓ | 多 dataset 时聚合结果 |
| `DatasetEnumerator` | ✓ | 枚举 dataset ID |
| `GroundTruthProvider` | ✓ | 加载 ground truth |
| `ExecutionBackend` | ✓ | 提交任务、等待事件 |
| `OptimizerBackend` | 使用内置 | 默认 `OptunaBackendAdapter` |
| `ParallelismPolicy` | 使用内置 | 默认 `AsyncFillParallelismPolicy` |
| `DispatchPolicy` | 使用内置 | 默认 `SubmitNowDispatchPolicy` |
| `RunCache` | 使用内置 | 默认 `FileRunCache` |
| `ObjectiveCache` | 使用内置 | 默认 `FileObjectiveCache` |
| `ResultStore` | 使用内置 | 默认 `FileResultStore` |
| `ProgressScorer` | 可选 | 用于 early pruning，可传 `None` |

---

## 8. 参考

- 完整 E2E 示例：`tests/e2e/test_random_sampler.py`、`tests/e2e/test_tpe_sampler.py`
- 开发设计文档：`docs/development-guide.md`
- Port 定义位置：`src/optimization_control_plane/ports/`
