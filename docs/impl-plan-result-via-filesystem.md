# 实现计划：结果落盘协议

## 概述

本文档描述如何将「控制端与执行后端通过文件系统传递结果数据」协议落地到代码库中。

---

## 阶段 1：领域模型与校验

### 1.1 `RunSpec` 新增 `result_output_path`

**文件**：`src/optimization_control_plane/domain/models.py`

```python
@dataclass(frozen=True)
class RunSpec:
    job: Job
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    result_output_path: str  # 新增：结果落盘路径，非空
```

**校验**：`validate_run_spec()` 增加对 `result_output_path` 的校验

**文件**：`src/optimization_control_plane/core/orchestration/_trial_utils.py`

- 要求 `result_output_path` 为非空字符串
- 可选的路径合法性检查（如禁止 `..`、协议规定的路径前缀等，视安全需求而定）

### 1.2 `RunSpecBuilder` 协议扩展

**文件**：`src/optimization_control_plane/ports/objective.py`

- `RunSpecBuilder.build()` 返回的 `RunSpec` 必须包含合法的 `result_output_path`
- 文档中明确说明：`RunSpecBuilder` 负责生成唯一的、可写的落盘路径（如基于 `run_key`、`trial_id`、`dataset_id` 等生成）

---

## 阶段 2：执行后端变更

### 2.1 将 `result_output_path` 传入任务

**文件**：`src/optimization_control_plane/adapters/execution/multiprocess_backend.py`

- 在 `job.env` 中注入 `OCP_RESULT_OUTPUT_PATH=result_output_path`
- 或通过 `job.args` 追加 `--ocp-result-path`, `result_output_path`

### 2.2 执行后端仅返回成功/失败

**当前行为**：
- 解析 stdout 中的 `__OCP_RESULT__`，构造 `RunResult` 放入 `ExecutionEvent`
- 若未解析到，exit 0 时构造默认 `RunResult`

**新行为（协议模式）**：
- `COMPLETED` 事件中 `run_result=None`
- 不再解析 `__OCP_RESULT__`（或保留为可选兼容，见阶段 4）

**实现策略**：
- 在 `_run_worker` 中，当 `run_spec.result_output_path` 存在时，采用「仅成功/失败」模式
- 发出 `COMPLETED` 时显式设置 `run_result=None`

### 2.3 `FakeExecutionBackend` / `TestonlyBackend` 适配

**文件**：`src/optimization_control_plane/adapters/execution/testonly_backend.py`

- `FakeRunScript` 若需模拟 `COMPLETED`，可按协议选择：
  - 不设置 `run_result`，由调用方从 `result_output_path` 读取；或
  - 在测试中预先写入 `result_output_path` 对应文件，再发出 `COMPLETED`

---

## 阶段 3：控制端读取逻辑

### 3.1 新增 `ResultFileReader` 或内联读取

**职责**：从 `result_output_path` 读取 JSON，反序列化为 `RunResult`。

**位置**：
- 方案 A：在 `_event_handler._handle_completed` 内直接实现
- 方案 B：抽取为 `ResultFileReader` 协议/实现，通过依赖注入

**读取逻辑**：
```python
def load_run_result(path: str) -> RunResult:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RunResult(
        metrics=data.get("metrics", {}),
        diagnostics=data.get("diagnostics", {}),
        artifact_refs=data.get("artifact_refs", []),
    )
```

### 3.2 `_handle_completed` 逻辑变更

**文件**：`src/optimization_control_plane/core/orchestration/_event_handler.py`

**当前**：
```python
run_result = event.run_result
assert run_result is not None, "COMPLETED event must carry run_result"
```

**新逻辑**：
```python
if event.run_result is not None:
    run_result = event.run_result
else:
    # 协议模式：从 result_output_path 读取
    path = entry.run_spec.result_output_path  # 需确保 entry 能访问 run_spec
    run_result = load_run_result(path)
```

### 3.3 `InflightRegistry` / `RunBinding` 暴露 `run_spec`

**文件**：`src/optimization_control_plane/core/orchestration/inflight_registry.py`

- `RunBinding` 已包含 `run_spec`
- `InflightRegistry.get_by_handle()` 返回的 entry 需能获取 `run_spec`（通常通过 `entry.leader` 或类似结构）

确认 `_handle_completed` 中 `entry` 能访问 `run_spec.result_output_path`。

---

## 阶段 4：错误处理与边界

### 4.1 读取失败时的行为

当 `run_result is None` 且从文件读取失败时：
- 不写入 `RunCache`，不调用 `ObjectiveEvaluator`
- 将此次 run 视为 `FAILED`，调用 `record_run_failure`
- `error_code` 区分：`RESULT_FILE_MISSING`、`RESULT_FILE_INVALID`、`RESULT_FILE_IO_ERROR`

### 4.2 兼容 Legacy 模式（可选）

若需同时支持「执行后端直接返回 run_result」：
- `COMPLETED` 时：优先使用 `event.run_result`；若为 `None` 且存在 `result_output_path`，则从文件读取
- 这样现有 `MultiprocessExecutionBackend` 可逐步迁移

---

## 阶段 5：测试与文档

### 5.1 单元测试

- `validate_run_spec`：`result_output_path` 空、非法路径的校验
- `load_run_result`：正常 JSON、缺失字段、损坏 JSON、文件不存在
- `MultiprocessExecutionBackend`：传入 `result_output_path` 时，env/args 正确注入；`COMPLETED` 时 `run_result is None`
- `_handle_completed`：从文件成功读取并完成流程；读取失败时正确 FAILED

### 5.2 集成 / E2E 测试

- 用户脚本将结果写入 `OCP_RESULT_OUTPUT_PATH` 指定的文件
- 控制端能正确读取并完成 trial

### 5.3 文档更新

- `docs/user-manual-hyperparameter-tuning.md`：说明 `result_output_path` 含义及用户脚本书写规范
- `docs/development-guide.md`：在 ExecutionBackend 相关章节引用 `protocol-result-via-filesystem.md`

---

## 阶段 6：迁移清单

| 序号 | 修改项 | 文件 |
|------|--------|------|
| 1 | RunSpec 增加 result_output_path | domain/models.py |
| 2 | validate_run_spec 校验 result_output_path | _trial_utils.py |
| 3 | 所有 RunSpecBuilder 实现生成 result_output_path | 各 adapter / 测试 conftest |
| 4 | MultiprocessBackend 注入路径、COMPLETED 不返回 run_result | multiprocess_backend.py |
| 5 | ResultFileReader 或内联 load_run_result | 新模块或 _event_handler |
| 6 | _handle_completed 支持从文件读取 | _event_handler.py |
| 7 | FakeExecutionBackend 适配 | testonly_backend.py |
| 8 | 单元 / 集成 / E2E 测试 | tests/ |

---

## 依赖与顺序

1. 阶段 1 必须最先完成，否则后续无法构造合法 `RunSpec`
2. 阶段 2 与 3 可部分并行，但 `_handle_completed` 需在读取逻辑就绪后联调
3. 阶段 4 的错误处理应在阶段 3 实现时一并完成
4. 阶段 5 贯穿开发过程

---

## 可选：配置开关

若希望通过配置切换协议模式：
- 在 `execution_config` 或环境变量中增加 `result_via_filesystem: true/false`
- `RunSpecBuilder` 根据配置决定是否填充 `result_output_path`
- 控制端根据配置决定从 `event.run_result` 还是文件读取

此方案可最大程度保持向后兼容，但增加实现复杂度。
