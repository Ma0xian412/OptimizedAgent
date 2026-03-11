# 协议：控制面与执行后端基于文件系统的数据交换

## 1. 概述

本协议约定：**控制面**与**执行后端**之间的结果数据交换必须通过文件系统完成，禁止在内存或 IPC 中直接传递 `RunResult` 等大块数据。

### 1.1 参与方

| 角色 | 职责 |
|------|------|
| 控制面 (Control Plane) | TrialOrchestrator、EventHandler 等，负责编排、读取结果、计算 objective |
| 执行后端 (Execution Backend) | 执行 Job、落盘结果、返回落盘路径 |

### 1.2 核心原则

1. **RunSpec** 必须包含结果落盘的**目标路径**字段
2. **执行后端**执行完成后，将结果写入该路径（或路径下），并返回**实际落盘路径**
3. **控制面**从该路径读取数据，不依赖执行后端直接传递 `RunResult` 对象

---

## 2. 数据流

```
┌─────────────────┐     ExecutionRequest      ┌─────────────────┐
│                 │ ───────────────────────> │                 │
│   控制面         │   (含 RunSpec 含目标路径)   │  执行后端        │
│                 │                           │                 │
│                 │   ExecutionEvent          │  写入文件系统    │
│                 │ <─────────────────────── │  返回 result_path
│                 │   (仅含 result_path)      │                 │
│                 │                           │                 │
│  从 result_path  │                           │                 │
│  读取 RunResult  │                           │                 │
└─────────────────┘                           └─────────────────┘
         │
         v
    文件系统 (共享或可访问)
```

---

## 3. 契约规格

### 3.1 RunSpec 扩展

在 `RunSpec` 中新增**必填**字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `result_output_path` | `str` | 是 | 结果落盘的**目标路径**。可为目录或文件路径；执行后端应在该路径（或其子路径）写入结果 |

**约定：**

- 控制面在构建 `RunSpec` 时，通过 `RunSpecBuilder` 或 `execution_config` 生成该路径
- 路径必须为绝对路径，或相对于执行后端可解析的基准目录
- 控制面与执行后端必须能访问同一文件命名空间（如共享 NFS、本地磁盘、或挂载卷）

### 3.2 结果文件格式

执行后端必须将 `RunResult` 序列化为以下 JSON 结构，写入 `result_output_path` 指定的位置（或该目录下的约定文件名）：

```json
{
  "metrics": { "metric_1": 0.12, "metric_2": 15.8 },
  "diagnostics": { "runtime_sec": 412.5 },
  "artifact_refs": ["s3://bucket/path/report.json"]
}
```

**文件名约定：**

- 若 `result_output_path` 为**目录**：必须写入 `run_result.json`（即 `{result_output_path}/run_result.json`）
- 若 `result_output_path` 为**文件**：直接写入该路径

### 3.3 ExecutionEvent (COMPLETED) 扩展

| 字段 | 类型 | 说明 |
|------|------|------|
| `result_path` | `str` | **实际落盘路径**。执行后端写入结果后的真实路径（绝对路径） |
| `run_result` | `RunResult \| None` | **弃用**。在新契约下为 `None`；控制面禁止使用此字段，必须从 `result_path` 读取 |

**语义：**

- `COMPLETED` 事件必须携带 `result_path`
- 控制面在收到 `COMPLETED` 后，从 `result_path` 读取 JSON，反序列化为 `RunResult`，再送入 `ObjectiveEvaluator`

### 3.4 执行后端契约

1. **写入**：执行完成后，将 `RunResult` 按 3.2 节格式写入 `RunSpec.result_output_path` 指定位置
2. **返回**：在 `COMPLETED` 事件中设置 `result_path` 为实际写入的绝对路径
3. **失败**：若写入失败，应发出 `FAILED` 事件，而非 `COMPLETED`

### 3.5 控制面契约

1. **构建**：`RunSpecBuilder` 必须设置 `result_output_path`，否则 `validate_run_spec` 应失败
2. **读取**：收到 `COMPLETED` 且 `result_path` 不为空时，从该路径读取 JSON 并解析为 `RunResult`
3. **错误**：若文件不存在、格式错误或权限不足，应记录错误并视作运行失败（或重试，由策略决定）

---

## 4. 实现计划

### 4.1 阶段一：模型与验证

1. **`domain/models.py`**
   - `RunSpec` 新增 `result_output_path: str`
   - `ExecutionEvent` 新增 `result_path: str | None = None`，`run_result` 保留但标记为可选（后续弃用）

2. **`core/orchestration/_trial_utils.py`**
   - `validate_run_spec()` 增加对 `result_output_path` 的校验：非空、合法路径

3. **`RunSpecBuilder` 协议**
   - 文档与类型提示明确：`build()` 返回的 `RunSpec` 必须包含有效的 `result_output_path`

### 4.2 阶段二：ResultLoader 抽象

1. **新增 `ports/result_loader.py`**
   ```python
   class ResultLoader(Protocol):
       def load_run_result(self, path: str) -> RunResult: ...
   ```

2. **实现 `adapters/storage/file_result_loader.py`**
   - 从 JSON 文件读取并反序列化为 `RunResult`
   - 处理文件不存在、JSON 解析异常

### 4.3 阶段三：事件处理改造

1. **`_event_handler.py`**
   - `_handle_completed()` 改造：
     - 若 `event.result_path` 存在：调用 `ResultLoader.load_run_result(event.result_path)` 得到 `RunResult`
     - 若仅有 `event.run_result`（兼容旧实现）：直接使用，并打弃用告警
   - 若两者皆无：断言失败或抛出明确异常

2. **`EventHandlerDeps`**
   - 注入 `result_loader: ResultLoader`

### 4.4 阶段四：ExecutionBackend 适配

1. **`ExecutionBackend` 协议**
   - 文档约定：`COMPLETED` 事件必须提供 `result_path`，`run_result` 不再传递

2. **`FakeExecutionBackend`**
   - 改为写入临时文件，并在 `COMPLETED` 事件中设置 `result_path`
   - 或：接受预写文件路径的脚本配置，用于测试

3. **真实执行后端适配器**（若存在）
   - 将训练/推理脚本的输出重定向到 `result_output_path`
   - 或：通过环境变量/参数将路径传给子进程，子进程负责写入
   - 完成时返回实际落盘路径

### 4.5 阶段五：RunSpecBuilder 改造

1. **所有 `RunSpecBuilder` 实现**
   - 在 `build()` 中根据 `run_key`、`request_id` 等生成 `result_output_path`
   - 示例：`{base_dir}/run_results/{run_key}.json` 或 `{base_dir}/run_results/{request_id}/run_result.json`

2. **`conftest.StubRunSpecBuilder`、E2E 中的 Builder**
   - 统一增加 `result_output_path` 构造逻辑

### 4.6 阶段六：测试与文档

1. 单元测试：`validate_run_spec` 对 `result_output_path` 的校验
2. 集成测试：`_handle_completed` 通过 `result_path` 读取的路径
3. E2E：完整流程验证
4. 更新 `development-guide.md`、`user-manual-hyperparameter-tuning.md` 中的 RunSpec、ExecutionBackend、数据流说明

---

## 5. 兼容性与迁移

### 5.1 过渡策略

- 支持双模式：`result_path` 优先，若无则回退到 `run_result`（并打弃用日志）
- 新实现强制使用 `result_path`；旧 `FakeExecutionBackend` 可在一段时间内继续提供 `run_result` 以兼容未迁移测试

### 5.2 最终态

- 移除对 `run_result` 的依赖，`ExecutionEvent.run_result` 可保留为 `None` 或从类型中移除
- 所有执行后端必须实现文件系统落盘并返回 `result_path`

---

## 6. 文件结构变更摘要

| 文件 | 变更 |
|------|------|
| `domain/models.py` | RunSpec.result_output_path, ExecutionEvent.result_path |
| `core/orchestration/_trial_utils.py` | validate_run_spec 增加 path 校验 |
| `ports/result_loader.py` | 新增 ResultLoader Protocol |
| `adapters/storage/file_result_loader.py` | 新增 FileResultLoader |
| `core/orchestration/_event_handler.py` | _handle_completed 使用 ResultLoader |
| `adapters/execution/testonly_backend.py` | FakeExecutionBackend 写文件+result_path |
| `ports/execution_backend.py` | 文档更新 |
| `conftest.py`, e2e/integration tests | StubRunSpecBuilder 等补充 result_output_path |

---

## 7. 风险与注意点

1. **路径可见性**：控制面与执行后端可能不在同一机器，需确保共享存储或挂载正确
2. **并发**：同一路径避免多 run 并发写入；建议路径包含 `run_key` 或 `request_id` 保证唯一
3. **清理**：结果文件的生命周期策略（何时删除）需在运维层面约定
4. **Checkpoint**：本协议仅约束最终 `RunResult`；若未来 Checkpoint 也走文件系统，需另行扩展
