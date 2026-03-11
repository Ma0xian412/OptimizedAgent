# 协议：控制端与执行后端通过文件系统传递结果数据

## 1. 协议目标

控制端（TrialOrchestrator）与执行后端（ExecutionBackend）之间的**结果数据**必须通过**文件系统**传递，而非通过执行后端的返回值或 IPC 通道传递。

## 2. 契约要点

| 角色 | 职责 |
|------|------|
| **RunSpec** | 必须包含 `result_output_path`：任务执行完毕后结果落盘的目标路径 |
| **执行后端** | 执行完命令后，仅通过 `ExecutionEvent` 返回**成功**或**失败**；不返回 `RunResult` 数据本身 |
| **控制端** | 收到 `COMPLETED` 后，从 `run_spec.result_output_path` 指定的路径自行读取 `RunResult` 数据 |

## 3. 数据流变化

### 3.1 现有流程（当前）

```
RunSpec → ExecutionBackend.submit()
         ↓
       job 执行，stdout 解析 __OCP_RESULT__ 或 exit code
         ↓
ExecutionEvent(kind=COMPLETED, run_result=RunResult)
         ↓
控制端：event.run_result → RunCache / ResultStore → ObjectiveEvaluator
```

### 3.2 新流程（协议生效后）

```
RunSpec(result_output_path="/path/to/result.json") → ExecutionBackend.submit()
         ↓
       job 执行，将结果写入 result_output_path；执行后端仅检测成功/失败
         ↓
ExecutionEvent(kind=COMPLETED, run_result=None)  # 不携带 run_result
         ↓
控制端：从 result_output_path 读取 → 解析为 RunResult → RunCache / ResultStore → ObjectiveEvaluator
```

## 4. 模型与接口变更

### 4.1 RunSpec 新增字段

```python
@dataclass(frozen=True)
class RunSpec:
    job: Job
    resource_request: ResourceRequest = field(default_factory=ResourceRequest)
    # 新增：结果落盘路径，执行成功后控制端从此路径读取 RunResult
    result_output_path: str  # 必填，非空
```

### 4.2 ExecutionEvent 语义变更

- `COMPLETED` 事件：`run_result` 字段改为可选，当协议生效时为 `None`
- 执行后端只需保证：成功时返回 `COMPLETED`，失败时返回 `FAILED` 或 `CANCELLED`

### 4.3 结果文件格式

与当前 `RunResult` JSON 结构一致：

```json
{
  "metrics": { "key": value, ... },
  "diagnostics": { "key": value, ... },
  "artifact_refs": [ "path_or_uri", ... ]
}
```

执行任务（用户脚本）负责按此格式写入 `result_output_path`。

## 5. 兼容性与可选实现

- 为支持渐进迁移，可引入**双模式**：
  - **filesystem 模式**：`result_output_path` 必填；`COMPLETED` 不含 `run_result`；控制端从文件读取
  - **legacy 模式**：`result_output_path` 可选；若执行后端仍通过 stdout 或其他方式返回 `run_result`，则可继续使用 `event.run_result`
- 或采用**单模式**：强制要求 `result_output_path`，执行后端一律不返回 `run_result`，控制端一律从文件读取

## 6. 错误与边界

| 场景 | 处理方式 |
|------|----------|
| 任务成功但文件不存在 | 视为 `FAILED`，error_code 如 `RESULT_FILE_MISSING` |
| 文件存在但格式错误 / 损坏 | 视为 `FAILED`，error_code 如 `RESULT_FILE_INVALID` |
| 文件路径不可读（权限等） | 视为 `FAILED`，error_code 如 `RESULT_FILE_IO_ERROR` |
| 任务失败（exit != 0） | 执行后端返回 `FAILED`，控制端不尝试读取文件 |

## 7. 与 CHECKPOINT 的关系

- **CHECKPOINT** 事件仍可经 stdout 协议（`__OCP_CHECKPOINT__`）传递，与结果落盘协议独立
- 若未来希望 CHECKPOINT 也落盘，可另行扩展 `checkpoint_output_path` 等字段

## 8. 执行后端实现指南

- 将 `result_output_path` 通过环境变量（如 `OCP_RESULT_OUTPUT_PATH`）或命令行参数传给任务
- 任务脚本在结束时将 `RunResult` JSON 写入该路径
- 执行后端：检测进程 exit code 或文件存在性，发出 `COMPLETED` 或 `FAILED`，不读取、不解析结果文件内容
