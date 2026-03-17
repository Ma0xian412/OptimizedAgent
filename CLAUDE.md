# CLAUDE.md — 适配器开发速查卡

## 1. 架构边界（最重要）

依赖方向严格单向：

```
adapters → core → ports ← domain
```

| 层 | 规则 |
|---|---|
| `domain/` | 零外部依赖，只用标准库；用 `@dataclass(frozen=True)` |
| `ports/` | 只写 `Protocol`，不含任何实现逻辑 |
| `core/` | 只 import `domain` 和 `ports`，**绝不** import `adapters` |
| `adapters/` | 实现 `ports` 里的 Protocol；可依赖外部库 |

违反方向 = 架构错误，必须拒绝。

---

## 2. Port 接口文件

实现新适配器前先读对应的 Port：

```
src/optimization_control_plane/ports/
  optimizer_backend.py   # OptimizerBackend, TrialContext
  execution_backend.py   # ExecutionBackend
  objective.py           # SearchSpace, RunSpecBuilder, ObjectiveEvaluator, …
  cache.py               # RunCache, ObjectiveCache
  result_store.py        # ResultStore
  policies.py            # ParallelismPolicy, DispatchPolicy
  dataset.py             # DatasetEnumerator
  groundtruth.py         # GroundTruthProvider
  run_result_loader.py   # RunResultLoader
```

---

## 3. 适配器目录约定

```
src/optimization_control_plane/adapters/
  optuna/          # OptimizerBackend 参考实现
  execution/       # ExecutionBackend（含 testonly_backend.py 测试桩）
  storage/         # 文件存储（RunCache / ObjectiveCache / ResultStore）
  policies/        # ParallelismPolicy / DispatchPolicy
  backtestsys/     # 完整领域适配器参考实现（最复杂，参考价值最高）
```

新适配器放在 `adapters/<name>/`，入口类命名 `<Name>Adapter` 或 `<Name>BackendAdapter`。

---

## 4. 代码规范

```bash
# 类型检查（所有方法必须有完整注解）
mypy --strict src/

# Lint（行长 120，规则集 E/F/W/I/N/UP/B/SIM）
ruff check src/ tests/
ruff format src/ tests/
```

- 文件头必须写 `from __future__ import annotations`
- 模块级 logger：`logger = logging.getLogger(__name__)`
- 不用 `structlog`，用标准 `logging`（现有代码已统一）
- 私有方法加 `_` 前缀；模块级私有常量加 `_` 前缀

---

## 5. 测试规范

```
tests/
  unit/        # 单元测试，所有外部依赖用测试桩替换
  integration/ # 集成测试
  e2e/         # 端到端测试
```

```bash
pytest                          # 运行全部（60 秒硬超时）
pytest tests/unit/              # 只跑单元测试
pytest -k "test_xxx"            # 按名过滤
```

测试桩：用 `adapters/execution/testonly_backend.py` 中的 `FakeExecutionBackend` 替代真实执行后端；
用 `FakeRunScript` 控制每个 run_key 返回的事件序列。

---

## 6. 新增适配器清单

1. 读 `ports/<target>.py`，确认要实现的所有方法签名
2. 在 `adapters/<name>/` 下新建文件，类实现对应 Protocol
3. 构造函数只接受配置参数（字符串、路径、数字），不在内部 `new` 其他适配器
4. 所有公开方法写完整类型注解，无 `Any` 返回
5. 运行 `mypy --strict src/` 直到零错误
6. 运行 `ruff check src/` 直到零警告
7. 在 `tests/unit/` 写测试，依赖均用测试桩注入
8. 如有新的第三方依赖，在 `pyproject.toml` 的 `[project.dependencies]` 追加

---

## 7. 常用模式

**值对象**：用 `frozen=True` dataclass，字段类型明确，不含方法逻辑。

**错误处理**：直接 `raise ValueError` / `raise KeyError`，不吞异常，不返回 `None` 作为错误信号。

**配置读取**：从 `ExperimentSpec.objective_config`（`dict[str, Any]`）读取，读不到就 `raise ValueError`，给出明确的 key 路径提示。

**幂等性**：`tell` / `put` 类操作若重复调用，先检查已有状态是否冲突，冲突则 `raise`，一致则静默跳过（参考 `OptunaBackendAdapter.tell`）。

**文件写入**：用原子写（先写 `.tmp` 再 rename），参考 `adapters/storage/_file_helpers.py`。
