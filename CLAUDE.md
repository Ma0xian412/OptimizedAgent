# CLAUDE.md

## 项目概述

`optimization-control-plane` — 单目标超参数优化控制面框架（V1）。  
基于 Optuna ask/tell 风格驱动 BackTestSys 回测引擎的超参搜索（如 `exchange.cancel_bias_k`、`tape.epsilon`）。  
**不是**回测执行引擎，也不是 Optuna `objective()` 封装。

## 技术栈

- **语言**: Python 3.10+
- **构建**: hatchling
- **依赖**: `optuna`, `structlog`
- **开发**: `pytest`, `pytest-timeout`, `mypy`, `ruff`
- **存储**: SQLite（Optuna study）、JSON 文件（缓存 / 审计）
- **配置**: XML（`config.xml`）

## 架构

**Ports & Adapters（六边形架构）**，依赖方向严格单向：

```
adapters/ → core/ → ports/ → domain/
```

- `domain/` — 纯领域模型（frozen dataclass、枚举、状态），零外部依赖
- `ports/` — Protocol 接口定义
- `core/` — 编排逻辑（TrialOrchestrator、InflightRegistry、ObjectiveDefinition），只依赖 domain + ports
- `adapters/` — 具体实现（Optuna、BackTestSys、文件存储、策略）

`core/` **绝不** import `adapters/`。

## 目录结构

```
src/optimization_control_plane/
├── domain/           # models.py, enums.py, state.py
├── ports/            # optimizer_backend, execution_backend, objective, cache, result_store, groundtruth, policies
├── core/
│   ├── objective_definition.py
│   └── orchestration/
│       ├── trial_orchestrator.py     # 主控制循环
│       ├── inflight_registry.py      # run_key 级 in-flight 去重
│       ├── trial_batching.py         # 数据集分片与 batch
│       ├── _request_planner.py       # 请求规划（内部）
│       ├── _event_handler.py         # 事件处理（内部）
│       └── _metrics.py              # 指标收集（内部）
└── adapters/
    ├── optuna/       # OptunaBackendAdapter, OptunaTrialContext, SamplerProfile
    ├── backtestsys/  # 执行后端、evaluator、groundtruth、key builder、search space、loss 组件
    ├── execution/    # FakeExecutionBackend（测试用）
    ├── storage/      # FileRunCache, FileObjectiveCache, FileResultStore
    └── policies/     # AsyncFillParallelismPolicy, SubmitNowDispatchPolicy

tests/
├── conftest.py
├── unit/             # key 稳定性、search space、inflight、evaluator、batching 等
├── unit/backtestsys/ # BackTestSys 适配器专用
├── integration/      # Optuna ask/tell、cache 命中、pruning、dedup、graceful stop 等
└── e2e/              # RandomSampler、TPESampler、follower fanout 完整流程
```

## 关键入口

- `main.py` — CLI 入口，解析 XML → `AppConfig` → 组装 `TrialOrchestrator` → `start()`
- `TrialOrchestrator.start(settings)` — 同步阻塞主循环
- `TrialOrchestrator.stop()` — 另一线程调用触发优雅停止

## 核心流程

1. `open_or_resume_experiment` → 创建/恢复 Optuna study
2. 主循环：`ParallelismPolicy.target_in_flight()` → 填充 trial
3. 每个 trial：`SearchSpace.sample()` → `RunSpecBuilder.build()` → `RunKeyBuilder` → `ObjectiveKeyBuilder`
4. 缓存查找顺序：ObjectiveCache → RunCache → 执行
5. in-flight 去重：相同 `run_key` 后续 trial 作为 follower 挂载，不占执行槽位
6. 事件处理：CHECKPOINT（pruning）、COMPLETED（cache + tell）、CANCELLED、FAILED
7. 数据集 fan-out：一个 trial 在 train 全部文件上分别运行并聚合 loss
8. baseline 预跑：优化前用 base config 在全部分片上跑一轮算 `base_loss`
9. test-only final report：优化后用最佳超参在 test 集上运行（不 tell）

## 常用命令

```bash
# 安装
pip install -e ".[dev]"

# 运行
python main.py --config config.xml

# 测试
pytest tests/ -v
pytest tests/unit/ -v               # 仅单元测试
pytest tests/integration/ -v        # 仅集成测试
pytest tests/e2e/ -v                # 仅端到端测试

# 静态检查
mypy src/
ruff check src/
ruff format src/
```

## 代码规范

- **类型**: `from __future__ import annotations`，mypy strict，完整类型标注
- **模型**: `@dataclass(frozen=True)` 不可变优先
- **命名**: 模块 `snake_case`，类 `PascalCase`，私有实现 `_leading_underscore`
- **导入**: 绝对导入 `optimization_control_plane.xxx`
- **文档/注释**: 中文为主

## 核心设计约束

- V1 只支持单目标优化、`ASYNC_FILL` 并发模式
- 一个 study 同时只允许一个 `TrialOrchestrator` 实例
- 只有 `TrialOrchestrator` 能调用 `report / should_prune / tell`
- loss 不在 backend 里算：中间 pruning 用 `ProgressScorer`，最终 objective 用 `ObjectiveEvaluator`
- `ExecutionBackend` 不直接与 `OptimizerBackend` 通信
- key 生成必须是纯函数、稳定序列化
- 所有缓存/存储写入必须幂等
- `tell()` 重复提交相同终态安全，冲突终态报错
- objective 变更必须新开 study，但可复用 `RunCache`

## 测试超时

pytest 全局 timeout = 60 秒（pyproject.toml 配置）。

## 详细设计文档

- `docs/development-guide.md` — 完整架构、类图、时序图、组件规格、开发任务清单
- `docs/config-guide.md` — `config.xml` 配置说明
