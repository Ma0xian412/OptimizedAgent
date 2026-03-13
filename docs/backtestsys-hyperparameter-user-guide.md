# BackTestSys 超参数调优用户指南（通用）

本文档面向“使用本仓库迭代框架调优 BackTestSys”的场景，重点是**通用方法**，不限定某个单一参数。

---

## 1. 目标与整体流程

使用流程可概括为 6 步：

1. 准备 BackTestSys 可执行环境（`main.py` + `config.xml`）
2. 准备数据（市场数据、策略输入、GroundTruth）
3. 在迭代入口脚本中组装 `TrialOrchestrator`
4. 定义 `settings`（搜索空间、目标函数、执行配置、停止条件）
5. 启动 `trial_ortrastrator.start(settings=...)`
6. 验证指标与产物，分析最优参数

本仓库提供了可直接运行的入口示例：`iter_backtestsys.py`。

---

## 2. 关键文件与职责

### 2.1 入口与配置

- `iter_backtestsys.py`
  - 组装并启动调优编排
- `config.xml`
  - BackTestSys 基础模板配置（每个 trial 会基于模板生成试验配置）

### 2.2 BackTestSys 适配器（已实现）

- `BackTestRunSpecBuilderAdapter`
- `BackTestRunKeyBuilderAdapter`
- `BackTestObjectiveKeyBuilderAdapter`
- `BackTestRunResultLoaderAdapter`
- `BackTestObjectiveEvaluatorAdapter`
- `BackTestTrialResultAggregatorAdapter`
- `BackTestDatasetEnumeratorAdapter`
- `BackTestGroundTruthProviderAdapter`

### 2.3 示例数据目录

- `mock_backtestsys/`
  - 多份市场数据（`market_data_ds_*.csv`）
  - 策略输入（`replay_orders.csv` / `replay_cancels.csv`）
  - 合约字典（`contracts.xml`）
  - GT 文件（`groundtruth/*.csv`）

---

## 3. 必要依赖

在仓库根目录执行：

```bash
python3 -m pip install optuna structlog pytest
```

---

## 4. `settings` 通用配置说明

`settings` 是调优入口最核心配置，建议至少包含：

- `spec_id`
- `meta`
- `objective_config`
- `execution_config`
- `sampler`
- `pruner`
- `parallelism`
- `stop`

### 4.1 `objective_config`（目标定义）

建议结构：

- `name` / `version` / `direction`
- `params`
  - loss 聚合所需权重与归一化基线（`weights` / `baseline` / `eps`）
- `groundtruth`
  - `doneinfo_path`
  - `executiondetail_path`
- `backtest_search_space`
  - 你希望调优的参数空间（通用，不限参数个数）

### 4.2 `execution_config`（执行定义）

必须包含：

- `executor_kind: "backtest"`
- `default_resources`（如 `cpu`）
- `backtest_run_spec`
  - `backtestsys_root`
  - `base_config_path`
  - `output_root_dir`
  - `dataset_inputs`（`dataset_id -> {market_data_path, order_file, cancel_file}`）
  - `python_executable`（建议显式传 `sys.executable`）

---

## 5. 搜索空间如何写成“通用调参”

你有两种方式：

### 方式 A：直接使用内置二选一适配器

适配器 1：`BackTestDelaySearchSpaceAdapter`

- 仅采样 `delay`（一个参数）
- 输出时自动映射为 `delay_in = delay_out = delay`
- 需在 `objective_config.backtest_fixed_params` 固定：
  - `time_scale_lambda`
  - `cancel_bias_k`

适配器 2：`BackTestCoreParamsSearchSpaceAdapter`

- 仅采样：
  - `time_scale_lambda`
  - `cancel_bias_k`
- 需在 `objective_config.backtest_fixed_params` 固定：
  - `delay`（输出映射为 `delay_in = delay_out = delay`）

每次实验只注入一个适配器到 `ObjectiveDefinition(search_space=...)`。

### 方式 B：自定义 SearchSpace（推荐）

适用于任意参数组合、参数耦合约束、条件空间。  
例如可以表达：

- 只调一部分参数，其他参数固定
- 参数等式约束（如 `delay_in = delay_out`）
- 条件分支空间（按策略类型采样不同参数）

实现方式：在入口脚本定义自定义 `SearchSpace` 类，并注入 `ObjectiveDefinition(search_space=...)`。

---

## 6. `config.xml` 模板与 trial 覆盖机制

`config.xml` 作为基础模板，建议固定：

- 策略类型
- 策略输入文件路径
- 合约字典路径
- 非调参字段默认值

每个 trial 会自动生成 trial 级 XML 并覆盖关键字段（由 `RunSpecBuilder` 决定）。  
你可以通过查看：

- `runtime/iter_backtestsys/artifacts/configs/*.xml`

确认每轮实际参数是否正确写入。

---

## 7. 如何运行调优

在仓库根目录执行：

```bash
python3 iter_backtestsys.py
```

完成后会输出 metrics 快照，重点看：

- `trials_asked_total`
- `trials_completed_total`
- `trials_failed_total`
- `execution_submitted_total`

---

## 8. 结果与产物位置

运行目录：

- `runtime/iter_backtestsys/`
  - `study.db`：Optuna 存储
  - `artifacts/configs/`：trial 配置文件
  - `artifacts/results/`：BackTestSys 回测结果
  - `ocp_data/`：run/objective 缓存与 trial 结果存储

BackTestSys 回测结果表（每次执行）：

- `DoneInfo.csv`
- `ExecutionDetail.csv`
- `OrderInfo.csv`
- `CancelRequest.csv`

---

## 9. 从 Mock 切换到真实数据

切换时重点修改 3 类内容：

1. `dataset_inputs`
   - 每个 `dataset_id` 同时配置：
     - `market_data_path`
     - `order_file`
     - `cancel_file`
2. `groundtruth`
   - 指向真实 GT 的 DoneInfo/ExecutionDetail
3. 基础 `config.xml`
   - 策略参数、策略输入路径、合约字典、日志/产物路径

编排主流程、适配器与 orchestrator 组装方式可保持不变。

---

## 10. 常见问题排查

### 10.1 `ModuleNotFoundError: optuna`

执行：

```bash
python3 -m pip install optuna structlog
```

### 10.2 报“缺失 BackTestSys 结果表”

优先检查：

- BackTestSys 子进程是否成功执行
- `--save-result` 输出目录是否可写
- 结果目录下是否有四张 CSV

### 10.3 轮次不足或提前停止

检查：

- `stop.max_trials`
- `stop.max_failures`
- `trials_failed_total` 是否增长

---

## 11. 最小验收清单（通用）

1. `python3 iter_backtestsys.py` 成功返回
2. `trials_completed_total` 达到预期轮次
3. `artifacts/configs/` 中 trial 参数写入符合预期
4. `artifacts/results/` 中每个执行结果包含 4 张 CSV
5. `ocp_data/trial_results/` 存在对应 trial 结果记录

---

## 12. 迭代结果可视化子项目

已提供独立子项目：

- `tools/backtestsys-visualizer`

能力：

- 可视化四个 `raw` 分项 loss
- 可视化四个 `normalized` 分项 loss
- 可视化总 loss 和 `best-so-far`
- 展示迭代明细（含参数）
- 支持跨 stage 汇总

启动页面：

```bash
python3 -m backtestsys_visualizer app --runtime-root ./runtime
```

导出 HTML/PNG/CSV：

```bash
python3 -m backtestsys_visualizer export --runtime-root ./runtime
```

