# BackTestSys 延迟参数调优使用文档（`delay_in = delay_out`）

本文档说明如何使用本仓库的迭代框架，针对 BackTestSys 做以下约束优化：

- 只调一个参数 `delay`
- 始终满足 `delay_in = delay_out = delay`

对应入口脚本为：`iter_backtestsys.py`。

---

## 1. 目录与文件说明

### 1.1 关键入口

- `iter_backtestsys.py`
  - 组装 `trial_ortrastrator`（变量名沿用当前代码）
  - 调用 `trial_ortrastrator.start(settings=...)` 启动完整编排

### 1.2 BackTestSys 基础配置

- `config.xml`
  - 这是 BackTestSys 的基础模板配置
  - 每个 trial 会基于该模板生成一个 trial 级 XML，并覆盖：
    - `runner.delay_in`
    - `runner.delay_out`
    - `data.path`
    - （本示例固定）`tape.time_scale_lambda`、`exchange.cancel_bias_k`

### 1.3 Mock 数据目录

- `mock_backtestsys/`
  - `market_data_ds_01.csv`
  - `market_data_ds_02.csv`
  - `market_data_ds_03.csv`
  - `replay_orders.csv`
  - `replay_cancels.csv`
  - `contracts.xml`
  - `groundtruth/PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv`
  - `groundtruth/PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv`

---

## 2. 环境准备

在仓库根目录执行：

```bash
python3 -m pip install optuna structlog pytest
```

说明：

- `optuna`：优化后端
- `structlog`：项目依赖
- `pytest`：用于运行单测（可选但建议）

---

## 3. 一次性跑通 10 轮调优

在仓库根目录执行：

```bash
python3 iter_backtestsys.py
```

默认行为：

- `max_trials = 10`
- 并行度 `max_in_flight_trials = 1`
- sampler 使用 `random(seed=42)`
- 数据集使用 3 份 mock dataset（`ds_01/ds_02/ds_03`）

运行完成后会打印 metrics 快照，例如：

```json
{
  "trials_asked_total": 10,
  "trials_completed_total": 10,
  "trials_pruned_total": 0,
  "trials_failed_total": 0
}
```

---

## 4. 如何确认 `delay_in = delay_out`

每个 trial 的实际配置文件会落盘到：

- `runtime/iter_backtestsys/artifacts/configs/*.xml`

检查任意 trial XML 中的：

- `<runner><delay_in>...</delay_in></runner>`
- `<runner><delay_out>...</delay_out></runner>`

它们应始终相等。

原因：

- `iter_backtestsys.py` 内自定义了 `DelayEqualitySearchSpaceAdapter`
- 该适配器只采样一个 `delay`
- 返回参数时强制：
  - `delay_in = delay`
  - `delay_out = delay`

---

## 5. 结果产物位置

### 5.1 OCP 运行目录

- `runtime/iter_backtestsys/`
  - `study.db`：Optuna SQLite
  - `artifacts/configs/`：trial 配置
  - `artifacts/results/`：BackTestSys 回测结果 CSV
  - `ocp_data/`：缓存和结果存储

### 5.2 BackTestSys 结果表

每次执行最终会有四张表：

- `DoneInfo.csv`
- `ExecutionDetail.csv`
- `OrderInfo.csv`
- `CancelRequest.csv`

---

## 6. 参数与范围调整

在 `iter_backtestsys.py` 中修改 `settings`：

- 路径：`_build_settings(...)`
- 字段：

```python
"backtest_search_space": {
    "delay": {"low": 0, "high": 500000}
}
```

例如改成 0~1,000,000：

```python
"delay": {"low": 0, "high": 1000000}
```

---

## 7. 切换真实数据/真实项目配置

从 mock 切换到真实 BackTestSys 数据时，只改三类配置：

1. `dataset_paths`
   - 从 mock csv 路径替换为真实数据路径
2. `groundtruth`
   - 指向真实 `PubOrderDoneInfoLog_*.csv` 与 `PubExecutionDetailLog_*.csv`
3. `config.xml` 模板内的策略与依赖路径
   - 如 `strategy.params.order_file/cancel_file`
   - `contract.contract_id`、`contract.contract_dictionary_path`

保持不变的部分：

- `DelayEqualitySearchSpaceAdapter`（约束逻辑）
- orchestrator 组装方式

---

## 8. 常见问题

### 8.1 报错 `ModuleNotFoundError: optuna`

安装依赖：

```bash
python3 -m pip install optuna structlog
```

### 8.2 找不到 BackTestSys 结果 CSV

本项目已兼容 BackTestSys 的“结果在子目录”输出模式。若仍报错，优先检查：

- BackTestSys 是否执行成功
- `--save-result` 指向目录是否有写权限
- 结果目录下是否确实生成了 4 张 CSV

### 8.3 轮次未达到 10

检查：

- `stop.max_trials` 是否为 10
- 是否存在提前失败（看 `trials_failed_total`）
- 是否手动中断运行

---

## 9. 建议的最小验证清单

每次改参数后建议做以下验证：

1. 执行 `python3 iter_backtestsys.py`
2. 确认 `trials_completed_total == 10`
3. 随机抽查 2~3 个 trial 配置，确认 `delay_in == delay_out`
4. 抽查至少一个 `artifacts/results/...` 目录，确认四张 CSV 都存在

