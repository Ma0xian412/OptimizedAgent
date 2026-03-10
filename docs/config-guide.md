# BackTestSys 优化配置说明（`config.xml`）

本文对应当前实现：`core` 已支持“一个 trial 对 train 全文件 fan-out 运行并聚合 loss”，并在优化结束后生成 **test-only final report**（不参与 `tell`）。

---

## 1. 启动方式

```bash
python3 main.py --config /workspace/config.xml
```

---

## 2. 顶层结构

`config.xml` 顶层节点 `<optimization>` 下包含：

- `<study>`：实验与并发
- `<paths>`：BackTestSys 路径与 GT 路径
- `<dataset_plan>`：数据自动发现与 train:test 切分（9:1）
- `<sampler>`：Optuna sampler
- `<pruner>`：Optuna pruner（当前仅终态事件）
- `<search_space>`：超参搜索空间
- `<base_overrides>`：固定覆盖项

---

## 3. study 节点

```xml
<study>
    <spec_id>backtestsys_countdiff_demo</spec_id>
    <dataset_version>mock_v1</dataset_version>
    <engine_version>backtestsys_main</engine_version>
    <storage_dsn>sqlite:////workspace/data/backtestsys_mock/study.db</storage_dsn>
    <max_trials>8</max_trials>
    <max_failures>8</max_failures>
    <max_in_flight_trials>4</max_in_flight_trials>
    <max_workers>4</max_workers>
</study>
```

- `max_trials`：trial 数（不是子 run 数）
- `max_in_flight_trials`：控制面并发槽位
- `max_workers`：BackTestSys 执行后端线程池大小

---

## 4. paths 节点

```xml
<paths>
    <data_dir>/workspace/data/backtestsys_mock</data_dir>
    <backtestsys_repo_root>/workspace/BackTestSys</backtestsys_repo_root>
    <backtestsys_base_config>/workspace/BackTestSys/config.xml</backtestsys_base_config>
    <groundtruth_dir>/workspace/tests/fixtures/backtestsys_gt</groundtruth_dir>
</paths>
```

- `groundtruth_dir` 目前为全局 GT 目录，需包含：
  - `doneinfo.csv`
  - `excutiondetail.csv`
- 自动发现模式下 `replay_order_file` / `replay_cancel_file` 可以不配置

---

## 5. dataset_plan 节点（核心）

```xml
<dataset_plan>
    <train_ratio>9</train_ratio>
    <test_ratio>1</test_ratio>
    <seed>42</seed>
    <auto_discovery>
        <data_dir>/workspace/tests/fixtures/backtestsys_gt/data_auto</data_dir>
        <data_glob>market_*.csv</data_glob>
        <data_date_regex>market_(?P&lt;date&gt;\d{8})\.csv</data_date_regex>
        <replay_order_dir>/workspace/tests/fixtures/backtestsys_gt/replay_auto</replay_order_dir>
        <replay_order_pattern>orders_{date}.csv</replay_order_pattern>
        <replay_cancel_dir>/workspace/tests/fixtures/backtestsys_gt/replay_auto</replay_cancel_dir>
        <replay_cancel_pattern>cancels_{date}.csv</replay_cancel_pattern>
    </auto_discovery>
</dataset_plan>
```

语义：

1. 在 `data_dir` 内按 `data_glob` 自动搜索数据文件。
2. 用 `data_date_regex` 从数据文件名提取日期（需包含命名组 `date` 或第一个捕获组）。
3. 通过 `replay_order_pattern` / `replay_cancel_pattern` 按同一天匹配 replay 文件。
4. 组装出 `dataset_plan.files` 后按 `seed` 做 deterministic shuffle。
5. 按 `train_ratio:test_ratio` 切分为 train/test。
6. 每个 **trial**（一组超参）会在 **train 全部文件** 上跑一遍并聚合 loss。
7. 优化结束后，用最佳 train trial 在 **test 全部文件** 上跑一遍，生成 final report（不 tell）。

约束：

- 自动发现后的数据文件至少 2 个
- `train_ratio`、`test_ratio` 必须 > 0
- 每个数据文件必须能匹配到同日期的 `order/cancel` 文件

---

## 6. sampler / pruner

```xml
<sampler>
    <type>tpe</type>
    <seed>42</seed>
    <n_startup_trials>4</n_startup_trials>
    <constant_liar>true</constant_liar>
</sampler>

<pruner>
    <type>median</type>
    <n_startup_trials>2</n_startup_trials>
    <n_warmup_steps>1</n_warmup_steps>
</pruner>
```

说明：当前执行后端主要上报终态事件，pruner 参数可保留但中途剪枝触发有限。

---

## 7. search_space

```xml
<search_space>
    <param name="exchange.cancel_bias_k" type="float" low="0.0" high="0.3" />
    <param name="tape.epsilon" type="float" low="0.6" high="1.4" />
    <param name="runner.delay_in" type="int" low="0" high="1" />
</search_space>
```

`name` 使用 BackTestSys 配置点路径；采样值会写入 `run_spec.config.overrides`。

---

## 8. base_overrides

```xml
<base_overrides>
    <override key="data.path" type="str">/workspace/tests/fixtures/backtestsys_gt/data_auto/market_20240101.csv</override>
    <override key="data.format" type="str">csv</override>
    <override key="runner.delay_out" type="int">0</override>
    <override key="strategy.name" type="str">ReplayStrategy_Impl</override>
</base_overrides>
```

说明：

- `data.path` 会被 `dataset_plan` 为每个子 run 动态覆盖。
- `strategy.order_file` 与 `strategy.cancel_file` 也会由自动发现结果按日期动态覆盖。
- 其余项作为每个子 run 的固定基础配置。

---

## 9. loss 与聚合

### 9.1 单文件 loss（run loss）

当前 evaluator 定义：

`|DoneInfo_count - GT_doneinfo_count| + |ExecutionDetail_count - GT_executiondetail_count|`

### 9.2 trial 聚合 loss（port + adapter）

core 调用 `TrialLossAggregator` 聚合一个 trial 的多个 run loss。  
当前 BackTestSys adapter 使用 `MeanTrialLossAggregator`（均值）：

`trial_loss = mean(run_loss_1 ... run_loss_n)`

---

## 10. test-only final report

- 优化结束后自动触发
- 使用最佳 train trial 的超参
- 在 test 文件集合上执行并聚合
- 结果写入 `ResultStore.run_record`（key: `final_report:test`）
- **不会调用 `OptimizerBackend.tell()`**

---

## 11. 最小示例

```bash
python3 main.py --config config.xml
```

