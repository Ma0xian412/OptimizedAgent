# BackTestSys 优化配置说明（`config.xml`）

本文对应当前实现：`core` 已支持“一个 trial 对 train 全文件 fan-out 运行并聚合 loss”，并在优化结束后生成 **test-only final report**（不参与 `tell`）。
另外在正式迭代前，会先用 **base config** 在全部数据（train+test）上跑一轮，计算并写入 evaluator 的 `base_loss`。

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
- `<loss>`：loss 权重、归一化 eps
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

## 7. loss 配置

```xml
<loss>
    <weights>
        <curve>1.0</curve>
        <terminal>1.0</terminal>
        <cancel>1.0</cancel>
        <post>1.0</post>
    </weights>
    <eps>
        <curve>1e-12</curve>
        <terminal>1e-12</terminal>
        <cancel>1e-12</cancel>
        <post>1e-12</post>
    </eps>
</loss>
```

支持组件：`curve` / `terminal` / `cancel` / `post`。

### 7.1 权重字段（weights）

- 路径：`loss.weights.<component>`
- 含义：各 loss 组件在线性加权中的原始权重（要求 `>= 0`）。
- 运行时仅对“可用组件”做重归一化：  
  设可用集合为 `A`，原始权重为 `w_i`，则  
  `w'_i = w_i / Σ_{j∈A}(w_j)`。
- 若 `A` 中权重和为 0，会报错（`sum of available component weights must be > 0`）。

### 7.2 eps 字段（eps）

- 路径：`loss.eps.<component>`
- 含义：baseline 归一化的分母平滑项（要求 `>= 0`）。
- 对应组件归一化公式：`normalized_i = raw_i / (baseline_i + eps_i)`。

### 7.3 baseline 归一化行为

- 未初始化 baseline（无 `base_loss`）时：直接使用 `raw_components` 参与加权。
- 已初始化 baseline（有 `base_loss`）时：
  1. 若存在 `baseline_components`，优先按组件分母归一化：  
     `baseline_i = baseline_components[i]`；
  2. 否则回退使用统一分母：  
     `baseline_i = base_loss`。

---

## 8. search_space

```xml
<search_space>
    <param name="exchange.cancel_bias_k" type="float" low="0.0" high="0.3" />
    <param name="tape.epsilon" type="float" low="0.6" high="1.4" />
    <param name="runner.delay_in" type="int" low="0" high="1" />
</search_space>
```

`name` 使用 BackTestSys 配置点路径；采样值会写入 `run_spec.config.overrides`。

---

## 9. base_overrides

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

## 10. loss 与聚合

### 10.1 单文件 loss（run loss）

当前 evaluator 定义：

`|DoneInfo_count - GT_doneinfo_count| + |ExecutionDetail_count - GT_executiondetail_count|`

### 10.2 trial 聚合 loss（port + adapter）

core 调用 `TrialLossAggregator` 聚合一个 trial 的多个 run loss。  
当前 BackTestSys adapter 使用 `MeanTrialLossAggregator`（均值）：

`trial_loss = mean(run_loss_1 ... run_loss_n)`

### 10.3 baseline loss 初始化

开始优化循环前，系统会执行 baseline 预跑：

1. 用 base config（无采样参数）在 dataset_plan 的全部分片上运行。
2. 对每个分片计算 run loss。
3. 通过 `TrialLossAggregator` 聚合为一个 `base_loss`。
4. 若 evaluator 实现了 `set_base_loss(loss, attrs)`，则保存该 `base_loss` 供后续评估使用。

---

## 11. test-only final report

- 优化结束后自动触发
- 使用最佳 train trial 的超参
- 在 test 文件集合上执行并聚合
- 结果写入 `ResultStore.run_record`（key: `final_report:test`）
- **不会调用 `OptimizerBackend.tell()`**

---

## 12. 最小示例

```bash
python3 main.py --config config.xml
```

