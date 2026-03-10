# BackTestSys 优化配置说明（`config.xml`）

本文说明控制面入口 `main.py` 对 `config.xml` 的配置要求、字段含义与示例。

---

## 1. 配置文件位置与启动方式

- 默认配置文件名：`config.xml`（仓库根目录）
- 启动命令：

```bash
python3 main.py --config /workspace/config.xml
```

---

## 2. 顶层结构

`config.xml` 顶层节点为 `<optimization>`，包含以下子节点：

- `<study>`：实验与并发控制
- `<paths>`：路径配置
- `<sampler>`：采样器配置（Optuna）
- `<pruner>`：剪枝器配置（Optuna）
- `<search_space>`：可优化参数空间
- `<base_overrides>`：每次回测都生效的固定覆盖项

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

字段说明：

- `spec_id`：实验标识（必填，非空字符串）
- `dataset_version`：数据版本（参与 `spec_hash`）
- `engine_version`：引擎版本（参与 `spec_hash`）
- `storage_dsn`：Optuna 存储地址（建议 sqlite）
- `max_trials`：最多 ask 多少个 trial
- `max_failures`：允许失败上限
- `max_in_flight_trials`：控制面最大并发 trial 数
- `max_workers`：BackTestSys 执行后端线程池 worker 数

---

## 4. paths 节点

```xml
<paths>
    <data_dir>/workspace/data/backtestsys_mock</data_dir>
    <backtestsys_repo_root>/workspace/BackTestSys</backtestsys_repo_root>
    <backtestsys_base_config>/workspace/BackTestSys/config.xml</backtestsys_base_config>
    <replay_order_file>/workspace/tests/fixtures/backtestsys_gt/orders.csv</replay_order_file>
    <replay_cancel_file>/workspace/tests/fixtures/backtestsys_gt/cancels.csv</replay_cancel_file>
    <groundtruth_dir>/workspace/tests/fixtures/backtestsys_gt</groundtruth_dir>
</paths>
```

字段说明：

- `data_dir`：控制面本地存储目录（run cache / objective cache / result store）
- `backtestsys_repo_root`：BackTestSys 仓库根目录
- `backtestsys_base_config`：BackTestSys 原始配置文件路径
- `replay_order_file`：`ReplayStrategy_Impl` 订单 CSV
- `replay_cancel_file`：`ReplayStrategy_Impl` 撤单 CSV
- `groundtruth_dir`：GT 文件目录，必须包含：
  - `doneinfo.csv`
  - `excutiondetail.csv`（注意文件名拼写是 `excutiondetail`）

---

## 5. sampler 节点

```xml
<sampler>
    <type>tpe</type>
    <seed>42</seed>
    <n_startup_trials>4</n_startup_trials>
    <constant_liar>true</constant_liar>
</sampler>
```

- `type`：`tpe` 或 `random`
- `seed`：随机种子
- `n_startup_trials`：TPE 冷启动 trial 数（可选）
- `constant_liar`：并发下 TPE 建议开启（可选）

---

## 6. pruner 节点

```xml
<pruner>
    <type>median</type>
    <n_startup_trials>2</n_startup_trials>
    <n_warmup_steps>1</n_warmup_steps>
</pruner>
```

- `type`：`median` 或 `nop`
- `n_startup_trials`、`n_warmup_steps`：`median` 参数（可选）

说明：当前 BackTestSys 执行适配器只上报终态事件，不产出 checkpoint，pruner 仍可保留配置但不会触发中途剪枝链路。

---

## 7. search_space 节点

```xml
<search_space>
    <param name="exchange.cancel_bias_k" type="float" low="0.0" high="0.3" />
    <param name="tape.epsilon" type="float" low="0.6" high="1.4" />
    <param name="runner.delay_in" type="int" low="0" high="1" />
</search_space>
```

每个 `<param>` 含义：

- `name`：参数名（会作为 BackTestSys 配置覆盖路径）
- `type`：`int` / `float` / `categorical`
- `low`、`high`：`int/float` 必填
- `choices`：`categorical` 必填，逗号分隔（例如 `choices="A,B,C"`）

---

## 8. base_overrides 节点

```xml
<base_overrides>
    <override key="data.path" type="str">/workspace/tests/fixtures/backtestsys_gt/market.csv</override>
    <override key="data.format" type="str">csv</override>
    <override key="runner.delay_out" type="int">0</override>
    <override key="strategy.name" type="str">ReplayStrategy_Impl</override>
</base_overrides>
```

说明：

- 这些项对所有 trial 固定生效
- `search_space` 采样出的参数会覆盖同名项（优先级更高）
- `key` 使用点路径，对应 BackTestSys `BacktestConfig` 的字段层级
- `type` 支持：`int` / `float` / `bool` / `str`

---

## 9. 当前 objective 语义

当前目标固定为 **minimize**：

`|DoneInfo_count - GT_doneinfo_count| + |ExecutionDetail_count - GT_executiondetail_count|`

其中：

- 回测结果来自 BackTestSys `app.run()` 的 `BacktestRunResult`
- GT 来自 `groundtruth_dir` 下的 `doneinfo.csv` 与 `excutiondetail.csv`

同时，若一次回测结果满足：

- `len(DoneInfo) + len(ExecutionDetail) == 0`

该 trial 会被标记为 `FAILED`。

---

## 10. 最小可运行示例

使用仓库当前提供的示例数据时，下面命令可直接跑通：

```bash
python3 main.py --config config.xml
```

