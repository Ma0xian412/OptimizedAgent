# Staged Calibration XML 配置使用指南

本文档说明如何使用 `iter_backtestsys.py --config <绝对路径>` 运行 staged calibration。

## 1. 运行入口

```bash
python3 /workspace/iter_backtestsys.py --config /workspace/staged_calibration.config.xml
```

可选进度参数：

```bash
python3 /workspace/iter_backtestsys.py \
  --config /workspace/staged_calibration.config.xml \
  --progress-interval-seconds 2 \
  --progress-format text
```

- `--progress-interval-seconds`：进度心跳间隔（秒，必须 > 0）
- `--progress-format`：`text` 或 `json`

注意：

- `--config` 必填。
- `--config` 必须是 **绝对路径**。

---

## 2. 预设配置文件

仓库已提供可直接运行的预设文件：

```text
/workspace/staged_calibration.config.xml
```

---

## 3. XML 字段说明（全部关键路径均要求绝对路径）

根节点：`<staged_calibration>`

### 3.1 全局路径

- `workspace_root`：工作区根目录
- `runtime_root`：运行输出根目录（每次运行会创建 `runtime_root/<run_tag>`）
- `backtestsys_root`：BackTestSys 工程目录（需包含 `main.py`）
- `base_config_path`：BackTestSys 基础 XML 模板
- `python_executable`：Python 解释器绝对路径（如 `/usr/bin/python3`）

### 3.2 运行控制

- `max_failures`
- `baseline_trials`
- `machine_delay_trials`
- `contract_core_trials`
- `verify_trials`

以上字段均为正整数。

### 3.3 资源配置

位于 `default_resources`，支持键：

- `cpu`
- `memory_mb`
- `memory_gb`
- `gpu`
- `max_runtime_seconds`

值均为正整数。

### 3.4 搜索空间

位于 `search_ranges`：

- `delay.low / delay.high`（int）
- `time_scale_lambda.low / time_scale_lambda.high`（float）
- `cancel_bias_k.low / cancel_bias_k.high`（float）

要求 `low <= high`。

### 3.5 数据集（每个 dataset 独立配置）

位于 `datasets/dataset`，每个 dataset 必须包含：

- `dataset_id`
- `market_data_path`（绝对路径）
- `order_file`（绝对路径）
- `cancel_file`（绝对路径）
- `machine`
- `contract`
- `groundtruth_doneinfo_path`（绝对路径）
- `groundtruth_executiondetail_path`（绝对路径）

---

## 4. 完整示例

```xml
<?xml version="1.0" encoding="UTF-8"?>
<staged_calibration>
  <workspace_root>/workspace</workspace_root>
  <runtime_root>/workspace/runtime</runtime_root>
  <backtestsys_root>/workspace/BackTestSys</backtestsys_root>
  <base_config_path>/workspace/config.xml</base_config_path>
  <python_executable>/usr/bin/python3</python_executable>

  <max_failures>2</max_failures>
  <baseline_trials>1</baseline_trials>
  <machine_delay_trials>12</machine_delay_trials>
  <contract_core_trials>12</contract_core_trials>
  <verify_trials>1</verify_trials>

  <default_resources>
    <cpu>1</cpu>
    <max_runtime_seconds>60</max_runtime_seconds>
  </default_resources>

  <search_ranges>
    <delay>
      <low>0</low>
      <high>500000</high>
    </delay>
    <time_scale_lambda>
      <low>-0.5</low>
      <high>0.5</high>
    </time_scale_lambda>
    <cancel_bias_k>
      <low>-1.0</low>
      <high>1.0</high>
    </cancel_bias_k>
  </search_ranges>

  <datasets>
    <dataset>
      <dataset_id>ds_01</dataset_id>
      <market_data_path>/workspace/mock_backtestsys/market_data_ds_01.csv</market_data_path>
      <order_file>/workspace/mock_backtestsys/replay_orders.csv</order_file>
      <cancel_file>/workspace/mock_backtestsys/replay_cancels.csv</cancel_file>
      <machine>m1</machine>
      <contract>c1</contract>
      <groundtruth_doneinfo_path>/workspace/mock_backtestsys/groundtruth/PubOrderDoneInfoLog_m1_20260312_TEST_CONTRACT.csv</groundtruth_doneinfo_path>
      <groundtruth_executiondetail_path>/workspace/mock_backtestsys/groundtruth/PubExecutionDetailLog_m1_20260312_TEST_CONTRACT.csv</groundtruth_executiondetail_path>
    </dataset>
  </datasets>
</staged_calibration>
```

---

## 5. 启动后输出说明

脚本会打印两段 JSON：

1. `effective config summary`：生效配置摘要（用于核对）
2. `staged calibration output`：本次运行结果（包含 `run_tag`、`runtime_root`、最优结果等）

此外，运行期间会持续输出进度事件（stage started/progress/finished/failed），并落盘到：

- `runtime_root/<run_tag>/progress.jsonl`

---

## 6. 常见错误与定位

### 6.1 `--config must be an absolute path`

原因：`--config` 使用了相对路径。  
处理：改为绝对路径。

### 6.2 `required path not found: ...`

原因：配置中的某个文件路径不存在。  
处理：检查对应字段路径是否正确，文件是否已准备好。

### 6.3 `config.<field> is required`

原因：缺少必填字段。  
处理：按本指南第 3 节补齐字段。

### 6.4 `config.<field> must be a positive int`

原因：某些轮次/资源字段不是正整数。  
处理：改为正整数。

