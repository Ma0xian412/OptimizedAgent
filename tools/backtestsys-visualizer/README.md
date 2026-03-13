# backtestsys-visualizer

用于可视化 BackTestSys 迭代调参结果，覆盖：

- 四个分项 loss（`raw`）：`curve / terminal / cancel / post`
- 四个分项 loss（`normalized`）
- 总 loss（`value`）与 `best-so-far`
- 迭代结果表（含参数）
- 跨 stage 汇总查看
- `RunResult vs GT` 对比指标随迭代变化（done state 一致率、done time 误差、成交率差等）

数据源同时支持：

1. `stages/*/ocp_data/trial_results/*.json`
2. `stages/*/study.db`

> 读取策略：同一 trial 优先使用 `trial_results` 的 `attrs`，缺失时回退到 `study.db` 的 `user_attrs/value`。

---

## 1. 安装

在仓库根目录执行：

```bash
python3 -m pip install -e "./tools/backtestsys-visualizer"
```

---

## 2. 启动交互页面（Streamlit）

```bash
python3 -m backtestsys_visualizer app --runtime-root ./runtime --run-tag iter_backtestsys_20260312_120000
```

如果省略 `--run-tag`，默认取 `runtime` 下最新的 `iter_backtestsys_*` 目录。

---

## 3. 导出 HTML/PNG/CSV

```bash
python3 -m backtestsys_visualizer export \
  --runtime-root ./runtime \
  --run-tag iter_backtestsys_20260312_120000 \
  --output-dir ./exports/backtestsys-visualizer
```

导出内容：

- `total_loss.html/.png`
- `raw_components.html/.png`
- `normalized_components.html/.png`
- `trial_points.csv`
- `run_gt_metrics.csv`
- `run_gt_state_match_rate.html/.png`
- `run_gt_done_time_mae.html/.png`
- `run_gt_fill_gap_ratio.html/.png`
- `summary.json`

只导出指定 stage：

```bash
python3 -m backtestsys_visualizer export \
  --runtime-root ./runtime \
  --run-tag iter_backtestsys_20260312_120000 \
  --stages baseline machine_delay_m1 final_verify
```

