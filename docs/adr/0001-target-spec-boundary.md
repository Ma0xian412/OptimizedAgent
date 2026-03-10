# ADR 0001: 引入 TargetSpec 并冻结目标边界

## 状态

已采纳（迭代 1）

## 背景

此前 `ExperimentSpec` 只有 `objective_config` 与 `execution_config`。在启动编排时，target 身份/配置可能被混放到其它配置块中，导致：

- 领域边界不清晰（target / objective / execution 职责耦合）。
- `spec_hash` 对 target 变化不敏感，影响去重与复现实验。
- 存在“从 execution/objective 推断 target”的隐式行为风险。

## 决策

1. 新增值对象 `TargetSpec`，仅表达 target 身份与配置。
2. `ExperimentSpec` 新增必填字段 `target_spec`。
3. `compute_spec_hash` 将 `target_spec` 纳入稳定序列化输入。
4. `TrialOrchestrator` 在 `start(settings=...)` 构建 spec 时强制要求 `target_spec`；
   缺失即报错，不做 fallback，不从 `execution_config` 或 `objective_config` 猜测。
5. `RunSpec` 仅预留可选 `target_spec` 字段，为后续执行链路全量迁移做准备。

## 迁移策略

- 采用 fail-fast：旧格式（无 `target_spec`）直接失败并暴露错误。
- 调用方必须显式提供 `target_spec`，先完成领域模型和入口校验，再在后续迭代扩展到执行链路。

## 结果

- target/objective/execution 边界明确。
- `spec_hash` 对 target 变化可感知。
- 迁移期问题可见，不会被 silent fallback 掩盖。
