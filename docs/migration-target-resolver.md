# 迁移说明：TargetResolver 与 resolved_target（迭代 6 收口）

## 目标

本文用于收口 `TargetSpec -> TargetResolver -> ResolvedTarget -> RunSpecBuilder -> ExecutionBackend` 迁移，
并明确旧格式的 fail-fast 语义与排查路径。

## 旧用法 -> 新用法映射

| 场景 | 旧用法（已废弃） | 新用法（当前） |
|---|---|---|
| Experiment 输入 | `target` 缺失，或 target 塞进 `execution_config` | 顶层必须显式提供 `target_spec` |
| target 解释位置 | core / execution 侧隐式猜测 target | 仅 `TargetResolver.resolve(target_spec, spec)` 解释 |
| RunSpec target 字段 | 旧字段 `target_spec`（位于 RunSpec） | `run_spec.resolved_target` |
| RunSpecBuilder 签名 | `build(target_spec, params, execution_config)` | `build(resolved_target, params, execution_config)` |
| ExecutionBackend target 来源 | 从 `run_spec.config` 等隐式字段推断 | 只消费 `run_spec.resolved_target` |

## 新链路（冻结语义）

```text
TargetSpec (experiment级声明)
  -> TargetResolver.resolve(...)
ResolvedTarget (experiment级 canonical)
  -> RunSpecBuilder.build(resolved_target, params, execution_config)
RunSpec (trial级执行输入，含 resolved_target)
  -> ExecutionBackend.submit(request)
```

关键约束：

1. `TargetSpec` 与 `ResolvedTarget` 是 experiment 级固定对象；
2. `params` 是 trial 级变量，不改变 target 身份；
3. core 不解释 target kind；
4. ExecutionBackend 不消费原始 `TargetSpec`。

## fail-fast 语义（旧格式）

以下输入会直接失败，不做 fallback：

1. 顶层缺失 `target_spec`；
2. 旧格式把 target 藏在 `execution_config.target`；
3. 旧格式把 target 藏在 `execution_config.target_spec`；
4. `TargetResolver` 解析失败（未知 kind / schema 非法）。

典型错误信息：

- `settings must include spec fields ... missing=['target_spec']`
- `legacy target format is not supported: found execution_config.target; use top-level target_spec only`
- `legacy target format is not supported: found execution_config.target_spec; use top-level target_spec only`

## 迁移示例

### 旧（错误）

```python
settings = {
    "spec_id": "exp_1",
    "meta": {"dataset_version": "ds_v1", "engine_version": "e_v1"},
    "objective_config": {...},
    "execution_config": {
        "executor_kind": "backtest",
        "target": {"kind": "package", "ref": "pkg_a"},  # 旧格式，禁止
    },
}
```

### 新（正确）

```python
settings = {
    "spec_id": "exp_1",
    "meta": {"dataset_version": "ds_v1", "engine_version": "e_v1"},
    "target_spec": {
        "target_id": "logical_target",
        "config": {
            "envelope": {"kind": "package", "ref": "pkg_a", "config": {"region": "apac"}}
        },
    },
    "objective_config": {...},
    "execution_config": {"executor_kind": "backtest", "default_resources": {"cpu": 1}},
}
```

## 失败案例与排查建议

### 案例 1：`missing=['target_spec']`

- 现象：启动时直接报缺字段；
- 原因：调用方仍在使用旧 payload（无顶层 `target_spec`）；
- 排查：检查 `start(settings=...)` 入参是否包含顶层 `target_spec`。

### 案例 2：`legacy target format ... execution_config.target(_spec)`

- 现象：即使有 `execution_config.target` 仍直接失败；
- 原因：旧格式已禁止，系统不再从 execution 配置读取 target；
- 排查：删除 `execution_config.target` / `execution_config.target_spec`，改为顶层 `target_spec`。

### 案例 3：resolver 报 kind/schema 错误

- 现象：`envelope.kind` 非 `package/project` 或字段缺失时报错；
- 原因：`SimpleTargetResolver` 对 envelope schema 做严格校验；
- 排查：确认 `target_spec.config.envelope` 下 `kind/ref/config` 完整且合法。
