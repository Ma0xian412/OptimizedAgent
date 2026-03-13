from __future__ import annotations

from typing import Any


def resolve_effective_params(
    run_cfg: dict[str, Any],
    trial_params: dict[str, object],
    dataset_input: dict[str, Any],
) -> dict[str, object]:
    binding = run_cfg.get("param_binding")
    if not isinstance(binding, dict) or binding.get("mode", "trial_global") == "trial_global":
        return dict(trial_params)
    machine = dataset_input.get("machine")
    contract = dataset_input.get("contract")
    machine_map = binding.get("machine_delay_map")
    core_map = binding.get("contract_core_map")
    if not isinstance(machine, str) or not isinstance(contract, str):
        return dict(trial_params)
    if not isinstance(machine_map, dict) or not isinstance(core_map, dict):
        return dict(trial_params)
    delay = machine_map.get(machine)
    core = core_map.get(contract)
    if not isinstance(delay, int) or isinstance(delay, bool):
        return dict(trial_params)
    if not isinstance(core, dict):
        return dict(trial_params)
    lam = core.get("time_scale_lambda")
    bias = core.get("cancel_bias_k")
    if not isinstance(lam, (int, float)) or not isinstance(bias, (int, float)):
        return dict(trial_params)
    return {
        "time_scale_lambda": float(lam),
        "cancel_bias_k": float(bias),
        "delay_in": int(delay),
        "delay_out": int(delay),
    }
