from __future__ import annotations

import optuna

from optimization_control_plane.domain.enums import SamplingMode
from optimization_control_plane.domain.models import SamplerProfile

_DEFAULT_BATCH_SIZE = 1
_DEFAULT_PENDING_POLICY = "allow"


def build_sampler_profile(sampler: optuna.samplers.BaseSampler) -> SamplerProfile:
    if isinstance(sampler, optuna.samplers.RandomSampler):
        return SamplerProfile(
            mode=SamplingMode.ASYNC_FILL,
            startup_trials=0,
            batch_size=_DEFAULT_BATCH_SIZE,
            pending_policy=_DEFAULT_PENDING_POLICY,
            recommended_max_inflight=None,
        )

    if isinstance(sampler, optuna.samplers.TPESampler):
        n_startup: int = getattr(sampler, "_n_startup_trials", 10)
        return SamplerProfile(
            mode=SamplingMode.ASYNC_FILL,
            startup_trials=n_startup,
            batch_size=_DEFAULT_BATCH_SIZE,
            pending_policy=_DEFAULT_PENDING_POLICY,
            recommended_max_inflight=None,
        )

    raise NotImplementedError(
        f"Sampler {type(sampler).__name__} is not supported in V1. "
        "Only RandomSampler and TPESampler are supported."
    )
