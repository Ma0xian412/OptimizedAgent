from __future__ import annotations

import hashlib

from optimization_control_plane.adapters.backtestsys.groundtruth_adapter import (
    BackTestSysGroundTruth,
    BackTestSysGroundTruthAdapter,
)
from optimization_control_plane.domain.models import (
    ExperimentSpec,
    GroundTruthData,
    stable_json_serialize,
)


class BackTestSysGroundTruthProvider:
    """Load BackTestSys GT once and expose it through the groundtruth port."""

    def __init__(self, adapter: BackTestSysGroundTruthAdapter) -> None:
        self._adapter = adapter

    def load(self, spec: ExperimentSpec) -> GroundTruthData:
        gt_dir = self._resolve_groundtruth_dir(spec)
        payload = self._adapter.load(gt_dir)
        fingerprint = self._fingerprint(gt_dir, payload)
        return GroundTruthData(payload=payload, fingerprint=fingerprint)

    @staticmethod
    def _resolve_groundtruth_dir(spec: ExperimentSpec) -> str:
        groundtruth_cfg = spec.objective_config.get("groundtruth")
        if not isinstance(groundtruth_cfg, dict):
            raise ValueError("spec.objective_config.groundtruth must be a dict")
        gt_dir = groundtruth_cfg.get("dir")
        if not isinstance(gt_dir, str) or not gt_dir:
            raise ValueError("spec.objective_config.groundtruth.dir must be a non-empty string")
        return gt_dir

    @staticmethod
    def _fingerprint(groundtruth_dir: str, payload: BackTestSysGroundTruth) -> str:
        raw = stable_json_serialize(
            {
                "groundtruth_dir": groundtruth_dir,
                "doneinfo_rows": payload.doneinfo.rows,
                "executiondetail_rows": payload.executiondetail.rows,
            }
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"
