from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtestsys_visualizer.models import stage_sort_key
from backtestsys_visualizer.run_gt_metrics import build_trial_metric_row, evaluate_dataset_metrics
from backtestsys_visualizer.run_gt_resolver import discover_stage_dirs, load_stage_trial_payloads


def load_run_gt_trial_dataframe(
    *,
    run_root: Path,
    selected_stages: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stage_dirs = discover_stage_dirs(run_root=run_root, selected_stages=selected_stages)
    for stage_dir in stage_dirs:
        rows.extend(_load_stage_metric_rows(stage_dir))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sorted_df = _sort_with_global_iteration(df)
    return sorted_df.drop(columns=["stage_order"])


def _load_stage_metric_rows(stage_dir: Path) -> list[dict[str, Any]]:
    payloads = load_stage_trial_payloads(stage_dir)
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        dataset_metrics = _build_dataset_metric_list(
            sim_payloads=list(payload.dataset_payloads),
            gt_tables=payload.gt_tables,
        )
        if not dataset_metrics:
            continue
        rows.append(
            build_trial_metric_row(
                stage_name=payload.stage_name,
                stage_order=stage_sort_key(payload.stage_name)[0],
                trial_number=payload.trial_number,
                expected_dataset_count=payload.expected_dataset_count,
                dataset_metrics=dataset_metrics,
            )
        )
    return rows


def _build_dataset_metric_list(
    *,
    sim_payloads: list[dict[str, Any]],
    gt_tables: dict[str, list[dict[str, str]]],
) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for payload in sim_payloads:
        dataset_metric = evaluate_dataset_metrics(sim_payload=payload, gt_tables=gt_tables)
        if dataset_metric is not None:
            metrics.append(dataset_metric)
    return metrics


def _sort_with_global_iteration(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values(["stage_order", "stage_iteration", "trial_id"]).reset_index(drop=True)
    sorted_df["global_iteration"] = sorted_df.index + 1
    return sorted_df

