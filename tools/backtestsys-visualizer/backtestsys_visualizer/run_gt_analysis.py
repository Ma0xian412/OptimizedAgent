from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtestsys_visualizer.models import stage_sort_key
from backtestsys_visualizer.run_gt_metrics import build_trial_metric_row, evaluate_dataset_metrics
from backtestsys_visualizer.run_gt_resolver import (
    DatasetTrialPayload,
    discover_stage_dirs,
    load_stage_trial_payloads,
)

_ORDER_WEIGHT_METRICS = (
    "state_match_rate",
    "done_time_mae",
    "sim_fill_ratio",
    "gt_fill_ratio",
    "fill_gap_ratio",
    "terminal_raw_from_tables",
)
_CANCEL_WEIGHT_METRICS = ("post_cancel_gap", "cancel_state_match_rate")
_GROUP_BY_COLUMNS = {
    "trial": (),
    "dataset": ("dataset_id",),
    "machine": ("machine",),
    "contract": ("contract",),
    "machine_contract": ("machine", "contract"),
}


def load_run_gt_trial_dataframe(
    *,
    run_root: Path,
    selected_stages: set[str] | None = None,
) -> pd.DataFrame:
    trial_df, _ = _load_run_gt_dataframes(run_root=run_root, selected_stages=selected_stages)
    return trial_df


def load_run_gt_dataset_dataframe(
    *,
    run_root: Path,
    selected_stages: set[str] | None = None,
) -> pd.DataFrame:
    _, dataset_df = _load_run_gt_dataframes(run_root=run_root, selected_stages=selected_stages)
    return dataset_df


def filter_run_gt_dataset_dataframe(
    dataset_df: pd.DataFrame,
    *,
    machines: set[str] | None = None,
    contracts: set[str] | None = None,
    dataset_ids: set[str] | None = None,
) -> pd.DataFrame:
    if dataset_df.empty:
        return dataset_df
    filtered = dataset_df
    if machines is not None:
        filtered = filtered[filtered["machine"].isin(machines)]
    if contracts is not None:
        filtered = filtered[filtered["contract"].isin(contracts)]
    if dataset_ids is not None:
        filtered = filtered[filtered["dataset_id"].isin(dataset_ids)]
    return filtered.reset_index(drop=True)


def aggregate_run_gt_dataset_dataframe(dataset_df: pd.DataFrame, *, group_by: str) -> pd.DataFrame:
    if dataset_df.empty:
        return pd.DataFrame()
    group_dims = _GROUP_BY_COLUMNS.get(group_by)
    if group_dims is None:
        raise ValueError(f"unsupported group_by: {group_by}")
    meta_cols = ["global_iteration", "stage", "stage_iteration", "trial_id", *group_dims]
    rows = [
        _aggregate_group_row(group_df, group_dims=group_dims)
        for _, group_df in dataset_df.groupby(meta_cols, dropna=False, sort=False)
    ]
    if not rows:
        return pd.DataFrame()
    sort_cols = ["global_iteration", *group_dims]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def _load_run_gt_dataframes(
    *,
    run_root: Path,
    selected_stages: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    stage_dirs = discover_stage_dirs(run_root=run_root, selected_stages=selected_stages)
    for stage_dir in stage_dirs:
        stage_trial_rows, stage_dataset_rows = _load_stage_metric_rows(stage_dir)
        trial_rows.extend(stage_trial_rows)
        dataset_rows.extend(stage_dataset_rows)
    if not trial_rows:
        return pd.DataFrame(), pd.DataFrame()
    sorted_trial_df = _sort_with_global_iteration(pd.DataFrame(trial_rows))
    dataset_df = _build_dataset_df(dataset_rows=dataset_rows, trial_df=sorted_trial_df)
    return sorted_trial_df.drop(columns=["stage_order"]), dataset_df


def _load_stage_metric_rows(stage_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = load_stage_trial_payloads(stage_dir)
    trial_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for payload in payloads:
        stage_order = stage_sort_key(payload.stage_name)[0]
        dataset_metrics = _build_dataset_metric_list(payloads=list(payload.dataset_payloads))
        if not dataset_metrics:
            continue
        trial_rows.append(
            build_trial_metric_row(
                stage_name=payload.stage_name,
                stage_order=stage_order,
                trial_number=payload.trial_number,
                expected_dataset_count=payload.expected_dataset_count,
                dataset_metrics=dataset_metrics,
            )
        )
        dataset_rows.extend(
            _build_dataset_rows(
                stage_name=payload.stage_name,
                stage_order=stage_order,
                trial_number=payload.trial_number,
                expected_dataset_count=payload.expected_dataset_count,
                dataset_metrics=dataset_metrics,
            )
        )
    return trial_rows, dataset_rows


def _build_dataset_metric_list(
    *,
    payloads: list[DatasetTrialPayload],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for payload in payloads:
        dataset_metric = evaluate_dataset_metrics(
            sim_payload=payload.sim_payload,
            gt_tables=payload.gt_tables,
            dataset_id=payload.dataset_id,
            machine=payload.machine,
            contract=payload.contract,
        )
        if dataset_metric is not None:
            metrics.append(dataset_metric)
    return metrics


def _build_dataset_rows(
    *,
    stage_name: str,
    stage_order: int,
    trial_number: int,
    expected_dataset_count: int,
    dataset_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in dataset_metrics:
        rows.append(
            {
                "stage": stage_name,
                "stage_order": stage_order,
                "trial_id": str(trial_number),
                "stage_iteration": int(trial_number),
                "expected_dataset_count": int(expected_dataset_count),
                **metric,
            }
        )
    return rows


def _build_dataset_df(*, dataset_rows: list[dict[str, Any]], trial_df: pd.DataFrame) -> pd.DataFrame:
    if not dataset_rows:
        return pd.DataFrame()
    key_df = trial_df[["stage", "trial_id", "global_iteration"]].copy()
    key_df["__key"] = list(zip(key_df["stage"], key_df["trial_id"]))
    key_to_iteration = dict(zip(key_df["__key"], key_df["global_iteration"]))
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df["__key"] = list(zip(dataset_df["stage"], dataset_df["trial_id"]))
    dataset_df["global_iteration"] = dataset_df["__key"].map(key_to_iteration)
    merged = dataset_df.dropna(subset=["global_iteration"]).copy()
    merged["global_iteration"] = merged["global_iteration"].astype(int)
    sorted_df = merged.sort_values(["global_iteration", "dataset_id"]).reset_index(drop=True)
    return sorted_df.drop(columns=["stage_order", "__key"])


def _aggregate_group_row(group_df: pd.DataFrame, *, group_dims: tuple[str, ...]) -> dict[str, Any]:
    first = group_df.iloc[0]
    order_total = float(group_df["order_count"].sum())
    cancel_total = float(group_df["cancel_order_count"].sum())
    expected_count = int(group_df["expected_dataset_count"].max())
    row: dict[str, Any] = {
        "global_iteration": int(first["global_iteration"]),
        "stage": str(first["stage"]),
        "stage_iteration": int(first["stage_iteration"]),
        "trial_id": str(first["trial_id"]),
        "expected_dataset_count": expected_count,
        "dataset_count": int(len(group_df)),
        "missing_dataset_count": max(expected_count - int(len(group_df)), 0),
        "order_count": int(order_total),
        "cancel_order_count": int(cancel_total),
    }
    for dim in group_dims:
        row[dim] = str(first[dim])
    for metric in _ORDER_WEIGHT_METRICS:
        row[metric] = _weighted_mean(group_df, value_key=metric, weight_key="order_count")
    for metric in _CANCEL_WEIGHT_METRICS:
        row[metric] = _weighted_mean(group_df, value_key=metric, weight_key="cancel_order_count")
    return row


def _weighted_mean(group_df: pd.DataFrame, *, value_key: str, weight_key: str) -> float:
    weights = group_df[weight_key].astype(float)
    values = group_df[value_key].astype(float)
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return 0.0
    numerator = float((values * weights).sum())
    return numerator / denominator


def _sort_with_global_iteration(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.sort_values(["stage_order", "stage_iteration", "trial_id"]).reset_index(drop=True)
    sorted_df["global_iteration"] = sorted_df.index + 1
    return sorted_df

