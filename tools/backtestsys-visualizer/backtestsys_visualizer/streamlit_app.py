from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from backtestsys_visualizer.charts import build_default_figures
from backtestsys_visualizer.loader import discover_run_dirs, load_trial_points, to_dataframe

_ENV_RUNTIME_ROOT = "BACKTESTSYS_VIS_RUNTIME_ROOT"
_ENV_RUN_TAG = "BACKTESTSYS_VIS_RUN_TAG"


def run() -> None:
    st.set_page_config(page_title="BackTestSys Visualizer", layout="wide")
    st.title("BackTestSys 迭代结果可视化")
    runtime_root = _read_runtime_root_from_sidebar()
    run_dirs = _read_run_dirs(runtime_root=runtime_root)
    if not run_dirs:
        st.warning(f"未发现 run 目录：{runtime_root}")
        return
    selected_run_dir = _select_run_dir(run_dirs)
    points = load_trial_points(run_root=selected_run_dir)
    if not points:
        st.warning(f"当前 run 没有可视化数据：{selected_run_dir}")
        return
    selected_stages = _select_stages(points)
    filtered_points = [item for item in points if item.stage_name in selected_stages]
    df = to_dataframe(filtered_points)
    if df.empty:
        st.warning("按当前 stage 过滤后没有可用数据")
        return
    _render_overview(df)
    _render_figures(df)
    _render_table(df)


def _read_runtime_root_from_sidebar() -> Path:
    default_root = os.environ.get(_ENV_RUNTIME_ROOT, "./runtime")
    runtime_root_raw = st.sidebar.text_input("runtime 根目录", value=default_root).strip()
    return Path(runtime_root_raw).expanduser().resolve()


def _read_run_dirs(*, runtime_root: Path) -> list[Path]:
    try:
        return discover_run_dirs(runtime_root=runtime_root)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return []


def _select_run_dir(run_dirs: list[Path]) -> Path:
    run_names = [item.name for item in run_dirs]
    env_run_tag = os.environ.get(_ENV_RUN_TAG)
    default_index = run_names.index(env_run_tag) if env_run_tag in run_names else 0
    selected_name = st.sidebar.selectbox("选择 run_tag", options=run_names, index=default_index)
    selected = next(item for item in run_dirs if item.name == selected_name)
    st.sidebar.caption(f"run 路径：{selected}")
    return selected


def _select_stages(points: list) -> set[str]:
    stage_options = sorted({item.stage_name for item in points})
    selected = st.sidebar.multiselect("选择 stage（默认全选）", options=stage_options, default=stage_options)
    return set(selected)


def _render_overview(df: pd.DataFrame) -> None:
    st.subheader("概览")
    trial_count = int(len(df))
    stage_count = int(df["stage"].nunique())
    best_series = df["best_so_far"].dropna()
    best_value = float(best_series.iloc[-1]) if not best_series.empty else None
    source_trial_results = int(df["from_trial_results"].sum())
    source_study_db = int(df["from_study_db"].sum())
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("迭代点数", trial_count)
    col2.metric("stage 数", stage_count)
    col3.metric("当前 best-so-far", "-" if best_value is None else f"{best_value:.8f}")
    col4.metric("来自 trial_results", source_trial_results)
    col5.metric("来自 study.db", source_study_db)


def _render_figures(df: pd.DataFrame) -> None:
    st.subheader("曲线")
    figures = build_default_figures(df)
    st.plotly_chart(figures["total_loss"], use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(figures["raw_components"], use_container_width=True)
    with right:
        st.plotly_chart(figures["normalized_components"], use_container_width=True)


def _render_table(df: pd.DataFrame) -> None:
    st.subheader("迭代结果明细")
    table_cols = [
        "global_iteration",
        "stage",
        "stage_iteration",
        "trial_id",
        "state",
        "total_loss",
        "best_so_far",
        "raw_curve",
        "raw_terminal",
        "raw_cancel",
        "raw_post",
        "normalized_curve",
        "normalized_terminal",
        "normalized_cancel",
        "normalized_post",
        "params_json",
    ]
    table_df = df[table_cols].sort_values("global_iteration")
    st.dataframe(table_df, use_container_width=True)


if __name__ == "__main__":
    run()

