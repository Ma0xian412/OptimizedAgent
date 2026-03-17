from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backtestsys_visualizer.models import COMPONENT_NAMES


def build_default_figures(df: pd.DataFrame) -> dict[str, go.Figure]:
    return {
        "total_loss": build_total_loss_figure(df),
        "raw_components": build_component_figure(df, metric_prefix="raw", title="Raw 分项 loss"),
        "normalized_components": build_component_figure(
            df,
            metric_prefix="normalized",
            title="Normalized 分项 loss",
        ),
    }


def build_run_gt_metric_figure(df: pd.DataFrame, *, metric: str, title: str) -> go.Figure:
    if df.empty:
        return _empty_figure("暂无数据")
    if metric not in df.columns:
        return _empty_figure(f"缺少字段: {metric}")
    data = df.dropna(subset=[metric]).sort_values("global_iteration")
    if data.empty:
        return _empty_figure("当前指标无有效数据")
    fig = px.line(
        data,
        x="global_iteration",
        y=metric,
        color="stage",
        markers=True,
        title=title,
        hover_data=["trial_id", "stage_iteration"],
    )
    fig.update_layout(xaxis_title="全局迭代序号", yaxis_title=metric)
    return fig


def build_total_loss_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("暂无数据")
    sorted_df = df.sort_values("global_iteration")
    value_df = sorted_df.dropna(subset=["total_loss"])
    best_df = sorted_df.dropna(subset=["best_so_far"])
    if value_df.empty:
        return _empty_figure("没有可用的总 loss 数据")
    fig = px.line(
        value_df,
        x="global_iteration",
        y="total_loss",
        color="stage",
        markers=True,
        title="总 loss 迭代曲线",
        hover_data=["trial_id", "stage_iteration", "state"],
    )
    if not best_df.empty:
        fig.add_trace(
            go.Scatter(
                x=best_df["global_iteration"],
                y=best_df["best_so_far"],
                mode="lines",
                name="best-so-far",
                line={"color": "black", "dash": "dash"},
            )
        )
    fig.update_layout(xaxis_title="全局迭代序号", yaxis_title="总 loss")
    return fig


def build_component_figure(
    df: pd.DataFrame,
    *,
    metric_prefix: str,
    title: str,
) -> go.Figure:
    if df.empty:
        return _empty_figure("暂无数据")
    value_cols = [f"{metric_prefix}_{name}" for name in COMPONENT_NAMES]
    melted = df[["global_iteration", "stage", *value_cols]].melt(
        id_vars=["global_iteration", "stage"],
        value_vars=value_cols,
        var_name="component",
        value_name="value",
    )
    filtered = melted.dropna(subset=["value"]).copy()
    if filtered.empty:
        return _empty_figure(f"{title}无有效数据")
    filtered["component"] = filtered["component"].str.replace(f"{metric_prefix}_", "", regex=False)
    fig = px.line(
        filtered,
        x="global_iteration",
        y="value",
        color="component",
        line_dash="stage",
        markers=True,
        title=title,
    )
    fig.update_layout(xaxis_title="全局迭代序号", yaxis_title=f"{metric_prefix} loss")
    return fig


def export_figures(
    *,
    figures: dict[str, go.Figure],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for name, figure in figures.items():
        html_path = output_dir / f"{name}.html"
        png_path = output_dir / f"{name}.png"
        figure.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        exported.append(html_path)
        try:
            figure.write_image(str(png_path))
        except Exception as exc:  # pragma: no cover - depends on local renderer
            raise RuntimeError(
                f"PNG 导出失败: {png_path}. 请确认已安装 kaleido 且环境支持图像导出。"
            ) from exc
        exported.append(png_path)
    return exported


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white")
    return fig

