from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from backtestsys_visualizer.charts import (
    build_default_figures,
    build_run_gt_metric_figure,
    export_figures,
)
from backtestsys_visualizer.loader import discover_run_dirs, load_trial_points, to_dataframe
from backtestsys_visualizer.run_gt_analysis import (
    aggregate_run_gt_dataset_dataframe,
    load_run_gt_dataset_dataframe,
    load_run_gt_trial_dataframe,
)

_ENV_RUNTIME_ROOT = "BACKTESTSYS_VIS_RUNTIME_ROOT"
_ENV_RUN_TAG = "BACKTESTSYS_VIS_RUN_TAG"


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "app":
        _run_app_command(args)
        return
    if args.command == "export":
        _run_export_command(args)
        return
    parser.print_help()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BackTestSys 可视化工具")
    subparsers = parser.add_subparsers(dest="command")

    app_parser = subparsers.add_parser("app", help="启动 Streamlit 页面")
    _add_common_path_options(app_parser)

    export_parser = subparsers.add_parser("export", help="导出 HTML/PNG 图和 CSV")
    _add_common_path_options(export_parser)
    export_parser.add_argument(
        "--output-dir",
        default="./exports/backtestsys-visualizer",
        help="导出目录",
    )
    export_parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="指定 stage 名称列表，不传则导出所有 stage",
    )
    return parser


def _add_common_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-root", default="./runtime", help="runtime 根目录")
    parser.add_argument("--run-tag", default=None, help="指定 run_tag，不传则选最新")


def _run_app_command(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(
        runtime_root=Path(args.runtime_root).expanduser().resolve(),
        run_tag=args.run_tag,
    )
    app_path = (Path(__file__).resolve().parent / "streamlit_app.py").resolve()
    env = dict(os.environ)
    env[_ENV_RUNTIME_ROOT] = str(run_dir.parent)
    env[_ENV_RUN_TAG] = run_dir.name
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(command, check=True, env=env)


def _run_export_command(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(
        runtime_root=Path(args.runtime_root).expanduser().resolve(),
        run_tag=args.run_tag,
    )
    selected_stages = set(args.stages) if args.stages else None
    points = load_trial_points(run_root=run_dir, selected_stages=selected_stages)
    if not points:
        raise ValueError(f"没有可导出的数据: {run_dir}")
    df = to_dataframe(points)
    output_dir = Path(args.output_dir).expanduser().resolve()
    figures = build_default_figures(df)
    exported_files = export_figures(figures=figures, output_dir=output_dir)
    csv_path = output_dir / "trial_points.csv"
    df.to_csv(csv_path, index=False)
    exported_files.append(csv_path)
    exported_files.extend(_export_run_gt_assets(run_dir=run_dir, output_dir=output_dir, selected_stages=selected_stages))
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_tag": run_dir.name,
                "row_count": int(len(df)),
                "stage_count": int(df["stage"].nunique()),
                "exported_files": [str(path) for path in exported_files],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"导出完成: {summary_path}")


def _export_run_gt_assets(
    *,
    run_dir: Path,
    output_dir: Path,
    selected_stages: set[str] | None,
) -> list[Path]:
    run_gt_df = load_run_gt_trial_dataframe(run_root=run_dir, selected_stages=selected_stages)
    run_gt_dataset_df = load_run_gt_dataset_dataframe(run_root=run_dir, selected_stages=selected_stages)
    if run_gt_df.empty or run_gt_dataset_df.empty:
        return []
    exported: list[Path] = []
    csv_path = output_dir / "run_gt_metrics.csv"
    run_gt_df.to_csv(csv_path, index=False)
    exported.append(csv_path)
    dataset_csv_path = output_dir / "run_gt_dataset_metrics.csv"
    run_gt_dataset_df.to_csv(dataset_csv_path, index=False)
    exported.append(dataset_csv_path)
    machine_csv_path = output_dir / "run_gt_group_metrics_by_machine.csv"
    aggregate_run_gt_dataset_dataframe(run_gt_dataset_df, group_by="machine").to_csv(machine_csv_path, index=False)
    exported.append(machine_csv_path)
    contract_csv_path = output_dir / "run_gt_group_metrics_by_contract.csv"
    aggregate_run_gt_dataset_dataframe(run_gt_dataset_df, group_by="contract").to_csv(contract_csv_path, index=False)
    exported.append(contract_csv_path)
    machine_contract_csv_path = output_dir / "run_gt_group_metrics_by_machine_contract.csv"
    aggregate_run_gt_dataset_dataframe(run_gt_dataset_df, group_by="machine_contract").to_csv(
        machine_contract_csv_path,
        index=False,
    )
    exported.append(machine_contract_csv_path)
    metric_title = {
        "state_match_rate": "DoneState 一致率",
        "done_time_mae": "DoneTime 绝对误差均值",
        "fill_gap_ratio": "成交率差值均值",
    }
    figures = {
        f"run_gt_{metric}": build_run_gt_metric_figure(run_gt_df, metric=metric, title=f"{title} 迭代曲线")
        for metric, title in metric_title.items()
    }
    exported.extend(export_figures(figures=figures, output_dir=output_dir))
    return exported


def _resolve_run_dir(*, runtime_root: Path, run_tag: str | None) -> Path:
    run_dirs = discover_run_dirs(runtime_root=runtime_root)
    if not run_dirs:
        raise FileNotFoundError(f"未发现 run 目录: {runtime_root}")
    if run_tag is None:
        return run_dirs[0]
    for run_dir in run_dirs:
        if run_dir.name == run_tag:
            return run_dir
    raise FileNotFoundError(f"run_tag 不存在: {run_tag}")


if __name__ == "__main__":
    main()

