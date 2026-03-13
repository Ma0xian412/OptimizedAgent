from __future__ import annotations

from optimization_control_plane.adapters.backtestsys.backtest_loss_math import (
    calculate_daily_raw_metrics,
    daily_intermediate_value,
)
from optimization_control_plane.adapters.backtestsys.backtest_loss_parsing import (
    _CANCEL_REQUEST_TABLE,
    _DONE_INFO_TABLE,
    _EXECUTION_DETAIL_TABLE,
    _LOSS_SCHEMA_VERSION,
    _ORDER_INFO_TABLE,
    index_done_info,
    index_earliest_cancel_time,
    index_execution_detail,
    index_order_info,
    order_key_sort_key,
    read_artifact_refs,
    read_tables,
)
from optimization_control_plane.domain.models import GroundTruthData, ObjectiveResult, RunResult


class BackTestObjectiveEvaluatorAdapter:
    """Evaluate one BackTestSys run into single-dataset raw objective metrics."""

    def evaluate(
        self,
        run_result: RunResult,
        spec: object,
        groundtruth: GroundTruthData,
    ) -> ObjectiveResult:
        del spec  # Evaluator is data-driven; config-level weighting is done by trial aggregator.
        sim_tables = read_tables(
            payload=run_result.payload,
            required_tables=(
                _DONE_INFO_TABLE,
                _EXECUTION_DETAIL_TABLE,
                _ORDER_INFO_TABLE,
                _CANCEL_REQUEST_TABLE,
            ),
            payload_name="run_result.payload",
        )
        gt_tables = read_tables(
            payload=groundtruth.payload,
            required_tables=(_DONE_INFO_TABLE, _EXECUTION_DETAIL_TABLE),
            payload_name="groundtruth.payload",
        )
        order_info_by_key = index_order_info(sim_tables[_ORDER_INFO_TABLE])
        sim_done_by_key = index_done_info(sim_tables[_DONE_INFO_TABLE], source_name="Sim")
        gt_done_by_key = index_done_info(gt_tables[_DONE_INFO_TABLE], source_name="GT")
        evaluation_keys = tuple(sorted(
            set(order_info_by_key) & set(sim_done_by_key) & set(gt_done_by_key),
            key=order_key_sort_key,
        ))
        if not evaluation_keys:
            raise ValueError("no evaluable orders found in intersection(OrderInfo, GT.DoneInfo, Sim.DoneInfo)")
        sim_exec_by_key = index_execution_detail(sim_tables[_EXECUTION_DETAIL_TABLE], source_name="Sim")
        gt_exec_by_key = index_execution_detail(gt_tables[_EXECUTION_DETAIL_TABLE], source_name="GT")
        cancel_time_by_key = index_earliest_cancel_time(sim_tables[_CANCEL_REQUEST_TABLE])
        raw_metrics = calculate_daily_raw_metrics(
            order_info_by_key=order_info_by_key,
            sim_done_by_key=sim_done_by_key,
            gt_done_by_key=gt_done_by_key,
            sim_exec_by_key=sim_exec_by_key,
            gt_exec_by_key=gt_exec_by_key,
            cancel_time_by_key=cancel_time_by_key,
            evaluation_keys=evaluation_keys,
        )
        attrs = {
            "value": daily_intermediate_value(raw_metrics),
            "loss_schema_version": _LOSS_SCHEMA_VERSION,
            "raw": {
                "curve": raw_metrics.curve,
                "terminal": raw_metrics.terminal,
                "post": raw_metrics.post,
            },
            "counts": {
                "order_count": raw_metrics.order_count,
                "cancel_order_count": raw_metrics.cancel_order_count,
            },
            "availability": {
                "curve": True,
                "terminal": True,
                "post": raw_metrics.post is not None,
            },
        }
        return ObjectiveResult(attrs=attrs, artifact_refs=read_artifact_refs(run_result.payload))
