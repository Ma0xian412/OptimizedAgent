from __future__ import annotations

import importlib
import uuid
from collections import deque
from typing import Any

from optimization_control_plane.domain.enums import EventKind
from optimization_control_plane.domain.models import (
    ExecutionEvent,
    ExecutionRequest,
    RunHandle,
    RunResult,
)

_PYTHON_CLASS_KIND = "python_class"
_PYTHON_CALLABLE_KIND = "python_callable"


class PythonBlackBoxExecutionBackend:
    """Execution backend that runs Python targets synchronously."""

    def __init__(self) -> None:
        self._pending_events: deque[ExecutionEvent] = deque()
        self._handles: dict[str, RunHandle] = {}

    def submit(self, request: ExecutionRequest) -> RunHandle:
        handle = self._new_handle(request.request_id)
        self._handles[handle.handle_id] = handle
        self._enqueue_final_event(handle.handle_id, request)
        return handle

    def wait_any(
        self,
        handles: list[RunHandle],
        timeout: float | None = None,
    ) -> ExecutionEvent | None:
        _ = timeout
        active_ids = {h.handle_id for h in handles}
        for _ in range(len(self._pending_events)):
            event = self._pending_events.popleft()
            if event.handle_id in active_ids:
                return event
            self._pending_events.append(event)
        return None

    def cancel(self, handle: RunHandle, reason: str) -> None:
        self._pending_events = deque(
            event for event in self._pending_events if event.handle_id != handle.handle_id
        )
        self._pending_events.append(
            ExecutionEvent(
                kind=EventKind.CANCELLED,
                handle_id=handle.handle_id,
                reason=reason,
            )
        )

    def _new_handle(self, request_id: str) -> RunHandle:
        return RunHandle(
            handle_id=f"pyh_{uuid.uuid4().hex[:12]}",
            request_id=request_id,
            state="RUNNING",
        )

    def _enqueue_final_event(self, handle_id: str, request: ExecutionRequest) -> None:
        try:
            run_result = _execute_target_once(request.run_spec.target_config, request.run_spec.config)
        except Exception as exc:
            self._pending_events.append(
                ExecutionEvent(
                    kind=EventKind.FAILED,
                    handle_id=handle_id,
                    error_code=type(exc).__name__,
                )
            )
            return
        self._pending_events.append(
            ExecutionEvent(
                kind=EventKind.COMPLETED,
                handle_id=handle_id,
                run_result=run_result,
            )
        )


def _execute_target_once(target_config: dict[str, Any], config: dict[str, Any]) -> RunResult:
    kind = _read_required_str(target_config, "kind")
    ref = _read_required_str(target_config, "ref")
    symbol = _resolve_ref(ref)
    if kind == _PYTHON_CLASS_KIND:
        return _execute_class_target(symbol, target_config, config)
    if kind == _PYTHON_CALLABLE_KIND:
        return _execute_callable_target(symbol, config)
    raise ValueError(f"unsupported target kind: {kind}")


def _execute_class_target(
    symbol: Any,
    target_config: dict[str, Any],
    config: dict[str, Any],
) -> RunResult:
    if not callable(symbol):
        raise TypeError("target class symbol must be callable")
    init_kwargs = target_config.get("init_kwargs", {})
    if not isinstance(init_kwargs, dict):
        raise ValueError("target_config.init_kwargs must be a dict")
    invoke_method = target_config.get("invoke_method", "run")
    if not isinstance(invoke_method, str) or not invoke_method:
        raise ValueError("target_config.invoke_method must be a non-empty string")
    instance = symbol(**init_kwargs)
    method = getattr(instance, invoke_method, None)
    if method is None or not callable(method):
        raise AttributeError(f"class target has no callable method: {invoke_method}")
    return _normalize_run_result(method(config))


def _execute_callable_target(symbol: Any, config: dict[str, Any]) -> RunResult:
    if not callable(symbol):
        raise TypeError("target callable symbol must be callable")
    return _normalize_run_result(symbol(config))


def _resolve_ref(ref: str) -> Any:
    module_name, symbol_name = _split_ref(ref)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"symbol not found in module: {ref}") from exc


def _split_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError("target_config.ref must be in 'module.path:Symbol' format")
    module_name, symbol_name = ref.split(":", 1)
    if not module_name or not symbol_name:
        raise ValueError("target_config.ref must include both module and symbol")
    return module_name, symbol_name


def _read_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"target_config.{key} must be a non-empty string")
    return value


def _normalize_run_result(raw: Any) -> RunResult:
    if isinstance(raw, RunResult):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("python target must return RunResult or dict")
    metrics = raw.get("metrics")
    diagnostics = raw.get("diagnostics")
    artifact_refs = raw.get("artifact_refs", [])
    if not isinstance(metrics, dict):
        raise TypeError("run result dict must include dict 'metrics'")
    if not isinstance(diagnostics, dict):
        raise TypeError("run result dict must include dict 'diagnostics'")
    if not isinstance(artifact_refs, list):
        raise TypeError("run result dict 'artifact_refs' must be a list")
    return RunResult(
        metrics=dict(metrics),
        diagnostics=dict(diagnostics),
        artifact_refs=[_to_artifact_ref(value) for value in artifact_refs],
    )


def _to_artifact_ref(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("artifact_refs items must be strings")
    return value
