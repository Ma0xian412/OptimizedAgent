from __future__ import annotations

from enum import Enum, unique


@unique
class SamplingMode(str, Enum):
    ASYNC_FILL = "ASYNC_FILL"
    GENERATIONAL_BATCH = "GENERATIONAL_BATCH"
    COORDINATED_BATCH = "COORDINATED_BATCH"


@unique
class DispatchDecision(str, Enum):
    SUBMIT_NOW = "SUBMIT_NOW"
    BUFFER = "BUFFER"


@unique
class EventKind(str, Enum):
    CHECKPOINT = "CHECKPOINT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@unique
class TrialState(str, Enum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"
