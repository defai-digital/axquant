from __future__ import annotations

from axquant.errors import PlanningError
from axquant.schema import MemoryBudgetBreakdown


def _nonnegative_bytes(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PlanningError(f"{label} must be a non-negative integer byte count")
    return value


def evaluate_budget(
    weight_bytes: int,
    kv_bytes: int,
    reserve_bytes: int,
    limit_bytes: int,
) -> MemoryBudgetBreakdown:
    """Evaluate the normative weight + KV + reserve deployment constraint."""

    weights = _nonnegative_bytes(weight_bytes, "weight_bytes")
    kv = _nonnegative_bytes(kv_bytes, "kv_bytes")
    reserve = _nonnegative_bytes(reserve_bytes, "reserve_bytes")
    if type(limit_bytes) is not int or limit_bytes <= 0:
        raise PlanningError("limit_bytes must be a positive integer byte count")
    remainder = limit_bytes - (weights + kv + reserve)
    return MemoryBudgetBreakdown(
        weight_bytes=weights,
        kv_bytes=kv,
        reserve_bytes=reserve,
        limit_bytes=limit_bytes,
        remainder_bytes=remainder,
        feasible=remainder >= 0,
    )
