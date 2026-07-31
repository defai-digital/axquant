from __future__ import annotations

import importlib
from typing import Any

from axquant.errors import PlanningError


def apply_mlx_dwq_clip(module: Any) -> dict[str, float | int]:
    """Apply the portable DWQ mutation used by both probing and conversion.

    The sampling contract is intentionally shared: a sensitivity measurement is only useful
    when it evaluates the exact weight mutation that the conversion predicate later executes.
    """
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise PlanningError("DWQ execution requires MLX") from exc
    weight = getattr(module, "weight", None)
    if weight is None:
        raise PlanningError("DWQ requires a module with a weight tensor")
    flat = weight.reshape(-1)
    elements = int(flat.size)
    if elements < 2:
        raise PlanningError("DWQ requires at least two weight elements")
    sample_limit = 65536
    stride = max(1, elements // sample_limit)
    sample = flat[::stride]
    ordered = mx.sort(sample)
    sample_count = int(ordered.size)
    lower_index = max(0, int(sample_count * 0.001))
    upper_index = min(sample_count - 1, int(sample_count * 0.999))
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    mx.eval(lower, upper)
    module.weight = mx.clip(weight, lower, upper)
    mx.eval(module.weight)
    return {
        "sample_count": sample_count,
        "sample_stride": stride,
        "clip_lower": float(lower.item()),
        "clip_upper": float(upper.item()),
    }
