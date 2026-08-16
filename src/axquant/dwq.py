from __future__ import annotations

import importlib
from typing import Any

from axquant.errors import PlanningError

# MLX encodes shape dims as int32. flatten/reshape(-1) on a fused Flash
# switch stack (often >2^31 elements) raises. Sample those tensors by
# striding each axis instead.
_MLX_FLAT_LIMIT = 2_147_483_647
_SAMPLE_LIMIT = 65536


def dwq_sample_strides(
    shape: tuple[int, ...],
    *,
    sample_limit: int = _SAMPLE_LIMIT,
) -> tuple[int, ...]:
    """Return per-axis strides so a strided view has at most *sample_limit* items."""

    if not shape:
        return ()
    if any(dim < 1 for dim in shape):
        raise PlanningError("DWQ sample strides require positive dimensions")
    strides = [1] * len(shape)
    counts = list(shape)
    product = 1
    for count in counts:
        product *= count
    while product > sample_limit:
        axis = max(range(len(counts)), key=lambda index: counts[index])
        strides[axis] += 1
        counts[axis] = (shape[axis] + strides[axis] - 1) // strides[axis]
        product = 1
        for count in counts:
            product *= count
    return tuple(strides)


def _element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def apply_mlx_dwq_clip(module: Any) -> dict[str, float | int | tuple[int, ...]]:
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
    shape = tuple(int(dim) for dim in weight.shape)
    elements = _element_count(shape)
    if elements < 2:
        raise PlanningError("DWQ requires at least two weight elements")
    if elements <= _MLX_FLAT_LIMIT:
        flat = weight.reshape(-1)
        stride = max(1, elements // _SAMPLE_LIMIT)
        sample = flat[::stride]
        sample_stride: int | tuple[int, ...] = stride
    else:
        strides = dwq_sample_strides(shape)
        sample = weight[tuple(slice(None, None, stride) for stride in strides)]
        sample = sample.reshape(-1)
        sample_stride = strides
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
        "sample_stride": sample_stride,
        "clip_lower": float(lower.item()),
        "clip_upper": float(upper.item()),
    }
