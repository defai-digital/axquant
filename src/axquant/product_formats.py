"""AXQ-070 product boundary: the only storage formats AXQuant writes.

MLX packs store MXFP4 (4-bit, group 32), MXFP8 (8-bit, group 32) and BF16.
CUDA packs store NVFP4 and BF16. Affine integer storage of any width, AWQ /
GPTQ / DWQ refined affine packing, FP8 and mixed budget lanes are not
produced. Readers (inspection, validation, probes of existing packs) keep
accepting historical affine tensors; this module only governs what is written.

In plan v2 an unrefined ``affine`` allocation names a storage slot by width:
4-bit is stored as MXFP4 and 8-bit as MXFP8. ``mxfp4`` allocations are 4-bit.
"""

from __future__ import annotations

from collections.abc import Iterable

from axquant.errors import PlanningError
from axquant.schema import Allocation, QuantMethod

MLX_GROUP_SIZE = 32
MLX_STORAGE_MODES: dict[int, str] = {4: "mxfp4", 8: "mxfp8"}
MLX_PRODUCT_BITS = (4, 8, 16)
MLX_PRODUCT_METHODS = (QuantMethod.AFFINE, QuantMethod.MXFP4, QuantMethod.BF16)
MLX_FORMATS_SUMMARY = "MLX MXFP4, MXFP8 and BF16; CUDA NVFP4 and BF16 (AXQ-070)"


def mlx_storage_violation(bits: int, method: QuantMethod, group_size: int | None) -> str | None:
    """Why one MLX allocation is outside the product formats, or ``None``."""

    if bits >= 16:
        return None
    if bits not in MLX_STORAGE_MODES:
        return f"{bits}-bit storage"
    if method not in (QuantMethod.AFFINE, QuantMethod.MXFP4):
        return f"{method.value} refinement"
    if method is QuantMethod.MXFP4 and bits != 4:
        return f"mxfp4 at {bits}-bit"
    if group_size != MLX_GROUP_SIZE:
        return f"{bits}-bit group {group_size}"
    return None


def mlx_candidate_allowed(bits: int, method: QuantMethod, group_size: int | None) -> bool:
    """Planner candidates: MXFP4 at 4-bit, the MXFP8 slot at 8-bit, BF16."""

    if bits >= 16:
        return True
    if bits == 4 and method is not QuantMethod.MXFP4:
        return False
    return mlx_storage_violation(bits, method, group_size) is None


def mlx_storage_mode(allocation: Allocation) -> str:
    """MLX ``to_quantized`` mode of one quantized allocation."""

    reason = mlx_storage_violation(allocation.bits, allocation.method, allocation.group_size)
    if reason is not None or allocation.bits >= 16:
        raise PlanningError(
            f"AXQ-070: AXQuant writes only {MLX_FORMATS_SUMMARY}; "
            f"{allocation.module_path} requests {reason or 'no quantization'}"
        )
    return MLX_STORAGE_MODES[allocation.bits]


def require_mlx_product_storage(allocations: Iterable[Allocation]) -> None:
    """Refuse a plan whose quantized allocations fall outside AXQ-070."""

    violations: dict[str, list[str]] = {}
    for allocation in allocations:
        reason = mlx_storage_violation(allocation.bits, allocation.method, allocation.group_size)
        if reason is not None:
            violations.setdefault(reason, []).append(allocation.module_path)
    if violations:
        detail = "; ".join(
            f"{reason}: {sorted(paths)[:5]}"
            + (f" and {len(paths) - 5} more" if len(paths) > 5 else "")
            for reason, paths in sorted(violations.items())
        )
        raise PlanningError(
            f"AXQ-070: AXQuant writes only {MLX_FORMATS_SUMMARY}. Quantized MLX "
            f"allocations must be 4-bit MXFP4 or 8-bit MXFP8 at group {MLX_GROUP_SIZE}; "
            f"this plan has {detail}"
        )


def mlx_default_quantization(allocations: Iterable[Allocation]) -> dict[str, int | str] | None:
    """Coherent top-level ``quantization`` default: the dominant quantized lane."""

    weights: dict[int, int] = {}
    for allocation in allocations:
        if allocation.bits in MLX_STORAGE_MODES:
            weights[allocation.bits] = weights.get(allocation.bits, 0) + int(allocation.parameters)
    if not weights:
        return None
    bits = max(sorted(weights), key=lambda width: weights[width])
    return {"bits": bits, "group_size": MLX_GROUP_SIZE, "mode": MLX_STORAGE_MODES[bits]}
