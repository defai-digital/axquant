"""Experimental ultra-low-bit (2/3-bit) policy labels (AXQ-028 QP3).

MLX affine kernels can execute 2- and 3-bit weights; AX Engine admits them only
behind documented experimental env gates. Plans that include these bits are
always development evidence until a full release audit passes.
"""

from __future__ import annotations

from axquant.schema import QuantizationPlan

EXPERIMENTAL_LOW_BITS: frozenset[int] = frozenset({2, 3})
PREFERRED_EXPERIMENTAL_GROUP_SIZE = 32

EXPERIMENTAL_WARNING = (
    "Experimental 2/3-bit weight assignments are development evidence only. "
    "They are not production claims without the ordinary release audit; "
    "AX Engine requires AX_ENGINE_2BIT_EXPERIMENTAL=1 / "
    "AX_ENGINE_3BIT_EXPERIMENTAL=1 for runtime admission."
)


def is_experimental_low_bit(bits: int) -> bool:
    return bits in EXPERIMENTAL_LOW_BITS


def plan_uses_experimental_low_bits(plan: QuantizationPlan) -> bool:
    return any(is_experimental_low_bit(assignment.bits) for assignment in plan.assignments)


def experimental_target_class(min_bits: int) -> str:
    if is_experimental_low_bit(min_bits):
        return f"{min_bits}bit-experimental"
    return f"{min_bits}bit"


def annotate_experimental_low_bit_plan(plan: QuantizationPlan) -> QuantizationPlan:
    """Stamp experimental labels on plans that assign 2- or 3-bit weights."""
    if not plan_uses_experimental_low_bits(plan):
        return plan
    experimental_bits = sorted(
        {
            assignment.bits
            for assignment in plan.assignments
            if is_experimental_low_bit(assignment.bits)
        }
    )
    warnings = list(plan.warnings)
    if EXPERIMENTAL_WARNING not in warnings:
        warnings.append(EXPERIMENTAL_WARNING)
    details = (
        "Experimental low-bit assignments present: "
        + ", ".join(f"{bits}-bit" for bits in experimental_bits)
        + f"; preferred fine group size is {PREFERRED_EXPERIMENTAL_GROUP_SIZE}."
    )
    if details not in warnings:
        warnings.append(details)
    min_quant = min(
        (assignment.bits for assignment in plan.assignments if assignment.bits < 16),
        default=16,
    )
    target_class = experimental_target_class(min_quant) if min_quant < 16 else plan.target_class
    return plan.model_copy(update={"warnings": warnings, "target_class": target_class})
