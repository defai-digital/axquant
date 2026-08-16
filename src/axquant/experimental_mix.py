"""Measured mixed 2/3/4-bit assignment on the robust trunk.

AXQ-owned. Not mlx-optiq. Not ``plan-joint``.

The 1.8/1.9 default planner upgrades tensors first and only then forces one
signature onto each fused MLX-LM switch stack. Heuristic Flash-0731 recipes
that pick 2/3/4 by layer index or projection name spent extra BPW without
improving short QA. This path spends the same 2/3/4 grid from a sensitivity
report, but upgrades each fused switch as one unit ordered by measured
(or explicitly unmeasured) ranking loss per extra storage bit.

Attention stays off the experimental 2/3-bit grid (RM-42). Fused and packed
expert stacks stay affine. ``plan-joint`` is unchanged (4.0/4.8/6.0 x KV).
"""

from __future__ import annotations

from axquant.errors import PlanningError
from axquant.planner import plan_quantization
from axquant.schema import PlanRequest, QuantizationPlan, QuantMethod, SensitivityReport

EXPERIMENTAL_MIX_CANDIDATE_BITS = (2, 3, 4, 8, 16)
EXPERIMENTAL_MIX_WARNING = (
    "Experimental trunk mix: fused switch modules upgrade 2/3/4-bit as one "
    "unit from sensitivity evidence. Development only. Not a certificate. "
    "Not mlx-optiq. plan-joint is not used as the 2-bit allocator."
)


def experimental_mix_request(request: PlanRequest) -> PlanRequest:
    """Fill the 2/3/4 trunk rungs and 8/16 floors; default method is affine."""

    if 2 not in request.candidate_bits:
        raise PlanningError("plan-experimental-mix requires 2-bit on the candidate grid")
    bits = tuple(sorted(set(request.candidate_bits) | set(EXPERIMENTAL_MIX_CANDIDATE_BITS)))
    methods = request.candidate_methods or (QuantMethod.AFFINE, QuantMethod.BF16)
    return request.model_copy(
        update={
            "candidate_bits": bits,
            "candidate_methods": methods,
        }
    )


def plan_experimental_mix(
    report: SensitivityReport,
    request: PlanRequest,
) -> QuantizationPlan:
    """Allocate a convert-ready experimental 2/3/4 mix from a sensitivity report.

    Tests and the CLI must call this function (or ``plan_quantization`` with
    ``allocation_units='fused-module'``). The fused-signature rule stays the
    planner's fail-closed check; this wrapper only chooses the unit grid.
    """

    mix_request = experimental_mix_request(request)
    plan = plan_quantization(report, mix_request, allocation_units="fused-module")
    warnings = list(plan.warnings)
    if EXPERIMENTAL_MIX_WARNING not in warnings:
        warnings.insert(0, EXPERIMENTAL_MIX_WARNING)
    return plan.model_copy(update={"warnings": warnings})
