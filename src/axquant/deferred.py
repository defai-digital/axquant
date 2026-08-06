"""Fail-closed gates for deferred expansion features (P2).

Vision-tower optimization, per-expert unfused MoE splits, and domain LoRA/SFT are
intentionally unavailable until their public MLX / AX Engine execution evidence exists.
"""

from __future__ import annotations

from axquant.errors import PlanningError
from axquant.schema import DeferredFeature, SupportTier

_MESSAGES: dict[DeferredFeature, str] = {
    DeferredFeature.VLM_OPTIMIZATION: (
        "Vision-tower optimization is deferred (P2): promoted VLM language paths may be "
        "quantized, but vision towers remain protected at BF16 in the main checkpoint or a "
        "sidecar. Enable only when a public MLX runtime provides a validated vision quant path."
    ),
    DeferredFeature.PER_EXPERT_UNFUSED: (
        "Per-expert (unfused) MoE precision is deferred (P2): packed expert stacks "
        "quantize as fused switch modules with one precision per group. Finer splits "
        "require MLX-LM-side support."
    ),
    DeferredFeature.LORA_DOMAIN_SFT: (
        "Domain LoRA/SFT is out of scope for AXQuant recovery (P2): recovery is "
        "retention-restore only (scales/biases). Use an external fine-tune stack for "
        "domain adaptation."
    ),
    DeferredFeature.FAMILY_WITHOUT_PROMOTION: (
        "Family expansion requires convertible/certified tier promotion evidence "
        "(AXQ-017). inspect-only families cannot convert."
    ),
}


def require_feature(feature: DeferredFeature) -> None:
    """Always fail closed for deferred features until explicitly implemented."""
    raise PlanningError(_MESSAGES[feature])


def assert_conversion_tier(support_tier: SupportTier, *, family: str) -> None:
    """Fail closed when a family lacks promotion evidence for conversion."""
    if support_tier is SupportTier.INSPECT_ONLY:
        raise PlanningError(
            f"{family} is inspect-only; conversion requires convertible or certified "
            f"tier promotion evidence. {_MESSAGES[DeferredFeature.FAMILY_WITHOUT_PROMOTION]}"
        )


def assert_not_vlm_optimization(*, vision_optimize: bool) -> None:
    if vision_optimize:
        require_feature(DeferredFeature.VLM_OPTIMIZATION)


def assert_not_per_expert_unfused(*, per_expert: bool) -> None:
    if per_expert:
        require_feature(DeferredFeature.PER_EXPERT_UNFUSED)


def deferred_feature_matrix() -> list[dict[str, str]]:
    """Return a stable matrix for CLI `deferred-features`."""
    return [
        {
            "feature": feature.value,
            "status": "deferred",
            "message": _MESSAGES[feature],
        }
        for feature in DeferredFeature
    ]
