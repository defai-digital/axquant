"""Capability-gated vision/audio certification policy (AXQuant 1.8.0).

Best practice:

* If a modality is **not supported** by the pack (no architecture declaration and
  no protected sidecar weights), certification **disables** it:
  ``status=not-applicable``, ``supported=false``. No smoke or quality suite runs.
* If a modality **is supported**, certification must either:
  1. run a bound **smoke** (load + generation/transcription) → ``smoke-certified``;
  2. run a bound **quality** suite with thresholds → ``quality-certified``; or
  3. explicitly leave the modality **present but uncertified**
     (``present-not-certified``) when weights are only protected/preserved and no
     multimodal quality claim is authorized.

Public language must not treat ``Vision present=True`` on a model card as a
quality pass. Text dual-suite Tier 1 never implies vision-tower or audio quality
(see Certification Specification §8).
"""

from __future__ import annotations

from typing import Any, Literal

from axquant.schema.public_certification import (
    ModalityClaimStatus,
    PublicModalitiesBlock,
    PublicModalityClaim,
)

ModalityName = Literal["vision", "audio"]

_SMOKE_EVIDENCE = {
    "vision": "runtime-smoke-mlx-vlm",
    "audio": "runtime-smoke-mlx-audio",
}
_QUALITY_EVIDENCE = {
    "vision": "multimodal-quality-vision",
    "audio": "multimodal-quality-audio",
}


def derive_modality_claim(
    *,
    supported: bool,
    smoke_passed: bool | None = None,
    quality_passed: bool | None = None,
    reason: str | None = None,
    runtime: str | None = None,
    modality: ModalityName = "vision",
) -> PublicModalityClaim:
    """Derive one modality claim from support flags and optional evidence.

    Precedence when supported: quality > smoke > present-not-certified.
    Unsupported always maps to not-applicable regardless of evidence flags.
    """

    if not supported:
        return PublicModalityClaim(
            status="not-applicable",
            supported=False,
            reason=reason or f"{modality} not supported on this pack",
            runtime=None,
            evidence_kind=None,
        )
    if quality_passed is True:
        return PublicModalityClaim(
            status="quality-certified",
            supported=True,
            reason=reason or f"{modality} quality suite passed thresholds",
            runtime=runtime,
            evidence_kind=_QUALITY_EVIDENCE[modality],
        )
    if smoke_passed is True:
        return PublicModalityClaim(
            status="smoke-certified",
            supported=True,
            reason=reason or f"{modality} runtime smoke passed",
            runtime=runtime or ("mlx-vlm" if modality == "vision" else "mlx-audio"),
            evidence_kind=_SMOKE_EVIDENCE[modality],
        )
    return PublicModalityClaim(
        status="present-not-certified",
        supported=True,
        reason=reason
        or (
            f"{modality} weights present or protected; no multimodal "
            f"{'quality or smoke' if smoke_passed is None else 'quality'} claim"
        ),
        runtime=runtime,
        evidence_kind=None,
    )


def build_modalities_block(
    *,
    vision_supported: bool,
    audio_supported: bool,
    vision_smoke_passed: bool | None = None,
    audio_smoke_passed: bool | None = None,
    vision_quality_passed: bool | None = None,
    audio_quality_passed: bool | None = None,
    vision_reason: str | None = None,
    audio_reason: str | None = None,
    vision_runtime: str | None = None,
    audio_runtime: str | None = None,
) -> PublicModalitiesBlock:
    """Build the Tier 1 ``modalities`` envelope from capability + evidence."""

    return PublicModalitiesBlock(
        policy="capability-gated-v1",
        vision=derive_modality_claim(
            supported=vision_supported,
            smoke_passed=vision_smoke_passed,
            quality_passed=vision_quality_passed,
            reason=vision_reason,
            runtime=vision_runtime,
            modality="vision",
        ),
        audio=derive_modality_claim(
            supported=audio_supported,
            smoke_passed=audio_smoke_passed,
            quality_passed=audio_quality_passed,
            reason=audio_reason,
            runtime=audio_runtime,
            modality="audio",
        ),
    )


def claim_allows_public_quality(status: ModalityClaimStatus) -> bool:
    """Whether public language may claim multimodal quality for this status."""

    return status == "quality-certified"


def claim_allows_public_smoke(status: ModalityClaimStatus) -> bool:
    """Whether public language may claim multimodal runtime smoke."""

    return status in {"smoke-certified", "quality-certified"}


def validate_modality_evidence_consistency(
    block: PublicModalitiesBlock,
    *,
    vision_supported: bool,
    audio_supported: bool,
) -> list[str]:
    """Return human-readable issues; empty means consistent."""

    issues: list[str] = []
    if block.vision.supported != vision_supported:
        issues.append(
            "vision.supported "
            f"({block.vision.supported}) disagrees with capability flag "
            f"({vision_supported})"
        )
    if block.audio.supported != audio_supported:
        issues.append(
            "audio.supported "
            f"({block.audio.supported}) disagrees with capability flag "
            f"({audio_supported})"
        )
    if vision_supported and block.vision.status == "not-applicable":
        issues.append("vision is supported but claim status is not-applicable")
    if audio_supported and block.audio.status == "not-applicable":
        issues.append("audio is supported but claim status is not-applicable")
    if not vision_supported and block.vision.status != "not-applicable":
        issues.append("vision is unsupported but claim status is not not-applicable")
    if not audio_supported and block.audio.status != "not-applicable":
        issues.append("audio is unsupported but claim status is not not-applicable")
    return issues


def modalities_to_public_dict(block: PublicModalitiesBlock) -> dict[str, Any]:
    """Stable JSON-ready dict for certificate writers."""

    return block.model_dump(mode="json")


def summarize_modalities_for_markdown(block: PublicModalitiesBlock | None) -> str:
    """One-line human summary for cert markdown / cards."""

    if block is None:
        return "Modalities: legacy record (capability-gated block not stated)."
    return (
        f"Vision: `{block.vision.status}`"
        f"{'' if block.vision.supported else ' (disabled)'}; "
        f"Audio: `{block.audio.status}`"
        f"{'' if block.audio.supported else ' (disabled)'}."
    )
