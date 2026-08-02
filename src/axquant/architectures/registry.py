from __future__ import annotations

from typing import Any

from axquant import __version__
from axquant.architectures.dense_family import DENSE_FAMILY_SPECS, DenseFamilyAdapter
from axquant.architectures.nemotron3 import Nemotron3Adapter
from axquant.architectures.qwen36 import Qwen36Adapter
from axquant.architectures.types import ArchitectureAdapter
from axquant.errors import ArtifactError
from axquant.schema import SupportMatrix, SupportMatrixEntry, SupportTier
from axquant.support_policy import policy_for_adapter

_ADAPTERS: tuple[ArchitectureAdapter, ...] = (
    Qwen36Adapter(),
    Nemotron3Adapter(),
    *(DenseFamilyAdapter(spec) for spec in DENSE_FAMILY_SPECS),
)


def declared_tier_for(adapter_id: str) -> SupportTier | None:
    """The current policy tier for a registered adapter id (AXQ-017).

    Tier is evidence-backed permission and lives in code, so artifacts written
    before a promotion (or before the tier field existed) resolve their tier
    from the current registry rather than from recorded history.
    """
    for adapter in _ADAPTERS:
        if adapter.adapter_id == adapter_id:
            return adapter.declared_tier
    return None


def _adapter_notes(adapter: ArchitectureAdapter) -> list[str]:
    notes: list[str] = []
    policy = policy_for_adapter(adapter.adapter_id)
    if policy is not None:
        notes.append(f"policy: {policy.summary}")
        notes.extend(f"do: {item}" for item in policy.do[:2])
        notes.extend(f"do-not: {item}" for item in policy.do_not[:2])
    if isinstance(adapter, DenseFamilyAdapter):
        notes.extend(adapter.spec.notes)
    elif isinstance(adapter, Nemotron3Adapter):
        notes.append("Thin convert scope: Nano-30B-A3B only; Super/Ultra are inspect-only.")
    elif isinstance(adapter, Qwen36Adapter):
        notes.append("Primary cert track for AX Engine + MTP.")
    return notes


def support_matrix() -> SupportMatrix:
    """The registry's declared family support matrix (AXQ-017) + investment policy."""
    entries: list[SupportMatrixEntry] = []
    for adapter in _ADAPTERS:
        policy = policy_for_adapter(adapter.adapter_id)
        entries.append(
            SupportMatrixEntry(
                adapter_id=adapter.adapter_id,
                product_family=adapter.product_family,
                support_tier=adapter.declared_tier,
                investment_posture=(
                    policy.investment_posture.value if policy is not None else "secondary"
                ),
                priority=policy.priority if policy is not None else 50,
                cert_track=policy.cert_track if policy is not None else False,
                notes=_adapter_notes(adapter),
            )
        )
    entries.sort(key=lambda entry: (entry.priority, entry.adapter_id))
    return SupportMatrix(axquant_version=__version__, entries=entries)


def adapter_for(
    model_reference: str,
    config: dict[str, Any],
) -> ArchitectureAdapter | None:
    matched = [adapter for adapter in _ADAPTERS if adapter.matches(model_reference, config)]
    if len(matched) > 1:
        contenders = ", ".join(adapter.adapter_id for adapter in matched)
        raise ArtifactError(f"ambiguous architecture adapters for {model_reference}: {contenders}")
    return matched[0] if matched else None
