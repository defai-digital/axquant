from __future__ import annotations

from typing import Any

from axquant import __version__
from axquant.architectures.dense_family import DENSE_FAMILY_SPECS, DenseFamilyAdapter
from axquant.architectures.qwen36 import Qwen36Adapter
from axquant.architectures.types import ArchitectureAdapter
from axquant.errors import ArtifactError
from axquant.schema import SupportMatrix, SupportMatrixEntry, SupportTier

_ADAPTERS: tuple[ArchitectureAdapter, ...] = (
    Qwen36Adapter(),
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
    if isinstance(adapter, DenseFamilyAdapter):
        return list(adapter.spec.notes)
    return []


def support_matrix() -> SupportMatrix:
    """The registry's declared family support matrix (AXQ-017)."""
    entries = [
        SupportMatrixEntry(
            adapter_id=adapter.adapter_id,
            product_family=adapter.product_family,
            support_tier=adapter.declared_tier,
            notes=_adapter_notes(adapter),
        )
        for adapter in _ADAPTERS
    ]
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
