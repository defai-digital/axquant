from __future__ import annotations

from typing import Any, Protocol

from axquant.schema import ArchitectureProfile, SupportTier, TensorRole


class ArchitectureAdapter(Protocol):
    adapter_id: str
    product_family: str
    # The best tier this adapter can report for an eligible checkpoint; a
    # concrete profile may still downgrade to inspect-only (fail closed).
    declared_tier: SupportTier

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        """Return whether the source belongs to this architecture."""

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        """Describe the supported optimization scope and protection policy."""

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        """Return the tensor role recognized by this architecture."""
