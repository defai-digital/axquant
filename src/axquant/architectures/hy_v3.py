"""Tencent Hy-MT2 / hy_v3 development conversion adapter."""

from __future__ import annotations

from typing import Any

from axquant.architectures.dense_family import classify_dense_tensor, valid_layer_count
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    OptimizationScope,
    SupportTier,
    TensorRole,
)

_SIGNATURE = {
    "num_hidden_layers": 48,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "hidden_size": 2048,
}
_NOTES = (
    "Hy-MT2-30B-A3B is development convert only; no translation-quality claim.",
    "No MTP tensors are present in this checkpoint.",
    "The first layer is dense; subsequent layers have routed and shared experts.",
    "expert_bias stays with the router; shared_mlp projections remain dense MLP.",
)


class HyV3Adapter:
    adapter_id = "hy-v3-v1"
    product_family = "hy-v3"
    declared_tier = SupportTier.CONVERTIBLE
    notes = _NOTES

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        return config.get("model_type") == "hy_v3"

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        supported = self.matches(model_reference, config) and all(
            type(config.get(key)) is int and config[key] == value
            for key, value in _SIGNATURE.items()
        )
        notes = list(self.notes)
        if not supported:
            notes.append("Non-matching Hy-MT2 signatures remain inventory-only / inspect-only.")
        return ArchitectureProfile(
            adapter_id=self.adapter_id,
            product_family=self.product_family,
            config_model_type=str(config.get("model_type")),
            support_level=(
                ArchitectureSupportLevel.SUPPORTED
                if supported
                else ArchitectureSupportLevel.INVENTORY_ONLY
            ),
            support_tier=(SupportTier.CONVERTIBLE if supported else SupportTier.INSPECT_ONLY),
            optimization_scope=(
                OptimizationScope.TEXT_PATH if supported else OptimizationScope.INVENTORY_ONLY
            ),
            dense=False,
            text_layer_count=valid_layer_count(config.get("num_hidden_layers")),
            mtp_declared=False,
            vision_present=False,
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        if name.endswith("expert_bias") or ".expert_bias" in name:
            return TensorRole.ROUTER
        return classify_dense_tensor(name, source_file)
