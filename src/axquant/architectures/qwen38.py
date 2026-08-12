from __future__ import annotations

import re
from typing import Any

from axquant.architectures.dense_family import classify_dense_tensor, valid_layer_count
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    OptimizationScope,
    SupportTier,
    TensorRole,
)

_QWEN38 = re.compile(r"qwen[._-]?3[._-]?8", re.IGNORECASE)
_QWEN_PRODUCT_VERSION = re.compile(r"qwen[._-]?3[._-]?[0-9]", re.IGNORECASE)
_SIZE_24T = re.compile(r"(?<![0-9])2[._-]?4t(?![0-9])", re.IGNORECASE)
_ACTIVE_95B = re.compile(r"(?<![0-9])a95b(?![0-9])", re.IGNORECASE)
_MODEL_TYPES = frozenset({"qwen3_5_moe_text", "qwen3_5_moe"})


def _identity_reference(model_reference: str, config: dict[str, Any]) -> str:
    if _QWEN_PRODUCT_VERSION.search(model_reference):
        return model_reference
    configured = config.get("_name_or_path")
    return str(configured) if isinstance(configured, str) else model_reference


def _has_super_size_hint(reference: str) -> bool:
    return bool(_SIZE_24T.search(reference) or _ACTIVE_95B.search(reference))


def _is_catalog_identity(reference: str) -> bool:
    return bool(
        _QWEN38.search(reference) and _SIZE_24T.search(reference) and _ACTIVE_95B.search(reference)
    )


class Qwen38Adapter:
    adapter_id = "qwen38-moe-v1"
    product_family = "qwen3.8"
    declared_tier = SupportTier.CONVERTIBLE

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        if config.get("model_type") not in _MODEL_TYPES:
            return False
        reference = _identity_reference(model_reference, config)
        return bool(_QWEN38.search(reference) and _has_super_size_hint(reference))

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        reference = _identity_reference(model_reference, config)
        text = config.get("text_config")
        scope = text if isinstance(text, dict) else config
        layer_count = valid_layer_count(scope.get("num_hidden_layers"))
        supported = config.get("model_type") in _MODEL_TYPES and _is_catalog_identity(reference)
        notes = [
            "Development evidence only for the Qwen 3.8 text MoE path.",
            "AX Engine layer-stack expert stream required; resident loading is not supported.",
            "No AX Engine certification track exists for this family yet.",
        ]
        if not supported:
            notes.append(
                "Only the catalog Qwen3.8-2.4T-A95B identity is convertible; other sizes "
                "remain inventory-only."
            )
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
            text_layer_count=layer_count,
            mtp_declared=bool(scope.get("mtp_num_hidden_layers")),
            vision_present=False,
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        value = name.lower()
        if "shared_expert_gate" in value:
            return TensorRole.ROUTER
        if "shared_expert" in value:
            return TensorRole.MLP
        return classify_dense_tensor(name, source_file)
