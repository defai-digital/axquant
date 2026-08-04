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

_QWEN36 = re.compile(r"qwen[._-]?3[._-]?6", re.IGNORECASE)
_QWEN36_27B = re.compile(
    r"qwen[._-]?3[._-]?6[._-]?27b(?=$|[._/-])",
    re.IGNORECASE,
)
_QWEN36_35B_A3B = re.compile(
    r"qwen[._-]?3[._-]?6[._-]?35b[._-]?a3b(?=$|[._/-])",
    re.IGNORECASE,
)
_QWEN36_CATALOG_SIZE = re.compile(
    r"qwen[._-]?3[._-]?6[._-]?\d+b(?:[._-]?a\d+b)?(?=$|[._/-])",
    re.IGNORECASE,
)
_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "num_local_experts",
    "n_routed_experts",
    "enable_moe_block",
)
_DENSE_27B_SIGNATURE = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 17408,
}
_MOE_35B_A3B_SIGNATURE = {
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
}


class Qwen36Adapter:
    adapter_id = "qwen36-v1"
    product_family = "qwen3.6"
    # Convertible until the formal release audit promotes the family (AXQ-017).
    declared_tier = SupportTier.CONVERTIBLE

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        # The official catalog ships the dense sizes as `qwen3_5` and the MoE
        # size as `qwen3_5_moe`; both carry the Qwen 3.6 product reference.
        if config.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
            return False
        references = [model_reference, str(config.get("_name_or_path", ""))]
        return any(_QWEN36.search(reference) for reference in references)

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        text = config.get("text_config")
        text_config = text if isinstance(text, dict) else {}
        dense = not any(text_config.get(key) for key in _MOE_CONFIG_KEYS)
        layer_count = valid_layer_count(text_config.get("num_hidden_layers"))
        # An explicit Qwen 3.6 model reference is authoritative.  Fall back to
        # `_name_or_path` only for anonymous local snapshot paths; otherwise a
        # stale config identity could make a differently named checkpoint
        # inherit catalog conversion scope.
        references = (
            (model_reference,)
            if _QWEN36.search(model_reference)
            else (str(config.get("_name_or_path", "")),)
        )
        reference_is_27b = any(_QWEN36_27B.search(reference) for reference in references)
        signature_is_27b = all(
            text_config.get(key) == value for key, value in _DENSE_27B_SIGNATURE.items()
        )
        # The official catalog's MoE size. Expert tensors quantize as fused
        # MLX-LM switch modules with a uniform per-group precision; the
        # router keeps its 8-bit floor. Conversion evidence for this path is
        # development-only until the family certifies.
        reference_is_35b_a3b = any(_QWEN36_35B_A3B.search(reference) for reference in references)
        reference_declares_catalog_size = any(
            _QWEN36_CATALOG_SIZE.search(reference) for reference in references
        )
        signature_is_35b_a3b = all(
            text_config.get(key) == value for key, value in _MOE_35B_A3B_SIGNATURE.items()
        )
        model_type = config.get("model_type")
        supported = (
            dense
            and model_type == "qwen3_5"
            and (reference_is_27b or not reference_declares_catalog_size)
            and signature_is_27b
        ) or (
            not dense
            and model_type == "qwen3_5_moe"
            and (reference_is_35b_a3b or not reference_declares_catalog_size)
            and signature_is_35b_a3b
        )
        vision_present = isinstance(config.get("vision_config"), dict)
        notes = [
            "AXQuant optimizes the Qwen 3.6 language path only.",
            "Vision tensors are preserved at BF16 and VLM quality is not claimed.",
        ]
        if supported and not dense:
            notes.append(
                "MoE experts quantize as fused switch modules with uniform per-group "
                "precision; artifacts are development evidence until certified."
            )
        if not supported:
            notes.append("This Qwen 3.6 checkpoint is inventory-only until explicitly validated.")
        return ArchitectureProfile(
            adapter_id=self.adapter_id,
            product_family="qwen3.6",
            config_model_type=(model_type if isinstance(model_type, str) else None),
            support_level=(
                ArchitectureSupportLevel.SUPPORTED
                if supported
                else ArchitectureSupportLevel.INVENTORY_ONLY
            ),
            # Convertible until a formal release audit promotes the family to
            # certified in code (AXQ-017).
            support_tier=(SupportTier.CONVERTIBLE if supported else SupportTier.INSPECT_ONLY),
            optimization_scope=(
                OptimizationScope.TEXT_PATH if supported else OptimizationScope.INVENTORY_ONLY
            ),
            dense=dense,
            text_layer_count=layer_count,
            mtp_declared=bool(text_config.get("mtp_num_hidden_layers")),
            vision_present=vision_present,
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        return classify_dense_tensor(name, source_file)
