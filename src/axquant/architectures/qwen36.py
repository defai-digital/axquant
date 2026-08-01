from __future__ import annotations

import re
from typing import Any

from axquant.architectures.dense_family import classify_dense_tensor
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    OptimizationScope,
    SupportTier,
    TensorRole,
)

_QWEN36 = re.compile(r"qwen[._-]?3[._-]?6", re.IGNORECASE)


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
        dense = not any(
            key in text_config
            for key in (
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
            )
        )
        text_layers = text_config.get("num_hidden_layers")
        layer_count = int(text_layers) if isinstance(text_layers, int) else None
        reference_is_27b = "27b" in model_reference.lower()
        signature_is_27b = (
            layer_count == 64
            and text_config.get("hidden_size") == 5120
            and text_config.get("intermediate_size") == 17408
        )
        # The official catalog's MoE size. Expert tensors quantize as fused
        # MLX-LM switch modules with a uniform per-group precision; the
        # router keeps its 8-bit floor. Conversion evidence for this path is
        # development-only until the family certifies.
        reference_is_35b_a3b = "35b-a3b" in model_reference.lower()
        supported = (dense and (reference_is_27b or signature_is_27b)) or (
            not dense and reference_is_35b_a3b
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
            config_model_type="qwen3_5",
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
