from __future__ import annotations

import re
from typing import Any

from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    OptimizationScope,
    TensorRole,
)

_QWEN36 = re.compile(r"qwen[._-]?3[._-]?6", re.IGNORECASE)
_MTP = re.compile(r"(^|[./_-])(mtp|multi[_-]?token)([./_-]|$)")


class Qwen36Adapter:
    adapter_id = "qwen36-v1"

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        if config.get("model_type") != "qwen3_5":
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
        supported = dense and (reference_is_27b or signature_is_27b)
        vision_present = isinstance(config.get("vision_config"), dict)
        notes = [
            "AXQuant optimizes the Qwen 3.6 language path only.",
            "Vision tensors are preserved at BF16 and VLM quality is not claimed.",
        ]
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
        value = f"{source_file}/{name}".lower()
        if _MTP.search(value):
            if any(token in value for token in ("output_head", "lm_head", "vocab_head")):
                return TensorRole.MTP_OUTPUT
            if any(token in value for token in ("proj", "projection", ".fc.")):
                return TensorRole.MTP_PROJECTION
            return TensorRole.MTP_BLOCK
        if any(
            token in value
            for token in (
                "/visual.",
                ".visual.",
                "vision_tower",
                "vision_model",
                "patch_embed",
                "merger.",
            )
        ):
            return TensorRole.VISION
        if "norm" in value:
            return TensorRole.NORM
        if any(token in value for token in ("lm_head", "output.weight", "output_layer")):
            return TensorRole.LM_HEAD
        if any(token in value for token in ("embed_tokens", "token_embedding")):
            return TensorRole.EMBEDDING
        if "router" in value:
            return TensorRole.ROUTER
        if "expert" in value:
            return TensorRole.EXPERT
        if any(
            token in value
            for token in (
                "self_attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "linear_attn",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "conv1d",
                "a_log",
                "dt_bias",
            )
        ):
            return TensorRole.ATTENTION
        if any(
            token in value
            for token in (
                "mlp",
                "feed_forward",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        ):
            return TensorRole.MLP
        return None
