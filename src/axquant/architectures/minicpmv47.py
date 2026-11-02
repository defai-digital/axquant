"""Narrow development adapter for the official MiniCPM-V 4.7 35B-A3B model."""

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

_REFERENCE_NAME = "minicpm-v-4.7-35b-a3b"
_TEXT_SIGNATURE = {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
}
_VISION_SIGNATURE = {
    "model_type": "minicpmv4_7_vision",
    "num_hidden_layers": 27,
    "hidden_size": 1152,
    "num_attention_heads": 16,
    "patch_size": 14,
}


def _is_requested_reference(model_reference: str) -> bool:
    normalized = model_reference.replace("\\", "/").rstrip("/").casefold()
    return normalized.rsplit("/", 1)[-1] == _REFERENCE_NAME


def _matches_signature(value: object, expected: dict[str, object]) -> bool:
    return isinstance(value, dict) and all(value.get(key) == item for key, item in expected.items())


class MiniCPMV47Adapter:
    """Convert only the exact 35B-A3B V4.7 multimodal checkpoint signature."""

    adapter_id = "minicpm-v47-v1"
    product_family = "minicpm-v-4.7"
    declared_tier = SupportTier.CONVERTIBLE

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        return (
            _is_requested_reference(model_reference) and config.get("model_type") == "minicpmv4_7"
        )

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        text_config = config.get("text_config")
        vision_config = config.get("vision_config")
        signature_supported = (
            _is_requested_reference(model_reference)
            and config.get("model_type") == "minicpmv4_7"
            and config.get("architectures") == ["MiniCPMV4_7ForConditionalGeneration"]
            and _matches_signature(text_config, _TEXT_SIGNATURE)
            and _matches_signature(vision_config, _VISION_SIGNATURE)
        )
        layer_count = (
            valid_layer_count(text_config.get("num_hidden_layers"))
            if isinstance(text_config, dict)
            else None
        )
        notes = [
            "Exact MiniCPM-V 4.7 35B-A3B development conversion through MLX-VLM.",
            "Vision tower and merger stay at source precision; "
            "image/video quality is not certified.",
            "Hybrid linear-attention tensors stay at source precision to match the "
            "MLX-VLM model policy.",
            "The source config declares MTP, but conversion must verify that indexed MTP weights "
            "exist.",
            "Do not infer support for MiniCPM-V 4.7 1B or other MiniCPM checkpoints.",
        ]
        if not signature_supported:
            notes.append(
                "This MiniCPM-V reference is inventory-only until its exact signature is validated."
            )
        return ArchitectureProfile(
            adapter_id=self.adapter_id,
            product_family=self.product_family,
            config_model_type=(
                config.get("model_type") if isinstance(config.get("model_type"), str) else None
            ),
            support_level=(
                ArchitectureSupportLevel.SUPPORTED
                if signature_supported
                else ArchitectureSupportLevel.INVENTORY_ONLY
            ),
            support_tier=(
                SupportTier.CONVERTIBLE if signature_supported else SupportTier.INSPECT_ONLY
            ),
            optimization_scope=(
                OptimizationScope.TEXT_PATH
                if signature_supported
                else OptimizationScope.INVENTORY_ONLY
            ),
            dense=False,
            text_layer_count=layer_count,
            mtp_declared=(
                bool(text_config.get("mtp_num_hidden_layers"))
                if isinstance(text_config, dict)
                else False
            ),
            vision_present=isinstance(vision_config, dict),
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        return classify_dense_tensor(name, source_file)

    def preserve_source_precision(self, name: str, source_file: str) -> bool:
        del source_file
        return ".linear_attn." in name.casefold()
