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


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, dict) else {}


def _signature_is_35b_a3b(text_config: dict[str, Any]) -> bool:
    return all(text_config.get(key) == value for key, value in _MOE_35B_A3B_SIGNATURE.items())


def _references_include_qwen36(model_reference: str, config: dict[str, Any]) -> bool:
    references = [model_reference, str(config.get("_name_or_path", ""))]
    return any(_QWEN36.search(reference) for reference in references)


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
        return _references_include_qwen36(model_reference, config)

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        text_config = _text_config(config)
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
        signature_is_35b_a3b = _signature_is_35b_a3b(text_config)
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


class Qwen35MoeAdapter:
    """Development convert for Qwen3.5-class 35B-A3B MoE and fine-tunes (e.g. Ornith).

    Official Qwen 3.6 catalog MoE stays on ``qwen36-v1``. This adapter covers the
    same fused-expert MLX layout when the checkpoint is *not* named as Qwen 3.6
    but still matches the validated 35B-A3B signature (Ornith-1.0-35B,
    Qwen3.5-35B-A3B, and compatible fine-tunes). Artifacts are development
    evidence only — not the Qwen 3.6 certification track.
    """

    adapter_id = "qwen35-moe-v1"
    product_family = "qwen3.5-moe"
    declared_tier = SupportTier.CONVERTIBLE

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        if config.get("model_type") != "qwen3_5_moe":
            return False
        # Leave official Qwen 3.6 product names on the primary adapter.
        return not _references_include_qwen36(model_reference, config)

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        text_config = _text_config(config)
        dense = not any(text_config.get(key) for key in _MOE_CONFIG_KEYS)
        layer_count = valid_layer_count(text_config.get("num_hidden_layers"))
        signature_is_35b_a3b = _signature_is_35b_a3b(text_config)
        model_type = config.get("model_type")
        supported = (
            not dense and model_type == "qwen3_5_moe" and signature_is_35b_a3b and layer_count == 40
        )
        vision_present = isinstance(config.get("vision_config"), dict)
        notes = [
            "AXQuant optimizes the Qwen3.5-class 35B-A3B MoE language path only "
            "(development convert; not the Qwen 3.6 certification track).",
            "Vision tensors are preserved at BF16 and VLM quality is not claimed.",
            "Fine-tunes such as Ornith-1.0-35B and Holo3-35B-A3B are eligible when the "
            "text_config matches the 35B-A3B MoE signature.",
        ]
        if supported:
            notes.append(
                "MoE experts quantize as fused switch modules with uniform per-group "
                "precision; label packs as development evidence until certified."
            )
        else:
            notes.append(
                "This qwen3_5_moe checkpoint is inventory-only until it matches the "
                "validated 35B-A3B MoE signature."
            )
        return ArchitectureProfile(
            adapter_id=self.adapter_id,
            product_family=self.product_family,
            config_model_type=(model_type if isinstance(model_type, str) else None),
            support_level=(
                ArchitectureSupportLevel.SUPPORTED
                if supported
                else ArchitectureSupportLevel.INVENTORY_ONLY
            ),
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
