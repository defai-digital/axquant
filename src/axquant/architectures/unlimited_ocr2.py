"""Unlimited-OCR-2 development storage exports; no inference execution claim."""

from __future__ import annotations

from typing import Any

from axquant.architectures.dense_family import DenseFamilyAdapter, DenseFamilySpec
from axquant.schema import ArchitectureProfile, SupportTier, TensorRole


class UnlimitedOCR2Adapter(DenseFamilyAdapter):
    """Preserve SR-SWA routing state while exporting the language matrices."""

    def __init__(self) -> None:
        super().__init__(
            DenseFamilySpec(
                adapter_id="unlimited-ocr2-v1",
                product_family="unlimited-ocr2",
                model_types=("unlimited-ocr",),
                reference_pattern=r"unlimited[._-]?ocr[._-]?2",
                support_tier=SupportTier.CONVERTIBLE,
                text_config_key="language_config",
                allow_moe=True,
                extra_role_patterns=(
                    ("dash_router", TensorRole.ROUTER),
                    (".indexer.", TensorRole.ROUTER),
                    ("sam_model", TensorRole.VISION),
                    ("vision_model", TensorRole.VISION),
                    ("projector", TensorRole.VISION),
                    ("image_newline", TensorRole.VISION),
                ),
                notes=(
                    "CUDA NVFP4 and source-layout MLX MXFP4/MXFP8 storage exports only.",
                    "SR-SWA router/indexer, vision, norms, embeddings and head "
                    "retain source precision.",
                    "No AX Engine, vLLM, SGLang or OCR quality claim without exact-pack evidence.",
                ),
            )
        )

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        sparse = config.get("sparse_attention_config")
        return config.get("model_type") == "unlimited-ocr" and (
            super().matches(model_reference, config) or isinstance(sparse, dict)
        )

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        profile = super().profile(model_reference, config)
        # Storage backends retain individual source experts; the generic dense
        # adapter's fused-runtime note does not apply to this family.
        profile = profile.model_copy(update={"notes": list(self.spec.notes)})
        signature = {
            "hidden_size": 1280,
            "num_hidden_layers": 12,
            "num_attention_heads": 10,
            "num_key_value_heads": 10,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "vocab_size": 129280,
        }
        sparse = config.get("sparse_attention_config")
        supported = (
            config.get("architectures") == ["UnlimitedOCRForCausalLM"]
            and all(
                type(config.get(key)) is int and config[key] == value
                for key, value in signature.items()
            )
            and isinstance(sparse, dict)
            and sparse.get("use_sparse_attention") is True
            and sparse.get("dash_router") == "entmax_topk_perlayer"
        )
        if not supported:
            from axquant.schema import ArchitectureSupportLevel, OptimizationScope

            return profile.model_copy(
                update={
                    "support_tier": SupportTier.INSPECT_ONLY,
                    "support_level": ArchitectureSupportLevel.INVENTORY_ONLY,
                    "optimization_scope": OptimizationScope.INVENTORY_ONLY,
                    "notes": [
                        *self.spec.notes,
                        "Source signature is not supported; inspection only.",
                    ],
                }
            )
        return profile
