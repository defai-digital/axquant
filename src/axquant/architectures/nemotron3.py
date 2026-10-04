"""Nemotron 3 hybrid MoE adapter (Nano / Lightning / Super / Ultra catalog).

Public generative Nemotron 3 checkpoints use ``model_type: nemotron_h`` with
routed experts under ``backbone.layers.*.mixer.experts.*``. MLX-LM loads them
via ``mlx_lm.models.nemotron_h`` and fuses experts into ``switch_mlp.fc1/fc2``.
"""

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

_NEMOTRON3 = re.compile(r"nemotron[._-]?3", re.IGNORECASE)
# Conversion scope is limited to exact NVIDIA catalog checkpoints below.
_NANO_MOE = re.compile(
    r"(nano[._-]?30b[._-]?a3b|(?<![0-9])30b[._-]?a3b(?![0-9]))",
    re.IGNORECASE,
)
_SUPER_OR_ULTRA = re.compile(
    r"(super[._-]?120b[._-]?a12b|ultra[._-]?550b[._-]?a55b|"
    r"(?<![0-9])120b[._-]?a12b(?![0-9])|(?<![0-9])550b[._-]?a55b(?![0-9]))",
    re.IGNORECASE,
)
_LIGHTNING = re.compile(r"nemotron[._-]?3[._-]?5[._-]?lightning[._-]?30b[._-]?a3b", re.I)
_SUPER = re.compile(r"nemotron[._-]?3[._-]?super[._-]?120b[._-]?a12b", re.I)
_MOE_KEYS = (
    "n_routed_experts",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "num_local_experts",
)
_NANO_30B_A3B_SIGNATURE = {
    "num_hidden_layers": 52,
    "hidden_size": 2688,
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "moe_intermediate_size": 1856,
}
_LIGHTNING_30B_A3B_SIGNATURE = {
    "num_hidden_layers": 52,
    "hidden_size": 2688,
    "n_routed_experts": 128,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "moe_intermediate_size": 1856,
    "moe_shared_expert_intermediate_size": 3712,
    "num_nextn_predict_layers": 1,
}
_SUPER_120B_A12B_SIGNATURE = {
    "num_hidden_layers": 88,
    "hidden_size": 4096,
    "n_routed_experts": 512,
    "num_experts_per_tok": 22,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2688,
    "moe_latent_size": 1024,
    "num_nextn_predict_layers": 1,
}
_NEMOTRON_EXTRA = (
    ("shared_experts", TensorRole.MLP),  # fires every token — denser than routed experts
    ("backbone.embeddings", TensorRole.EMBEDDING),
    ("mixer.norm", TensorRole.NORM),
    ("mixer.in_proj", TensorRole.ATTENTION),
    ("mixer.out_proj", TensorRole.ATTENTION),
    ("mixer.conv1d", TensorRole.ATTENTION),
    ("mixer.A_log", TensorRole.ATTENTION),
    ("mixer.D", TensorRole.ATTENTION),
    ("mixer.dt_bias", TensorRole.ATTENTION),
    ("mixer.norm", TensorRole.NORM),
    (".mixer.gate", TensorRole.ROUTER),
    ("router", TensorRole.ROUTER),
    ("e_score_correction_bias", TensorRole.ROUTER),
)


def _matches_signature(scope: dict[str, Any], signature: dict[str, int]) -> bool:
    return all(
        type(scope.get(key)) is int and scope[key] == value for key, value in signature.items()
    )


class Nemotron3Adapter:
    adapter_id = "nemotron3-v1"
    product_family = "nemotron3"
    # Exact Nano, 3.5 Lightning, and 3 Super catalog checkpoints are convertible.
    declared_tier = SupportTier.CONVERTIBLE

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        if config.get("model_type") not in ("nemotron_h", "nemotron3", "nemotron"):
            return False
        references = [model_reference, str(config.get("_name_or_path", ""))]
        return any(_NEMOTRON3.search(reference) for reference in references)

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        scope = config
        text = config.get("text_config")
        if isinstance(text, dict):
            scope = text
        moe = any(scope.get(key) for key in _MOE_KEYS) or any(config.get(key) for key in _MOE_KEYS)
        layers = valid_layer_count(scope.get("num_hidden_layers"))
        if layers is None:
            layers = valid_layer_count(config.get("num_hidden_layers"))
        references = " ".join(
            [
                model_reference,
                str(config.get("_name_or_path", "")),
                str(config.get("architectures", "")),
            ]
        )
        is_lightning = bool(_LIGHTNING.search(references))
        is_super = bool(_SUPER.search(references))
        is_nano = (
            bool(_NANO_MOE.search(references))
            and not bool(_SUPER_OR_ULTRA.search(references))
            and not is_lightning
        )
        signature_is_nano = _matches_signature(scope, _NANO_30B_A3B_SIGNATURE)
        signature_is_lightning = _matches_signature(scope, _LIGHTNING_30B_A3B_SIGNATURE)
        signature_is_super = _matches_signature(scope, _SUPER_120B_A12B_SIGNATURE)
        # Each catalog marker requires its complete source signature; a stale
        # _name_or_path or conflicting model id cannot promote a checkpoint.
        supported = bool(
            moe
            and layers is not None
            and (
                (is_nano and signature_is_nano)
                or (is_lightning and signature_is_lightning)
                or (is_super and signature_is_super)
            )
        )
        notes = [
            "Nemotron 3 generative catalog is hybrid MoE (nemotron_h).",
            "Routed experts fuse to switch_mlp.fc1/fc2 under MLX-LM sanitize.",
            "Shared experts are treated as dense MLP (higher protection than routed experts).",
            "MXFP8 packing uses MLX-LM SwitchLinear group_size=32; format-level evidence only.",
            "MTP tensors are preserved in a tagged sidecar; runtime support is external.",
        ]
        if supported:
            notes.append(
                "Catalog checkpoint is convertible (development evidence until certified)."
            )
        elif _SUPER_OR_ULTRA.search(references):
            notes.append(
                "Only the exact Nemotron 3 Super-120B-A12B catalog signature is convertible; "
                "other Super and all Ultra references remain inspect-only."
            )
        else:
            notes.append(
                "This Nemotron 3 checkpoint is inventory-only until it matches the "
                "catalog convert targets (Nano-30B-A3B, 3.5 Lightning-30B-A3B, "
                "or 3 Super-120B-A12B)."
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
            dense=not moe,
            text_layer_count=layers,
            mtp_declared=any(
                bool(scope.get(key))
                for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers")
            ),
            vision_present=isinstance(config.get("vision_config"), dict),
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        value = name.lower()
        if value.startswith("mtp."):
            if any(token in value for token in ("norm", "layernorm", "conv1d")):
                return TensorRole.MTP_BLOCK
            return TensorRole.MTP_PROJECTION
        # Experts before generic mixer / mlp rules.
        if ".experts." in value or value.endswith(".experts"):
            return TensorRole.EXPERT
        if "shared_experts" in value:
            return TensorRole.MLP
        if "gate_proj" not in value and (
            "mixer.gate" in value or "router" in value or "e_score_correction" in value
        ):
            return TensorRole.ROUTER
        return classify_dense_tensor(name, source_file, _NEMOTRON_EXTRA)
