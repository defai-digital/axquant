"""Nemotron 3 hybrid MoE adapter (Nano / Super / Ultra catalog).

Public generative Nemotron 3 checkpoints use ``model_type: nemotron_h`` with
routed experts under ``backbone.layers.*.mixer.experts.*``. MLX-LM loads them
via ``mlx_lm.models.nemotron_h`` and fuses experts into ``switch_mlp.fc1/fc2``.
"""

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

_NEMOTRON3 = re.compile(r"nemotron[._-]?3", re.IGNORECASE)
# Thin-support best practice: only Nano-30B-A3B is a convert product target.
# Super/Ultra remain inspect-only (OptiQ owns SSD-stream / huge-MoE product story).
_NANO_MOE = re.compile(
    r"(nano[._-]?30b[._-]?a3b|(?<![0-9])30b[._-]?a3b(?![0-9]))",
    re.IGNORECASE,
)
_SUPER_OR_ULTRA = re.compile(
    r"(super[._-]?120b[._-]?a12b|ultra[._-]?550b[._-]?a55b|"
    r"(?<![0-9])120b[._-]?a12b(?![0-9])|(?<![0-9])550b[._-]?a55b(?![0-9]))",
    re.IGNORECASE,
)
_MOE_KEYS = (
    "n_routed_experts",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "num_local_experts",
)
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


class Nemotron3Adapter:
    adapter_id = "nemotron3-v1"
    product_family = "nemotron3"
    # Declared family tier is convertible because Nano is in scope; Super/Ultra
    # checkpoints still profile as inspect-only (thin-support best practice).
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
        layer_count = scope.get("num_hidden_layers")
        if not isinstance(layer_count, int):
            layer_count = config.get("num_hidden_layers")
        layers = int(layer_count) if isinstance(layer_count, int) else None
        references = " ".join(
            [
                model_reference,
                str(config.get("_name_or_path", "")),
                str(config.get("architectures", "")),
            ]
        )
        is_nano = bool(_NANO_MOE.search(references))
        is_super_ultra = bool(_SUPER_OR_ULTRA.search(references))
        # Thin support: only Nano-30B-A3B MoE converts. Super/Ultra are inventory.
        supported = bool(moe and is_nano and layers is not None)
        notes = [
            "Nemotron 3 generative catalog is hybrid MoE (nemotron_h).",
            "Routed experts fuse to switch_mlp.fc1/fc2 under MLX-LM sanitize.",
            "Shared experts are treated as dense MLP (higher protection than routed experts).",
            "Investment posture: thin — Nano convert only; not OptiQ Super/stream parity.",
        ]
        if supported:
            notes.append("Nano-30B-A3B is convertible (development evidence until certified).")
        elif is_super_ultra:
            notes.append(
                "Super/Ultra are inspect-only under thin-support policy: SSD expert "
                "streaming and huge-MoE product features are deferred (not AX Engine path)."
            )
        else:
            notes.append(
                "This Nemotron 3 checkpoint is inventory-only until it matches the "
                "thin-support convert target (Nano-30B-A3B MoE)."
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
            mtp_declared=False,
            vision_present=isinstance(config.get("vision_config"), dict),
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        value = f"{source_file}/{name}".lower()
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
