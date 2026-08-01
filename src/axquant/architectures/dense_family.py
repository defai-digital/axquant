"""Declarative dense-transformer family adapters (AXQ-018).

Most dense families differ only in config ``model_type`` values, reference
naming, and the location of the text-layer count. ``DenseFamilySpec`` captures
those differences as data and ``DenseFamilyAdapter`` implements the adapter
protocol once, so tensor-role classification — the protection-critical logic —
is reviewed in exactly one place. Classification stays fail-closed: a tensor no
rule matches remains unclassified and blocks conversion downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    OptimizationScope,
    SupportTier,
    TensorRole,
)

_MTP = re.compile(r"(^|[./_-])(mtp|multi[_-]?token)([./_-]|$)")
_MOE_CONFIG_KEYS = ("num_experts", "num_experts_per_tok", "moe_intermediate_size")

_MTP_OUTPUT_TOKENS = ("output_head", "lm_head", "vocab_head")
_MTP_PROJECTION_TOKENS = ("proj", "projection", ".fc.")
_VISION_TOKENS = (
    "/visual.",
    ".visual.",
    "vision_tower",
    "vision_model",
    "patch_embed",
    "merger.",
)
_LM_HEAD_TOKENS = ("lm_head", "output.weight", "output_layer")
_EMBEDDING_TOKENS = ("embed_tokens", "token_embedding")
_ATTENTION_TOKENS = (
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
_MLP_TOKENS = ("mlp", "feed_forward", "gate_proj", "up_proj", "down_proj")


def classify_dense_tensor(
    name: str,
    source_file: str,
    extra_patterns: tuple[tuple[str, TensorRole], ...] = (),
) -> TensorRole | None:
    """Shared fail-closed role classification for dense transformer tensors."""
    value = f"{source_file}/{name}".lower()
    for token, role in extra_patterns:
        if token in value:
            return role
    if _MTP.search(value):
        if any(token in value for token in _MTP_OUTPUT_TOKENS):
            return TensorRole.MTP_OUTPUT
        if any(token in value for token in _MTP_PROJECTION_TOKENS):
            return TensorRole.MTP_PROJECTION
        return TensorRole.MTP_BLOCK
    if any(token in value for token in _VISION_TOKENS):
        return TensorRole.VISION
    if "norm" in value:
        return TensorRole.NORM
    if any(token in value for token in _LM_HEAD_TOKENS):
        return TensorRole.LM_HEAD
    if any(token in value for token in _EMBEDDING_TOKENS):
        return TensorRole.EMBEDDING
    if "router" in value:
        return TensorRole.ROUTER
    if "expert" in value:
        return TensorRole.EXPERT
    if any(token in value for token in _ATTENTION_TOKENS):
        return TensorRole.ATTENTION
    if any(token in value for token in _MLP_TOKENS):
        return TensorRole.MLP
    return None


@dataclass(frozen=True)
class DenseFamilySpec:
    adapter_id: str
    product_family: str
    model_types: tuple[str, ...]
    reference_pattern: str
    support_tier: SupportTier
    layer_count_keys: tuple[str, ...] = ("num_hidden_layers",)
    text_config_key: str | None = None
    exclude_reference_pattern: str | None = None
    extra_role_patterns: tuple[tuple[str, TensorRole], ...] = ()
    notes: tuple[str, ...] = ()


class DenseFamilyAdapter:
    """`ArchitectureAdapter` implementation driven by a `DenseFamilySpec`."""

    def __init__(self, spec: DenseFamilySpec) -> None:
        self.spec = spec
        self.adapter_id = spec.adapter_id
        self.product_family = spec.product_family
        self.declared_tier = spec.support_tier
        self._reference = re.compile(spec.reference_pattern, re.IGNORECASE)
        self._exclude = (
            re.compile(spec.exclude_reference_pattern, re.IGNORECASE)
            if spec.exclude_reference_pattern
            else None
        )

    def _references(self, model_reference: str, config: dict[str, Any]) -> list[str]:
        return [model_reference, str(config.get("_name_or_path", ""))]

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool:
        if config.get("model_type") not in self.spec.model_types:
            return False
        references = self._references(model_reference, config)
        if self._exclude is not None and any(
            self._exclude.search(reference) for reference in references
        ):
            return False
        return any(self._reference.search(reference) for reference in references)

    def _text_scope(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.spec.text_config_key is not None:
            nested = config.get(self.spec.text_config_key)
            if isinstance(nested, dict):
                return nested
        return config

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile:
        scope = self._text_scope(config)
        dense = not any(key in scope for key in _MOE_CONFIG_KEYS)
        layer_count: int | None = None
        for key in self.spec.layer_count_keys:
            value = scope.get(key)
            if isinstance(value, int):
                layer_count = value
                break
        # Fail closed: a non-dense checkpoint or a missing layer count downgrades
        # the family's default tier to inventory-only regardless of the spec.
        eligible = dense and layer_count is not None
        tier = self.spec.support_tier if eligible else SupportTier.INSPECT_ONLY
        supported = tier is not SupportTier.INSPECT_ONLY
        notes = list(self.spec.notes)
        if not supported:
            notes.append(
                f"The {self.spec.product_family} family is inventory-only until its "
                "tier-promotion evidence exists (AXQ-017)."
            )
        return ArchitectureProfile(
            adapter_id=self.adapter_id,
            product_family=self.spec.product_family,
            config_model_type=str(config.get("model_type")),
            support_level=(
                ArchitectureSupportLevel.SUPPORTED
                if supported
                else ArchitectureSupportLevel.INVENTORY_ONLY
            ),
            support_tier=tier,
            optimization_scope=(
                OptimizationScope.TEXT_PATH if supported else OptimizationScope.INVENTORY_ONLY
            ),
            dense=dense,
            text_layer_count=layer_count,
            mtp_declared=bool(scope.get("mtp_num_hidden_layers")),
            vision_present=isinstance(config.get("vision_config"), dict),
            notes=notes,
        )

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None:
        return classify_dense_tensor(name, source_file, self.spec.extra_role_patterns)


_QWEN36_REFERENCE = r"qwen[._-]?3[._-]?6"

DENSE_FAMILY_SPECS: tuple[DenseFamilySpec, ...] = (
    DenseFamilySpec(
        adapter_id="qwen35-dense-v1",
        product_family="qwen3.5",
        model_types=("qwen3_5",),
        reference_pattern=r"qwen[._-]?3[._-]?5",
        exclude_reference_pattern=_QWEN36_REFERENCE,
        # AXQ-017 promotion (2026-08-01): real-checkpoint evidence on
        # Qwen/Qwen3.5-9B@c20223623576 — full 775-tensor classification
        # (integrated MTP + vision protected), a complete one-command
        # development conversion, and passing MLX-LM/AX Engine runtime
        # smokes. Recorded in the expansion implementation plan's E5 log.
        support_tier=SupportTier.CONVERTIBLE,
        text_config_key="text_config",
        notes=("Qwen 3.5 dense checkpoints share the Qwen 3.6 tensor conventions.",),
    ),
    DenseFamilySpec(
        adapter_id="gemma4-dense-v1",
        product_family="gemma-4",
        model_types=("gemma4", "gemma4_text"),
        reference_pattern=r"gemma[._-]?4",
        support_tier=SupportTier.INSPECT_ONLY,
    ),
    DenseFamilySpec(
        adapter_id="minicpm5-dense-v1",
        product_family="minicpm5",
        model_types=("minicpm5",),
        reference_pattern=r"minicpm[._-]?5",
        support_tier=SupportTier.INSPECT_ONLY,
    ),
    DenseFamilySpec(
        adapter_id="nemotron3-dense-v1",
        product_family="nemotron3",
        model_types=("nemotron3",),
        reference_pattern=r"nemotron[._-]?3",
        support_tier=SupportTier.INSPECT_ONLY,
    ),
)
