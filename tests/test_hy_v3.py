"""Metadata-only Hy-MT2 regression coverage; no weights or network required."""

from typing import Any

import pytest

from axquant.architectures.hy_v3 import HyV3Adapter
from axquant.architectures.registry import adapter_for
from axquant.schema import ArchitectureSupportLevel, OptimizationScope, SupportTier, TensorRole

MODEL = "tencent/Hy-MT2-30B-A3B"
# Source metadata: d3ead4dba61c09aac60a261a96ad1df3e705febb (Apache-2.0).
CONFIG: dict[str, Any] = {
    "model_type": "hy_v3",
    "architectures": ["HYV3ForCausalLM"],
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6912,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_shared_experts": 1,
    "moe_intermediate_size": 768,
    "router_scaling_factor": 2.826,
    "qk_norm": True,
    "tie_word_embeddings": False,
}


def test_real_config_selects_convertible_adapter() -> None:
    adapter = adapter_for(MODEL, CONFIG)
    assert isinstance(adapter, HyV3Adapter)
    assert adapter.adapter_id == "hy-v3-v1"
    profile = adapter.profile(MODEL, CONFIG)
    assert profile.product_family == "hy-v3"
    assert profile.support_level is ArchitectureSupportLevel.SUPPORTED
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.optimization_scope is OptimizationScope.TEXT_PATH
    assert profile.text_layer_count == 48
    assert profile.dense is False
    assert profile.mtp_declared is False


@pytest.mark.parametrize("model_type", ["hy_v2", "llama", "HY_V3", None])
def test_mismatched_model_type_does_not_match(model_type: str | None) -> None:
    config = {**CONFIG, "model_type": model_type}
    assert not HyV3Adapter().matches(MODEL, config)
    assert adapter_for(MODEL, config) is None


@pytest.mark.parametrize(
    "key", ["num_hidden_layers", "num_experts", "num_experts_per_tok", "hidden_size"]
)
@pytest.mark.parametrize("replacement", [None, 1, "48", 48.0, True])
def test_invalid_signature_stays_inventory_only(key: str, replacement: Any) -> None:
    config = {**CONFIG, key: replacement}
    adapter = adapter_for(MODEL, config)
    assert isinstance(adapter, HyV3Adapter)
    profile = adapter.profile(MODEL, config)
    assert profile.support_level is ArchitectureSupportLevel.INVENTORY_ONLY
    assert profile.support_tier is SupportTier.INSPECT_ONLY
    assert profile.optimization_scope is OptimizationScope.INVENTORY_ONLY


def test_all_checkpoint_index_templates_classify() -> None:
    adapter = HyV3Adapter()
    tensors: dict[str, TensorRole] = {
        "model.embed_tokens.weight": TensorRole.EMBEDDING,
        "model.norm.weight": TensorRole.NORM,
        "lm_head.weight": TensorRole.LM_HEAD,
    }
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        for projection in ("q", "k", "v", "o"):
            tensors[f"{prefix}.self_attn.{projection}_proj.weight"] = TensorRole.ATTENTION
        for norm in ("q_norm", "k_norm"):
            tensors[f"{prefix}.self_attn.{norm}.weight"] = TensorRole.NORM
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{prefix}.{norm}.weight"] = TensorRole.NORM
        if layer == 0:
            for projection in ("gate", "up", "down"):
                tensors[f"{prefix}.mlp.{projection}_proj.weight"] = TensorRole.MLP
            continue
        tensors[f"{prefix}.mlp.router.gate.weight"] = TensorRole.ROUTER
        tensors[f"{prefix}.mlp.expert_bias"] = TensorRole.ROUTER
        for projection in ("gate", "up", "down"):
            tensors[f"{prefix}.mlp.shared_mlp.{projection}_proj.weight"] = TensorRole.MLP
            for expert in range(128):
                name = f"{prefix}.mlp.experts.{expert}.{projection}_proj.weight"
                tensors[name] = TensorRole.EXPERT
    assert len(tensors) == 18673
    for name, expected in tensors.items():
        assert adapter.classify_tensor(name, "model-00001-of-00014.safetensors") is expected, name


@pytest.mark.parametrize("name", ["expert_bias", "model.layers.1.mlp.expert_bias.weight"])
def test_expert_bias_override(name: str) -> None:
    assert HyV3Adapter().classify_tensor(name, "model.safetensors") is TensorRole.ROUTER


def test_unknown_tensor_stays_unclassified() -> None:
    assert HyV3Adapter().classify_tensor("unknown.weight", "model.safetensors") is None
