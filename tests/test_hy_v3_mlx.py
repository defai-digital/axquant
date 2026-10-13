"""Tiny-weights tests for the hy_v3 MLX model (Tencent Hy-MT2).

Builds a 2-layer, 4-expert model with random unfused weights, checks that
``sanitize`` stacks only the MoE layer into ``switch_mlp`` while leaving the
dense layer, shared expert, router, and expert bias untouched, then loads
the weights and runs a 2-token forward. A numpy reference pins the routing
math: sigmoid scores, expert-bias top-k selection, renormalization, and
router scaling. Gated to hosts with a working Metal device.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from axquant import converter

try:
    import mlx.core as mx
    import mlx_lm.utils
    import numpy as np

    from axquant.mlx_models.hy_v3 import (
        HyV3SparseMoeBlock,
        Model,
        ModelArgs,
        register_hy_v3_model,
    )

    mx.zeros((1,))  # Metal allocator probe; raises on headless hosts.
except Exception as exc:
    pytest.skip(f"hy_v3 MLX tests need a Metal device: {exc}", allow_module_level=True)

pytestmark = pytest.mark.integration

_HIDDEN = 32
_INTERMEDIATE = 64
_MOE_INTERMEDIATE = 16
_LAYERS = 2
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_EXPERTS = 4
_TOP_K = 2
_VOCAB = 32

_TINY_CONFIG: dict[str, Any] = {
    "model_type": "hy_v3",
    "hidden_size": _HIDDEN,
    "intermediate_size": _INTERMEDIATE,
    "moe_intermediate_size": _MOE_INTERMEDIATE,
    "num_hidden_layers": _LAYERS,
    "num_attention_heads": _HEADS,
    "num_key_value_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "num_experts": _EXPERTS,
    "num_experts_per_tok": _TOP_K,
    "num_shared_experts": 1,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "vocab_size": _VOCAB,
    "tie_word_embeddings": False,
    "router_scaling_factor": 2.826,
    "first_k_dense_replace": 1,
}


def _tiny_weights() -> dict[str, mx.array]:
    mx.random.seed(0)
    weights: dict[str, mx.array] = {}
    weights["model.embed_tokens.weight"] = mx.random.normal((_VOCAB, _HIDDEN)) * 0.02
    for layer in range(_LAYERS):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.self_attn.q_proj.weight"] = (
            mx.random.normal((_HEADS * _HEAD_DIM, _HIDDEN)) * 0.02
        )
        weights[f"{prefix}.self_attn.k_proj.weight"] = (
            mx.random.normal((_KV_HEADS * _HEAD_DIM, _HIDDEN)) * 0.02
        )
        weights[f"{prefix}.self_attn.v_proj.weight"] = (
            mx.random.normal((_KV_HEADS * _HEAD_DIM, _HIDDEN)) * 0.02
        )
        weights[f"{prefix}.self_attn.o_proj.weight"] = (
            mx.random.normal((_HIDDEN, _HEADS * _HEAD_DIM)) * 0.02
        )
        weights[f"{prefix}.self_attn.q_norm.weight"] = mx.ones((_HEAD_DIM,))
        weights[f"{prefix}.self_attn.k_norm.weight"] = mx.ones((_HEAD_DIM,))
        weights[f"{prefix}.input_layernorm.weight"] = mx.ones((_HIDDEN,))
        weights[f"{prefix}.post_attention_layernorm.weight"] = mx.ones((_HIDDEN,))
    # Layer 0 is dense and stays unstacked.
    weights["model.layers.0.mlp.gate_proj.weight"] = (
        mx.random.normal((_INTERMEDIATE, _HIDDEN)) * 0.02
    )
    weights["model.layers.0.mlp.up_proj.weight"] = mx.random.normal((_INTERMEDIATE, _HIDDEN)) * 0.02
    weights["model.layers.0.mlp.down_proj.weight"] = (
        mx.random.normal((_HIDDEN, _INTERMEDIATE)) * 0.02
    )
    # Layer 1 is MoE with unfused experts plus a shared expert.
    for expert in range(_EXPERTS):
        weights[f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"] = (
            mx.random.normal((_MOE_INTERMEDIATE, _HIDDEN)) * 0.02
        )
        weights[f"model.layers.1.mlp.experts.{expert}.up_proj.weight"] = (
            mx.random.normal((_MOE_INTERMEDIATE, _HIDDEN)) * 0.02
        )
        weights[f"model.layers.1.mlp.experts.{expert}.down_proj.weight"] = (
            mx.random.normal((_HIDDEN, _MOE_INTERMEDIATE)) * 0.02
        )
    weights["model.layers.1.mlp.shared_mlp.gate_proj.weight"] = (
        mx.random.normal((_MOE_INTERMEDIATE, _HIDDEN)) * 0.02
    )
    weights["model.layers.1.mlp.shared_mlp.up_proj.weight"] = (
        mx.random.normal((_MOE_INTERMEDIATE, _HIDDEN)) * 0.02
    )
    weights["model.layers.1.mlp.shared_mlp.down_proj.weight"] = (
        mx.random.normal((_HIDDEN, _MOE_INTERMEDIATE)) * 0.02
    )
    weights["model.layers.1.mlp.router.gate.weight"] = mx.random.normal((_EXPERTS, _HIDDEN)) * 0.02
    weights["model.layers.1.mlp.expert_bias"] = mx.zeros((_EXPERTS,))
    weights["model.norm.weight"] = mx.ones((_HIDDEN,))
    weights["lm_head.weight"] = mx.random.normal((_VOCAB, _HIDDEN)) * 0.02
    return weights


def test_model_args_ignores_unknown_keys() -> None:
    config = dict(
        _TINY_CONFIG,
        architectures=["HYV3ForCausalLM"],
        attn_impl="eager",
        transformers_version="4.57.1",
    )
    args = ModelArgs.from_dict(config)
    assert args.hidden_size == _HIDDEN
    assert args.num_experts == _EXPERTS
    assert args.router_scaling_factor == 2.826


def test_model_args_defaults() -> None:
    config = {
        key: value
        for key, value in _TINY_CONFIG.items()
        if key not in {"router_scaling_factor", "first_k_dense_replace"}
    }
    args = ModelArgs.from_dict(config)
    assert args.router_scaling_factor == 2.826
    assert args.first_k_dense_replace == 1


def test_register_hy_v3_model() -> None:
    register_hy_v3_model()
    assert sys.modules["mlx_lm.models.hy_v3"] is sys.modules["axquant.mlx_models.hy_v3"]
    arch = importlib.import_module("mlx_lm.models.hy_v3")
    assert arch.Model is Model
    assert arch.ModelArgs is ModelArgs
    model_cls, args_cls = mlx_lm.utils._get_classes({"model_type": "hy_v3"})
    assert (model_cls, args_cls) == (Model, ModelArgs)


def test_converter_mlx_api_registers_hy_v3() -> None:
    sys.modules.pop("mlx_lm.models.hy_v3", None)
    converter._mlx_api()
    assert sys.modules["mlx_lm.models.hy_v3"] is sys.modules["axquant.mlx_models.hy_v3"]


def test_sanitize_stacks_only_moe_layers() -> None:
    model = Model(ModelArgs.from_dict(_TINY_CONFIG))
    clean = model.sanitize(_tiny_weights())
    assert "model.layers.1.mlp.switch_mlp.gate_proj.weight" in clean
    assert tuple(clean["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape) == (
        _EXPERTS,
        _MOE_INTERMEDIATE,
        _HIDDEN,
    )
    assert tuple(clean["model.layers.1.mlp.switch_mlp.up_proj.weight"].shape) == (
        _EXPERTS,
        _MOE_INTERMEDIATE,
        _HIDDEN,
    )
    assert tuple(clean["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape) == (
        _EXPERTS,
        _HIDDEN,
        _MOE_INTERMEDIATE,
    )
    assert not [key for key in clean if ".mlp.experts." in key]
    assert tuple(clean["model.layers.0.mlp.gate_proj.weight"].shape) == (
        _INTERMEDIATE,
        _HIDDEN,
    )
    assert tuple(clean["model.layers.0.mlp.down_proj.weight"].shape) == (
        _HIDDEN,
        _INTERMEDIATE,
    )
    assert tuple(clean["model.layers.1.mlp.shared_mlp.gate_proj.weight"].shape) == (
        _MOE_INTERMEDIATE,
        _HIDDEN,
    )
    assert tuple(clean["model.layers.1.mlp.shared_mlp.down_proj.weight"].shape) == (
        _HIDDEN,
        _MOE_INTERMEDIATE,
    )
    assert tuple(clean["model.layers.1.mlp.router.gate.weight"].shape) == (
        _EXPERTS,
        _HIDDEN,
    )
    assert tuple(clean["model.layers.1.mlp.expert_bias"].shape) == (_EXPERTS,)


def test_load_weights_and_forward_two_tokens() -> None:
    model = Model(ModelArgs.from_dict(_TINY_CONFIG))
    weights = model.sanitize(_tiny_weights())
    model.load_weights(list(weights.items()), strict=True)
    out = model(mx.array([[1, 2]]))
    mx.eval(out)
    assert tuple(out.shape) == (1, 2, _VOCAB)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_moe_routing_matches_sigmoid_bias_spec() -> None:
    args = ModelArgs.from_dict(_TINY_CONFIG)
    block = HyV3SparseMoeBlock(args)
    mx.random.seed(1)
    x = mx.random.normal((1, 3, _HIDDEN)) * 0.1
    out = block(x)
    mx.eval(out)

    xn = np.array(x, dtype=np.float32)
    logits = xn @ np.array(block.router.gate.weight, dtype=np.float32).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    bias = np.array(block.expert_bias, dtype=np.float32)
    inds = np.argsort(scores + bias, axis=-1)[..., -_TOP_K:]
    top_scores = np.take_along_axis(scores, inds, axis=-1)
    top_scores = top_scores / top_scores.sum(axis=-1, keepdims=True)
    top_scores = top_scores * np.float32(args.router_scaling_factor)

    def silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-z))

    up = np.array(block.switch_mlp.up_proj.weight, dtype=np.float32)
    gate = np.array(block.switch_mlp.gate_proj.weight, dtype=np.float32)
    down = np.array(block.switch_mlp.down_proj.weight, dtype=np.float32)
    expected = np.zeros_like(xn)
    for token in range(3):
        for slot in range(_TOP_K):
            expert = int(inds[0, token, slot])
            hidden = silu(xn[0, token] @ gate[expert].T) * (xn[0, token] @ up[expert].T)
            expected[0, token] += top_scores[0, token, slot] * (hidden @ down[expert].T)
    shared_gate = np.array(block.shared_mlp.gate_proj.weight, dtype=np.float32)
    shared_up = np.array(block.shared_mlp.up_proj.weight, dtype=np.float32)
    shared_down = np.array(block.shared_mlp.down_proj.weight, dtype=np.float32)
    expected += (silu(xn @ shared_gate.T) * (xn @ shared_up.T)) @ shared_down.T

    np.testing.assert_allclose(np.array(out, dtype=np.float32), expected, rtol=1e-4, atol=1e-5)
