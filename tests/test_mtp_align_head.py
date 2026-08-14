"""Tiny-shape MTP head forward tests."""

from __future__ import annotations

import pytest

from axquant.mtp_align.qwen_mtp_head import QwenMtpHead, QwenMtpHeadConfig


def _tiny_weights():
    mx = pytest.importorskip("mlx.core")
    cfg = QwenMtpHeadConfig(
        hidden_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    h, e, i = cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size
    # q_proj is interleaved query||gate → 2 * n_heads * head_dim
    w = {
        "mtp.fc.weight": mx.random.normal((h, 2 * h)) * 0.02,
        "mtp.layers.0.input_layernorm.weight": mx.ones((h,)),
        "mtp.layers.0.mlp.experts.down_proj": mx.random.normal((e, h, i)) * 0.02,
        "mtp.layers.0.mlp.experts.gate_up_proj": mx.random.normal((e, 2 * i, h)) * 0.02,
        "mtp.layers.0.mlp.gate.weight": mx.random.normal((e, h)) * 0.02,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": mx.random.normal((h, i)) * 0.02,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": mx.random.normal((i, h)) * 0.02,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": mx.random.normal((i, h)) * 0.02,
        "mtp.layers.0.mlp.shared_expert_gate.weight": mx.random.normal((1, h)) * 0.02,
        "mtp.layers.0.post_attention_layernorm.weight": mx.ones((h,)),
        "mtp.layers.0.self_attn.k_norm.weight": mx.ones((cfg.head_dim,)),
        "mtp.layers.0.self_attn.k_proj.weight": mx.random.normal(
            (cfg.num_key_value_heads * cfg.head_dim, h)
        )
        * 0.02,
        "mtp.layers.0.self_attn.o_proj.weight": mx.random.normal(
            (h, cfg.num_attention_heads * cfg.head_dim)
        )
        * 0.02,
        "mtp.layers.0.self_attn.q_norm.weight": mx.ones((cfg.head_dim,)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.random.normal(
            (cfg.num_attention_heads * cfg.head_dim * 2, h)
        )
        * 0.02,
        "mtp.layers.0.self_attn.v_proj.weight": mx.random.normal(
            (cfg.num_key_value_heads * cfg.head_dim, h)
        )
        * 0.02,
        "mtp.norm.weight": mx.ones((h,)),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((h,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((h,)),
    }
    return cfg, w


def test_tiny_head_draft_logits_shape() -> None:
    mx = pytest.importorskip("mlx.core")
    cfg, w = _tiny_weights()
    head = QwenMtpHead.from_weight_dict(w, config=cfg)
    hidden = mx.random.normal((2, cfg.hidden_size))
    embed = mx.random.normal((2, cfg.hidden_size))
    lm_head = mx.random.normal((cfg.vocab_size, cfg.hidden_size)) * 0.02
    logits = head.draft_logits(main_hidden=hidden, prev_token_embed=embed, lm_head_weight=lm_head)
    assert tuple(logits.shape) == (2, cfg.vocab_size)


def test_fc_norm_params_subset() -> None:
    pytest.importorskip("mlx.core")
    cfg, w = _tiny_weights()
    head = QwenMtpHead.from_weight_dict(w, config=cfg)
    params = head.trainable_fc_norm_params()
    assert set(params) == {
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
    }
