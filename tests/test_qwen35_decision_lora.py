from __future__ import annotations

import numpy as np

from scripts.qwen35_decision_lora import lora_delta, weight_key_for_lora


def test_weight_key_maps_mlx_module_path_onto_snapshot_weight() -> None:
    key = "language_model.model.layers.0.mlp.gate_proj.lora_a"
    assert weight_key_for_lora(key) == "model.language_model.layers.0.mlp.gate_proj.weight"
    linear = "language_model.model.layers.3.linear_attn.in_proj_qkv.lora_b"
    expected = "model.language_model.layers.3.linear_attn.in_proj_qkv.weight"
    assert weight_key_for_lora(linear) == expected


def test_lora_delta_matches_mlx_lm_forward() -> None:
    rng = np.random.default_rng(0)
    in_dim, out_dim, rank = 4, 3, 2
    weight = rng.normal(size=(out_dim, in_dim))
    lora_a = rng.normal(size=(in_dim, rank))
    lora_b = rng.normal(size=(rank, out_dim))
    scale = 1.0
    inputs = rng.normal(size=(5, in_dim))
    fused = weight + lora_delta(lora_a, lora_b, scale=scale, rank=rank)
    baseline = inputs @ weight.T + scale * (inputs @ lora_a) @ lora_b
    assert fused.shape == weight.shape
    assert np.allclose(inputs @ fused.T, baseline)
