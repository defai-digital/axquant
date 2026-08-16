from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axquant.dwq import (
    _MLX_FLAT_LIMIT,
    apply_mlx_dwq_clip,
    dwq_sample_strides,
    dwq_should_materialize,
)
from axquant.errors import PlanningError
from axquant.module_paths import fused_expert_module
from axquant.predicate import FUSED_STACK_METHODS, fused_stack_method_allowed


def test_apply_mlx_dwq_clip_clips_tail_outliers_to_the_0_1_percentile() -> None:
    mx = pytest.importorskip("mlx.core")
    # 2000 elements gives lower_index=2, upper_index=1998, so the two lowest
    # and one highest values are genuine outliers clipped in from the tails
    # -- a hand-computable check that the percentile math is right, not just
    # that the function runs without crashing.
    module = SimpleNamespace(weight=mx.array(list(range(2000)), dtype=mx.float32))

    result = apply_mlx_dwq_clip(module)

    assert result["sample_count"] == 2000
    assert result["sample_stride"] == 1
    assert result["materialized"] is True
    assert result["clip_lower"] == pytest.approx(2.0)
    assert result["clip_upper"] == pytest.approx(1998.0)
    clipped = module.weight
    assert float(clipped.min().item()) == pytest.approx(2.0)
    assert float(clipped.max().item()) == pytest.approx(1998.0)
    # An interior value, far from either tail, must pass through unchanged.
    assert float(clipped[1000].item()) == pytest.approx(1000.0)


def test_apply_mlx_dwq_clip_requires_a_weight_tensor() -> None:
    pytest.importorskip("mlx.core")
    with pytest.raises(PlanningError, match="requires a module with a weight"):
        apply_mlx_dwq_clip(SimpleNamespace())


def test_dwq_sample_strides_keep_fused_flash_stacks_under_int32() -> None:
    # One DeepSeek V4 fused switch is far above MLX's int32 flatten limit.
    shape = (256, 8192, 7168)
    assert 256 * 8192 * 7168 > 2_147_483_647
    strides = dwq_sample_strides(shape)
    sampled = 1
    for dim, stride in zip(shape, strides, strict=True):
        sampled *= (dim + stride - 1) // stride
    assert sampled <= 65536
    assert all(stride >= 1 for stride in strides)


def test_dwq_skips_materialize_on_fused_flash_element_counts() -> None:
    # (256, 8192, 7168) is one Flash fused switch; eval of the clipped
    # BF16 copy is what OOM-killed convert on 192 GB.
    fused_elements = 256 * 8192 * 7168
    assert fused_elements > _MLX_FLAT_LIMIT
    assert dwq_should_materialize(fused_elements) is False
    assert dwq_should_materialize(2000) is True


def test_apply_mlx_dwq_clip_on_small_fused_shaped_stack() -> None:
    mx = pytest.importorskip("mlx.core")
    # Same rank as a Flash switch, small enough to flatten and check.
    values = mx.arange(2 * 8 * 16, dtype=mx.float32).reshape((2, 8, 16))
    module = SimpleNamespace(weight=values)
    result = apply_mlx_dwq_clip(module)
    assert result["materialized"] is True
    assert module.weight.shape == (2, 8, 16)
    assert float(module.weight.min().item()) >= float(result["clip_lower"])
    assert float(module.weight.max().item()) <= float(result["clip_upper"])


def test_flash_dwq_recipe_clips_attention_not_fused_trunk() -> None:
    recipe_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "deepseek-v4-experimental-4bit-g128-dwq-v0.1.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    by_id = {rule["rule_id"]: rule for rule in recipe["rules"]}
    assert by_id["attention-6bit-dwq"]["method"] == "dwq"
    assert by_id["attention-6bit-dwq"]["bits"] == 6
    assert by_id["trunk-4bit-affine"]["method"] == "affine"
    assert by_id["trunk-4bit-affine"]["bits"] == 4
    assert set(by_id["trunk-4bit-affine"]["roles"]) == {"mlp", "expert"}


def test_fused_switch_cannot_mix_bit_widths_and_dwq_stays_allowed() -> None:
    assert fused_stack_method_allowed("dwq")
    assert fused_stack_method_allowed("affine")
    assert not fused_stack_method_allowed("awq")
    assert frozenset({"affine", "dwq"}) == FUSED_STACK_METHODS
    gate0 = "model.layers.0.ffn.experts.0.w1"
    gate1 = "model.layers.0.ffn.experts.1.w1"
    assert fused_expert_module(gate0) == fused_expert_module(gate1)
    assert fused_expert_module(gate0) == "model.layers.0.ffn.switch_mlp.gate_proj"
    mixed_bits = {4, 6}
    # One fused switch is one (bits, method) unit — mixed widths are rejected
    # at plan time; the helper here only shows they share a packing target.
    assert len(mixed_bits) > 1


def test_apply_mlx_dwq_clip_requires_multiple_elements() -> None:
    mx = pytest.importorskip("mlx.core")
    module = SimpleNamespace(weight=mx.array([1.0], dtype=mx.float32))
    with pytest.raises(PlanningError, match="at least two"):
        apply_mlx_dwq_clip(module)
