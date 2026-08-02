from __future__ import annotations

from types import SimpleNamespace

import pytest

from axquant.dwq import apply_mlx_dwq_clip
from axquant.errors import PlanningError


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


def test_apply_mlx_dwq_clip_requires_multiple_elements() -> None:
    mx = pytest.importorskip("mlx.core")
    module = SimpleNamespace(weight=mx.array([1.0], dtype=mx.float32))
    with pytest.raises(PlanningError, match="at least two"):
        apply_mlx_dwq_clip(module)
