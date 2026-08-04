from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from axquant import awq
from axquant.awq import (
    apply_channel_scales,
    apply_mlx_awq_scale,
    refine_weight_with_awq,
)
from axquant.errors import PlanningError


def test_refine_weight_with_awq_at_alpha_zero_matches_plain_affine_bound() -> None:
    # alpha=0 makes every channel_scale exactly 1.0 (x**0 == 1 for any
    # positive x), so AWQ degenerates to plain per-group affine quantization
    # with no scaling. That makes the round-trip error hand-verifiable
    # against the same invariant (error <= scale/2) that caught the
    # DwqPlugin zero-point bug in quantizers.py -- a correct AWQ
    # implementation must satisfy it too in this degenerate case.
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((4, 32)).astype(np.float32)
    activations = rng.standard_normal((16, 32)).astype(np.float32)

    refined, metadata = refine_weight_with_awq(
        weight, activations, bits=8, group_size=32, alpha_grid=(0.0,)
    )

    assert metadata["awq_alpha"] == 0.0
    assert refined.shape == weight.shape
    w_min = weight.min(axis=-1, keepdims=True)
    w_max = weight.max(axis=-1, keepdims=True)
    scale = (w_max - w_min) / 255.0
    per_element_error = np.abs(weight - refined)
    assert np.all(per_element_error <= scale / 2 + 1e-3)


def test_refine_weight_with_awq_rejects_mismatched_calibration_channels() -> None:
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((4, 32)).astype(np.float32)
    activations = rng.standard_normal((16, 16)).astype(np.float32)  # wrong width
    with pytest.raises(PlanningError, match="calibration channels"):
        refine_weight_with_awq(weight, activations, bits=8, group_size=32)


def test_apply_mlx_awq_scale_mutates_module_weight_in_place() -> None:
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((8, 32)).astype(np.float32)
    activations = rng.standard_normal((16, 32)).astype(np.float32)
    module = SimpleNamespace(weight=mx.array(weight))

    metadata = apply_mlx_awq_scale(module, activations=activations, bits=8, group_size=32)

    assert metadata["bits"] == 8
    mutated = np.asarray(module.weight)
    assert mutated.shape == weight.shape
    assert not np.allclose(mutated, weight)
    # Refinement must stay close to the source weights, not diverge.
    assert float(np.mean((weight - mutated) ** 2)) < float(np.mean(weight**2))


def test_apply_mlx_awq_scale_requires_a_weight_tensor() -> None:
    pytest.importorskip("mlx.core")
    with pytest.raises(PlanningError, match="requires a module with a weight"):
        apply_mlx_awq_scale(SimpleNamespace(), activations=None, bits=8, group_size=32)


def test_awq_rejects_non_finite_inputs_and_alpha_grid() -> None:
    weight = np.ones((2, 32), dtype=np.float32)
    activations = np.ones((4, 32), dtype=np.float32)

    invalid_weight = weight.copy()
    invalid_weight[0, 0] = np.nan
    with pytest.raises(PlanningError, match="finite values"):
        refine_weight_with_awq(
            invalid_weight,
            activations,
            bits=4,
            group_size=32,
        )

    invalid_activations = activations.copy()
    invalid_activations[0, 0] = np.inf
    with pytest.raises(PlanningError, match="finite values"):
        refine_weight_with_awq(
            weight,
            invalid_activations,
            bits=4,
            group_size=32,
        )

    with pytest.raises(PlanningError, match="finite and within"):
        refine_weight_with_awq(
            weight,
            activations,
            bits=4,
            group_size=32,
            alpha_grid=(float("nan"),),
        )


def test_awq_rejects_scalar_activations_and_invalid_channel_scales() -> None:
    weight = np.ones((2, 32), dtype=np.float32)
    with pytest.raises(PlanningError, match="calibration channels"):
        refine_weight_with_awq(weight, np.array(1.0), bits=4, group_size=32)
    with pytest.raises(PlanningError, match="finite positive"):
        apply_channel_scales(weight, np.zeros((32,), dtype=np.float32))


def test_apply_mlx_awq_scale_does_not_mutate_when_materialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingMlx:
        @staticmethod
        def array(value: np.ndarray, *, dtype: np.dtype[np.float32]) -> np.ndarray:
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def eval(value: np.ndarray) -> None:
            del value
            raise RuntimeError("materialization failed")

    monkeypatch.setattr(awq.importlib, "import_module", lambda _name: _FailingMlx)
    original = np.ones((2, 32), dtype=np.float32)
    module = SimpleNamespace(weight=original)

    with pytest.raises(PlanningError, match="materialization failed"):
        apply_mlx_awq_scale(
            module,
            activations=np.ones((4, 32), dtype=np.float32),
            bits=4,
            group_size=32,
        )

    assert module.weight is original
