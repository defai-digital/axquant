from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from axquant import gptq
from axquant.errors import PlanningError
from axquant.gptq import apply_mlx_gptq_refine, learn_gptq_refined_weight


def _rtn_group_affine(weight: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Plain round-to-nearest group-affine quantize-dequantize (no compensation)."""
    out_features, in_features = weight.shape
    w_grouped = weight.reshape(out_features, in_features // group_size, group_size)
    w_min = w_grouped.min(axis=-1, keepdims=True)
    w_max = w_grouped.max(axis=-1, keepdims=True)
    scale = (w_max - w_min) / ((1 << bits) - 1)
    scale = np.where(scale == 0, 1.0, scale)
    zero = np.clip(np.round(-w_min / scale), 0, (1 << bits) - 1)
    q = np.clip(np.round(w_grouped / scale) + zero, 0, (1 << bits) - 1)
    return ((q - zero) * scale).reshape(weight.shape).astype(np.float32)


def _correlated_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    weight = rng.standard_normal((64, 128)).astype(np.float32)
    mixing = rng.standard_normal((128, 128)).astype(np.float32)
    z = rng.standard_normal((256, 128)).astype(np.float32)
    activations = (z @ mixing).astype(np.float32)
    return weight, activations


def test_gptq_beats_rtn_under_correlated_activations() -> None:
    weight, activations = _correlated_fixture()

    refined, metadata = learn_gptq_refined_weight(weight, activations, bits=4, group_size=32)
    rtn = _rtn_group_affine(weight, bits=4, group_size=32)

    assert refined.shape == weight.shape
    # GPTQ optimizes the activation-weighted objective tr(E H E^T): its error
    # compensation deliberately shifts unquantized columns, so plain weight MSE
    # can be worse than RTN while output reconstruction clearly improves.
    reference = activations @ weight.T
    gptq_mse = float(np.mean((reference - activations @ refined.T) ** 2))
    rtn_mse = float(np.mean((reference - activations @ rtn.T) ** 2))
    assert gptq_mse < rtn_mse
    assert metadata["mean_quant_error"] >= 0.0
    assert metadata["calibration_rows"] == 256
    assert metadata["bits"] == 4
    assert metadata["group_size"] == 32


def test_gptq_rejects_unsupported_bits() -> None:
    weight, activations = _correlated_fixture()
    with pytest.raises(PlanningError, match="does not support 5-bit"):
        learn_gptq_refined_weight(weight, activations, bits=5, group_size=32)


def test_gptq_rejects_unsupported_group_size() -> None:
    weight, activations = _correlated_fixture()
    with pytest.raises(PlanningError, match="does not support group size 48"):
        learn_gptq_refined_weight(weight, activations, bits=4, group_size=48)


def test_gptq_rejects_mismatched_calibration_channels() -> None:
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((8, 64)).astype(np.float32)
    activations = rng.standard_normal((16, 32)).astype(np.float32)  # wrong width
    with pytest.raises(PlanningError, match="calibration channels"):
        learn_gptq_refined_weight(weight, activations, bits=4, group_size=32)


def test_gptq_is_deterministic() -> None:
    weight, activations = _correlated_fixture()
    refined_a, _ = learn_gptq_refined_weight(weight, activations, bits=4, group_size=32)
    refined_b, _ = learn_gptq_refined_weight(weight, activations, bits=4, group_size=32)
    np.testing.assert_array_equal(refined_a, refined_b)


def test_gptq_accepts_dict_form_activations() -> None:
    weight, activations = _correlated_fixture()
    refined, metadata = learn_gptq_refined_weight(
        weight, {"activations": activations}, bits=4, group_size=32
    )
    assert refined.shape == weight.shape
    assert metadata["calibration_rows"] == 256


def test_apply_mlx_gptq_refine_mutates_module_weight_in_place() -> None:
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(13)
    weight = rng.standard_normal((8, 64)).astype(np.float32)
    activations = rng.standard_normal((32, 64)).astype(np.float32)
    module = SimpleNamespace(weight=mx.array(weight))

    metadata = apply_mlx_gptq_refine(module, activations=activations, bits=4, group_size=32)

    assert "gptq_damping" in metadata
    mutated = np.asarray(module.weight)
    assert mutated.shape == weight.shape
    assert module.weight.dtype == mx.array(weight).dtype
    assert not np.allclose(mutated, weight)
    # Refinement must stay close to the source weights, not diverge.
    assert float(np.mean((weight - mutated) ** 2)) < float(np.mean(weight**2))


def test_apply_mlx_gptq_refine_requires_a_weight_tensor() -> None:
    pytest.importorskip("mlx.core")
    with pytest.raises(PlanningError, match="requires a module with a weight"):
        apply_mlx_gptq_refine(SimpleNamespace(), activations=None, bits=4, group_size=32)


def test_gptq_rejects_non_finite_inputs() -> None:
    weight = np.ones((2, 32), dtype=np.float32)
    activations = np.ones((4, 32), dtype=np.float32)
    invalid_weight = weight.copy()
    invalid_weight[0, 0] = np.nan
    with pytest.raises(PlanningError, match="finite values"):
        learn_gptq_refined_weight(
            invalid_weight,
            activations,
            bits=4,
            group_size=32,
        )
    invalid_activations = activations.copy()
    invalid_activations[0, 0] = np.inf
    with pytest.raises(PlanningError, match="finite values"):
        learn_gptq_refined_weight(
            weight,
            invalid_activations,
            bits=4,
            group_size=32,
        )


@pytest.mark.parametrize("block_size", [0, -1, True])
def test_gptq_rejects_invalid_block_size(block_size: int) -> None:
    with pytest.raises(PlanningError, match="block_size must be a positive integer"):
        learn_gptq_refined_weight(
            np.ones((2, 32), dtype=np.float32),
            np.ones((4, 32), dtype=np.float32),
            bits=4,
            group_size=32,
            block_size=block_size,
        )


@pytest.mark.parametrize("damping", [-0.01, float("nan"), float("inf"), True])
def test_gptq_rejects_invalid_damping(damping: float) -> None:
    with pytest.raises(PlanningError, match="damping must be a finite non-negative number"):
        learn_gptq_refined_weight(
            np.ones((2, 32), dtype=np.float32),
            np.ones((4, 32), dtype=np.float32),
            bits=4,
            group_size=32,
            damping=damping,
        )


def test_gptq_zero_damping_escalates_if_factorization_needs_it() -> None:
    refined, metadata = learn_gptq_refined_weight(
        np.ones((2, 32), dtype=np.float32),
        np.ones((4, 32), dtype=np.float32),
        bits=4,
        group_size=32,
        damping=0.0,
    )
    assert np.all(np.isfinite(refined))
    assert metadata["gptq_damping"] >= 0.0


def test_gptq_rejects_scalar_activations() -> None:
    with pytest.raises(PlanningError, match="calibration channels"):
        learn_gptq_refined_weight(
            np.ones((2, 32), dtype=np.float32),
            np.array(1.0),
            bits=4,
            group_size=32,
        )


def test_apply_mlx_gptq_refine_does_not_mutate_when_materialization_fails(
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

    monkeypatch.setattr(gptq.importlib, "import_module", lambda _name: _FailingMlx)
    original = np.ones((2, 32), dtype=np.float32)
    module = SimpleNamespace(weight=original)

    with pytest.raises(PlanningError, match="materialization failed"):
        apply_mlx_gptq_refine(
            module,
            activations=np.ones((4, 32), dtype=np.float32),
            bits=4,
            group_size=32,
        )

    assert module.weight is original
