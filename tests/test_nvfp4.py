from __future__ import annotations

import builtins
from dataclasses import replace

import numpy as np
import pytest

from axquant.errors import ArtifactError
from axquant.nvfp4 import (
    Nvfp4Tensor,
    _quantize_nvfp4_torch,
    dequantize_nvfp4,
    quantize_nvfp4_cuda,
    quantize_nvfp4_reference,
)


def test_exact_e2m1_bytes_and_inverse_scale() -> None:
    values = np.array(
        [[0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]],
        dtype=np.float32,
    )
    encoded = quantize_nvfp4_reference(values)
    np.testing.assert_array_equal(
        encoded.packed, [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]]
    )
    np.testing.assert_array_equal(encoded.scale_bytes, [[126]])
    assert encoded.global_scale == 448.0
    np.testing.assert_array_equal(dequantize_nvfp4(encoded), values)


def test_e2m1_midpoints_round_to_even() -> None:
    values = np.array([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6]], dtype=np.float32)
    values = np.concatenate((values, -values), axis=1)
    encoded = quantize_nvfp4_reference(values)
    expected = np.array([[0, 1, 1, 2, 2, 4, 4, 6, -0.0, -1, -1, -2, -2, -4, -4, -6]])
    np.testing.assert_array_equal(dequantize_nvfp4(encoded), expected)
    assert encoded.packed[0, 4] & 15 == 8


def test_e4m3_scale_midpoints_round_to_even() -> None:
    values = np.zeros((1, 48), dtype=np.float32)
    values[0, 0], values[0, 16], values[0, 32] = 6, 1.0625 * 6 / 448, 1.1875 * 6 / 448
    encoded = quantize_nvfp4_reference(values)
    np.testing.assert_array_equal(encoded.scale_bytes, [[126, 56, 58]])


@pytest.mark.parametrize("scale", [0.0, 1e-20, 1e-6, 1.0, 1e6, 1e30])
def test_round_trip_is_finite_with_bounded_error(scale: float) -> None:
    values = (np.random.default_rng(7).normal(size=(9, 64)) * scale).astype(np.float32)
    encoded = quantize_nvfp4_reference(values)
    decoded = dequantize_nvfp4(encoded)
    assert np.isfinite(decoded).all()
    assert encoded.packed.nbytes + encoded.scale_bytes.nbytes + 4 == values.size * 9 // 16 + 4
    if scale:
        assert np.linalg.norm((decoded / scale) - (values / scale)) < 0.15 * np.linalg.norm(
            values / scale
        )
    else:
        np.testing.assert_array_equal(decoded, values)
        assert encoded.global_scale == 1.0


def test_zero_blocks_and_fp8_underflow_do_not_produce_nan() -> None:
    source = np.zeros((1, 48), dtype=np.float32)
    source[0, 0], source[0, 16] = 6, 1e-10
    encoded = quantize_nvfp4_reference(source)
    assert encoded.scale_bytes[0, 1] == 1
    assert encoded.scale_bytes[0, 2] == 56
    assert np.isfinite(dequantize_nvfp4(encoded)).all()


@pytest.mark.parametrize(
    "value",
    [np.zeros((4, 17)), np.zeros((0, 16)), np.zeros((16,)), np.zeros((1, 16), dtype=np.int32)],
)
def test_invalid_source_shapes_and_dtypes_fail(value) -> None:
    with pytest.raises(ArtifactError):
        quantize_nvfp4_reference(value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e100, 1e-100])
def test_nonfinite_and_unrepresentable_weights_fail(value: float) -> None:
    with pytest.raises(ArtifactError):
        quantize_nvfp4_reference(np.full((1, 16), value, dtype=np.float64))


@pytest.mark.parametrize(
    "field,value", [("global_scale", 0), ("global_scale", np.inf), ("shape", (1, 17))]
)
def test_invalid_decode_metadata_fails(field: str, value) -> None:
    encoded = quantize_nvfp4_reference(np.ones((1, 16), dtype=np.float32))
    with pytest.raises(ArtifactError):
        dequantize_nvfp4(replace(encoded, **{field: value}))


@pytest.mark.parametrize("code", [0, 127, 128, 255])
def test_invalid_scale_codes_fail(code: int) -> None:
    encoded = Nvfp4Tensor(
        np.zeros((1, 8), dtype=np.uint8),
        np.array([[code]], dtype=np.uint8),
        1,
        (1, 16),
        "numpy-reference",
    )
    with pytest.raises(ArtifactError):
        dequantize_nvfp4(encoded)


def test_cuda_dependency_error_is_explicit(monkeypatch) -> None:
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("optional dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(ArtifactError, match="CUDA-enabled PyTorch"):
        quantize_nvfp4_cuda(np.zeros((1, 16), dtype=np.float32))


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan, 1000])
def test_invalid_shared_scale_fails(scale: float) -> None:
    with pytest.raises(ArtifactError):
        quantize_nvfp4_reference(np.full((1, 16), 6.0, dtype=np.float32), global_scale=scale)


def test_shared_scale_matches_torch_operations() -> None:
    torch = pytest.importorskip("torch")
    source = np.random.default_rng(19).normal(size=(3, 32)).astype(np.float32)
    global_scale = 2688 / 20
    reference = quantize_nvfp4_reference(source, global_scale=global_scale)
    actual = _quantize_nvfp4_torch(
        source, torch=torch, target=torch.device("cpu"), rows_per_chunk=1, global_scale=global_scale
    )
    np.testing.assert_array_equal(actual.packed, reference.packed)
    np.testing.assert_array_equal(actual.scale_bytes, reference.scale_bytes)


def test_cuda_never_falls_back_to_cpu(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ArtifactError, match="available CUDA"):
        quantize_nvfp4_cuda(np.zeros((1, 16), dtype=np.float32))


@pytest.mark.parametrize("rows_per_chunk", [1, 3, 256])
@pytest.mark.parametrize("scale", [0, 1e-20, 1, 1e30])
def test_torch_operations_match_reference_on_cpu(rows_per_chunk: int, scale: float) -> None:
    torch = pytest.importorskip("torch")
    source = np.random.default_rng(19).normal(size=(7, 64)).astype(np.float32) * scale
    source[0] = 0
    source[1, :16] = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 2) * scale
    reference = quantize_nvfp4_reference(source)
    actual = _quantize_nvfp4_torch(
        source, torch=torch, target=torch.device("cpu"), rows_per_chunk=rows_per_chunk
    )
    assert actual.backend == "torch-reference"
    assert actual.global_scale == reference.global_scale
    np.testing.assert_array_equal(actual.packed, reference.packed)
    np.testing.assert_array_equal(actual.scale_bytes, reference.scale_bytes)


@pytest.mark.integration
@pytest.mark.parametrize("rows_per_chunk", [1, 3, 256])
@pytest.mark.parametrize("global_scale", [None, 2688 / 20])
def test_real_cuda_matches_reference(rows_per_chunk: int, global_scale: float | None) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real CUDA device required")
    source = np.random.default_rng(19).normal(size=(7, 64)).astype(np.float32)
    source[0] = 0
    reference = quantize_nvfp4_reference(source, global_scale=global_scale)
    actual = quantize_nvfp4_cuda(source, rows_per_chunk=rows_per_chunk, global_scale=global_scale)
    assert actual.backend == "torch-cuda"
    assert actual.global_scale == reference.global_scale
    np.testing.assert_array_equal(actual.packed, reference.packed)
    np.testing.assert_array_equal(actual.scale_bytes, reference.scale_bytes)
