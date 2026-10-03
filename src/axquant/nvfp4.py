"""AXQuant-owned NVFP4 RTN codec; optional CUDA execution is imported lazily.

Checkpoint scales use the compressed-tensors inverse global-scale convention:
weight = E2M1 * E4M3_block_scale / FP32_global_scale. Blocks contain 16 values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from axquant.errors import ArtifactError

BLOCK_SIZE = 16
_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass(frozen=True)
class Nvfp4Tensor:
    """Unswizzled checkpoint bytes, not a runtime-specific Tensor Core layout."""

    packed: Any
    scale_bytes: Any
    global_scale: float
    shape: tuple[int, int]
    backend: str


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ArtifactError("NVFP4 numerical execution requires axquant[cuda]") from exc
    return np


def _scale_values(np: Any) -> Any:
    codes = np.arange(127, dtype=np.int32)
    exponent, fraction = codes >> 3, codes & 7
    return np.where(
        exponent == 0,
        fraction / 512.0,
        (1.0 + fraction / 8.0) * np.exp2(exponent - 7),
    ).astype(np.float32)


def _global_scale(maximum: float) -> float:
    scale = 1.0 if maximum == 0.0 else 2688.0 / maximum
    np = _numpy()
    if not math.isfinite(scale) or not 0 < scale <= float(np.finfo(np.float32).max):
        raise ArtifactError("NVFP4 inverse global scale cannot be represented in FP32")
    result = float(np.float32(scale))
    if result == 0.0:
        raise ArtifactError("NVFP4 inverse global scale underflows FP32")
    return result


def _shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) != 2 or min(shape) < 1 or shape[1] % BLOCK_SIZE:
        raise ArtifactError("NVFP4 requires a nonempty matrix with columns divisible by 16")
    return shape[0], shape[1]


def quantize_nvfp4_reference(weight: Any, *, global_scale: float | None = None) -> Nvfp4Tensor:
    """CPU oracle with explicit IEEE round-to-nearest-even for both formats."""
    np = _numpy()
    source = np.asarray(weight)
    if source.dtype.kind != "f":
        raise ArtifactError("NVFP4 requires floating-point source weights")
    shape = _shape(tuple(source.shape))
    if not np.isfinite(source).all() or np.abs(source).max() > np.finfo(np.float32).max:
        raise ArtifactError("NVFP4 source weights must be finite and representable in FP32")
    converted = source.astype(np.float32)
    if np.any((source != 0) & (converted == 0)):
        raise ArtifactError("NVFP4 source weights underflow FP32")
    source = converted
    maximum = float(np.abs(source).max())
    global_scale = _resolve_global_scale(maximum, global_scale)
    blocks = source.reshape(shape[0], -1, BLOCK_SIZE)
    maxima = np.max(np.abs(blocks), axis=-1)
    # Multiply after dividing, so a finite FP32 weight cannot overflow here.
    ideal = np.clip((maxima / np.float32(6.0)) * np.float32(global_scale), 1 / 512, 448)
    ideal = np.where(maxima == 0, np.float32(1), ideal)
    values = _scale_values(np)
    upper = np.clip(np.searchsorted(values, ideal), 1, 126)
    lower = upper - 1
    down, up = ideal - values[lower], values[upper] - ideal
    use_upper = (up < down) | ((up == down) & ((upper & 1) == 0))
    scales = np.where(use_upper, upper, lower).astype(np.uint8)
    normalized = (blocks / values[scales][..., None]) * np.float32(global_scale)
    magnitude = np.abs(normalized)
    boundaries = np.asarray(_MIDPOINTS, dtype=np.float32)
    codes = np.searchsorted(boundaries, magnitude).astype(np.uint8)
    # At midpoint i, choose i+1 only when the lower code is odd.
    odd_ties = (codes < 7) & ((codes & 1) == 1)
    midpoint = boundaries[np.minimum(codes, 6)]
    codes += (odd_ties & (magnitude == midpoint)).astype(np.uint8)
    codes |= np.signbit(normalized).astype(np.uint8) << 3
    codes = codes.reshape(shape)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return Nvfp4Tensor(packed, scales, global_scale, shape, "numpy-reference")


def quantize_nvfp4_cuda(
    weight: Any,
    *,
    device: str = "cuda",
    rows_per_chunk: int = 256,
    global_scale: float | None = None,
) -> Nvfp4Tensor:
    """Quantize and pack on CUDA with bounded working memory; never fall back to CPU."""
    if rows_per_chunk < 1:
        raise ArtifactError("NVFP4 rows_per_chunk must be positive")
    try:
        import torch
    except ImportError as exc:
        raise ArtifactError(
            "CUDA execution requires axquant[cuda] and a CUDA-enabled PyTorch"
        ) from exc
    try:
        target = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ArtifactError(f"invalid CUDA device: {device}") from exc
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ArtifactError("NVFP4 CUDA execution requires an available CUDA device")
    try:
        return _quantize_nvfp4_torch(
            weight,
            torch=torch,
            target=target,
            rows_per_chunk=rows_per_chunk,
            global_scale=global_scale,
        )
    except RuntimeError as exc:
        raise ArtifactError(f"NVFP4 CUDA quantization failed: {exc}") from exc


def _quantize_nvfp4_torch(
    weight: Any,
    *,
    torch: Any,
    target: Any,
    rows_per_chunk: int,
    global_scale: float | None = None,
) -> Nvfp4Tensor:
    """Shared tensor operations, also exercised on real PyTorch CPU in tests."""
    np = _numpy()
    source = weight if isinstance(weight, torch.Tensor) else torch.as_tensor(weight)
    shape = _shape(tuple(source.shape))
    if not source.is_floating_point():
        raise ArtifactError("NVFP4 requires floating-point source weights")
    maximum = 0.0
    with torch.no_grad():
        for start in range(0, shape[0], rows_per_chunk):
            original = source[start : start + rows_per_chunk].to(device=target)
            chunk = original.float()
            if not bool(torch.isfinite(chunk).all().item()):
                raise ArtifactError("NVFP4 source weights must be finite and representable in FP32")
            if bool(((original != 0) & (chunk == 0)).any().item()):
                raise ArtifactError("NVFP4 source weights underflow FP32")
            maximum = max(maximum, float(chunk.abs().amax().item()))
        global_scale = _resolve_global_scale(maximum, global_scale)
        packed = np.empty((shape[0], shape[1] // 2), dtype=np.uint8)
        scales = np.empty((shape[0], shape[1] // BLOCK_SIZE), dtype=np.uint8)
        boundaries = torch.tensor(_MIDPOINTS, device=target, dtype=torch.float32)
        for start in range(0, shape[0], rows_per_chunk):
            chunk = source[start : start + rows_per_chunk].to(device=target, dtype=torch.float32)
            blocks = chunk.reshape(chunk.shape[0], -1, BLOCK_SIZE)
            maxima = blocks.abs().amax(dim=-1)
            ideal = ((maxima / 6.0) * global_scale).clamp(min=1 / 512, max=448)
            ideal = torch.where(maxima == 0, torch.ones_like(ideal), ideal)
            block_scales = ideal.to(torch.float8_e4m3fn)
            normalized = (blocks / block_scales.float().unsqueeze(-1)) * global_scale
            magnitude = normalized.abs().contiguous()
            codes = torch.bucketize(magnitude, boundaries).to(torch.uint8)
            midpoint = boundaries[codes.clamp(max=6).long()]
            codes += ((codes < 7) & ((codes & 1) == 1) & (magnitude == midpoint)).to(torch.uint8)
            codes |= torch.signbit(normalized).to(torch.uint8) << 3
            codes = codes.reshape(chunk.shape)
            result = codes[:, ::2] | (codes[:, 1::2] << 4)
            end = start + chunk.shape[0]
            packed[start:end] = result.cpu().numpy()
            scales[start:end] = block_scales.view(torch.uint8).cpu().numpy()
    backend = "torch-cuda" if target.type == "cuda" else "torch-reference"
    return Nvfp4Tensor(packed, scales, global_scale, shape, backend)


def _resolve_global_scale(maximum: float, override: float | None) -> float:
    own = _global_scale(maximum)
    if override is None:
        return own
    np = _numpy()
    if not math.isfinite(override) or not 0 < override <= float(np.finfo(np.float32).max):
        raise ArtifactError("NVFP4 shared inverse global scale must be positive finite FP32")
    scale = float(np.float32(override))
    if scale == 0 or (maximum > 0 and scale > own * (1 + 1e-6)):
        raise ArtifactError("NVFP4 shared inverse scale would clip the source tensor")
    return scale


def dequantize_nvfp4(tensor: Nvfp4Tensor) -> Any:
    """Decode checkpoint bytes for numerical validation without MLX or CUDA."""
    np = _numpy()
    rows, columns = _shape(tensor.shape)
    packed, scales = np.asarray(tensor.packed), np.asarray(tensor.scale_bytes)
    if packed.dtype != np.uint8 or packed.shape != (rows, columns // 2):
        raise ArtifactError("invalid NVFP4 packed byte shape or dtype")
    if scales.dtype != np.uint8 or scales.shape != (rows, columns // BLOCK_SIZE):
        raise ArtifactError("invalid NVFP4 block-scale shape or dtype")
    if not math.isfinite(tensor.global_scale) or tensor.global_scale <= 0:
        raise ArtifactError("invalid NVFP4 inverse global scale")
    if np.any(scales == 0) or np.any(scales > 126):
        raise ArtifactError("NVFP4 block scales must be positive finite E4M3 values")
    codes = np.empty((rows, columns), dtype=np.uint8)
    codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
    decoded = np.asarray(_FP4_VALUES, dtype=np.float32)[codes & 7]
    decoded = np.where(codes & 8, -decoded, decoded).reshape(rows, -1, BLOCK_SIZE)
    # Keep intermediate scale arithmetic in FP64 for tiny source weights.
    output = decoded.astype(np.float64) * _scale_values(np)[scales][..., None]
    return (output / tensor.global_scale).reshape(rows, columns).astype(np.float32)
