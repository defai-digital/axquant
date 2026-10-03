"""Independent OCP MXFP6 reference codec; no native inference backend is implied."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from axquant.errors import BackendUnavailableError, QuantizerError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

Mxfp6Format = Literal["e2m3", "e3m2"]
GROUP_SIZE = 32
PACKED_BLOCK_BYTES = 24


def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailableError(
            "MXFP6 reference operations require the axquant[mxfp6] extra"
        ) from exc


def _format_parameters(element_format: Mxfp6Format) -> tuple[int, int]:
    if element_format == "e2m3":
        return 3, 1
    if element_format == "e3m2":
        return 2, 3
    raise QuantizerError(f"unsupported MXFP6 element format: {element_format}")


def fp6_values(element_format: Mxfp6Format) -> NDArray[np.float64]:
    """Decode all 64 FP6 codes, including both signed zeros and subnormals."""
    _require_numpy()
    import numpy as np

    mantissa_bits, bias = _format_parameters(element_format)
    codes = np.arange(64, dtype=np.uint8)
    exponent = (codes & 31) >> mantissa_bits
    mantissa = codes & ((1 << mantissa_bits) - 1)
    fraction = mantissa.astype(np.float64) / (1 << mantissa_bits)
    values = np.ldexp(
        np.where(exponent == 0, fraction, 1.0 + fraction),
        np.maximum(exponent.astype(np.int32), 1) - bias,
    )
    return cast("NDArray[np.float64]", np.copysign(values, np.where(codes & 32, -1.0, 1.0)))


def encode_fp6(values: NDArray[np.generic], element_format: Mxfp6Format) -> NDArray[np.uint8]:
    """Round finite real values to nearest FP6, ties to even, with saturation."""
    _require_numpy()
    import numpy as np

    values = np.asarray(values)
    if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        raise QuantizerError("FP6 encoding requires finite floating-point values")
    table = fp6_values(element_format)[:32]
    magnitudes = np.abs(values.astype(np.float64))
    upper = np.minimum(np.searchsorted(table, magnitudes), 31)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitudes - table[lower]
    upper_distance = table[upper] - magnitudes
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    codes = np.where(choose_upper, upper, lower).astype(np.uint8)
    return cast("NDArray[np.uint8]", codes | (np.signbit(values).astype(np.uint8) << 5))


def pack_fp6(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack consecutive groups of four 6-bit codes into three little-endian bytes."""
    _require_numpy()
    import numpy as np

    codes = np.asarray(codes)
    if (
        codes.dtype != np.uint8
        or codes.ndim < 1
        or any(size <= 0 for size in codes.shape)
        or codes.shape[-1] % GROUP_SIZE
    ):
        raise QuantizerError("FP6 codes must be uint8 with a last dimension divisible by 32")
    if np.any(codes > 63):
        raise QuantizerError("FP6 codes must fit in six bits")
    groups = codes.reshape(*codes.shape[:-1], -1, 4).astype(np.uint32)
    words = groups[..., 0] | (groups[..., 1] << 6) | (groups[..., 2] << 12)
    words |= groups[..., 3] << 18
    packed = np.stack((words & 255, (words >> 8) & 255, (words >> 16) & 255), axis=-1)
    return packed.astype(np.uint8).reshape(*codes.shape[:-1], codes.shape[-1] * 3 // 4)


def unpack_fp6(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
    _require_numpy()
    import numpy as np

    packed = np.asarray(packed)
    if (
        packed.dtype != np.uint8
        or packed.ndim < 1
        or any(size <= 0 for size in packed.shape)
        or packed.shape[-1] % PACKED_BLOCK_BYTES
    ):
        raise QuantizerError("packed FP6 must be uint8 with a last dimension divisible by 24")
    groups = packed.reshape(*packed.shape[:-1], -1, 3).astype(np.uint32)
    words = groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16)
    codes = np.stack(tuple((words >> shift) & 63 for shift in (0, 6, 12, 18)), axis=-1)
    return codes.astype(np.uint8).reshape(*packed.shape[:-1], packed.shape[-1] * 4 // 3)


@dataclass(frozen=True)
class Mxfp6Tensor:
    packed: NDArray[np.uint8]
    scales: NDArray[np.uint8]
    shape: tuple[int, ...]
    element_format: Mxfp6Format


def quantize_mxfp6(
    weights: NDArray[np.generic], *, element_format: Mxfp6Format = "e2m3"
) -> Mxfp6Tensor:
    """Quantize rows in blocks of 32 using independent E8M0 shared scales.

    The scale exponent is floor(log2(amax)) minus the format's maximum normal
    exponent, clamped to [-127, 127]. Zero blocks use exponent zero. FP6
    overflow saturates; NaN/Inf input is rejected rather than emitting evidence.
    """
    _require_numpy()
    import numpy as np

    weights = np.asarray(weights)
    _format_parameters(element_format)
    if weights.ndim < 2 or any(size <= 0 for size in weights.shape):
        raise QuantizerError("MXFP6 requires nonempty tensors with at least two dimensions")
    if weights.shape[-1] % GROUP_SIZE:
        raise QuantizerError("MXFP6 requires a last dimension divisible by 32")
    if weights.dtype.kind != "f" or not np.all(np.isfinite(weights)):
        raise QuantizerError("MXFP6 requires finite floating-point weights")
    if np.any(np.abs(weights) > np.finfo(np.float32).max):
        raise QuantizerError("MXFP6 reference input must fit in finite float32")
    blocks = weights.astype(np.float64).reshape(*weights.shape[:-1], -1, GROUP_SIZE)
    maximum = np.max(np.abs(blocks), axis=-1)
    _, exponent = np.frexp(maximum)
    max_normal_exponent = 2 if element_format == "e2m3" else 4
    exponent = np.where(maximum == 0, 0, exponent - 1 - max_normal_exponent)
    exponent = np.clip(exponent, -127, 127).astype(np.int32)
    normalized = np.ldexp(blocks, -exponent[..., None])
    codes = encode_fp6(normalized, element_format).reshape(weights.shape)
    return Mxfp6Tensor(
        packed=pack_fp6(codes),
        scales=(exponent + 127).astype(np.uint8),
        shape=tuple(weights.shape),
        element_format=element_format,
    )


def dequantize_mxfp6(tensor: Mxfp6Tensor) -> NDArray[np.float32]:
    """Reference decode; finite float32 overflow saturates and E8M0 NaN is rejected."""
    _require_numpy()
    import numpy as np

    shape = tensor.shape
    if (
        len(shape) < 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
        or shape[-1] % GROUP_SIZE
    ):
        raise QuantizerError("invalid MXFP6 logical shape")
    packed_shape = (*shape[:-1], shape[-1] * 3 // 4)
    scale_shape = (*shape[:-1], shape[-1] // GROUP_SIZE)
    if tensor.packed.shape != packed_shape:
        raise QuantizerError("packed MXFP6 shape does not match the logical shape")
    if tensor.scales.shape != scale_shape or tensor.scales.dtype != np.uint8:
        raise QuantizerError("MXFP6 scales must be uint8 with one scale per block of 32")
    if np.any(tensor.scales == 255):
        raise QuantizerError("E8M0 NaN scales are not valid finite-weight artifacts")
    codes = unpack_fp6(tensor.packed)
    table = fp6_values(tensor.element_format)
    blocks = table[codes].reshape(*shape[:-1], -1, GROUP_SIZE)
    decoded = np.ldexp(blocks, tensor.scales.astype(np.int32)[..., None] - 127)
    maximum = float(np.finfo(np.float32).max)
    return np.clip(decoded, -maximum, maximum).reshape(shape).astype(np.float32)


def mxfp6_storage_bytes(shape: tuple[int, ...]) -> int:
    if (
        len(shape) < 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
        or shape[-1] % GROUP_SIZE
    ):
        raise QuantizerError("MXFP6 storage requires a nonempty block-aligned matrix shape")
    return math.prod(shape) // GROUP_SIZE * (PACKED_BLOCK_BYTES + 1)
