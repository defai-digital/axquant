"""Portable AWQ activation scaling shared by the plugin, predicate, and conversion.

AWQ searches per-channel scales from calibration activations, quantizes in the
scaled space, then unscales the dequantized reconstruction so the refined float
weights approximate the original matrix. Conversion packs those refined floats
with the same portable affine contract MLX-LM already accepts for affine/DWQ.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from axquant.errors import PlanningError, QuantizerError
from axquant.numeric import as_finite_float32_matrix
from axquant.package_data import load_package_yaml


def _load_awq_defaults() -> tuple[tuple[float, ...], frozenset[int], frozenset[int]]:
    raw = load_package_yaml("quantizer_defaults.yaml")
    if not isinstance(raw, dict) or not isinstance(raw.get("awq"), dict):
        raise PlanningError("quantizer_defaults.yaml missing awq section")
    section = raw["awq"]
    grid = section.get("alpha_grid")
    bits = section.get("supported_bits")
    groups = section.get("supported_group_sizes")
    if not isinstance(grid, list) or not grid:
        raise PlanningError("quantizer_defaults.yaml awq.alpha_grid is invalid")
    if not isinstance(bits, list) or not bits:
        raise PlanningError("quantizer_defaults.yaml awq.supported_bits is invalid")
    if not isinstance(groups, list) or not groups:
        raise PlanningError("quantizer_defaults.yaml awq.supported_group_sizes is invalid")
    return (
        tuple(float(value) for value in grid),
        frozenset(int(value) for value in bits),
        frozenset(int(value) for value in groups),
    )


_DEFAULT_ALPHA_GRID, _SUPPORTED_BITS, _SUPPORTED_GROUP_SIZES = _load_awq_defaults()


def _as_numpy(weight: Any) -> Any:
    return as_finite_float32_matrix(weight, component="AWQ")


def _resolve_alpha_grid(value: Any) -> tuple[float, ...]:
    try:
        raw_values = tuple(value)
    except TypeError as exc:
        raise PlanningError("AWQ alpha_grid must contain numeric values") from exc
    resolved: list[float] = []
    for raw in raw_values:
        if isinstance(raw, bool):
            raise PlanningError("AWQ alpha_grid must contain numeric values")
        try:
            alpha = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PlanningError("AWQ alpha_grid must contain numeric values") from exc
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise PlanningError("AWQ alpha_grid values must be finite and within [0, 1]")
        resolved.append(alpha)
    if not resolved:
        raise PlanningError("AWQ alpha_grid must not be empty")
    return tuple(resolved)


def learn_awq_channel_scales(
    weight: Any,
    activations: Any,
    *,
    bits: int,
    group_size: int,
    alpha_grid: tuple[float, ...] = _DEFAULT_ALPHA_GRID,
) -> tuple[Any, dict[str, float | int | list[float]]]:
    """Search AWQ channel scales using the same reconstruction objective as AwqPlugin."""
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("AWQ execution requires numpy") from exc

    if bits not in _SUPPORTED_BITS:
        raise PlanningError(f"AWQ does not support {bits}-bit quantization")
    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise PlanningError(f"AWQ does not support group size {group_size}")
    if activations is None:
        raise PlanningError("AWQ requires calibration activations")
    w = _as_numpy(weight)
    if w.ndim != 2:
        raise PlanningError("AWQ currently requires a two-dimensional weight matrix")
    calibration_config = activations if isinstance(activations, dict) else {}
    activation_value = calibration_config.get("activations", activations)
    try:
        act = np.asarray(activation_value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanningError(f"AWQ calibration activations are not numeric: {exc}") from exc
    if act.ndim == 0 or act.size == 0 or act.shape[-1] != w.shape[-1]:
        raise PlanningError("AWQ calibration channels must match the weight input dimension")
    if not bool(np.all(np.isfinite(act))):
        raise PlanningError("AWQ calibration activations must contain only finite values")
    alpha_grid_value = calibration_config.get("alpha_grid", alpha_grid)
    resolved_grid = _resolve_alpha_grid(alpha_grid_value)

    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise PlanningError(
            f"input features {in_features} not divisible by group size {group_size}"
        )

    if act.ndim == 2:
        channel_magnitudes = np.mean(np.abs(act), axis=0)
    else:
        channel_magnitudes = np.mean(np.abs(act.reshape(-1, in_features)), axis=0)
    if not bool(np.all(np.isfinite(channel_magnitudes))):
        raise PlanningError("AWQ activation statistics are non-finite")
    observed_max = float(np.max(channel_magnitudes))
    max_mag = observed_max if observed_max > 0 else 1.0
    activation_basis = np.clip(channel_magnitudes / max_mag, 1e-5, None)
    activation_rows = act.reshape(-1, in_features)[:256]
    reference_output = activation_rows @ w.T
    if not bool(np.all(np.isfinite(reference_output))):
        raise PlanningError("AWQ reference output is non-finite")

    best: tuple[float, float, Any] | None = None
    for alpha in resolved_grid:
        channel_scales = np.clip(activation_basis**alpha, 1e-5, None)
        w_scaled = w * channel_scales[np.newaxis, :]
        w_grouped = w_scaled.reshape(out_features, in_features // group_size, group_size)
        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        q_scale = (w_max - w_min) / ((1 << bits) - 1)
        q_scale = np.where(q_scale == 0, 1.0, q_scale)
        zero_point = np.round(-w_min / q_scale)
        quantized = np.clip(
            np.round(w_grouped / q_scale + zero_point),
            0,
            (1 << bits) - 1,
        )
        reconstructed = ((quantized - zero_point) * q_scale).reshape(w.shape)
        reconstructed = reconstructed / channel_scales[np.newaxis, :]
        candidate_output = activation_rows @ reconstructed.T
        reconstruction_mse = float(np.mean((reference_output - candidate_output) ** 2))
        if not math.isfinite(reconstruction_mse):
            raise PlanningError("AWQ reconstruction objective is non-finite")
        candidate = (reconstruction_mse, alpha, channel_scales)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise PlanningError("AWQ scale search produced no candidates")
    reconstruction_mse, alpha, scales_per_channel = best
    metadata: dict[str, float | int | list[float]] = {
        "awq_alpha": float(alpha),
        "awq_channel_scales": scales_per_channel.astype(np.float16).tolist(),
        "activation_reconstruction_mse": reconstruction_mse,
        "calibration_rows": len(activation_rows),
        "bits": bits,
        "group_size": group_size,
    }
    return scales_per_channel.astype(np.float32), metadata


def refine_weight_with_awq(
    weight: Any,
    activations: Any,
    *,
    bits: int,
    group_size: int,
    alpha_grid: tuple[float, ...] = _DEFAULT_ALPHA_GRID,
) -> tuple[Any, dict[str, float | int | list[float]]]:
    """Apply AWQ scale search and return the unscaled float reconstruction.

    The refined matrix approximates the source weights after AWQ-aware rounding
    in the scaled domain. Conversion then packs it with portable affine mode.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("AWQ execution requires numpy") from exc

    w = _as_numpy(weight)
    channel_scales, metadata = learn_awq_channel_scales(
        w,
        activations,
        bits=bits,
        group_size=group_size,
        alpha_grid=alpha_grid,
    )
    out_features, in_features = w.shape
    w_scaled = w * channel_scales[np.newaxis, :]
    w_grouped = w_scaled.reshape(out_features, in_features // group_size, group_size)
    w_min = w_grouped.min(axis=-1, keepdims=True)
    w_max = w_grouped.max(axis=-1, keepdims=True)
    q_scale = (w_max - w_min) / ((1 << bits) - 1)
    q_scale = np.where(q_scale == 0, 1.0, q_scale)
    zero_point = np.round(-w_min / q_scale)
    quantized = np.clip(
        np.round(w_grouped / q_scale + zero_point),
        0,
        (1 << bits) - 1,
    )
    reconstructed = ((quantized - zero_point) * q_scale).reshape(w.shape)
    refined = reconstructed / channel_scales[np.newaxis, :]
    if not bool(np.all(np.isfinite(refined))):
        raise PlanningError("AWQ refinement produced non-finite weights")
    return refined.astype(np.float32), metadata


def apply_channel_scales(weight: Any, channel_scales: Any) -> Any:
    """Multiply a 2-D weight matrix by per-input-channel scales."""
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("AWQ execution requires numpy") from exc
    w = _as_numpy(weight)
    scales = np.asarray(channel_scales, dtype=np.float32)
    if w.ndim != 2:
        raise PlanningError("AWQ channel scaling requires a two-dimensional weight matrix")
    if scales.ndim != 1 or scales.shape[0] != w.shape[-1]:
        raise PlanningError("AWQ channel scales must match the weight input dimension")
    if not bool(np.all(np.isfinite(scales))) or bool(np.any(scales <= 0)):
        raise PlanningError("AWQ channel scales must contain finite positive values")
    scaled = w * scales[np.newaxis, :]
    if not bool(np.all(np.isfinite(scaled))):
        raise PlanningError("AWQ channel scaling produced non-finite weights")
    return scaled.astype(np.float32)


def apply_mlx_awq_scale(
    module: Any,
    *,
    activations: Any,
    bits: int,
    group_size: int,
    alpha_grid: tuple[float, ...] = _DEFAULT_ALPHA_GRID,
) -> dict[str, float | int | list[float]]:
    """Mutate ``module.weight`` with portable AWQ refinement before affine packing."""
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise PlanningError("AWQ execution requires MLX") from exc
    weight = getattr(module, "weight", None)
    if weight is None:
        raise PlanningError("AWQ requires a module with a weight tensor")
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("AWQ execution requires numpy") from exc

    try:
        # bfloat16 arrays reject numpy's buffer protocol; cast via MLX then.
        try:
            weight_f32 = np.asarray(weight, dtype=np.float32)
        except (RuntimeError, TypeError):
            weight_f32 = np.asarray(mx.array(weight, dtype=mx.float32))
        refined, metadata = refine_weight_with_awq(
            weight_f32,
            activations,
            bits=bits,
            group_size=group_size,
            alpha_grid=alpha_grid,
        )
    except (PlanningError, QuantizerError, ValueError, TypeError, RuntimeError) as exc:
        raise PlanningError(str(exc)) from exc
    try:
        candidate = mx.array(refined, dtype=weight.dtype)
        mx.eval(candidate)
    except Exception as exc:
        raise PlanningError(f"AWQ MLX weight materialization failed: {exc}") from exc
    module.weight = candidate
    return metadata
