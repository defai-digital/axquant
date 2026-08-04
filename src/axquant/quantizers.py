"""Quantizer plugin registry and implementations.

Provides a plugin system for quantization methods (affine, AWQ, DWQ, GPTQ).
Each plugin declares its capabilities and can be registered/looked up
by method ID.  MLX is a lazy optional dependency.

NOT WIRED INTO CONVERSION: this module is a standalone, numpy-only reference
implementation kept for its algorithmic documentation and its own test
coverage (`tests/test_quantizers.py`). The real conversion path quantizes
through MLX-LM's `quant_predicate` (`predicate.py`/`converter.py`), and real
AWQ/DWQ refinement lives in `awq.py`/`dwq.py`. Nothing outside this module
and its test imports `AffinePlugin`/`AwqPlugin`/`DwqPlugin`/`GptqPlugin` or the plugin
registry; do not assume registering a plugin here makes it reachable from
`axquant convert`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import structlog

from axquant.errors import QuantizerError
from axquant.schema import QuantizerExecutionRecord, QuantMethod

log = structlog.get_logger()


@dataclass(frozen=True)
class QuantizedWeight:
    """Result of a quantization operation."""

    data: Any
    scales: Any | None = None
    biases: Any | None = None
    bits: int = 4
    group_size: int = 64
    method: QuantMethod = QuantMethod.AFFINE
    metadata: dict[str, Any] | None = None


def _finite_weight_matrix(np: Any, weight: Any, label: str) -> Any:
    try:
        result = np.asarray(weight, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuantizerError(f"{label} weight tensor is not numeric: {exc}") from exc
    if result.ndim != 2:
        raise QuantizerError(f"{label} quantization requires a two-dimensional weight matrix")
    if result.size == 0:
        raise QuantizerError(f"{label} weight matrix must not be empty")
    if not bool(np.all(np.isfinite(result))):
        raise QuantizerError(f"{label} weight matrix must contain only finite values")
    return result


def _stored_affine_parameters(np: Any, scale: Any, zero_point: Any, label: str) -> tuple[Any, Any]:
    if (
        not bool(np.all(np.isfinite(scale)))
        or bool(np.any(scale <= 0))
        or not bool(np.all(np.isfinite(zero_point)))
    ):
        raise QuantizerError(f"{label} produced invalid affine parameters")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored_scale = scale.astype(np.float16)
        stored_zero = zero_point.astype(np.float16)
    if (
        not bool(np.all(np.isfinite(stored_scale)))
        or bool(np.any(stored_scale <= 0))
        or not bool(np.all(np.isfinite(stored_zero)))
    ):
        raise QuantizerError(f"{label} affine parameters are not representable in float16")
    return stored_scale, stored_zero


def _dequantize_affine(
    np: Any,
    quantized: QuantizedWeight,
    *,
    expected_method: QuantMethod,
    supported_bits: tuple[int, ...],
    supported_group_sizes: tuple[int, ...],
    label: str,
) -> Any:
    if quantized.method != expected_method:
        actual_method = (
            quantized.method.value
            if isinstance(quantized.method, QuantMethod)
            else str(quantized.method)
        )
        raise QuantizerError(f"{label} cannot dequantize method {actual_method!r}")
    if quantized.bits not in supported_bits:
        raise QuantizerError(f"{label} quantized weight has unsupported bit width")
    if quantized.group_size not in supported_group_sizes:
        raise QuantizerError(f"{label} quantized weight has unsupported group size")
    if quantized.scales is None or quantized.biases is None:
        raise QuantizerError(f"{label} quantized weight is missing affine parameters")
    try:
        data = np.asarray(quantized.data, dtype=np.float32)
        scales = np.asarray(quantized.scales, dtype=np.float32)
        biases = np.asarray(quantized.biases, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuantizerError(f"{label} quantized payload is not numeric: {exc}") from exc
    metadata = quantized.metadata
    original_shape = metadata.get("original_shape") if isinstance(metadata, dict) else None
    if (
        not isinstance(original_shape, (list, tuple))
        or len(original_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in original_shape
        )
    ):
        raise QuantizerError(f"{label} quantized weight has an invalid original shape")
    out_features, in_features = original_shape
    if in_features % quantized.group_size != 0:
        raise QuantizerError(f"{label} original shape is incompatible with its group size")
    expected_data_shape = (
        out_features,
        in_features // quantized.group_size,
        quantized.group_size,
    )
    expected_parameter_shape = (*expected_data_shape[:-1], 1)
    if data.shape != expected_data_shape:
        raise QuantizerError(f"{label} quantized code shape does not match its metadata")
    if scales.shape != expected_parameter_shape or biases.shape != expected_parameter_shape:
        raise QuantizerError(f"{label} affine parameter shapes do not match its codes")
    qmax = (1 << quantized.bits) - 1
    if (
        not bool(np.all(np.isfinite(data)))
        or bool(np.any(data < 0))
        or bool(np.any(data > qmax))
        or not bool(np.all(data == np.round(data)))
    ):
        raise QuantizerError(f"{label} quantized codes are invalid")
    if (
        not bool(np.all(np.isfinite(scales)))
        or bool(np.any(scales <= 0))
        or not bool(np.all(np.isfinite(biases)))
    ):
        raise QuantizerError(f"{label} affine parameters are invalid")
    dequantized = (data - biases) * scales
    if not bool(np.all(np.isfinite(dequantized))):
        raise QuantizerError(f"{label} dequantization produced non-finite values")
    return dequantized.reshape(original_shape)


@runtime_checkable
class QuantizerPlugin(Protocol):
    """Protocol for quantizer plugins."""

    @property
    def method_id(self) -> QuantMethod:
        """The quantization method this plugin implements."""
        ...

    @property
    def supported_bits(self) -> tuple[int, ...]:
        """Supported bit widths."""
        ...

    @property
    def supported_group_sizes(self) -> tuple[int, ...]:
        """Supported group sizes."""
        ...

    @property
    def requires_calibration(self) -> bool:
        """Whether this plugin requires calibration data."""
        ...

    def quantize(
        self,
        weight: Any,
        *,
        bits: int,
        group_size: int,
        calibration: Any | None = None,
    ) -> QuantizedWeight:
        """Quantize a weight tensor."""
        ...

    def dequantize(self, quantized: QuantizedWeight) -> Any:
        """Dequantize back to full precision."""
        ...


class _PluginRegistry:
    """Internal plugin registry."""

    def __init__(self) -> None:
        self._plugins: dict[QuantMethod, QuantizerPlugin] = {}

    def register(self, plugin: QuantizerPlugin) -> None:
        method = plugin.method_id
        if method in self._plugins:
            raise QuantizerError(f"quantizer plugin already registered for method: {method}")
        self._plugins[method] = plugin
        log.debug("quantizer_registered", method=method.value)

    def get(self, method: QuantMethod) -> QuantizerPlugin | None:
        return self._plugins.get(method)

    def require(self, method: QuantMethod) -> QuantizerPlugin:
        plugin = self._plugins.get(method)
        if plugin is None:
            raise QuantizerError(
                f"no quantizer plugin registered for method '{method.value}'; "
                "plans using this method cannot be converted until the plugin is available"
            )
        return plugin

    def is_registered(self, method: QuantMethod) -> bool:
        return method in self._plugins

    def registered_methods(self) -> list[QuantMethod]:
        return list(self._plugins.keys())


# Global registry instance
registry = _PluginRegistry()


def register_plugin(plugin: QuantizerPlugin) -> None:
    """Register a quantizer plugin in the global registry."""
    registry.register(plugin)


def get_plugin(method: QuantMethod) -> QuantizerPlugin | None:
    """Look up a plugin by method, returning None if not registered."""
    return registry.get(method)


def require_plugin(method: QuantMethod) -> QuantizerPlugin:
    """Look up a plugin by method, raising QuantizerError if not registered."""
    return registry.require(method)


def is_method_available(method: QuantMethod) -> bool:
    """Check whether a quantization method has a registered plugin."""
    return registry.is_registered(method)


class AffinePlugin:
    """Standard affine quantization plugin (passthrough to MLX-LM)."""

    @property
    def method_id(self) -> QuantMethod:
        return QuantMethod.AFFINE

    @property
    def supported_bits(self) -> tuple[int, ...]:
        return (4, 6, 8)

    @property
    def supported_group_sizes(self) -> tuple[int, ...]:
        return (32, 64, 128)

    @property
    def requires_calibration(self) -> bool:
        return False

    def quantize(
        self,
        weight: Any,
        *,
        bits: int,
        group_size: int,
        calibration: Any | None = None,
    ) -> QuantizedWeight:
        if bits not in self.supported_bits:
            raise QuantizerError(f"affine plugin does not support {bits}-bit quantization")
        if group_size not in self.supported_group_sizes:
            raise QuantizerError(f"affine plugin does not support group size {group_size}")

        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("affine quantization requires numpy") from None

        w = _finite_weight_matrix(np, weight, "affine")
        original_shape = w.shape

        # Reshape for group quantization
        out_features, in_features = w.shape
        if in_features % group_size != 0:
            raise QuantizerError(
                f"input features {in_features} not divisible by group size {group_size}"
            )
        w_grouped = w.reshape(out_features, in_features // group_size, group_size)

        # Compute per-group scale and zero-point (affine)
        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        scale = (w_max - w_min) / ((1 << bits) - 1)
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.round(-w_min / scale)
        stored_scale, stored_zero = _stored_affine_parameters(np, scale, zero_point, "affine")

        # Quantize
        quantized = np.clip(np.round(w_grouped / scale + zero_point), 0, (1 << bits) - 1)

        return QuantizedWeight(
            data=quantized.astype(np.uint8),
            scales=stored_scale,
            biases=stored_zero,
            bits=bits,
            group_size=group_size,
            method=QuantMethod.AFFINE,
            metadata={"original_shape": list(original_shape)},
        )

    def dequantize(self, quantized: QuantizedWeight) -> Any:
        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("dequantization requires numpy") from None

        return _dequantize_affine(
            np,
            quantized,
            expected_method=self.method_id,
            supported_bits=self.supported_bits,
            supported_group_sizes=self.supported_group_sizes,
            label="affine",
        )


class AwqPlugin:
    """Activation-Aware Weight Quantization plugin.

    Computes per-channel scaling from activation magnitudes, applies
    scale before affine quantization, and inverse scale after.
    Requires calibration activations. Shares the portable scale search
    with convert-time AWQ refinement in ``axquant.awq``.
    """

    @property
    def method_id(self) -> QuantMethod:
        return QuantMethod.AWQ

    @property
    def supported_bits(self) -> tuple[int, ...]:
        return (4, 6, 8)

    @property
    def supported_group_sizes(self) -> tuple[int, ...]:
        return (32, 64, 128)

    @property
    def requires_calibration(self) -> bool:
        return True

    def quantize(
        self,
        weight: Any,
        *,
        bits: int,
        group_size: int,
        calibration: Any | None = None,
    ) -> QuantizedWeight:
        if bits not in self.supported_bits:
            raise QuantizerError(f"AWQ plugin does not support {bits}-bit quantization")
        if group_size not in self.supported_group_sizes:
            raise QuantizerError(f"AWQ plugin does not support group size {group_size}")
        if calibration is None:
            raise QuantizerError("AWQ requires calibration activations")

        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("AWQ quantization requires numpy") from None

        from axquant.awq import learn_awq_channel_scales
        from axquant.errors import PlanningError

        w = _finite_weight_matrix(np, weight, "AWQ")
        original_shape = w.shape
        calibration_config = calibration if isinstance(calibration, dict) else {}
        alpha_grid_value = calibration_config.get(
            "alpha_grid",
            (0.0, 0.25, 0.5, 0.75, 1.0),
        )
        try:
            alpha_grid = tuple(float(alpha) for alpha in alpha_grid_value)
        except (TypeError, ValueError):
            raise QuantizerError("AWQ alpha_grid must contain numeric values") from None
        try:
            channel_scales, search_meta = learn_awq_channel_scales(
                w,
                calibration,
                bits=bits,
                group_size=group_size,
                alpha_grid=alpha_grid,
            )
        except PlanningError as exc:
            raise QuantizerError(str(exc)) from exc

        out_features, in_features = w.shape
        w_scaled = w * channel_scales[np.newaxis, :]
        w_grouped = w_scaled.reshape(out_features, in_features // group_size, group_size)
        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        q_scale = (w_max - w_min) / ((1 << bits) - 1)
        q_scale = np.where(q_scale == 0, 1.0, q_scale)
        zero_point = np.round(-w_min / q_scale)
        stored_scale, stored_zero = _stored_affine_parameters(np, q_scale, zero_point, "AWQ")
        quantized = np.clip(
            np.round(w_grouped / q_scale + zero_point),
            0,
            (1 << bits) - 1,
        )

        channel_scales_meta = search_meta["awq_channel_scales"]
        if not isinstance(channel_scales_meta, list):
            raise QuantizerError("AWQ search metadata is missing channel scales")
        return QuantizedWeight(
            data=quantized.astype(np.uint8),
            scales=stored_scale,
            biases=stored_zero,
            bits=bits,
            group_size=group_size,
            method=QuantMethod.AWQ,
            metadata={
                "original_shape": list(original_shape),
                "awq_channel_scales": list(channel_scales_meta),
                "awq_alpha": float(search_meta["awq_alpha"]),  # type: ignore[arg-type]
                "awq_alpha_grid": list(alpha_grid),
                "calibration_rows": int(search_meta["calibration_rows"]),  # type: ignore[arg-type]
                "activation_reconstruction_mse": float(
                    search_meta["activation_reconstruction_mse"]  # type: ignore[arg-type]
                ),
            },
        )

    def dequantize(self, quantized: QuantizedWeight) -> Any:
        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("dequantization requires numpy") from None

        dequantized = _dequantize_affine(
            np,
            quantized,
            expected_method=self.method_id,
            supported_bits=self.supported_bits,
            supported_group_sizes=self.supported_group_sizes,
            label="AWQ",
        )

        # Apply inverse AWQ channel scales
        metadata = quantized.metadata
        channel_scale_value = (
            metadata.get("awq_channel_scales") if isinstance(metadata, dict) else None
        )
        try:
            channel_scales = np.asarray(channel_scale_value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QuantizerError(f"AWQ channel scale metadata is invalid: {exc}") from exc
        if (
            channel_scales.ndim != 1
            or channel_scales.shape[0] != dequantized.shape[-1]
            or not bool(np.all(np.isfinite(channel_scales)))
            or bool(np.any(channel_scales <= 0))
        ):
            raise QuantizerError("AWQ channel scale metadata is invalid")
        dequantized = dequantized / channel_scales[np.newaxis, :]
        if not bool(np.all(np.isfinite(dequantized))):
            raise QuantizerError("AWQ dequantization produced non-finite values")

        return dequantized


class GptqPlugin:
    """GPTQ second-order weight refinement plugin.

    Quantizes weight columns left-to-right onto the static per-group affine
    grid of the original weight, compensating each column's rounding error in
    the remaining columns via the damped inverse-Hessian Cholesky factor.
    Requires calibration activations. Shares the portable refinement with
    convert-time GPTQ refinement in ``axquant.gptq``.
    """

    @property
    def method_id(self) -> QuantMethod:
        return QuantMethod.GPTQ

    @property
    def supported_bits(self) -> tuple[int, ...]:
        return (2, 3, 4, 6, 8)

    @property
    def supported_group_sizes(self) -> tuple[int, ...]:
        return (32, 64, 128)

    @property
    def requires_calibration(self) -> bool:
        return True

    def quantize(
        self,
        weight: Any,
        *,
        bits: int,
        group_size: int,
        calibration: Any | None = None,
    ) -> QuantizedWeight:
        if bits not in self.supported_bits:
            raise QuantizerError(f"GPTQ plugin does not support {bits}-bit quantization")
        if group_size not in self.supported_group_sizes:
            raise QuantizerError(f"GPTQ plugin does not support group size {group_size}")
        if calibration is None:
            raise QuantizerError("GPTQ requires calibration activations")

        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("GPTQ quantization requires numpy") from None

        from axquant.errors import PlanningError
        from axquant.gptq import learn_gptq_refined_weight

        w = _finite_weight_matrix(np, weight, "GPTQ")
        original_shape = w.shape
        try:
            refined, refine_meta = learn_gptq_refined_weight(
                w,
                calibration,
                bits=bits,
                group_size=group_size,
            )
        except PlanningError as exc:
            raise QuantizerError(str(exc)) from exc

        # The refined weight sits on the static per-group grid of the original
        # weight, so re-deriving that grid makes the integer codes exact.
        out_features, in_features = w.shape
        w_grouped = w.reshape(out_features, in_features // group_size, group_size)
        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        q_scale = (w_max - w_min) / ((1 << bits) - 1)
        q_scale = np.where(q_scale == 0, 1.0, q_scale)
        zero_point = np.clip(np.round(-w_min / q_scale), 0, (1 << bits) - 1)
        stored_scale, stored_zero = _stored_affine_parameters(np, q_scale, zero_point, "GPTQ")
        refined_grouped = refined.reshape(out_features, in_features // group_size, group_size)
        quantized = np.clip(
            np.round(refined_grouped / q_scale) + zero_point,
            0,
            (1 << bits) - 1,
        )

        return QuantizedWeight(
            data=quantized.astype(np.uint8),
            scales=stored_scale,
            biases=stored_zero,
            bits=bits,
            group_size=group_size,
            method=QuantMethod.GPTQ,
            metadata={
                "original_shape": list(original_shape),
                "gptq_damping": float(refine_meta["gptq_damping"]),
                "calibration_rows": int(refine_meta["calibration_rows"]),
                "mean_quant_error": float(refine_meta["mean_quant_error"]),
            },
        )

    def dequantize(self, quantized: QuantizedWeight) -> Any:
        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("dequantization requires numpy") from None

        return _dequantize_affine(
            np,
            quantized,
            expected_method=self.method_id,
            supported_bits=self.supported_bits,
            supported_group_sizes=self.supported_group_sizes,
            label="GPTQ",
        )


class DwqPlugin:
    """Distribution-Wise (Data-Free) Weight Quantization plugin.

    Uses weight distribution statistics for rounding optimization.
    Does not require calibration data.
    """

    @property
    def method_id(self) -> QuantMethod:
        return QuantMethod.DWQ

    @property
    def supported_bits(self) -> tuple[int, ...]:
        return (4, 6, 8)

    @property
    def supported_group_sizes(self) -> tuple[int, ...]:
        return (32, 64, 128)

    @property
    def requires_calibration(self) -> bool:
        return False

    def quantize(
        self,
        weight: Any,
        *,
        bits: int,
        group_size: int,
        calibration: Any | None = None,
    ) -> QuantizedWeight:
        if bits not in self.supported_bits:
            raise QuantizerError(f"DWQ plugin does not support {bits}-bit quantization")
        if group_size not in self.supported_group_sizes:
            raise QuantizerError(f"DWQ plugin does not support group size {group_size}")

        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("DWQ quantization requires numpy") from None

        w = _finite_weight_matrix(np, weight, "DWQ")
        original_shape = w.shape

        # Distribution-aware clipping using percentile bounds
        # Instead of min/max, use percentile-based range to reduce outlier impact
        lower_pct = 0.1
        upper_pct = 99.9
        w_flat = w.flatten()
        w_lower = float(np.percentile(w_flat, lower_pct))
        w_upper = float(np.percentile(w_flat, upper_pct))

        # Clip to distribution bounds
        w_clipped = np.clip(w, w_lower, w_upper)

        # Group-wise quantization with distribution-aware rounding
        out_features, in_features = w_clipped.shape
        if in_features % group_size != 0:
            raise QuantizerError(
                f"input features {in_features} not divisible by group size {group_size}"
            )
        w_grouped = w_clipped.reshape(out_features, in_features // group_size, group_size)

        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        scale = (w_max - w_min) / ((1 << bits) - 1)
        scale = np.where(scale == 0, 1.0, scale)

        # Zero-point must be rounded first and then used directly in the
        # forward quantization (matching AffinePlugin). Rounding `w_min` only
        # implicitly, via `(w - w_min) / scale`, is not the algebraic inverse
        # of `dequantize`'s `(data - zero_point) * scale` once rounding is
        # involved, and silently produces round-trip errors beyond the
        # theoretical max of `scale / 2` for a correct affine quantizer.
        zero_point = np.round(-w_min / scale)
        stored_scale, stored_zero = _stored_affine_parameters(np, scale, zero_point, "DWQ")
        quantized = np.clip(np.round(w_grouped / scale + zero_point), 0, (1 << bits) - 1)

        return QuantizedWeight(
            data=quantized.astype(np.uint8),
            scales=stored_scale,
            biases=stored_zero,
            bits=bits,
            group_size=group_size,
            method=QuantMethod.DWQ,
            metadata={
                "original_shape": list(original_shape),
                "clip_lower": w_lower,
                "clip_upper": w_upper,
            },
        )

    def dequantize(self, quantized: QuantizedWeight) -> Any:
        try:
            import numpy as np
        except ImportError:
            raise QuantizerError("dequantization requires numpy") from None

        return _dequantize_affine(
            np,
            quantized,
            expected_method=self.method_id,
            supported_bits=self.supported_bits,
            supported_group_sizes=self.supported_group_sizes,
            label="DWQ",
        )


def record_execution(
    method: QuantMethod,
    module_path: str,
    bits: int,
    group_size: int | None,
    success: bool,
    *,
    fallback: bool = False,
    note: str | None = None,
) -> QuantizerExecutionRecord:
    """Create a quantizer execution record for audit trails."""
    return QuantizerExecutionRecord(
        method=method,
        module_path=module_path,
        bits=bits,
        group_size=group_size,
        success=success,
        fallback=fallback,
        note=note,
    )


def _register_defaults() -> None:
    """Register the default plugins."""
    registry.register(AffinePlugin())
    registry.register(AwqPlugin())
    registry.register(DwqPlugin())
    registry.register(GptqPlugin())


# Register defaults on module import
_register_defaults()
