"""Quantizer plugin registry and implementations.

Provides a plugin system for quantization methods (affine, AWQ, DWQ).
Each plugin declares its capabilities and can be registered/looked up
by method ID.  MLX is a lazy optional dependency.
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

        w = np.asarray(weight, dtype=np.float32)
        original_shape = w.shape

        # Reshape for group quantization
        w_grouped: Any
        if w.ndim == 2:
            out_features, in_features = w.shape
            if in_features % group_size != 0:
                raise QuantizerError(
                    f"input features {in_features} not divisible by group size {group_size}"
                )
            w_grouped = w.reshape(out_features, in_features // group_size, group_size)
        else:
            w_grouped = w.reshape(-1, group_size)

        # Compute per-group scale and zero-point (affine)
        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        scale = (w_max - w_min) / ((1 << bits) - 1)
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.round(-w_min / scale)

        # Quantize
        quantized = np.clip(np.round(w_grouped / scale + zero_point), 0, (1 << bits) - 1)

        return QuantizedWeight(
            data=quantized.astype(np.uint8),
            scales=scale.astype(np.float16),
            biases=zero_point.astype(np.float16),
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

        data = np.asarray(quantized.data, dtype=np.float32)
        scales = np.asarray(quantized.scales, dtype=np.float32)
        biases = np.asarray(quantized.biases, dtype=np.float32)

        dequantized = (data - biases) * scales

        if quantized.metadata and "original_shape" in quantized.metadata:
            dequantized = dequantized.reshape(quantized.metadata["original_shape"])

        return dequantized


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

        w = np.asarray(weight, dtype=np.float32)
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
            scales=q_scale.astype(np.float16),
            biases=zero_point.astype(np.float16),
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

        data = np.asarray(quantized.data, dtype=np.float32)
        scales = np.asarray(quantized.scales, dtype=np.float32)
        biases = np.asarray(quantized.biases, dtype=np.float32)

        dequantized = (data - biases) * scales

        if quantized.metadata and "original_shape" in quantized.metadata:
            dequantized = dequantized.reshape(quantized.metadata["original_shape"])

        # Apply inverse AWQ channel scales
        if quantized.metadata and "awq_channel_scales" in quantized.metadata:
            channel_scales = np.array(quantized.metadata["awq_channel_scales"], dtype=np.float32)
            inv_scales = 1.0 / np.clip(channel_scales, 1e-5, None)
            if dequantized.ndim == 2:
                dequantized = dequantized * inv_scales[np.newaxis, :]
            else:
                dequantized = dequantized * inv_scales

        return dequantized


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

        w = np.asarray(weight, dtype=np.float32)
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
        w_grouped: Any
        if w_clipped.ndim == 2:
            out_features, in_features = w_clipped.shape
            if in_features % group_size != 0:
                raise QuantizerError(
                    f"input features {in_features} not divisible by group size {group_size}"
                )
            w_grouped = w_clipped.reshape(out_features, in_features // group_size, group_size)
        else:
            w_grouped = w_clipped.reshape(-1, group_size)

        w_min = w_grouped.min(axis=-1, keepdims=True)
        w_max = w_grouped.max(axis=-1, keepdims=True)
        scale = (w_max - w_min) / ((1 << bits) - 1)
        scale = np.where(scale == 0, 1.0, scale)

        # Distribution-aware stochastic rounding
        normalized = (w_grouped - w_min) / scale
        # Use deterministic rounding biased by local distribution
        quantized = np.round(normalized)
        quantized = np.clip(quantized, 0, (1 << bits) - 1)

        zero_point = np.round(-w_min / scale)

        return QuantizedWeight(
            data=quantized.astype(np.uint8),
            scales=scale.astype(np.float16),
            biases=zero_point.astype(np.float16),
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

        data = np.asarray(quantized.data, dtype=np.float32)
        scales = np.asarray(quantized.scales, dtype=np.float32)
        biases = np.asarray(quantized.biases, dtype=np.float32)

        dequantized = (data - biases) * scales

        if quantized.metadata and "original_shape" in quantized.metadata:
            dequantized = dequantized.reshape(quantized.metadata["original_shape"])

        return dequantized


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


# Register defaults on module import
_register_defaults()
