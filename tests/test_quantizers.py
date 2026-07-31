"""Tests for the quantizer plugin system (v0.4)."""

from __future__ import annotations

import numpy as np
import pytest

from axquant.errors import QuantizerError
from axquant.quantizers import (
    AffinePlugin,
    AwqPlugin,
    DwqPlugin,
    _PluginRegistry,
    get_plugin,
    is_method_available,
    record_execution,
    registry,
    require_plugin,
)
from axquant.schema import QuantMethod


class TestPluginRegistry:
    def test_default_plugins_registered(self) -> None:
        assert is_method_available(QuantMethod.AFFINE)
        assert is_method_available(QuantMethod.AWQ)
        assert is_method_available(QuantMethod.DWQ)

    def test_bf16_not_registered(self) -> None:
        assert not is_method_available(QuantMethod.BF16)

    def test_gptq_not_registered(self) -> None:
        assert not is_method_available(QuantMethod.GPTQ)

    def test_get_plugin(self) -> None:
        plugin = get_plugin(QuantMethod.AFFINE)
        assert plugin is not None
        assert plugin.method_id == QuantMethod.AFFINE

    def test_get_missing_plugin(self) -> None:
        plugin = get_plugin(QuantMethod.GPTQ)
        assert plugin is None

    def test_require_plugin(self) -> None:
        plugin = require_plugin(QuantMethod.AWQ)
        assert plugin.method_id == QuantMethod.AWQ

    def test_require_missing_plugin_raises(self) -> None:
        with pytest.raises(QuantizerError, match="no quantizer plugin registered"):
            require_plugin(QuantMethod.GPTQ)

    def test_duplicate_registration_raises(self) -> None:
        fresh_registry = _PluginRegistry()
        fresh_registry.register(AffinePlugin())
        with pytest.raises(QuantizerError, match="already registered"):
            fresh_registry.register(AffinePlugin())

    def test_registered_methods(self) -> None:
        methods = registry.registered_methods()
        assert QuantMethod.AFFINE in methods
        assert QuantMethod.AWQ in methods
        assert QuantMethod.DWQ in methods


class TestAffinePlugin:
    @pytest.fixture
    def plugin(self) -> AffinePlugin:
        return AffinePlugin()

    def test_properties(self, plugin: AffinePlugin) -> None:
        assert plugin.method_id == QuantMethod.AFFINE
        assert 4 in plugin.supported_bits
        assert 6 in plugin.supported_bits
        assert 8 in plugin.supported_bits
        assert 64 in plugin.supported_group_sizes
        assert not plugin.requires_calibration

    def test_quantize_2d(self, plugin: AffinePlugin) -> None:
        weight = np.random.randn(16, 64).astype(np.float32)
        result = plugin.quantize(weight, bits=4, group_size=64)
        assert result.bits == 4
        assert result.group_size == 64
        assert result.method == QuantMethod.AFFINE
        assert result.data is not None
        assert result.scales is not None

    def test_quantize_unsupported_bits(self, plugin: AffinePlugin) -> None:
        weight = np.random.randn(16, 64).astype(np.float32)
        with pytest.raises(QuantizerError, match="does not support"):
            plugin.quantize(weight, bits=2, group_size=64)

    def test_quantize_unsupported_group(self, plugin: AffinePlugin) -> None:
        weight = np.random.randn(16, 64).astype(np.float32)
        with pytest.raises(QuantizerError, match="group size"):
            plugin.quantize(weight, bits=4, group_size=17)

    def test_quantize_indivisible_features(self, plugin: AffinePlugin) -> None:
        weight = np.random.randn(16, 63).astype(np.float32)
        with pytest.raises(QuantizerError, match="not divisible"):
            plugin.quantize(weight, bits=4, group_size=64)

    def test_roundtrip(self, plugin: AffinePlugin) -> None:
        weight = np.random.randn(8, 64).astype(np.float32)
        quantized = plugin.quantize(weight, bits=8, group_size=64)
        reconstructed = plugin.dequantize(quantized)
        # 8-bit should have low error
        error = np.mean((weight - reconstructed) ** 2)
        assert error < 0.1


class TestAwqPlugin:
    @pytest.fixture
    def plugin(self) -> AwqPlugin:
        return AwqPlugin()

    def test_properties(self, plugin: AwqPlugin) -> None:
        assert plugin.method_id == QuantMethod.AWQ
        assert plugin.requires_calibration

    def test_quantize_requires_calibration(self, plugin: AwqPlugin) -> None:
        weight = np.random.randn(16, 64).astype(np.float32)
        with pytest.raises(QuantizerError, match="requires calibration"):
            plugin.quantize(weight, bits=4, group_size=64, calibration=None)

    def test_quantize_with_calibration(self, plugin: AwqPlugin) -> None:
        weight = np.random.randn(16, 64).astype(np.float32)
        activations = np.random.randn(32, 64).astype(np.float32)
        result = plugin.quantize(weight, bits=4, group_size=64, calibration=activations)
        assert result.method == QuantMethod.AWQ
        assert result.metadata is not None
        assert "awq_channel_scales" in result.metadata
        assert result.metadata["awq_alpha"] in (0.0, 0.25, 0.5, 0.75, 1.0)
        assert result.metadata["activation_reconstruction_mse"] >= 0.0

    def test_awq_scaling_applied(self, plugin: AwqPlugin) -> None:
        weight = np.random.randn(8, 64).astype(np.float32)
        # Create activations with one very salient channel
        activations = np.zeros((32, 64), dtype=np.float32)
        activations[:, 0] = 100.0  # Channel 0 is highly active
        activations[:, 1:] = 0.01
        result = plugin.quantize(weight, bits=4, group_size=64, calibration=activations)
        scales = result.metadata["awq_channel_scales"]
        # Channel 0 should have a larger scale
        assert scales[0] > scales[1]

    def test_awq_learns_from_declared_alpha_grid(self, plugin: AwqPlugin) -> None:
        rng = np.random.default_rng(11)
        weight = rng.standard_normal((8, 64), dtype=np.float32)
        activations = rng.standard_normal((32, 64), dtype=np.float32)
        result = plugin.quantize(
            weight,
            bits=4,
            group_size=64,
            calibration={"activations": activations, "alpha_grid": [0.0, 1.0]},
        )

        assert result.metadata["awq_alpha"] in (0.0, 1.0)
        assert result.metadata["calibration_rows"] == 32

    def test_roundtrip(self, plugin: AwqPlugin) -> None:
        weight = np.random.randn(8, 64).astype(np.float32)
        activations = np.random.randn(32, 64).astype(np.float32)
        quantized = plugin.quantize(weight, bits=8, group_size=64, calibration=activations)
        reconstructed = plugin.dequantize(quantized)
        error = np.mean((weight - reconstructed) ** 2)
        assert error < 0.5  # AWQ may have slightly higher error due to scaling


class TestDwqPlugin:
    @pytest.fixture
    def plugin(self) -> DwqPlugin:
        return DwqPlugin()

    def test_properties(self, plugin: DwqPlugin) -> None:
        assert plugin.method_id == QuantMethod.DWQ
        assert not plugin.requires_calibration

    def test_quantize_no_calibration_needed(self, plugin: DwqPlugin) -> None:
        weight = np.random.randn(16, 64).astype(np.float32)
        result = plugin.quantize(weight, bits=4, group_size=64)
        assert result.method == QuantMethod.DWQ
        assert result.metadata is not None
        assert "clip_lower" in result.metadata
        assert "clip_upper" in result.metadata

    def test_distribution_clipping(self, plugin: DwqPlugin) -> None:
        # Create weight with outliers
        weight = np.random.randn(16, 64).astype(np.float32)
        weight[0, 0] = 1000.0  # Extreme outlier
        result = plugin.quantize(weight, bits=4, group_size=64)
        # Clip bounds should not include the extreme outlier
        assert result.metadata["clip_upper"] < 1000.0

    def test_roundtrip(self, plugin: DwqPlugin) -> None:
        weight = np.random.randn(8, 64).astype(np.float32)
        quantized = plugin.quantize(weight, bits=8, group_size=64)
        reconstructed = plugin.dequantize(quantized)
        error = np.mean((weight - reconstructed) ** 2)
        assert error < 0.1


class TestExecutionRecord:
    def test_record_success(self) -> None:
        record = record_execution(QuantMethod.AFFINE, "model.layers.0.mlp", 4, 64, success=True)
        assert record.success
        assert not record.fallback
        assert record.method == QuantMethod.AFFINE

    def test_record_fallback(self) -> None:
        record = record_execution(
            QuantMethod.AWQ,
            "model.layers.0.mlp",
            4,
            64,
            success=True,
            fallback=True,
            note="AWQ failed, fell back to affine",
        )
        assert record.fallback
        assert record.note is not None

    def test_record_failure(self) -> None:
        record = record_execution(QuantMethod.DWQ, "model.layers.0.attn", 6, 32, success=False)
        assert not record.success
