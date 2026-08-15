from __future__ import annotations

import importlib.util
from pathlib import Path


def _patcher():
    path = Path(__file__).resolve().parents[1] / "scripts" / "patch_mlx_lm_deepseek_v4_fp8.py"
    spec = importlib.util.spec_from_file_location("patch_mlx_lm_deepseek_v4_fp8", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stock_snippet() -> str:
    return """        elif quant_method == "compressed-tensors":
            quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method in ("awq", "gptq"):
            # Transform AutoAWQ/GPTQ packed weights to MLX format
            weights, quantization = _transform_awq_weights(weights, quantization_config)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)

    if config.get("quantize_activations", False):
        pass
"""


def test_apply_patch_inserts_fp8_deepseek_v4_hook() -> None:
    patcher = _patcher()
    updated = patcher.apply_patch(_stock_snippet())
    assert patcher.already_patched(updated)
    assert 'quant_method == "fp8"' in updated
    assert "make_quantization_config" in updated
    assert updated.count('elif quant_method in ("awq", "gptq"):') == 1


def test_apply_patch_is_idempotent() -> None:
    patcher = _patcher()
    once = patcher.apply_patch(_stock_snippet())
    assert patcher.apply_patch(once) == once


def test_script_lives_in_repo() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "patch_mlx_lm_deepseek_v4_fp8.py").is_file()
