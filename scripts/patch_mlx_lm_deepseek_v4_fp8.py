#!/usr/bin/env python3
"""Install the mlx-lm DeepSeek V4 Flash-0731 FP8 load hook.

Stock mlx-lm 0.31.3 vendors no ``deepseek_v4`` and, even after the model
file is copied in, ``load_model`` never calls ``make_quantization_config``.
Flash-0731 ships HF ``quant_method=fp8`` (F8_E4M3 weights + F8_E8M0
scales). Without the hook, sanitize emits leftover ``.scales`` and load
fails with ``Received N parameters not in model``.

The Blaizzy mlx-lm PR adds::

    elif quant_method == "fp8" and model_type == "deepseek_v4":
        quantization = make_quantization_config(model)
        _quantize(quantization)

This script inserts that branch into the active ``mlx_lm.utils`` if missing.
Idempotent. Does not replace the whole utils module.

Usage (Studio venv)::

    /Users/devop/code/axquant/.venv/bin/python \\
      scripts/patch_mlx_lm_deepseek_v4_fp8.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

MARKER = 'quant_method == "fp8"'
NEEDLE = """        elif quant_method in ("awq", "gptq"):
            # Transform AutoAWQ/GPTQ packed weights to MLX format
            weights, quantization = _transform_awq_weights(weights, quantization_config)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
"""
HOOK = """        elif quant_method in ("awq", "gptq"):
            # Transform AutoAWQ/GPTQ packed weights to MLX format
            weights, quantization = _transform_awq_weights(weights, quantization_config)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method == "fp8" and config.get("model_type", None) == "deepseek_v4":
            from .models.deepseek_v4 import make_quantization_config

            quantization = make_quantization_config(model)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
"""


def utils_path() -> Path:
    utils = importlib.import_module("mlx_lm.utils")
    path = getattr(utils, "__file__", None)
    if not path:
        raise SystemExit("mlx_lm.utils has no __file__")
    return Path(path)


def already_patched(text: str) -> bool:
    return MARKER in text and "make_quantization_config" in text


def apply_patch(text: str) -> str:
    if already_patched(text):
        return text
    if NEEDLE not in text:
        raise SystemExit(
            "mlx_lm.utils load_model does not contain the expected awq/gptq "
            "quant_method branch; refuse to patch"
        )
    return text.replace(NEEDLE, HOOK, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        help="Override mlx_lm/utils.py path (default: import mlx_lm.utils)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    target = args.path or utils_path()
    original = target.read_text(encoding="utf-8")
    if already_patched(original):
        print(f"already patched: {target}")
        return 0
    updated = apply_patch(original)
    if args.dry_run:
        print(f"would patch: {target}")
        return 0
    backup = target.with_suffix(target.suffix + ".pre-flash0731-fp8")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(updated, encoding="utf-8")
    print(f"patched: {target}")
    print(f"backup:  {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
