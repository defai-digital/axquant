"""Merge an MLX-LM LoRA into Qwen3.5 language weights.

OptiQ saves factors on the MLX module path
``language_model.model.layers.{i}...`` while this snapshot stores the base
matrices as ``model.language_model.layers.{i}...weight``. The update matches
``mlx_lm.tuner.lora.LoRALinear.fuse``:

    delta = (scale * lora_b.T) @ lora_a.T
    lora_a is (in, rank), lora_b is (rank, out), weight is (out, in).
"""

from __future__ import annotations

import numpy as np

LORA_PREFIX = "language_model.model."
WEIGHT_PREFIX = "model.language_model."


def weight_key_for_lora(lora_key: str) -> str:
    """Return the base weight name for one ``lora_a`` or ``lora_b`` key."""

    if lora_key.endswith(".lora_a"):
        stem = lora_key[: -len(".lora_a")]
    elif lora_key.endswith(".lora_b"):
        stem = lora_key[: -len(".lora_b")]
    else:
        raise ValueError(f"not a LoRA factor: {lora_key}")
    if not stem.startswith(LORA_PREFIX):
        raise ValueError(f"unexpected LoRA module path: {lora_key}")
    return f"{WEIGHT_PREFIX}{stem.removeprefix(LORA_PREFIX)}.weight"


def lora_delta(
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    *,
    scale: float,
    rank: int,
) -> np.ndarray:
    """Return the ``(out, in)`` update added to a Linear weight."""

    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("LoRA factors must be matrices")
    if lora_a.shape[1] != rank or lora_b.shape[0] != rank:
        raise ValueError(
            f"LoRA rank {rank} does not match factors {lora_a.shape} and {lora_b.shape}"
        )
    if not np.isfinite(scale):
        raise ValueError("LoRA scale must be finite")
    delta = (scale * np.swapaxes(lora_b, 0, 1)) @ np.swapaxes(lora_a, 0, 1)
    if delta.shape != (lora_b.shape[1], lora_a.shape[0]):
        raise ValueError(f"LoRA delta shape {delta.shape} is inconsistent")
    return delta
