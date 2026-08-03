"""Portable GPTQ second-order weight refinement shared by the plugin and tests.

GPTQ (Frantar et al., 2022) accumulates a Hessian proxy ``H = 2 X^T X`` from
calibration activations, then rounds each weight column onto a static per-group
asymmetric affine grid while propagating the induced error into the remaining
columns through the Cholesky factor of the damped inverse Hessian. The refined
float weights sit exactly on the dequantized grid, so conversion packs them
with the same portable affine contract MLX-LM already accepts for affine/DWQ
-- a weight pre-transform executed before portable MLX-LM affine packing, same
contract as ``awq.py`` states.
"""

from __future__ import annotations

import importlib
from typing import Any

from axquant.errors import PlanningError, QuantizerError

_MAX_CALIBRATION_ROWS = 4096
_SUPPORTED_BITS = (2, 3, 4, 6, 8)
_SUPPORTED_GROUP_SIZES = (32, 64, 128)


def _as_numpy(weight: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires numpy") from exc
    return np.asarray(weight, dtype=np.float32)


def _static_group_grid(weight: Any, bits: int, group_size: int) -> tuple[Any, Any]:
    """Per-group scale/zero from the original weight, kept at group level.

    Returns arrays of shape ``(out_features, in_features // group_size)`` so
    the caller never materializes full ``(out, in)`` grid copies.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires numpy") from exc

    out_features, in_features = weight.shape
    w_grouped = weight.reshape(out_features, in_features // group_size, group_size)
    w_min = w_grouped.min(axis=-1)
    w_max = w_grouped.max(axis=-1)
    scale = (w_max - w_min) / ((1 << bits) - 1)
    scale = np.where(scale == 0, 1.0, scale)
    zero = np.clip(np.round(-w_min / scale), 0, (1 << bits) - 1)
    return scale.astype(np.float32), zero.astype(np.float32)


def _cholesky_inv_upper(hessian: Any) -> Any:
    """Upper Cholesky factor of ``H^-1``: float32 first, float64 on failure."""
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires numpy") from exc

    try:
        lower = np.linalg.cholesky(np.asarray(hessian, dtype=np.float32))
    except np.linalg.LinAlgError:
        lower = np.linalg.cholesky(hessian.astype(np.float64))
    lower_inv = np.linalg.inv(lower)
    del lower
    hessian_inv = lower_inv.T @ lower_inv
    del lower_inv
    hessian_inv = (hessian_inv + hessian_inv.T) / 2.0
    upper = np.linalg.cholesky(hessian_inv).T
    del hessian_inv
    return upper.astype(np.float32, copy=False)


def learn_gptq_refined_weight(
    weight: Any,
    activations: Any,
    *,
    bits: int,
    group_size: int,
    damping: float = 0.01,
    block_size: int = 128,
) -> tuple[Any, dict[str, float | int]]:
    """Refine a weight matrix with classic GPTQ error compensation.

    Columns are quantized left-to-right onto the static per-group affine grid
    of the original weight; each column's rounding error is spread over the
    not-yet-quantized columns via the damped inverse-Hessian Cholesky factor,
    in blocks of ``block_size`` columns (LazyBatchUpdates). The refined matrix
    approximates the source weights under the calibration activations.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires numpy") from exc

    if bits not in _SUPPORTED_BITS:
        raise PlanningError(f"GPTQ does not support {bits}-bit quantization")
    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise PlanningError(f"GPTQ does not support group size {group_size}")
    if activations is None:
        raise PlanningError("GPTQ requires calibration activations")

    w = _as_numpy(weight)
    if w.ndim != 2:
        raise PlanningError("GPTQ currently requires a two-dimensional weight matrix")
    calibration_config = activations if isinstance(activations, dict) else {}
    activation_value = calibration_config.get("activations", activations)
    act = np.asarray(activation_value, dtype=np.float32)
    if act.size == 0 or act.shape[-1] != w.shape[-1]:
        raise PlanningError("GPTQ calibration channels must match the weight input dimension")

    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise PlanningError(
            f"input features {in_features} not divisible by group size {group_size}"
        )

    x = act.reshape(-1, in_features)[:_MAX_CALIBRATION_ROWS]
    calibration_rows = int(x.shape[0])

    # Hessian proxy accumulated in float32 (float64 only on factorization
    # retry); dead columns get a unit diagonal. ``x`` is freed as soon as the
    # Hessian exists.
    hessian = (2.0 * x.T) @ x
    del x
    dead = np.diag(hessian) == 0.0
    hessian[dead, dead] = 1.0

    # Damped factorization, escalating damping x10 up to three attempts. The
    # diagonal is damped in place on a copy -- no identity matrix is ever
    # materialized, keeping peak memory at a few in^2 float32 buffers.
    final_damping = float(damping)
    mean_diag = float(np.mean(np.diag(hessian)))
    hinv_chol: Any | None = None
    for _ in range(3):
        damped = hessian.copy()
        damped[np.diag_indices_from(damped)] += final_damping * mean_diag
        try:
            hinv_chol = _cholesky_inv_upper(damped)
            del damped
            break
        except np.linalg.LinAlgError:
            final_damping *= 10.0
    del hessian
    if hinv_chol is None:
        raise PlanningError("GPTQ Hessian factorization failed even after damping escalation")

    group_scale, group_zero = _static_group_grid(w, bits, group_size)
    qmax = (1 << bits) - 1
    working = w.copy()
    refined = np.zeros_like(w)
    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        block_errors = np.zeros((out_features, block_end - block_start), dtype=np.float32)
        for j in range(block_start, block_end):
            w_col = working[:, j]
            scale_col = group_scale[:, j // group_size]
            zero_col = group_zero[:, j // group_size]
            q_col = np.clip(np.round(w_col / scale_col) + zero_col, 0, qmax)
            deq_col = (q_col - zero_col) * scale_col
            refined[:, j] = deq_col
            err = (w_col - deq_col) / hinv_chol[j, j]
            block_errors[:, j - block_start] = err
            if j + 1 < block_end:
                working[:, j + 1 : block_end] -= np.outer(
                    err, hinv_chol[j, j + 1 : block_end]
                ).astype(np.float32)
        if block_end < in_features:
            # float32 throughout: no per-block float64 temporaries.
            working[:, block_end:] -= block_errors @ hinv_chol[block_start:block_end, block_end:]

    metadata: dict[str, float | int] = {
        "gptq_damping": final_damping,
        "calibration_rows": calibration_rows,
        "bits": bits,
        "group_size": group_size,
        "mean_quant_error": float(np.mean(np.abs(w - refined))),
    }
    return refined.astype(np.float32), metadata


def apply_mlx_gptq_refine(
    module: Any,
    *,
    activations: Any,
    bits: int,
    group_size: int,
    damping: float = 0.01,
) -> dict[str, float | int]:
    """Mutate ``module.weight`` with portable GPTQ refinement before affine packing."""
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires MLX") from exc
    weight = getattr(module, "weight", None)
    if weight is None:
        raise PlanningError("GPTQ requires a module with a weight tensor")
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires numpy") from exc

    try:
        refined, metadata = learn_gptq_refined_weight(
            np.asarray(weight, dtype=np.float32),
            activations,
            bits=bits,
            group_size=group_size,
            damping=damping,
        )
    except (PlanningError, QuantizerError, ValueError, TypeError) as exc:
        raise PlanningError(str(exc)) from exc
    module.weight = mx.array(refined, dtype=weight.dtype)
    mx.eval(module.weight)
    return metadata
