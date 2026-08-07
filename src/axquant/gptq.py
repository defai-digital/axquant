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
import math
import operator
from typing import Any

from axquant.errors import PlanningError, QuantizerError
from axquant.numeric import as_finite_float32_matrix
from axquant.package_data import load_package_yaml


def _gptq_defaults() -> dict[str, Any]:
    raw = load_package_yaml("quantizer_defaults.yaml")
    section = raw.get("gptq") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise PlanningError("quantizer_defaults.yaml missing gptq section")
    return section


_GPTQ_DEFAULTS = _gptq_defaults()
_MAX_CALIBRATION_ROWS = int(_GPTQ_DEFAULTS["max_calibration_rows"])
_SUPPORTED_BITS = tuple(int(value) for value in _GPTQ_DEFAULTS["supported_bits"])
_SUPPORTED_GROUP_SIZES = tuple(int(value) for value in _GPTQ_DEFAULTS["supported_group_sizes"])
_DEFAULT_DAMPING = float(_GPTQ_DEFAULTS.get("default_damping", 0.01))
_DEFAULT_BLOCK_SIZE = int(_GPTQ_DEFAULTS.get("default_block_size", 128))


def _as_numpy(weight: Any) -> Any:
    return as_finite_float32_matrix(weight, component="GPTQ")


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
    # The range must include zero: the zero point below is clamped to
    # [0, qmax], and a group that does not straddle zero would otherwise
    # land its true zero point outside that window, collapsing the
    # dequantization grid onto [0, w_max - w_min] away from the weights.
    w_min = np.minimum(w_grouped.min(axis=-1), 0.0)
    w_max = np.maximum(w_grouped.max(axis=-1), 0.0)
    scale = (w_max - w_min) / ((1 << bits) - 1)
    scale = np.where(scale == 0, 1.0, scale)
    zero = np.clip(np.round(-w_min / scale), 0, (1 << bits) - 1)
    if not bool(np.all(np.isfinite(scale))) or bool(np.any(scale <= 0)):
        raise PlanningError("GPTQ affine grid contains invalid scales")
    if not bool(np.all(np.isfinite(zero))):
        raise PlanningError("GPTQ affine grid contains invalid zero points")
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
    with np.errstate(over="ignore", invalid="ignore"):
        result = upper.astype(np.float32, copy=False)
    if not bool(np.all(np.isfinite(result))) or bool(np.any(np.diag(result) <= 0)):
        raise np.linalg.LinAlgError("GPTQ inverse-Hessian factor is non-finite")
    return result


def _group_preserving_permutation(diag: Any, group_size: int) -> tuple[Any, Any]:
    """Column ordering for act-order that never moves a column across groups.

    Whole groups are ordered by descending aggregate Hessian-diagonal mass and
    columns inside each group by descending diagonal, so high-salience columns
    quantize first and push their rounding error into lower-salience columns —
    while every column keeps its original group, scale, and packed position.
    Returns ``(perm, group_order)``; both use stable sorts for determinism.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError("GPTQ execution requires numpy") from exc

    n_groups = diag.shape[0] // group_size
    group_mass = diag.reshape(n_groups, group_size).sum(axis=1)
    group_order = np.argsort(-group_mass, kind="stable")
    perm_parts = []
    for group_index in group_order:
        base = int(group_index) * group_size
        inner = np.argsort(-diag[base : base + group_size], kind="stable")
        perm_parts.append(base + inner)
    return np.concatenate(perm_parts), group_order


def learn_gptq_refined_weight(
    weight: Any,
    activations: Any,
    *,
    bits: int,
    group_size: int,
    damping: float = _DEFAULT_DAMPING,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    act_order: bool = False,
) -> tuple[Any, dict[str, float | int]]:
    """Refine a weight matrix with classic GPTQ error compensation.

    Columns are quantized left-to-right onto the static per-group affine grid
    of the original weight; each column's rounding error is spread over the
    not-yet-quantized columns via the damped inverse-Hessian Cholesky factor,
    in blocks of ``block_size`` columns (LazyBatchUpdates). The refined matrix
    approximates the source weights under the calibration activations.

    With ``act_order`` the processing order follows the group-preserving
    activation ordering (ADR-0002): group membership and the packed layout are
    identical to the static order, only the error-propagation sequence changes,
    so the output still satisfies the portable affine contract.
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
    if isinstance(damping, bool):
        raise PlanningError("GPTQ damping must be a finite non-negative number")
    try:
        resolved_damping = float(damping)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanningError("GPTQ damping must be a finite non-negative number") from exc
    if not math.isfinite(resolved_damping) or resolved_damping < 0:
        raise PlanningError("GPTQ damping must be a finite non-negative number")
    if isinstance(block_size, bool):
        raise PlanningError("GPTQ block_size must be a positive integer")
    try:
        resolved_block_size = operator.index(block_size)
    except TypeError as exc:
        raise PlanningError("GPTQ block_size must be a positive integer") from exc
    if resolved_block_size <= 0:
        raise PlanningError("GPTQ block_size must be a positive integer")
    if not isinstance(act_order, bool):
        raise PlanningError("GPTQ act_order must be a boolean")

    w = _as_numpy(weight)
    if w.ndim != 2:
        raise PlanningError("GPTQ currently requires a two-dimensional weight matrix")
    calibration_config = activations if isinstance(activations, dict) else {}
    activation_value = calibration_config.get("activations", activations)
    try:
        act = np.asarray(activation_value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanningError(f"GPTQ calibration activations are not numeric: {exc}") from exc
    if act.ndim == 0 or act.size == 0 or act.shape[-1] != w.shape[-1]:
        raise PlanningError("GPTQ calibration channels must match the weight input dimension")
    if not bool(np.all(np.isfinite(act))):
        raise PlanningError("GPTQ calibration activations must contain only finite values")

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
    if not bool(np.all(np.isfinite(hessian))):
        raise PlanningError("GPTQ calibration Hessian is non-finite")
    dead = np.diag(hessian) == 0.0
    hessian[dead, dead] = 1.0

    perm = None
    group_order = None
    if act_order:
        perm, group_order = _group_preserving_permutation(np.diag(hessian).copy(), group_size)
        hessian = hessian[np.ix_(perm, perm)]

    # Damped factorization, escalating damping x10 up to three attempts. The
    # diagonal is damped in place on a copy -- no identity matrix is ever
    # materialized, keeping peak memory at a few in^2 float32 buffers.
    final_damping = resolved_damping
    mean_diag = float(np.mean(np.diag(hessian)))
    if not math.isfinite(mean_diag) or mean_diag <= 0:
        raise PlanningError("GPTQ calibration Hessian has an invalid diagonal")
    hinv_chol: Any | None = None
    for _ in range(3):
        damped = hessian.copy()
        damped[np.diag_indices_from(damped)] += final_damping * mean_diag
        try:
            if not bool(np.all(np.isfinite(damped))):
                raise np.linalg.LinAlgError("GPTQ damped Hessian is non-finite")
            hinv_chol = _cholesky_inv_upper(damped)
            del damped
            break
        except np.linalg.LinAlgError:
            final_damping = (
                final_damping * 10.0 if final_damping > 0 else float(np.finfo(np.float32).eps)
            )
    del hessian
    if hinv_chol is None:
        raise PlanningError("GPTQ Hessian factorization failed even after damping escalation")

    group_scale, group_zero = _static_group_grid(w, bits, group_size)
    if act_order:
        # Groups travel as units, so reordering the per-group grid columns by
        # ``group_order`` keeps ``j // group_size`` lookups correct in the
        # permuted domain; the grid values themselves are untouched.
        group_scale = group_scale[:, group_order]
        group_zero = group_zero[:, group_order]
    qmax = (1 << bits) - 1
    working = w[:, perm].copy() if act_order else w.copy()
    refined = np.zeros_like(w)
    for block_start in range(0, in_features, resolved_block_size):
        block_end = min(block_start + resolved_block_size, in_features)
        block_errors = np.zeros((out_features, block_end - block_start), dtype=np.float32)
        for j in range(block_start, block_end):
            w_col = working[:, j]
            if not bool(np.all(np.isfinite(w_col))):
                raise PlanningError("GPTQ error compensation produced non-finite weights")
            scale_col = group_scale[:, j // group_size]
            zero_col = group_zero[:, j // group_size]
            # Same encode as AWQ/portable affine: round(w/s + z). With integer
            # zeros this matches round(w/s)+z except at banker's-rounding
            # half-integers, where the joint form is the MSE-correct grid code.
            q_col = np.clip(np.round(w_col / scale_col + zero_col), 0, qmax)
            deq_col = (q_col - zero_col) * scale_col
            refined[:, j] = deq_col
            err = (w_col - deq_col) / hinv_chol[j, j]
            if not bool(np.all(np.isfinite(err))):
                raise PlanningError("GPTQ error compensation produced a non-finite error")
            block_errors[:, j - block_start] = err
            if j + 1 < block_end:
                working[:, j + 1 : block_end] -= np.outer(
                    err, hinv_chol[j, j + 1 : block_end]
                ).astype(np.float32)
        if block_end < in_features:
            # float32 throughout: no per-block float64 temporaries.
            working[:, block_end:] -= block_errors @ hinv_chol[block_start:block_end, block_end:]
            if not bool(np.all(np.isfinite(working[:, block_end:]))):
                raise PlanningError("GPTQ block update produced non-finite weights")

    if act_order:
        unpermuted = np.empty_like(refined)
        unpermuted[:, perm] = refined
        refined = unpermuted

    if not bool(np.all(np.isfinite(refined))):
        raise PlanningError("GPTQ refinement produced non-finite weights")
    with np.errstate(over="ignore", invalid="ignore"):
        absolute_error = np.abs(w - refined)
    if not bool(np.all(np.isfinite(absolute_error))):
        raise PlanningError("GPTQ refinement error is non-finite")
    mean_quant_error = float(np.mean(absolute_error, dtype=np.float64))
    del absolute_error
    if not math.isfinite(mean_quant_error):
        raise PlanningError("GPTQ refinement error is non-finite")
    metadata: dict[str, float | int] = {
        "gptq_damping": final_damping,
        "calibration_rows": calibration_rows,
        "bits": bits,
        "group_size": group_size,
        "mean_quant_error": mean_quant_error,
        "act_order": int(act_order),
    }
    return refined.astype(np.float32), metadata


def apply_mlx_gptq_refine(
    module: Any,
    *,
    activations: Any,
    bits: int,
    group_size: int,
    damping: float = _DEFAULT_DAMPING,
    act_order: bool = False,
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
        # bfloat16 arrays reject numpy's buffer protocol; cast via MLX then.
        try:
            weight_f32 = np.asarray(weight, dtype=np.float32)
        except (RuntimeError, TypeError):
            weight_f32 = np.asarray(mx.array(weight, dtype=mx.float32))
        refined, metadata = learn_gptq_refined_weight(
            weight_f32,
            activations,
            bits=bits,
            group_size=group_size,
            damping=damping,
            act_order=act_order,
        )
    except (PlanningError, QuantizerError, ValueError, TypeError, RuntimeError) as exc:
        raise PlanningError(str(exc)) from exc
    try:
        candidate = mx.array(refined, dtype=weight.dtype)
        mx.eval(candidate)
    except Exception as exc:
        raise PlanningError(f"GPTQ MLX weight materialization failed: {exc}") from exc
    module.weight = candidate
    return metadata
