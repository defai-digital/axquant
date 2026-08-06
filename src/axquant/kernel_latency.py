"""Kernel-latency measurement harness for latency-aware planning (ADR-0003).

Times decode-shaped (batch 1) and prefill-shaped GEMMs per executable
``(bits, group size)`` configuration on the current host. Every AXQuant
quantized method packs to the same portable affine layout, so one timing per
``(bits, group)`` covers affine, DWQ, AWQ, GPTQ, and act-order GPTQ alike —
the table records the packing-equivalence class, not the refinement recipe.

The resulting ``KernelLatencyTable`` is a frozen planning input: planning
never times kernels live (determinism), and a plan built with a table records
the table digest and host scope.
"""

from __future__ import annotations

import importlib
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

from axquant.errors import PlanningError
from axquant.schema import (
    KernelLatencyEntry,
    KernelLatencyTable,
    QuantMethod,
    RuntimeName,
)
from axquant.versioning import collect_versions

# Prefill rows chosen to be firmly compute-bound; decode is a single row.
_PREFILL_ROWS = 512
_DEFAULT_HIDDEN_SIZES = (2048, 4096)
_DEFAULT_ITERATIONS = 20
_DEFAULT_WARMUP = 5


def packing_equivalence_method(method: QuantMethod) -> QuantMethod:
    """Kernel-level equivalence class of a method's packed layout."""
    if method == QuantMethod.BF16:
        return QuantMethod.BF16
    return QuantMethod.AFFINE


def _timed_median_us(operation: Callable[[], Any], mx: Any, iterations: int) -> tuple[float, float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        mx.eval(operation())
        samples.append((time.perf_counter() - start) * 1e6)
    median = statistics.median(samples)
    if median <= 0.0:
        raise PlanningError("kernel latency sample collapsed to zero")
    quartiles = statistics.quantiles(samples, n=4) if len(samples) >= 4 else [median] * 3
    dispersion = (quartiles[2] - quartiles[0]) / median
    return median, max(0.0, dispersion)


def measure_mlx_kernel_latency(
    *,
    host_id: str,
    chip: str,
    os_version: str,
    bits_grid: Sequence[int] = (2, 3, 4, 6, 8),
    group_sizes: Sequence[int] = (32, 64, 128),
    hidden_sizes: Sequence[int] = _DEFAULT_HIDDEN_SIZES,
    iterations: int = _DEFAULT_ITERATIONS,
    warmup: int = _DEFAULT_WARMUP,
    seed: int = 0,
) -> KernelLatencyTable:
    """Measure the MLX affine-kernel latency grid on the current host.

    The caller is responsible for naming the host truthfully; authorizing
    performance evidence remains bound to the formal host by claims policy.
    """
    try:
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise PlanningError("kernel latency measurement requires MLX") from exc
    if iterations < 1:
        raise PlanningError("kernel latency iterations must be positive")

    entries: list[KernelLatencyEntry] = []
    warnings: list[str] = []
    mx.random.seed(seed)
    for hidden_size in sorted(set(hidden_sizes)):
        weight = mx.random.normal((hidden_size, hidden_size)).astype(mx.bfloat16)
        decode_x = mx.random.normal((1, hidden_size)).astype(mx.bfloat16)
        prefill_x = mx.random.normal((_PREFILL_ROWS, hidden_size)).astype(mx.bfloat16)
        mx.eval(weight, decode_x, prefill_x)

        def _measure(operation: Callable[[], Any]) -> tuple[float, float]:
            for _ in range(warmup):
                mx.eval(operation())
            return _timed_median_us(operation, mx, iterations)

        def _decode_matmul(dx: Any = decode_x, w: Any = weight) -> Any:
            return dx @ w.T

        def _prefill_matmul(px: Any = prefill_x, w: Any = weight) -> Any:
            return px @ w.T

        decode_us, decode_disp = _measure(_decode_matmul)
        prefill_us, _ = _measure(_prefill_matmul)
        entries.append(
            KernelLatencyEntry(
                runtime=RuntimeName.MLX_LM,
                bits=16,
                group_size=None,
                method=QuantMethod.BF16,
                hidden_size=hidden_size,
                decode_median_us=decode_us,
                prefill_median_us=prefill_us,
                dispersion=decode_disp,
                iterations=iterations,
            )
        )

        for bits in sorted(set(bits_grid)):
            for group_size in sorted(set(group_sizes)):
                if hidden_size % group_size != 0:
                    warnings.append(
                        f"skipped bits={bits} group={group_size} at hidden={hidden_size}: "
                        "hidden size not divisible by group size"
                    )
                    continue
                try:
                    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
                    mx.eval(packed, scales, biases)
                except (ValueError, RuntimeError) as exc:
                    warnings.append(
                        f"skipped bits={bits} group={group_size} at hidden={hidden_size}: {exc}"
                    )
                    continue

                def _qmm(
                    x: Any,
                    *,
                    _p: Any = packed,
                    _s: Any = scales,
                    _b: Any = biases,
                    _g: int = group_size,
                    _bits: int = bits,
                ) -> Any:
                    return mx.quantized_matmul(
                        x, _p, _s, _b, transpose=True, group_size=_g, bits=_bits
                    )

                def _decode_qmm(dx: Any = decode_x, op: Callable[[Any], Any] = _qmm) -> Any:
                    return op(dx)

                def _prefill_qmm(px: Any = prefill_x, op: Callable[[Any], Any] = _qmm) -> Any:
                    return op(px)

                try:
                    decode_us, decode_disp = _measure(_decode_qmm)
                    prefill_us, _ = _measure(_prefill_qmm)
                except (ValueError, RuntimeError) as exc:
                    warnings.append(
                        f"skipped bits={bits} group={group_size} at hidden={hidden_size}: {exc}"
                    )
                    continue
                entries.append(
                    KernelLatencyEntry(
                        runtime=RuntimeName.MLX_LM,
                        bits=bits,
                        group_size=group_size,
                        method=QuantMethod.AFFINE,
                        hidden_size=hidden_size,
                        decode_median_us=decode_us,
                        prefill_median_us=prefill_us,
                        dispersion=decode_disp,
                        iterations=iterations,
                    )
                )
    if not entries:
        raise PlanningError("kernel latency measurement produced no entries")
    return KernelLatencyTable(
        host_id=host_id,
        chip=chip,
        os_version=os_version,
        software_versions=collect_versions(),
        warmup_iterations=warmup,
        entries=entries,
        warnings=warnings,
    )


def decode_latency_provider(
    table: KernelLatencyTable,
    *,
    runtime: RuntimeName = RuntimeName.MLX_LM,
) -> Callable[[int, int | None, QuantMethod, int], float | None]:
    """Planner-facing lookup mapping a candidate to measured decode latency.

    Methods collapse to their packing-equivalence class before lookup, so a
    table measured once per (bits, group) serves every refinement method.
    """

    def lookup(
        bits: int,
        group_size: int | None,
        method: QuantMethod,
        hidden_size: int,
    ) -> float | None:
        return table.decode_latency_us(
            runtime=runtime,
            bits=bits,
            group_size=None if bits == 16 else group_size,
            method=packing_equivalence_method(method),
            hidden_size=hidden_size,
        )

    return lookup
