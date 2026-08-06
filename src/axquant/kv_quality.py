"""Measured-KV serving-quality report builder (RM-21 / WS-6, report-only).

Turns dual-profile quality evaluations executed under a planned per-layer
KV-cache table (``runtime-check --runtime mlx-lm-kv`` path) into a
digest-bound, report-only artifact. The report proposes thresholds from data;
it never enforces a gate — enforcement is a separate later step once the
thresholds are grounded (ADR-0001 additive two-step rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from axquant.errors import ArtifactError
from axquant.schema import (
    KvCachePlan,
    KvServingQualityProfileResult,
    KvServingQualityReport,
    ModelIdentity,
)
from axquant.serde import stable_sha256


def build_kv_serving_quality_report(
    *,
    model: ModelIdentity,
    kv_plan: KvCachePlan,
    execution_summary: Mapping[str, Any],
    results: Sequence[KvServingQualityProfileResult],
    kv_sensitivity_sha256: str | None = None,
    warnings: Sequence[str] = (),
) -> KvServingQualityReport:
    """Bind measured KV quality results to their executed plan.

    Fails closed unless the runtime-fidelity summary (from ``axquant.kv_exec``)
    proves every planned layer actually executed at its planned precision —
    quality numbers measured against a silently-reverted cache prove nothing.
    """
    if not bool(execution_summary.get("ok")):
        raise ArtifactError(
            "KV serving-quality evidence requires a passing mlx-lm-kv execution run"
        )
    if not bool(execution_summary.get("per_layer_execution")):
        raise ArtifactError(
            "KV serving-quality evidence requires exact per-layer execution: "
            "a layer that reverted to BF16 invalidates the measurement"
        )
    quantized_active = execution_summary.get("quantized_layers_active")
    if not isinstance(quantized_active, int) or quantized_active < 1:
        raise ArtifactError(
            "KV serving-quality evidence requires at least one active quantized KV layer"
        )
    return KvServingQualityReport(
        model=model,
        kv_plan_sha256=stable_sha256(kv_plan),
        kv_sensitivity_sha256=kv_sensitivity_sha256,
        results=list(results),
        warnings=list(warnings),
    )
