"""Bind weight + KV sensitivity digests for unified planning lineage (P1)."""

from __future__ import annotations

from pathlib import Path

from axquant.errors import PlanningError
from axquant.schema import (
    KvCachePlan,
    KvSensitivityReport,
    QuantizationPlan,
    SensitivityReport,
    UnifiedSensitivityBinding,
)
from axquant.serde import load_model, stable_sha256


def bind_unified_sensitivity(
    weight_sensitivity: SensitivityReport | str | Path,
    *,
    kv_sensitivity: KvSensitivityReport | str | Path | None = None,
    plan: QuantizationPlan | str | Path | None = None,
) -> UnifiedSensitivityBinding:
    """Create a digest binding between weight (and optional KV) sensitivity reports.

    When a plan is supplied, measured KV plans must match the KV report digest.
    """
    weight = (
        weight_sensitivity
        if isinstance(weight_sensitivity, SensitivityReport)
        else load_model(weight_sensitivity, SensitivityReport)
    )
    weight_digest = stable_sha256(weight)
    kv_digest: str | None = None
    kv_basis: str = "off"
    if kv_sensitivity is not None:
        kv = (
            kv_sensitivity
            if isinstance(kv_sensitivity, KvSensitivityReport)
            else load_model(kv_sensitivity, KvSensitivityReport)
        )
        if kv.model.model_id != weight.model.model_id:
            raise PlanningError(
                "KV sensitivity model_id does not match weight sensitivity model_id"
            )
        if kv.profile != weight.profile:
            raise PlanningError("KV sensitivity profile does not match weight sensitivity profile")
        if kv.inventory_sha256 != weight.inventory_sha256:
            raise PlanningError(
                "KV sensitivity inventory digest does not match weight sensitivity inventory"
            )
        kv_digest = stable_sha256(kv)
        kv_basis = "measured"

    if plan is not None:
        plan_model = (
            plan if isinstance(plan, QuantizationPlan) else load_model(plan, QuantizationPlan)
        )
        if plan_model.analysis_sha256 != weight_digest:
            raise PlanningError("plan analysis_sha256 does not match the weight sensitivity digest")
        if plan_model.kv_cache is not None:
            kv_plan: KvCachePlan = plan_model.kv_cache
            if kv_plan.allocation_basis == "measured":
                if kv_digest is None:
                    raise PlanningError(
                        "measured KV plan requires the producing kv sensitivity report"
                    )
                if kv_plan.sensitivity_sha256 != kv_digest:
                    raise PlanningError(
                        "plan kv_cache.sensitivity_sha256 does not match the KV report digest"
                    )
                kv_basis = "measured"
            else:
                kv_basis = kv_plan.allocation_basis

    notes = [
        "One sensitivity lineage drives weight allocation and optional KV allocation.",
        "Evidence kinds are not upgraded by binding; each report keeps its own label.",
    ]
    return UnifiedSensitivityBinding(
        source_model=weight.model,
        profile=weight.profile,
        weight_sensitivity_sha256=weight_digest,
        weight_evidence_kind=weight.evidence_kind,
        kv_sensitivity_sha256=kv_digest,
        kv_allocation_basis=kv_basis,
        inventory_sha256=weight.inventory_sha256,
        notes=notes,
    )


def attach_binding_warning(plan: QuantizationPlan, binding: UnifiedSensitivityBinding) -> None:
    """Record the binding digests on the plan warnings list (immutable-safe copy not required)."""
    note = (
        f"unified sensitivity binding: weight={binding.weight_sensitivity_sha256[:12]}… "
        f"kv_basis={binding.kv_allocation_basis}"
    )
    if note not in plan.warnings:
        plan.warnings.append(note)
