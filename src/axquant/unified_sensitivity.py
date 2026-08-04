"""Bind weight + KV sensitivity digests for unified planning lineage (P1)."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from axquant.errors import PlanningError
from axquant.schema import (
    EvidenceKind,
    KvCachePlan,
    KvSensitivityReport,
    QuantizationPlan,
    QuantMethod,
    SensitivityReport,
    UnifiedSensitivityBinding,
)
from axquant.serde import load_model, stable_sha256


def _same_model_lineage(left: object, right: object) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in ("model_id", "revision", "format")
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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
    try:
        weight = SensitivityReport.model_validate(weight.model_dump(mode="python"))
    except ValidationError as exc:
        raise PlanningError(f"invalid weight sensitivity report: {exc}") from exc
    if not weight.entries or any(not entry.candidates for entry in weight.entries):
        raise PlanningError("weight sensitivity must contain non-empty entries and candidates")
    if not _is_sha256(weight.inventory_sha256):
        raise PlanningError("weight sensitivity has an invalid inventory SHA-256 binding")
    tensor_names = [entry.tensor.name for entry in weight.entries]
    if len(tensor_names) != len(set(tensor_names)):
        raise PlanningError("weight sensitivity contains duplicate tensor entries")
    weight_digest = stable_sha256(weight)
    kv_digest: str | None = None
    kv_basis: str = "off"
    kv_report_basis: str | None = None
    if kv_sensitivity is not None:
        kv = (
            kv_sensitivity
            if isinstance(kv_sensitivity, KvSensitivityReport)
            else load_model(kv_sensitivity, KvSensitivityReport)
        )
        try:
            kv = KvSensitivityReport.model_validate(kv.model_dump(mode="python"))
        except ValidationError as exc:
            raise PlanningError(f"invalid KV sensitivity report: {exc}") from exc
        if not _same_model_lineage(kv.model, weight.model):
            raise PlanningError("KV sensitivity model lineage does not match weight sensitivity")
        if kv.profile != weight.profile:
            raise PlanningError("KV sensitivity profile does not match weight sensitivity profile")
        if kv.inventory_sha256 != weight.inventory_sha256:
            raise PlanningError(
                "KV sensitivity inventory digest does not match weight sensitivity inventory"
            )
        if kv.architecture_profile != weight.architecture_profile:
            raise PlanningError(
                "KV sensitivity architecture profile does not match weight sensitivity"
            )
        if any(
            not entry.candidates
            or not any(
                candidate.bits == 16
                and candidate.method is QuantMethod.BF16
                and candidate.supported
                for candidate in entry.candidates
            )
            for entry in kv.entries
        ):
            raise PlanningError("KV sensitivity entries require a supported BF16 baseline")
        kv_digest = stable_sha256(kv)
        kv_report_basis = (
            "architecture-prior"
            if kv.evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR
            else "measured"
        )
        kv_basis = kv_report_basis

    if plan is not None:
        plan_model = (
            plan if isinstance(plan, QuantizationPlan) else load_model(plan, QuantizationPlan)
        )
        try:
            plan_model = QuantizationPlan.model_validate(plan_model.model_dump(mode="python"))
        except ValidationError as exc:
            raise PlanningError(f"invalid quantization plan: {exc}") from exc
        if plan_model.analysis_sha256 != weight_digest:
            raise PlanningError("plan analysis_sha256 does not match the weight sensitivity digest")
        if not _same_model_lineage(plan_model.source_model, weight.model):
            raise PlanningError("plan source model does not match the weight sensitivity lineage")
        if plan_model.profile != weight.profile:
            raise PlanningError("plan profile does not match the weight sensitivity profile")
        if plan_model.evidence_kind != weight.evidence_kind:
            raise PlanningError("plan evidence kind does not match the weight sensitivity evidence")
        if plan_model.calibration != weight.calibration:
            raise PlanningError(
                "plan calibration evidence does not match the weight sensitivity report"
            )
        if plan_model.kv_cache is not None:
            kv_plan: KvCachePlan = plan_model.kv_cache
            if kv_plan.allocation_basis == "measured":
                if kv_digest is None:
                    raise PlanningError(
                        "measured KV plan requires the producing kv sensitivity report"
                    )
                if kv_report_basis != "measured":
                    raise PlanningError(
                        "measured KV plan requires a measured KV sensitivity report"
                    )
                if kv_plan.sensitivity_sha256 != kv_digest:
                    raise PlanningError(
                        "plan kv_cache.sensitivity_sha256 does not match the KV report digest"
                    )
                kv_basis = "measured"
            else:
                kv_basis = kv_plan.allocation_basis
        else:
            kv_basis = "off"

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
    try:
        validated_plan = QuantizationPlan.model_validate(plan.model_dump(mode="python"))
        validated_binding = UnifiedSensitivityBinding.model_validate(
            binding.model_dump(mode="python")
        )
    except ValidationError as exc:
        raise PlanningError(f"invalid plan or unified sensitivity binding: {exc}") from exc
    if validated_plan.analysis_sha256 != validated_binding.weight_sensitivity_sha256:
        raise PlanningError("unified sensitivity binding does not match the plan analysis")
    if not _same_model_lineage(validated_plan.source_model, validated_binding.source_model):
        raise PlanningError("unified sensitivity binding source model does not match the plan")
    if validated_plan.profile != validated_binding.profile:
        raise PlanningError("unified sensitivity binding profile does not match the plan")
    if validated_plan.kv_cache is None:
        if validated_binding.kv_allocation_basis != "off":
            raise PlanningError("weight-only plan requires an off KV sensitivity binding")
    elif validated_plan.kv_cache.allocation_basis != validated_binding.kv_allocation_basis:
        raise PlanningError("unified sensitivity KV allocation basis does not match the plan")
    elif (
        validated_plan.kv_cache.allocation_basis == "measured"
        and validated_plan.kv_cache.sensitivity_sha256 != validated_binding.kv_sensitivity_sha256
    ):
        raise PlanningError("unified sensitivity KV digest does not match the measured plan")
    note = (
        "unified sensitivity binding: "
        f"weight={validated_binding.weight_sensitivity_sha256[:12]}… "
        f"kv_basis={validated_binding.kv_allocation_basis}"
    )
    if note not in plan.warnings:
        plan.warnings.append(note)
