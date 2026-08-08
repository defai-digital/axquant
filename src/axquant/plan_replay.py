from __future__ import annotations

from copy import deepcopy
from math import isclose
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from axquant.errors import PlanningError
from axquant.naming import target_class_for_bpw
from axquant.planner import _current_policy_profile, storage_bpw, strategy_for_measurement
from axquant.profiles import objective_for
from axquant.schema import (
    Allocation,
    ArchitectureSupportLevel,
    PrecisionShare,
    QuantizationPlan,
    SensitivityReport,
)
from axquant.schema._base import utc_now
from axquant.serde import file_sha256, read_data, stable_sha256
from axquant.versioning import collect_versions


def _distribution(
    assignments: list[Allocation],
    *,
    mtp_only: bool = False,
) -> dict[str, PrecisionShare]:
    selected = [
        assignment
        for assignment in assignments
        if assignment.parameters > 0 and (not mtp_only or assignment.role.is_mtp)
    ]
    total = sum(assignment.parameters for assignment in selected)
    if total <= 0:
        return {}
    by_precision: dict[str, int] = {}
    for assignment in selected:
        label = "bf16" if assignment.bits == 16 else f"{assignment.bits}bit"
        by_precision[label] = by_precision.get(label, 0) + assignment.parameters
    return {
        label: PrecisionShare(parameters=parameters, fraction=parameters / total)
        for label, parameters in sorted(by_precision.items())
    }


def _normalized_source_plan(payload: Any) -> QuantizationPlan:
    """Load a legacy plan while repairing only schema-derived summaries.

    Older refinement plans can predate target-class policy and strict distribution
    validators. Their tensor assignments remain the load-bearing input; BPW values
    must already agree with those assignments and are never silently repaired.
    """

    if not isinstance(payload, dict):
        raise PlanningError("measured plan replay source must be a JSON object")
    if payload.get("schema_version") != "axquant.plan.v1":
        raise PlanningError("measured plan replay requires schema axquant.plan.v1")
    normalized = deepcopy(payload)
    raw_assignments = normalized.get("assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise PlanningError("measured plan replay source has no assignments")
    try:
        assignments = [Allocation.model_validate(item) for item in raw_assignments]
    except ValidationError as exc:
        raise PlanningError(f"measured plan replay source assignments are invalid: {exc}") from exc

    total_parameters = sum(assignment.parameters for assignment in assignments)
    if total_parameters <= 0:
        raise PlanningError("measured plan replay source has no logical parameters")
    nominal_bpw = (
        sum(assignment.bits * assignment.parameters for assignment in assignments)
        / total_parameters
    )
    effective_bpw = (
        sum(
            storage_bpw(assignment.bits, assignment.group_size) * assignment.parameters
            for assignment in assignments
        )
        / total_parameters
    )
    for field, expected in (("nominal_bpw", nominal_bpw), ("effective_bpw", effective_bpw)):
        observed = normalized.get(field)
        if not isinstance(observed, (int, float)) or not isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise PlanningError(
                f"measured plan replay source {field} does not match its assignments"
            )

    target_bpw = normalized.get("target_bpw")
    if not isinstance(target_bpw, (int, float)):
        raise PlanningError("measured plan replay source has no numeric target_bpw")
    normalized["target_class"] = target_class_for_bpw(float(target_bpw))
    normalized["weight_distribution"] = _distribution(assignments)
    normalized["mtp_distribution"] = _distribution(assignments, mtp_only=True)
    try:
        return QuantizationPlan.model_validate(normalized)
    except ValidationError as exc:
        raise PlanningError(f"measured plan replay source is invalid: {exc}") from exc


def replay_measured_plan(
    report: SensitivityReport,
    source_payload: Any,
    *,
    source_file_sha256: str,
    ax_engine_executable: str = "ax-engine-bench",
) -> QuantizationPlan:
    """Replay a precision allocation only when current measured evidence supports it exactly.

    This is intentionally stricter than a manual recipe. Every source assignment must bind to
    the same tensor identity and to an exactly matching measured candidate signature, metrics,
    and objective loss in the supplied sensitivity report.
    """

    if len(source_file_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_file_sha256
    ):
        raise PlanningError("measured plan replay source SHA-256 is invalid")
    source = _normalized_source_plan(source_payload)
    if not source.evidence_kind.release_quality:
        raise PlanningError(
            "measured plan replay source lacks release-quality sensitivity evidence"
        )
    if not report.evidence_kind.release_quality:
        raise PlanningError(
            "measured plan replay report lacks release-quality sensitivity evidence"
        )
    if report.architecture_profile.support_level is not ArchitectureSupportLevel.SUPPORTED:
        raise PlanningError("measured plan replay requires a supported architecture adapter")
    if source.source_model != report.model:
        raise PlanningError("measured plan replay source model differs from sensitivity evidence")
    if source.architecture_profile != report.architecture_profile:
        raise PlanningError(
            "measured plan replay architecture profile differs from sensitivity evidence"
        )
    if source.profile != report.profile:
        raise PlanningError("measured plan replay profile differs from sensitivity evidence")
    if source.calibration != report.calibration:
        raise PlanningError("measured plan replay calibration differs from sensitivity evidence")
    if source.kv_cache is not None:
        raise PlanningError("measured plan replay does not migrate KV-cache allocations")

    current_objective = objective_for(report.profile)
    if source.objective != current_objective:
        raise PlanningError("measured plan replay objective differs from current profile policy")
    weights = current_objective.normalized()
    entries = {entry.tensor.name: entry for entry in report.entries}
    if len(entries) != len(report.entries):
        raise PlanningError("measured plan replay sensitivity contains duplicate tensors")
    assignments_by_tensor = {assignment.tensor: assignment for assignment in source.assignments}
    if set(assignments_by_tensor) != set(entries):
        missing = sorted(set(entries) - set(assignments_by_tensor))
        unexpected = sorted(set(assignments_by_tensor) - set(entries))
        raise PlanningError(
            "measured plan replay tensor coverage differs from sensitivity evidence: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    replayed_assignments: list[Allocation] = []
    for tensor_name in (assignment.tensor for assignment in source.assignments):
        assignment = assignments_by_tensor[tensor_name]
        entry = entries[tensor_name]
        tensor = entry.tensor
        if (
            assignment.module_path != tensor.module_path
            or assignment.role != tensor.role
            or assignment.parameters != tensor.parameters
        ):
            raise PlanningError(
                f"measured plan replay tensor metadata differs for {assignment.tensor}"
            )
        signature_matches = [
            candidate
            for candidate in entry.candidates
            if (
                candidate.bits,
                candidate.method,
                candidate.group_size,
            )
            == (assignment.bits, assignment.method, assignment.group_size)
        ]
        if not signature_matches:
            raise PlanningError(
                f"measured plan replay has no sensitivity candidate for {assignment.tensor}: "
                f"{assignment.bits}-bit/{assignment.method.value}/gs{assignment.group_size}"
            )
        candidate = next(
            (item for item in signature_matches if item.metrics == assignment.metrics),
            None,
        )
        if candidate is None:
            raise PlanningError(f"measured plan replay metrics differ for {assignment.tensor}")
        if not candidate.supported:
            raise PlanningError(
                f"measured plan replay selected an unsupported candidate for {assignment.tensor}"
            )
        metric_values = candidate.metrics.model_dump()
        predicted_loss = sum(float(metric_values[key]) * weight for key, weight in weights.items())
        if not isclose(
            assignment.predicted_loss,
            predicted_loss,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise PlanningError(
                f"measured plan replay objective loss differs for {assignment.tensor}"
            )
        scale_strategy, outlier_strategy = strategy_for_measurement(candidate)
        replayed_assignments.append(
            Allocation(
                tensor=tensor.name,
                module_path=tensor.module_path,
                role=tensor.role,
                parameters=tensor.parameters,
                bits=candidate.bits,
                method=candidate.method,
                group_size=candidate.group_size,
                predicted_loss=predicted_loss,
                metrics=candidate.metrics,
                reason="exact measured-plan replay from checksum-bound source",
                scale_strategy=scale_strategy,
                outlier_strategy=outlier_strategy,
                strategy_metadata={
                    "storage_bpw": storage_bpw(candidate.bits, candidate.group_size),
                    "selected_from_candidates": len(entry.candidates),
                    "measured_plan_replay": True,
                },
            )
        )

    total_parameters = sum(assignment.parameters for assignment in replayed_assignments)
    nominal_bpw = (
        sum(assignment.bits * assignment.parameters for assignment in replayed_assignments)
        / total_parameters
    )
    effective_bpw = (
        sum(
            storage_bpw(assignment.bits, assignment.group_size) * assignment.parameters
            for assignment in replayed_assignments
        )
        / total_parameters
    )
    group_sizes = tuple(
        sorted(
            {
                assignment.group_size
                for assignment in replayed_assignments
                if assignment.group_size is not None
            }
        )
    )
    warnings = list(dict.fromkeys([*report.warnings, *source.warnings]))
    warnings.extend(
        [
            "Measured precision allocation replayed with exact tensor, candidate-signature, "
            "metric, and objective-loss matching against the bound sensitivity report.",
            f"Measured plan replay source file SHA-256: {source_file_sha256}.",
            f"Measured plan replay source analysis SHA-256: {source.analysis_sha256}.",
            "Replayed candidates still require complete-model quality, runtime, size, and MTP "
            "validation; replay does not promote or waive any release gate.",
        ]
    )
    replayed = source.model_copy(
        update={
            "architecture_profile": _current_policy_profile(report.architecture_profile),
            "target_class": target_class_for_bpw(source.target_bpw),
            "nominal_bpw": nominal_bpw,
            "effective_bpw": effective_bpw,
            "candidate_group_sizes": group_sizes,
            "objective": current_objective,
            "software_versions": collect_versions(ax_engine_executable=ax_engine_executable),
            "analysis_sha256": stable_sha256(report),
            "evidence_kind": report.evidence_kind,
            "calibration": report.calibration,
            "assignments": replayed_assignments,
            "weight_distribution": _distribution(replayed_assignments),
            "mtp_distribution": _distribution(replayed_assignments, mtp_only=True),
            "method_near_ties": [],
            "method_near_ties_omitted": 0,
            "cost_model": "abstract-bpw",
            "kernel_latency_sha256": None,
            "kernel_latency_host_id": None,
            "created_at": utc_now(),
            "warnings": list(dict.fromkeys(warnings)),
        }
    )
    try:
        return QuantizationPlan.model_validate(replayed.model_dump(mode="python"))
    except ValidationError as exc:
        raise PlanningError(f"measured plan replay produced an invalid plan: {exc}") from exc


def replay_measured_plan_file(
    report: SensitivityReport,
    source_plan: str | Path,
    *,
    ax_engine_executable: str = "ax-engine-bench",
) -> QuantizationPlan:
    source_path = Path(source_plan).expanduser().resolve()
    return replay_measured_plan(
        report,
        read_data(source_path),
        source_file_sha256=file_sha256(source_path),
        ax_engine_executable=ax_engine_executable,
    )
