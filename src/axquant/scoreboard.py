"""Certification scoreboard artifact (P0).

Compacts plan size, quality retention, and MTP gates into one auditable page.
Missing mandatory rows are listed with reasons (never silently dropped).
MTP speed ownership is recorded as AX Engine when the planner/artifact side is closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, PlanningError
from axquant.schema import (
    ArtifactSizeEvidence,
    EvaluationBundle,
    MtpAbComparison,
    ProfileName,
    QualityComparisonReport,
    QuantizationPlan,
    ScoreboardMetricRow,
    ScoreboardReport,
    ValidationReport,
)
from axquant.serde import load_model, stable_sha256


def _optional_load(path: str | Path | None, model_type: type[Any]) -> Any | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactError(f"scoreboard input does not exist: {resolved}")
    return load_model(resolved, model_type)


def _row(
    metric_id: str,
    label: str,
    *,
    status: str,
    value: float | str | None = None,
    threshold: float | str | None = None,
    unit: str | None = None,
    owner: str = "axquant",
    reason: str | None = None,
    notes: list[str] | None = None,
) -> ScoreboardMetricRow:
    return ScoreboardMetricRow(
        metric_id=metric_id,
        label=label,
        status=status,
        value=value,
        threshold=threshold,
        unit=unit,
        owner=owner,
        reason=reason,
        notes=notes or [],
    )


def build_scoreboard(
    *,
    plan: str | Path | QuantizationPlan,
    profile: ProfileName | None = None,
    title: str | None = None,
    candidate_size: str | Path | None = None,
    size_reference: str | Path | None = None,
    quality_comparison: str | Path | None = None,
    validation_report: str | Path | None = None,
    mtp_ab: str | Path | None = None,
    candidate_evaluation: str | Path | None = None,
    reference_evaluation: str | Path | None = None,
    minimum_quality_retention: float = 0.98,
    max_size_ratio_to_uniform4: float = 1.10,
    minimum_mtp_acceptance_retention: float = 0.95,
    minimum_mtp_speedup: float = 1.20,
) -> ScoreboardReport:
    """Build a scoreboard from a plan plus optional evidence artifacts."""
    plan_model = plan if isinstance(plan, QuantizationPlan) else load_model(plan, QuantizationPlan)
    active_profile = profile or plan_model.profile
    rows: list[ScoreboardMetricRow] = []

    rows.append(
        _row(
            "effective_bpw",
            "Effective BPW",
            status="available",
            value=round(plan_model.effective_bpw, 6),
            threshold=plan_model.target_bpw,
            unit="bpw",
            notes=[
                f"target_bpw={plan_model.target_bpw}",
                f"evidence={plan_model.evidence_kind.value}",
            ],
        )
    )
    rows.append(
        _row(
            "evidence_kind",
            "Plan evidence kind",
            status="available",
            value=plan_model.evidence_kind.value,
            notes=["Release claims require measured or imported evidence."],
        )
    )

    cand_size = _optional_load(candidate_size, ArtifactSizeEvidence)
    ref_size = _optional_load(size_reference, ArtifactSizeEvidence)
    if cand_size is not None and ref_size is not None and ref_size.weight_bytes > 0:
        ratio = cand_size.weight_bytes / ref_size.weight_bytes
        status = "pass" if ratio <= max_size_ratio_to_uniform4 else "fail"
        rows.append(
            _row(
                "size_ratio_vs_uniform4",
                "Size ratio vs uniform-4 reference",
                status=status,
                value=round(ratio, 6),
                threshold=max_size_ratio_to_uniform4,
                unit="ratio",
            )
        )
    else:
        rows.append(
            _row(
                "size_ratio_vs_uniform4",
                "Size ratio vs uniform-4 reference",
                status="unavailable",
                threshold=max_size_ratio_to_uniform4,
                unit="ratio",
                reason="provide --candidate-size and --size-reference ArtifactSizeEvidence",
            )
        )

    quality = _optional_load(quality_comparison, QualityComparisonReport)
    if quality is not None and quality.aggregate.retention is not None:
        retention = float(quality.aggregate.retention)
        status = "pass" if retention >= minimum_quality_retention else "fail"
        rows.append(
            _row(
                "quality_retention",
                "Quality retention",
                status=status,
                value=round(retention, 6),
                threshold=minimum_quality_retention,
                unit="ratio",
            )
        )
    elif quality is not None:
        rows.append(
            _row(
                "quality_retention",
                "Quality retention",
                status="unavailable",
                threshold=minimum_quality_retention,
                reason="quality comparison present but aggregate.retention is null",
            )
        )
    else:
        rows.append(
            _row(
                "quality_retention",
                "Quality retention",
                status="unavailable",
                threshold=minimum_quality_retention,
                unit="ratio",
                reason="provide --quality-comparison",
            )
        )

    validation = _optional_load(validation_report, ValidationReport)
    if validation is not None:
        rows.append(
            _row(
                "validation_gate",
                "Validation report",
                status="pass" if validation.passed else "fail",
                value="passed" if validation.passed else "failed",
            )
        )
    else:
        rows.append(
            _row(
                "validation_gate",
                "Validation report",
                status="unavailable",
                reason="provide --validation-report",
            )
        )

    mtp = _optional_load(mtp_ab, MtpAbComparison)
    if mtp is not None:
        # MtpAbComparison records greedy exactness; acceptance retention is not a
        # separate field — exactness is the quality-side MTP gate in this toolkit.
        rows.append(
            _row(
                "mtp_exactness",
                "MTP greedy exactness",
                status="pass" if mtp.exactness_pass else "fail",
                value="exact" if mtp.exactness_pass else "divergent",
                threshold="exact",
                notes=[
                    f"divergent_trials={mtp.divergent_trial_count}",
                    f"measured_trials={mtp.measured_trial_count}",
                    f"policy_acceptance_floor={minimum_mtp_acceptance_retention}",
                ],
            )
        )
        if mtp.speedup is not None:
            status = "pass" if float(mtp.speedup) >= minimum_mtp_speedup else "fail"
            rows.append(
                _row(
                    "mtp_speedup",
                    "MTP speedup",
                    status=status,
                    value=round(float(mtp.speedup), 6),
                    threshold=minimum_mtp_speedup,
                    unit="ratio",
                    owner="ax-engine",
                    notes=[
                        "Planner/artifact side is independent of decode pipeline speed.",
                        "Residual gap vs 1.20x is AX Engine ownership (async draft / overlap).",
                        f"speedup_pass={mtp.speedup_pass}",
                    ],
                )
            )
        else:
            rows.append(
                _row(
                    "mtp_speedup",
                    "MTP speedup",
                    status="unavailable",
                    threshold=minimum_mtp_speedup,
                    unit="ratio",
                    owner="ax-engine",
                    reason="MTP A/B present but speedup not recorded",
                    notes=["Speed gate owner: AX Engine runtime, not the quant planner."],
                )
            )
    else:
        rows.append(
            _row(
                "mtp_exactness",
                "MTP greedy exactness",
                status="unavailable",
                threshold="exact",
                reason="provide --mtp-ab",
            )
        )
        rows.append(
            _row(
                "mtp_speedup",
                "MTP speedup",
                status="unavailable",
                threshold=minimum_mtp_speedup,
                unit="ratio",
                owner="ax-engine",
                reason="provide --mtp-ab (engine-owned speed gate)",
                notes=[
                    "Speed residual is engine pipelining work, not bit allocation.",
                ],
            )
        )

    # Optional evaluation presence checks (do not require parsing full quality).
    for metric_id, label, path in (
        ("candidate_evaluation", "Candidate evaluation bundle", candidate_evaluation),
        ("reference_evaluation", "Reference evaluation bundle", reference_evaluation),
    ):
        if path is None:
            rows.append(
                _row(
                    metric_id,
                    label,
                    status="unavailable",
                    reason=f"provide --{metric_id.replace('_', '-')}",
                )
            )
        else:
            _optional_load(path, EvaluationBundle)
            rows.append(_row(metric_id, label, status="available", value=str(Path(path).name)))

    mandatory = [
        "effective_bpw",
        "size_ratio_vs_uniform4",
        "quality_retention",
        "mtp_exactness",
        "mtp_speedup",
    ]
    missing = [
        row.metric_id for row in rows if row.metric_id in mandatory and row.status == "unavailable"
    ]
    fails = [row.metric_id for row in rows if row.status == "fail"]
    if fails:
        overall = "fail"
    elif missing:
        overall = "incomplete"
    else:
        overall = "pass"

    warnings: list[str] = []
    if plan_model.evidence_kind.value == "architecture_prior":
        warnings.append(
            "Plan evidence is architecture_prior; scoreboard cannot support release claims."
        )
    if any(row.owner == "ax-engine" and row.status in {"fail", "unavailable"} for row in rows):
        warnings.append(
            "MTP speed gate is owned by AX Engine; freeze planner/recipe while the runtime "
            "closes residual speedup."
        )

    return ScoreboardReport(
        title=title or f"AXQuant scoreboard ({active_profile.value})",
        profile=active_profile,
        source_model=plan_model.source_model,
        plan_sha256=stable_sha256(plan_model),
        evidence_kind=plan_model.evidence_kind,
        overall_status=overall,
        rows=rows,
        missing_mandatory=missing,
        warnings=warnings,
    )


def scoreboard_markdown(report: ScoreboardReport) -> str:
    """Render the scoreboard as Markdown."""
    lines = [
        f"# {report.title}",
        "",
        f"- Model: `{report.source_model.model_id}`",
        f"- Revision: `{report.source_model.revision or 'unpinned'}`",
        f"- Profile: `{report.profile.value}`",
        f"- Plan digest: `{report.plan_sha256}`",
        f"- Evidence: `{report.evidence_kind.value}`",
        f"- **Overall:** `{report.overall_status}`",
        "",
        "| Metric | Status | Value | Threshold | Owner | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        value = "—" if row.value is None else str(row.value)
        threshold = "—" if row.threshold is None else str(row.threshold)
        if row.unit and row.value is not None:
            value = f"{value} {row.unit}"
        note = row.reason or ("; ".join(row.notes) if row.notes else "")
        lines.append(
            f"| {row.label} | `{row.status}` | {value} | {threshold} | `{row.owner}` | {note} |"
        )
    if report.missing_mandatory:
        lines.append("")
        lines.append("## Missing mandatory rows")
        lines.append("")
        for item in report.missing_mandatory:
            lines.append(f"- `{item}`")
    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append(
        "Unavailable rows are listed with reasons (AXQ-022 discipline). "
        "MTP speedup is AX Engine-owned."
    )
    lines.append("")
    return "\n".join(lines)


def require_scoreboard_inputs_for_certification(report: ScoreboardReport) -> None:
    """Fail closed when a scoreboard is incomplete for certification narrative."""
    if report.overall_status == "incomplete":
        missing = ", ".join(report.missing_mandatory) or "unknown"
        raise PlanningError(f"scoreboard is incomplete for certification; missing: {missing}")
    if report.overall_status == "fail":
        failed = [row.metric_id for row in report.rows if row.status == "fail"]
        raise PlanningError(f"scoreboard has failing gates: {', '.join(failed)}")
