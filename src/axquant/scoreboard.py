"""Certification scoreboard artifact (P0).

Compacts plan size, quality retention, and optional second-tier MTP gates into one auditable page.
Missing mandatory rows are listed with reasons (never silently dropped). Checkpoint certification
is the first tier; MTP acceleration is mandatory only when explicitly requested as a second tier.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, PlanningError
from axquant.identity import same_model_identity
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


def _positive_finite(value: float, label: str, *, at_most_one: bool = False) -> None:
    if not math.isfinite(value) or value <= 0.0 or (at_most_one and value > 1.0):
        suffix = " within (0, 1]" if at_most_one else " positive and finite"
        raise PlanningError(f"{label} must be{suffix}")


def _optional_load(path: str | Path | None, model_type: type[Any]) -> Any | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactError(f"scoreboard input does not exist: {resolved}")
    return load_model(resolved, model_type)


def _positive_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or numerator <= 0.0 or denominator <= 0.0:
        return None
    return numerator / denominator


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
    evaluation_profile: ProfileName | None = None,
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
    minimum_mtp_prompt_median_speedup: float = 1.10,
    require_mtp_acceleration: bool = False,
) -> ScoreboardReport:
    """Build a scoreboard from a plan plus optional evidence artifacts."""
    plan_model = plan if isinstance(plan, QuantizationPlan) else load_model(plan, QuantizationPlan)
    if profile is not None and profile != plan_model.profile:
        raise PlanningError("scoreboard profile does not match the quantization plan")
    active_profile = evaluation_profile or plan_model.profile
    _positive_finite(minimum_quality_retention, "minimum quality retention")
    _positive_finite(max_size_ratio_to_uniform4, "maximum size ratio")
    _positive_finite(
        minimum_mtp_acceptance_retention,
        "minimum MTP acceptance retention",
        at_most_one=True,
    )
    _positive_finite(minimum_mtp_speedup, "minimum MTP speedup")
    _positive_finite(
        minimum_mtp_prompt_median_speedup,
        "minimum MTP prompt-median speedup",
    )
    rows: list[ScoreboardMetricRow] = []
    size_reference_kind = "uniform-6bit" if plan_model.target_class == "6bit" else "uniform-4bit"
    size_reference_label = "uniform-6" if size_reference_kind == "uniform-6bit" else "uniform-4"
    size_metric_id = (
        "size_ratio_vs_uniform6"
        if size_reference_kind == "uniform-6bit"
        else "size_ratio_vs_uniform4"
    )

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
    if cand_size is not None and cand_size.kind != "candidate":
        raise ArtifactError("candidate size evidence has the wrong kind")
    if ref_size is not None and ref_size.kind != size_reference_kind:
        raise ArtifactError(
            f"{plan_model.target_class} scoreboard requires {size_reference_kind} size evidence"
        )
    if (
        cand_size is not None
        and ref_size is not None
        and cand_size.logical_parameters != ref_size.logical_parameters
    ):
        raise ArtifactError("candidate and size reference logical parameter counts differ")
    if cand_size is not None and ref_size is not None:
        ratio = cand_size.weight_bytes / ref_size.weight_bytes
        status = "pass" if ratio <= max_size_ratio_to_uniform4 else "fail"
        rows.append(
            _row(
                size_metric_id,
                f"Size ratio vs {size_reference_label} reference",
                status=status,
                value=round(ratio, 6),
                threshold=max_size_ratio_to_uniform4,
                unit="ratio",
            )
        )
    else:
        rows.append(
            _row(
                size_metric_id,
                f"Size ratio vs {size_reference_label} reference",
                status="unavailable",
                threshold=max_size_ratio_to_uniform4,
                unit="ratio",
                reason="provide --candidate-size and --size-reference ArtifactSizeEvidence",
            )
        )

    quality = _optional_load(quality_comparison, QualityComparisonReport)
    if quality is not None:
        aggregate = quality.aggregate
        if abs(aggregate.delta - (aggregate.candidate - aggregate.reference)) > 1e-9:
            raise ArtifactError("quality comparison aggregate delta is inconsistent")
        expected_retention = (
            aggregate.candidate / aggregate.reference if aggregate.reference > 0.0 else None
        )
        if (expected_retention is None) != (aggregate.retention is None) or (
            expected_retention is not None
            and aggregate.retention is not None
            and abs(aggregate.retention - expected_retention) > 1e-9
        ):
            raise ArtifactError("quality comparison aggregate retention is inconsistent")
        if cand_size is not None and not same_model_identity(
            quality.candidate_model, cand_size.model
        ):
            raise ArtifactError("quality comparison and size evidence use different candidates")
        if ref_size is not None and not same_model_identity(
            quality.reference_model, ref_size.model
        ):
            raise ArtifactError("quality comparison and size evidence use different references")
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
        if validation.profile != active_profile:
            raise ArtifactError("validation report profile does not match the scoreboard")
        if (
            plan_model.target_class in {"4bit", "6bit"}
            and validation.target_class != plan_model.target_class
        ):
            raise ArtifactError("validation report target class does not match the scoreboard")
        if cand_size is not None and not same_model_identity(
            validation.candidate_model, cand_size.model
        ):
            raise ArtifactError("validation report and size evidence use different candidates")
        if ref_size is not None and not same_model_identity(
            validation.reference_model, ref_size.model
        ):
            raise ArtifactError("validation report and size evidence use different references")
        if quality is not None and (
            not same_model_identity(validation.candidate_model, quality.candidate_model)
            or not same_model_identity(validation.reference_model, quality.reference_model)
        ):
            raise ArtifactError("validation report and quality comparison identities differ")
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
        if mtp.workload is not None and mtp.workload != active_profile.value:
            raise ArtifactError("MTP A/B workload does not match the scoreboard profile")
        if mtp.model is not None:
            candidate_identities = [
                evidence.model for evidence in (cand_size,) if evidence is not None
            ]
            if quality is not None:
                candidate_identities.append(quality.candidate_model)
            if validation is not None:
                candidate_identities.append(validation.candidate_model)
            if any(
                not same_model_identity(identity, mtp.model) for identity in candidate_identities
            ):
                raise ArtifactError("MTP A/B and candidate evidence identify different checkpoints")
        # The producer legitimately emits exactness_pass=False with zero
        # divergent trials (failed-trial / trial-set / MTP-inactive issues),
        # so only the tampered direction is inconsistent.
        if mtp.exactness_pass and mtp.divergent_trial_count > 0:
            raise ArtifactError("MTP A/B exactness status is inconsistent with divergent trials")
        measured_prompt_speedup = _positive_ratio(
            mtp.mtp_tokens_per_second_p50,
            mtp.direct_tokens_per_second_p50,
        )
        if (
            mtp.prompt_median_speedup is not None
            and measured_prompt_speedup is not None
            and abs(mtp.prompt_median_speedup - measured_prompt_speedup) > 1e-9
        ):
            raise ArtifactError("MTP A/B prompt-median speedup is inconsistent")
        prompt_speedup = (
            mtp.prompt_median_speedup
            if mtp.prompt_median_speedup is not None
            else measured_prompt_speedup
        )
        measured_weighted_speedup = _positive_ratio(
            mtp.mtp_token_weighted_decode_tps,
            mtp.direct_token_weighted_decode_tps,
        )
        if (
            mtp.token_weighted_decode_speedup is not None
            and measured_weighted_speedup is not None
            and abs(mtp.token_weighted_decode_speedup - measured_weighted_speedup) > 1e-9
        ):
            raise ArtifactError("MTP A/B token-weighted speedup is inconsistent")
        measured_speedup = (
            measured_prompt_speedup
            if mtp.speedup_metric == "prompt-median-tps"
            else measured_weighted_speedup
        )
        if mtp.speedup is None:
            if mtp.speedup_pass:
                raise ArtifactError("MTP A/B speed status is inconsistent with missing speedup")
        elif measured_speedup is not None and abs(mtp.speedup - measured_speedup) > 1e-9:
            raise ArtifactError("MTP A/B speedup does not match its selected metric")
        prompt_guardrail_pass = (
            prompt_speedup is not None and prompt_speedup >= mtp.minimum_prompt_median_speedup
        )
        if (
            mtp.prompt_median_speedup is not None
            and mtp.prompt_median_speedup_pass != prompt_guardrail_pass
        ):
            raise ArtifactError("MTP A/B prompt-median guardrail status is inconsistent")
        if measured_speedup is None and mtp.speedup_metric == "prompt-median-tps":
            expected_speedup_pass = mtp.speedup is not None and mtp.speedup >= mtp.minimum_speedup
        else:
            expected_speedup_pass = (
                mtp.speedup is not None
                and measured_speedup is not None
                and mtp.speedup >= mtp.minimum_speedup
                and prompt_guardrail_pass
            )
        if mtp.speedup_pass != expected_speedup_pass:
            raise ArtifactError("MTP A/B speed status is inconsistent with its thresholds")
        if mtp.release_ready:
            if mtp.measured_trial_count < 1 or mtp.failed_trial_count:
                raise ArtifactError("release-ready MTP A/B evidence has no complete trials")
            if (
                mtp.direct_tokens_per_second_p50 is None
                or mtp.direct_tokens_per_second_p50 <= 0.0
                or mtp.mtp_tokens_per_second_p50 is None
                or mtp.mtp_tokens_per_second_p50 <= 0.0
                or mtp.speedup is None
            ):
                raise ArtifactError("release-ready MTP A/B evidence is missing throughput")
            if mtp.speedup_metric == "token-weighted-decode-tps" and (
                mtp.direct_token_weighted_decode_tps is None
                or mtp.direct_token_weighted_decode_tps <= 0.0
                or mtp.mtp_token_weighted_decode_tps is None
                or mtp.mtp_token_weighted_decode_tps <= 0.0
                or mtp.token_weighted_decode_speedup is None
            ):
                raise ArtifactError("release-ready MTP A/B evidence is missing weighted throughput")
        # Keep exactness and both speed gates orthogonal in the scoreboard. The
        # producer's release_ready bit proves the evidence is fully bound, but
        # its configured thresholds may be weaker than the public policy here.
        exactness_status = (
            "fail" if not mtp.exactness_pass else "pass" if mtp.release_ready else "unavailable"
        )
        rows.append(
            _row(
                "mtp_exactness",
                "MTP greedy exactness",
                status=exactness_status,
                value="exact" if mtp.exactness_pass else "divergent",
                threshold="exact",
                reason=(
                    "MTP A/B evidence is not release-ready"
                    if mtp.exactness_pass and not mtp.release_ready
                    else None
                ),
                notes=[
                    f"divergent_trials={mtp.divergent_trial_count}",
                    f"measured_trials={mtp.measured_trial_count}",
                    f"policy_acceptance_floor={minimum_mtp_acceptance_retention}",
                ],
            )
        )

        weighted_speedup = (
            mtp.token_weighted_decode_speedup
            if mtp.token_weighted_decode_speedup is not None
            else measured_weighted_speedup
        )
        if weighted_speedup is not None:
            status = (
                "fail"
                if float(weighted_speedup) < minimum_mtp_speedup
                else "pass"
                if mtp.release_ready
                else "unavailable"
            )
            rows.append(
                _row(
                    "mtp_speedup",
                    "MTP token-weighted decode speedup",
                    status=status,
                    value=round(float(weighted_speedup), 6),
                    threshold=minimum_mtp_speedup,
                    unit="ratio",
                    owner="ax-engine",
                    reason=(
                        "MTP A/B evidence is not release-ready" if status == "unavailable" else None
                    ),
                    notes=[
                        "Planner/artifact side is independent of decode pipeline speed.",
                        f"Residual gap vs {minimum_mtp_speedup:.2f}x is AX Engine ownership "
                        "(async draft / overlap).",
                        "policy_metric=token-weighted-decode-tps",
                        f"evidence_selected_metric={mtp.speedup_metric}",
                    ],
                )
            )
        else:
            rows.append(
                _row(
                    "mtp_speedup",
                    "MTP token-weighted decode speedup",
                    status="unavailable",
                    threshold=minimum_mtp_speedup,
                    unit="ratio",
                    owner="ax-engine",
                    reason="MTP A/B present but token-weighted decode speedup is not recorded",
                    notes=["Speed gate owner: AX Engine runtime, not the quant planner."],
                )
            )

        if prompt_speedup is not None:
            prompt_status = (
                "fail"
                if float(prompt_speedup) < minimum_mtp_prompt_median_speedup
                else "pass"
                if mtp.release_ready
                else "unavailable"
            )
            rows.append(
                _row(
                    "mtp_prompt_median_speedup",
                    "MTP prompt-median speedup",
                    status=prompt_status,
                    value=round(float(prompt_speedup), 6),
                    threshold=minimum_mtp_prompt_median_speedup,
                    unit="ratio",
                    owner="ax-engine",
                    reason=(
                        "MTP A/B evidence is not release-ready"
                        if prompt_status == "unavailable"
                        else None
                    ),
                    notes=[
                        "This guardrail protects typical-prompt latency from a "
                        "weighted-average win.",
                        f"evidence_prompt_floor={mtp.minimum_prompt_median_speedup}",
                    ],
                )
            )
        else:
            rows.append(
                _row(
                    "mtp_prompt_median_speedup",
                    "MTP prompt-median speedup",
                    status="unavailable",
                    threshold=minimum_mtp_prompt_median_speedup,
                    unit="ratio",
                    owner="ax-engine",
                    reason="MTP A/B present but prompt-median speedup is not recorded",
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
        rows.append(
            _row(
                "mtp_prompt_median_speedup",
                "MTP prompt-median speedup",
                status="unavailable",
                threshold=minimum_mtp_prompt_median_speedup,
                unit="ratio",
                owner="ax-engine",
                reason="provide --mtp-ab (engine-owned prompt guardrail)",
            )
        )

    # Optional evaluation presence checks (do not require parsing full quality).
    for metric_id, label, path, candidate in (
        (
            "candidate_evaluation",
            "Candidate evaluation bundle",
            candidate_evaluation,
            True,
        ),
        (
            "reference_evaluation",
            "Reference evaluation bundle",
            reference_evaluation,
            False,
        ),
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
            evaluation = _optional_load(path, EvaluationBundle)
            if evaluation is None:
                raise ArtifactError(f"{label.lower()} could not be loaded")
            if evaluation.workload != active_profile.value:
                raise ArtifactError(f"{label.lower()} profile does not match the scoreboard")
            expected_identities = []
            if candidate:
                if cand_size is not None:
                    expected_identities.append(cand_size.model)
                if quality is not None:
                    expected_identities.append(quality.candidate_model)
                if validation is not None:
                    expected_identities.append(validation.candidate_model)
                if mtp is not None and mtp.model is not None:
                    expected_identities.append(mtp.model)
            else:
                if quality is not None:
                    expected_identities.append(quality.reference_model)
                if validation is not None:
                    expected_identities.append(validation.reference_model)
            if any(
                not same_model_identity(identity, evaluation.model)
                for identity in expected_identities
            ):
                raise ArtifactError(f"{label.lower()} identity differs from bound evidence")
            rows.append(_row(metric_id, label, status="available", value=str(Path(path).name)))

    mandatory = [
        "effective_bpw",
        size_metric_id,
        "quality_retention",
    ]
    if require_mtp_acceleration:
        mandatory.extend(["mtp_exactness", "mtp_speedup", "mtp_prompt_median_speedup"])
    missing = [
        row.metric_id for row in rows if row.metric_id in mandatory and row.status == "unavailable"
    ]
    fails = [
        row.metric_id
        for row in rows
        if row.status == "fail"
        and (
            require_mtp_acceleration
            or row.metric_id not in {"mtp_exactness", "mtp_speedup", "mtp_prompt_median_speedup"}
        )
    ]
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
        if require_mtp_acceleration:
            warnings.append(
                "Second-tier MTP acceleration is incomplete or failing; AX Engine owns the "
                "remaining exactness and speed work."
            )
        else:
            warnings.append(
                "MTP acceleration is not part of this first-tier checkpoint verdict; sidecar "
                "presence does not certify speculative-decode speed or exactness."
            )

    return ScoreboardReport(
        certification_tier=("mtp-acceleration" if require_mtp_acceleration else "checkpoint"),
        title=title or f"AXQuant scoreboard ({active_profile.value})",
        plan_profile=plan_model.profile,
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
        f"- Plan profile: `{(report.plan_profile or report.profile).value}`",
        f"- Evaluation profile: `{report.profile.value}`",
        f"- Certification tier: `{report.certification_tier}`",
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
    if not report.evidence_kind.release_quality:
        raise PlanningError("scoreboard plan evidence is not eligible for certification claims")
    if report.overall_status == "incomplete":
        missing = ", ".join(report.missing_mandatory) or "unknown"
        raise PlanningError(f"scoreboard is incomplete for certification; missing: {missing}")
    if report.overall_status == "fail":
        failed = [row.metric_id for row in report.rows if row.status == "fail"]
        raise PlanningError(f"scoreboard has failing gates: {', '.join(failed)}")
