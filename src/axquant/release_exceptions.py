from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from axquant.errors import ValidationGateError
from axquant.identity import same_model_identity
from axquant.profiles import thresholds_for
from axquant.schema import (
    ArtifactSizeEvidence,
    QuantizationPlan,
    ReleaseException,
    ReleaseExceptionTarget,
    ValidationIssue,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256

_WEIGHT_SIZE_RATIO = "artifact.weight_size_ratio"
_CANDIDATE_MEASURED_BPW = "artifact.candidate_measured_bpw"


def _same_number(left: float | int, right: float | int) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    left_value = float(left)
    right_value = float(right)
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)
    )


def _targets_by_metric(
    exception: ReleaseException,
) -> dict[str, ReleaseExceptionTarget]:
    return {target.metric: target for target in exception.targets}


def validate_release_exception_semantics(
    exception: ReleaseException,
    *,
    plan: QuantizationPlan,
    validation: ValidationReport,
    now: datetime | None = None,
) -> None:
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        raise ValueError("release exception verification time must include a timezone")
    if checked_at < exception.approved_at:
        raise ValidationGateError("release exception approval is not yet effective")
    if checked_at >= exception.expires_at:
        raise ValidationGateError("release exception has expired")
    if stable_sha256(plan) != exception.plan_sha256:
        raise ValidationGateError("release exception does not bind the selected plan")
    if validation.thresholds != thresholds_for(validation.profile):
        raise ValidationGateError(
            "release exception validation does not use authoritative profile thresholds"
        )
    if not same_model_identity(validation.candidate_model, exception.candidate_model):
        raise ValidationGateError("release exception identifies a different candidate")

    targets = _targets_by_metric(exception)
    for metric, target in targets.items():
        observed = validation.comparisons.get(metric)
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not _same_number(
                observed,
                target.observed_value,
            )
        ):
            raise ValidationGateError(
                f"release exception observed value does not match validation: {metric}"
            )

    size_target = targets[_WEIGHT_SIZE_RATIO]
    if size_target.required_minimum is not None:
        raise ValidationGateError("weight-size exception cannot alter a minimum requirement")
    if size_target.required_maximum is None or not _same_number(
        size_target.required_maximum,
        validation.thresholds.max_weight_size_ratio,
    ):
        raise ValidationGateError("release exception does not bind the active weight-size limit")
    if size_target.observed_value <= validation.thresholds.max_weight_size_ratio:
        raise ValidationGateError("release exception weight-size target is not a failed gate")


def verify_release_exception(
    exception: ReleaseException,
    *,
    plan: QuantizationPlan,
    validation: ValidationReport,
    evidence_files: Mapping[str, str | Path],
    now: datetime | None = None,
) -> None:
    validate_release_exception_semantics(
        exception,
        plan=plan,
        validation=validation,
        now=now,
    )
    if set(evidence_files) != set(exception.evidence_sha256):
        raise ValidationGateError(
            "release exception evidence files do not match the approved evidence set"
        )

    resolved: dict[str, Path] = {}
    for name, value in evidence_files.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValidationGateError(f"release exception evidence is missing: {name}")
        try:
            digest = file_sha256(path)
        except OSError as exc:
            raise ValidationGateError(
                f"release exception evidence cannot be read: {name}: {exc}"
            ) from exc
        if digest != exception.evidence_sha256[name]:
            raise ValidationGateError(f"release exception evidence checksum changed: {name}")
        resolved[name] = path

    evidence_plan = load_model(resolved["plan"], QuantizationPlan)
    if stable_sha256(evidence_plan) != exception.plan_sha256 or evidence_plan != plan:
        raise ValidationGateError("release exception plan evidence differs from validation")

    candidate_size = load_model(resolved["candidate_size"], ArtifactSizeEvidence)
    size_reference = load_model(resolved["size_reference"], ArtifactSizeEvidence)
    if candidate_size.kind != "candidate" or size_reference.kind != "uniform-4bit":
        raise ValidationGateError("release exception size evidence has invalid kinds")
    if not same_model_identity(candidate_size.model, exception.candidate_model):
        raise ValidationGateError("release exception candidate-size identity differs")
    if candidate_size.logical_parameters != size_reference.logical_parameters:
        raise ValidationGateError("release exception size evidence parameter counts differ")

    ratio = candidate_size.weight_bytes / size_reference.weight_bytes
    expected_values = {
        _WEIGHT_SIZE_RATIO: ratio,
        _CANDIDATE_MEASURED_BPW: candidate_size.measured_bpw,
    }
    targets = _targets_by_metric(exception)
    for metric, expected in expected_values.items():
        if not _same_number(targets[metric].observed_value, expected):
            raise ValidationGateError(
                f"release exception target does not match size evidence: {metric}"
            )
    comparison_sources = {
        "artifact.candidate_source_sha256": candidate_size.source_sha256,
        "artifact.uniform4_source_sha256": size_reference.source_sha256,
        "artifact.candidate_weight_bytes": candidate_size.weight_bytes,
        "artifact.uniform4_weight_bytes": size_reference.weight_bytes,
        "artifact.logical_parameters": candidate_size.logical_parameters,
    }
    for metric, evidence_value in comparison_sources.items():
        if validation.comparisons.get(metric) != evidence_value:
            raise ValidationGateError(
                f"release exception evidence does not match validation: {metric}"
            )


def apply_release_exception(
    validation: ValidationReport,
    exception: ReleaseException,
    *,
    plan: QuantizationPlan,
    evidence_files: Mapping[str, str | Path],
    now: datetime | None = None,
) -> ValidationReport:
    if validation.release_exceptions:
        raise ValidationGateError("validation already contains a release exception")
    verify_release_exception(
        exception,
        plan=plan,
        validation=validation,
        evidence_files=evidence_files,
        now=now,
    )
    matching_errors = [
        issue
        for issue in validation.issues
        if issue.metric == _WEIGHT_SIZE_RATIO and issue.severity == "error"
    ]
    if len(matching_errors) != 1:
        raise ValidationGateError(
            "release exception requires exactly one failed weight-size validation issue"
        )

    issues = [
        (
            ValidationIssue(
                severity="warning",
                metric=issue.metric,
                message=f"{issue.message}; governed exception {exception.exception_id}",
            )
            if issue is matching_errors[0]
            else issue
        )
        for issue in validation.issues
    ]
    return ValidationReport.model_validate(
        {
            **validation.model_dump(mode="json"),
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "release_exceptions": [exception.model_dump(mode="json")],
            "passed": not any(issue.severity == "error" for issue in issues),
        }
    )


def release_exception_allows_size(
    validation: ValidationReport,
    *,
    plan: QuantizationPlan,
    now: datetime | None = None,
) -> ReleaseException:
    if len(validation.release_exceptions) != 1:
        raise ValidationGateError(
            "publication size overage requires one governed release exception"
        )
    exception = validation.release_exceptions[0]
    validate_release_exception_semantics(
        exception,
        plan=plan,
        validation=validation,
        now=now,
    )
    matching_warnings = [
        issue
        for issue in validation.issues
        if issue.metric == _WEIGHT_SIZE_RATIO
        and issue.severity == "warning"
        and exception.exception_id in issue.message
    ]
    if len(matching_warnings) != 1:
        raise ValidationGateError(
            "publication validation does not record the governed size exception"
        )
    if not validation.passed or any(issue.severity == "error" for issue in validation.issues):
        raise ValidationGateError("publication validation contains unexcepted failures")
    return exception
