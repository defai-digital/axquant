from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.errors import ValidationGateError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.profiles import thresholds_for
from axquant.release_exceptions import (
    apply_release_exception,
    release_exception_allows_size,
    verify_release_exception,
)
from axquant.schema import (
    ArtifactSizeEvidence,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    ReleaseException,
    ReleaseExceptionTarget,
    ValidationIssue,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data, write_text

_SOURCE_REVISION = "a" * 40
_BASELINE_REVISION = "b" * 40
_CANDIDATE_REVISION = "c" * 40


def _plan(qwen36_model_dir: Path):
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision=_SOURCE_REVISION,
    )
    sensitivity = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
    )
    return plan_quantization(
        sensitivity,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )


def _evidence(tmp_path: Path, qwen36_model_dir: Path):
    plan = _plan(qwen36_model_dir)
    candidate_model = ModelIdentity(
        model_id="AutomatosX/Qwen3.6-27B-AXQuant",
        revision=_CANDIDATE_REVISION,
    )
    candidate_size = ArtifactSizeEvidence(
        kind="candidate",
        model=candidate_model,
        logical_parameters=1000,
        weight_bytes=720,
        measured_bpw=5.76,
        source_sha256="a" * 64,
    )
    size_reference = ArtifactSizeEvidence(
        kind="uniform-4bit",
        model=ModelIdentity(
            model_id="Qwen/Qwen3.6-27B-MLX-4bit",
            revision=_BASELINE_REVISION,
        ),
        logical_parameters=1000,
        weight_bytes=500,
        measured_bpw=4.0,
        source_sha256="b" * 64,
    )
    paths = {
        "plan": tmp_path / "plan.json",
        "candidate_size": tmp_path / "candidate-size.json",
        "size_reference": tmp_path / "size-reference.json",
        "tradeoff": tmp_path / "tradeoff.json",
    }
    write_data(paths["plan"], plan)
    write_data(paths["candidate_size"], candidate_size)
    write_data(paths["size_reference"], size_reference)
    write_data(paths["tradeoff"], {"quality_retention": 1.0, "peak_memory_ratio": 0.85})
    validation = ValidationReport(
        reference_model=ModelIdentity(
            model_id="Qwen/Qwen3.6-27B-MLX-6bit",
            revision=_BASELINE_REVISION,
        ),
        candidate_model=candidate_model,
        profile=ProfileName.AGENT_CODING,
        passed=False,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        issues=[
            ValidationIssue(
                severity="error",
                metric="artifact.weight_size_ratio",
                message="ratio 1.4400 exceeds 1.1000",
            )
        ],
        comparisons={
            "artifact.weight_size_ratio": 1.44,
            "artifact.candidate_measured_bpw": 5.76,
            "artifact.candidate_source_sha256": candidate_size.source_sha256,
            "artifact.uniform4_source_sha256": size_reference.source_sha256,
            "artifact.candidate_weight_bytes": candidate_size.weight_bytes,
            "artifact.uniform4_weight_bytes": size_reference.weight_bytes,
            "artifact.logical_parameters": candidate_size.logical_parameters,
        },
    )
    approved_at = datetime.now(UTC) - timedelta(days=1)
    exception = ReleaseException(
        exception_id="AXQ-SIZE-001",
        candidate_model=candidate_model,
        plan_sha256=stable_sha256(plan),
        targets=[
            ReleaseExceptionTarget(
                metric="artifact.weight_size_ratio",
                observed_value=1.44,
                required_maximum=1.1,
                requirement="candidate must be no more than 110% of uniform 4-bit",
            ),
            ReleaseExceptionTarget(
                metric="artifact.candidate_measured_bpw",
                observed_value=5.76,
                required_minimum=4.3,
                required_maximum=4.8,
                requirement="candidate must remain within the target BPW range",
            ),
        ],
        measured_tradeoff="Quality retained and peak memory is 85% of uniform 6-bit.",
        owner="AutomatosX release owner",
        approved_by="Release authority",
        approval_reference="decision-001",
        approved_at=approved_at,
        expires_at=approved_at + timedelta(days=30),
        evidence_sha256={name: file_sha256(path) for name, path in paths.items()},
    )
    return plan, validation, exception, paths


def test_governed_size_exception_changes_only_the_size_gate(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> None:
    plan, validation, exception, paths = _evidence(tmp_path, qwen36_model_dir)

    result = apply_release_exception(
        validation,
        exception,
        plan=plan,
        evidence_files=paths,
    )

    assert result.passed
    assert result.release_exceptions == [exception]
    assert result.issues == [
        ValidationIssue(
            severity="warning",
            metric="artifact.weight_size_ratio",
            message="ratio 1.4400 exceeds 1.1000; governed exception AXQ-SIZE-001",
        )
    ]
    assert release_exception_allows_size(result, plan=plan) == exception


def test_release_exception_rejects_mutable_candidate_revision(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> None:
    _plan_model, _validation, exception, _paths = _evidence(tmp_path, qwen36_model_dir)
    payload = exception.model_dump(mode="json")
    payload["candidate_model"]["revision"] = "main"

    with pytest.raises(ValueError, match="candidate revision must be immutable"):
        ReleaseException.model_validate(payload)


def test_governed_size_exception_does_not_waive_an_unrelated_failure(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> None:
    plan, validation, exception, paths = _evidence(tmp_path, qwen36_model_dir)
    validation.issues.append(
        ValidationIssue(
            severity="error",
            metric="hardware.effective_speedup",
            message="speedup is below the required minimum",
        )
    )

    result = apply_release_exception(
        validation,
        exception,
        plan=plan,
        evidence_files=paths,
    )

    assert not result.passed
    assert any(
        issue.metric == "hardware.effective_speedup" and issue.severity == "error"
        for issue in result.issues
    )


def test_release_exception_rejects_tampered_evidence(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> None:
    plan, validation, exception, paths = _evidence(tmp_path, qwen36_model_dir)
    write_text(paths["tradeoff"], "tampered\n")

    with pytest.raises(ValidationGateError, match="checksum changed: tradeoff"):
        verify_release_exception(
            exception,
            plan=plan,
            validation=validation,
            evidence_files=paths,
        )


def test_release_exception_rejects_stale_thresholds_and_boolean_metrics(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> None:
    plan, validation, exception, paths = _evidence(tmp_path, qwen36_model_dir)
    with pytest.raises(ValidationGateError, match="authoritative profile thresholds"):
        verify_release_exception(
            exception,
            plan=plan,
            validation=validation.model_copy(
                update={
                    "thresholds": validation.thresholds.model_copy(
                        update={"min_effective_speedup": 1.0}
                    )
                }
            ),
            evidence_files=paths,
        )

    comparisons = dict(validation.comparisons)
    comparisons["artifact.weight_size_ratio"] = True
    with pytest.raises(ValidationGateError, match="observed value does not match"):
        verify_release_exception(
            exception,
            plan=plan,
            validation=validation.model_copy(update={"comparisons": comparisons}),
            evidence_files=paths,
        )


def test_release_exception_cli_records_computed_size_values(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> None:
    _plan_model, _validation, _exception, paths = _evidence(tmp_path, qwen36_model_dir)
    output = tmp_path / "approved-exception.json"
    now = datetime.now(UTC)

    assert (
        main(
            [
                "release-exception",
                "--exception-id",
                "AXQ-SIZE-CLI",
                "--plan",
                str(paths["plan"]),
                "--candidate-size",
                str(paths["candidate_size"]),
                "--size-reference",
                str(paths["size_reference"]),
                "--tradeoff-evidence",
                str(paths["tradeoff"]),
                "--measured-tradeoff",
                "Measured quality and memory tradeoff.",
                "--owner",
                "AutomatosX release owner",
                "--approved-by",
                "Release authority",
                "--approval-reference",
                "decision-cli",
                "--approved-at",
                now.isoformat(),
                "--expires-at",
                (now + timedelta(days=30)).isoformat(),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    exception = load_model(output, ReleaseException)
    targets = {target.metric: target for target in exception.targets}
    assert targets["artifact.weight_size_ratio"].observed_value == 1.44
    assert targets["artifact.candidate_measured_bpw"].observed_value == 5.76
    assert exception.evidence_sha256["tradeoff"] == file_sha256(paths["tradeoff"])
