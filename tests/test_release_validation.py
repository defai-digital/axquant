from __future__ import annotations

from pathlib import Path

import pytest

from axquant.cli import main
from axquant.release_validation import build_release_validation_index
from axquant.schema import (
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    EvaluationBundle,
    IntegrityMetrics,
    ModelIdentity,
    ProfileName,
    ReleaseValidationIndex,
    ReleaseValidationInput,
    ReleaseValidationRequest,
    RuntimeName,
    SoftwareVersions,
    ValidationIssue,
    ValidationReport,
    ValidationThresholds,
)
from axquant.serde import file_sha256, load_model, write_data


def _profile_evidence(
    tmp_path: Path,
    *,
    profile: ProfileName,
    dataset_sha256: str,
    candidate: ModelIdentity,
) -> ReleaseValidationInput:
    reference = ModelIdentity(
        model_id="Qwen/Qwen3.6-27B-MLX-6bit",
        revision="6" * 40,
    )
    benchmark_entries: list[BenchmarkEvidenceEntry] = []
    for kind in BenchmarkEvidenceKind:
        if kind in {
            BenchmarkEvidenceKind.MIXED_PRECISION,
            BenchmarkEvidenceKind.AWQ,
            BenchmarkEvidenceKind.DWQ,
        }:
            benchmark_entries.append(
                BenchmarkEvidenceEntry(
                    kind=kind,
                    status="unavailable",
                    unavailable_reason=f"{kind.value} is unavailable",
                )
            )
            continue
        model = (
            reference
            if kind == BenchmarkEvidenceKind.UNIFORM_6BIT
            else candidate
            if kind
            in {
                BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
                BenchmarkEvidenceKind.AXQUANT_MTP_ON,
            }
            else ModelIdentity(
                model_id=f"Qwen/Qwen3.6-27B-{kind.value}",
                revision="b" * 40,
            )
        )
        evaluation_path = tmp_path / f"{profile.value}-{kind.value}.json"
        write_data(
            evaluation_path,
            EvaluationBundle(
                model=model,
                runtime=RuntimeName.AX_ENGINE,
                baseline_kind=kind.value,
                mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
                integrity=IntegrityMetrics(
                    safetensors_valid=True,
                    index_complete=True,
                    config_valid=True,
                    source_revision_pinned=True,
                ),
                workload=profile.value,
                dataset_sha256=dataset_sha256,
                software_versions=SoftwareVersions(
                    axquant="test",
                    python="3.13",
                    mlx="test",
                    mlx_lm="test",
                    ax_engine="test",
                    safetensors="test",
                    pydantic="test",
                ),
                random_seed=7,
            ),
        )
        benchmark_entries.append(
            BenchmarkEvidenceEntry(
                kind=kind,
                status="available",
                evaluation_file=str(evaluation_path),
                evaluation_sha256=file_sha256(evaluation_path),
                model=model,
                runtime=RuntimeName.AX_ENGINE,
                mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
            )
        )
    benchmark_path = tmp_path / f"{profile.value}-benchmark-index.json"
    write_data(
        benchmark_path,
        BenchmarkEvidenceIndex(
            profile=profile,
            dataset_sha256=dataset_sha256,
            random_seed=7,
            entries=benchmark_entries,
            release_ready=True,
            issues=[],
        ),
    )
    validation_path = tmp_path / f"{profile.value}-validation.json"
    write_data(
        validation_path,
        ValidationReport(
            reference_model=reference,
            candidate_model=candidate,
            profile=profile,
            passed=True,
            thresholds=ValidationThresholds(),
            issues=[],
            comparisons={},
        ),
    )
    return ReleaseValidationInput(
        profile=profile,
        validation_file=str(validation_path),
        benchmark_index_file=str(benchmark_path),
    )


def _request(tmp_path: Path) -> ReleaseValidationRequest:
    candidate = ModelIdentity(
        model_id="AutomatosX/candidate",
        revision="c" * 40,
        local_path="/models/candidate",
    )
    return ReleaseValidationRequest(
        entries=[
            _profile_evidence(
                tmp_path,
                profile=ProfileName.AGENT_CODING,
                dataset_sha256="a" * 64,
                candidate=candidate,
            ),
            _profile_evidence(
                tmp_path,
                profile=ProfileName.GENERAL,
                dataset_sha256="b" * 64,
                candidate=candidate,
            ),
        ]
    )


def test_release_validation_index_requires_both_disjoint_profiles(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    output = tmp_path / "release-validation-index.json"
    write_data(request_path, _request(tmp_path))

    assert (
        main(
            [
                "validation-index",
                "--request",
                str(request_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    index = load_model(output, ReleaseValidationIndex)
    assert index.release_ready
    assert {entry.profile for entry in index.entries} == {
        ProfileName.AGENT_CODING,
        ProfileName.GENERAL,
    }


def test_release_validation_rejects_mutable_candidate_revision(tmp_path: Path) -> None:
    request = _request(tmp_path)
    agent = next(entry for entry in request.entries if entry.profile == ProfileName.AGENT_CODING)
    validation = load_model(agent.validation_file, ValidationReport)
    validation.candidate_model.revision = "main"
    write_data(agent.validation_file, validation)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_release_validation_index(request_path)

    assert not index.release_ready
    assert "agent-coding candidate revision is not immutable" in index.issues


def test_release_validation_index_rejects_reused_dataset(tmp_path: Path) -> None:
    request = _request(tmp_path)
    general = next(entry for entry in request.entries if entry.profile == ProfileName.GENERAL)
    benchmark = load_model(general.benchmark_index_file, BenchmarkEvidenceIndex)
    benchmark.dataset_sha256 = "a" * 64
    write_data(general.benchmark_index_file, benchmark)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_release_validation_index(request_path)

    assert not index.release_ready
    assert "required profiles must use distinct benchmark datasets" in index.issues


def test_release_validation_rechecks_benchmark_evaluation_checksums(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    agent = next(entry for entry in request.entries if entry.profile == ProfileName.AGENT_CODING)
    benchmark = load_model(agent.benchmark_index_file, BenchmarkEvidenceIndex)
    evidence = next(
        entry for entry in benchmark.entries if entry.kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON
    )
    assert evidence.evaluation_file is not None
    Path(evidence.evaluation_file).write_text("{}", encoding="utf-8")
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_release_validation_index(request_path)

    assert not index.release_ready
    assert any(
        "agent-coding/axquant-mtp-on benchmark evaluation checksum changed" in issue
        for issue in index.issues
    )


def test_release_validation_rederives_required_baseline_availability(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    agent = next(entry for entry in request.entries if entry.profile == ProfileName.AGENT_CODING)
    benchmark = load_model(agent.benchmark_index_file, BenchmarkEvidenceIndex)
    benchmark.entries = [
        BenchmarkEvidenceEntry(
            kind=entry.kind,
            status="unavailable",
            unavailable_reason="forged unavailable baseline",
        )
        if entry.kind == BenchmarkEvidenceKind.BF16
        else entry
        for entry in benchmark.entries
    ]
    write_data(agent.benchmark_index_file, benchmark)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_release_validation_index(request_path)

    assert not index.release_ready
    assert any(
        "required benchmark evidence is unavailable: bf16" in issue for issue in index.issues
    )


def test_release_validation_request_cannot_omit_general(tmp_path: Path) -> None:
    payload = _request(tmp_path).model_dump(mode="json")
    payload["entries"] = [
        entry for entry in payload["entries"] if entry["profile"] != ProfileName.GENERAL
    ]

    with pytest.raises(ValueError, match=r"at least 2|exactly agent-coding and general"):
        ReleaseValidationRequest.model_validate(payload)


def test_release_validation_index_rejects_different_candidates(tmp_path: Path) -> None:
    request = _request(tmp_path)
    general = next(entry for entry in request.entries if entry.profile == ProfileName.GENERAL)
    validation = load_model(general.validation_file, ValidationReport)
    validation.candidate_model = ModelIdentity(
        model_id="AutomatosX/other-candidate",
        revision="d" * 40,
    )
    write_data(general.validation_file, validation)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_release_validation_index(request_path)

    assert not index.release_ready
    assert "general candidate evidence model differs" in index.issues
    assert "required profiles validate different candidate models" in index.issues


@pytest.mark.parametrize(
    ("passed", "issues"),
    [
        (
            True,
            [
                ValidationIssue(
                    severity="error",
                    metric="quality.perplexity",
                    message="quality threshold failed",
                )
            ],
        ),
        (False, []),
    ],
)
def test_validation_report_status_must_match_error_issues(
    passed: bool,
    issues: list[ValidationIssue],
) -> None:
    reference = ModelIdentity(model_id="reference", revision="b" * 40)
    candidate = ModelIdentity(model_id="candidate", revision="c" * 40)

    with pytest.raises(ValueError, match="validation report status is inconsistent"):
        ValidationReport(
            reference_model=reference,
            candidate_model=candidate,
            profile=ProfileName.AGENT_CODING,
            passed=passed,
            thresholds=ValidationThresholds(),
            issues=issues,
            comparisons={},
        )


def test_release_validation_index_requires_every_entry_to_pass(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    write_data(request_path, _request(tmp_path))
    payload = build_release_validation_index(request_path).model_dump(mode="json")
    payload["entries"][0]["passed"] = False

    with pytest.raises(ValueError, match="release validation status is inconsistent"):
        ReleaseValidationIndex.model_validate(payload)
