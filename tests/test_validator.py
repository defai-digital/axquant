from __future__ import annotations

from typing import Literal

import pytest

from axquant.profiles import thresholds_for
from axquant.schema import (
    ArtifactSizeEvidence,
    CalibrationManifest,
    EvaluationBundle,
    HardwareMetrics,
    IntegrityMetrics,
    ModelIdentity,
    MtpMetrics,
    ProfileName,
    QualityMetrics,
    SoftwareVersions,
)
from axquant.validator import validate_evaluations


def _calibration() -> CalibrationManifest:
    return CalibrationManifest(
        model=ModelIdentity(model_id="Qwen/Qwen3.6-27B", revision="source-revision"),
        profile=ProfileName.AGENT_CODING,
        dataset_id="calibration-v1",
        dataset_sha256="calibration-sha",
        samples=128,
        domains=["coding", "json", "tool", "multilingual", "long-context"],
        sequence_length=2048,
        random_seed=0,
        calibration_evaluation_separation_attested=True,
    )


def _versions() -> SoftwareVersions:
    return SoftwareVersions(
        axquant="0.1.0a0",
        python="3.13",
        mlx="0.32",
        mlx_lm="0.31",
        ax_engine="6.11.1",
        safetensors="0.6",
        pydantic="2.11",
    )


def _size(
    kind: Literal["uniform-4bit", "candidate"],
    weight_bytes: int,
) -> ArtifactSizeEvidence:
    model = (
        ModelIdentity(model_id="org/candidate", revision="candidate-rev")
        if kind == "candidate"
        else ModelIdentity(model_id="org/uniform-4bit", revision="uniform-4bit-revision")
    )
    return ArtifactSizeEvidence(
        kind=kind,
        model=model,
        logical_parameters=1000,
        weight_bytes=weight_bytes,
        measured_bpw=8.0 * weight_bytes / 1000,
        source_sha256=f"{kind}-sha",
    )


def _evaluation(
    *,
    mode: str,
    acceptance: float = 0.79,
) -> EvaluationBundle:
    candidate = mode != "reference"
    mtp_enabled = mode == "mtp"
    return EvaluationBundle(
        model=ModelIdentity(
            model_id="org/candidate" if candidate else "org/uniform-6",
            revision="candidate-rev" if candidate else "baseline-rev",
        ),
        baseline_kind=(
            "uniform-6bit"
            if mode == "reference"
            else "axquant-mtp-on"
            if mode == "mtp"
            else "axquant-mtp-off"
        ),
        mtp_enabled=mtp_enabled,
        quality=QualityMetrics(
            perplexity=10.1 if candidate else 10.0,
            task_scores={
                task: 0.795 if candidate else 0.8
                for task in ("coding", "tool", "json", "multilingual", "long_context")
            },
            json_valid_rate=0.995,
            syntax_valid_rate=0.99,
        ),
        mtp=MtpMetrics(
            token_accuracy={"1": 0.78 if candidate else 0.8, "2": 0.68 if candidate else 0.7},
            average_accepted_tokens=1.5,
            acceptance_rate=acceptance if candidate else 0.8,
            rejection_rate=1.0 - (acceptance if candidate else 0.8),
            effective_tokens_per_forward=1.6,
            repetition_rate=0.011 if candidate else 0.01,
            divergence_rate=0.006 if candidate else 0.005,
        ),
        hardware=HardwareMetrics(
            peak_memory_bytes=800 if candidate else 1000,
            decode_tokens_per_second=105 if mode == "mtp" else 100,
            mtp_effective_tokens_per_second=125 if mode == "mtp" else None,
            device_name="Test Mac",
            chip="M4 Max",
            unified_memory_bytes=128 * 1024**3,
            os_version="macOS",
            kernel_fallbacks=0,
        ),
        integrity=IntegrityMetrics(
            safetensors_valid=True,
            index_complete=True,
            config_valid=True,
            mtp_layout_valid=True,
            source_revision_pinned=True,
        ),
        workload="agent-coding-v1",
        dataset_sha256="a" * 64,
        software_versions=_versions(),
        random_seed=7,
        benchmark_metadata={
            "prompt_count": 5,
            "warmup_trials": 2,
            "measured_trials": 5,
            "successful_measured_trials": 5,
            "failed_trials": 0,
            "timed_out_trials": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 512,
            "draft_depth": 3,
            "power_mode": "AC power",
            "quantizer": "axquant",
            "quantizer_version": "0.1.0a0",
            "ax_engine_version": "6.11.1",
            "quality_dataset_sha256": "b" * 64,
        },
    )


def test_size_evidence_rejects_inconsistent_measured_bpw() -> None:
    with pytest.raises(ValueError, match="does not match byte accounting"):
        ArtifactSizeEvidence(
            kind="candidate",
            model=ModelIdentity(model_id="org/candidate", revision="candidate-rev"),
            logical_parameters=1000,
            weight_bytes=1000,
            measured_bpw=7.0,
            source_sha256="candidate-sha",
        )


def test_validation_passes_complete_candidate() -> None:
    profile = ProfileName.AGENT_CODING
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=profile,
        thresholds=thresholds_for(profile),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )
    assert report.passed is True
    assert report.comparisons["hardware.effective_speedup"] == 1.25
    assert report.comparisons["hardware.kernel_fallbacks"] == 0
    assert report.comparisons["software.mlx_lm"] == "0.31"
    assert report.comparisons["hardware.power_mode"] == "AC power"


def test_validation_rejects_kernel_fallbacks() -> None:
    candidate = _evaluation(mode="mtp")
    candidate.hardware.kernel_fallbacks = 1
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        candidate,
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )
    assert report.passed is False
    assert any(issue.metric == "candidate.hardware.kernel_fallbacks" for issue in report.issues)


def test_validation_rejects_power_mode_mismatch() -> None:
    candidate = _evaluation(mode="mtp")
    candidate.benchmark_metadata["power_mode"] = "battery"
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        candidate,
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )
    assert report.passed is False
    assert any(issue.metric == "candidate.benchmark_metadata.power_mode" for issue in report.issues)


def test_validation_fails_mtp_retention_regression() -> None:
    profile = ProfileName.AGENT_CODING
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp", acceptance=0.70),
        profile=profile,
        thresholds=thresholds_for(profile),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )
    assert report.passed is False
    assert any(issue.metric == "mtp.acceptance_rate" for issue in report.issues)


def test_validation_rejects_size_evidence_for_another_candidate() -> None:
    candidate_size = _size("candidate", 1050).model_copy(
        update={"model": ModelIdentity(model_id="org/another", revision="another-rev")}
    )
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=candidate_size,
    )
    assert report.passed is False
    assert any(issue.metric == "artifact.candidate_size.model" for issue in report.issues)


def test_validation_fails_calibration_evaluation_overlap() -> None:
    profile = ProfileName.AGENT_CODING
    calibration = _calibration().model_copy(update={"dataset_sha256": "b" * 64})

    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=profile,
        thresholds=thresholds_for(profile),
        calibration=calibration,
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )

    assert report.passed is False
    assert any(issue.metric == "calibration.quality_dataset_sha256" for issue in report.issues)


def test_validation_fails_weight_size_gate() -> None:
    profile = ProfileName.AGENT_CODING
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=profile,
        thresholds=thresholds_for(profile),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1200),
    )
    assert report.passed is False
    assert report.comparisons["artifact.weight_size_ratio"] == 1.2
    assert any(issue.metric == "artifact.weight_size_ratio" for issue in report.issues)
