from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from axquant.profiles import thresholds_for
from axquant.schema import (
    ArtifactSizeEvidence,
    CalibrationManifest,
    EvaluationBundle,
    HardwareMetrics,
    IntegrityMetrics,
    ModelIdentity,
    MtpAbComparison,
    MtpMetrics,
    ProfileName,
    QualityMetrics,
    RuntimeName,
    SoftwareVersions,
)
from axquant.validator import validate_evaluations

_SOURCE_REVISION = "a" * 40
_BASELINE_REVISION = "b" * 40
_CANDIDATE_REVISION = "c" * 40


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_quality_metrics_rejects_out_of_range_task_scores(score: float) -> None:
    with pytest.raises(ValidationError, match=r"task scores must be within \[0, 1\]"):
        QualityMetrics(task_scores={"coding": score})


def test_mtp_metrics_rejects_inconsistent_acceptance_and_rejection_rates() -> None:
    with pytest.raises(ValidationError, match="acceptance and rejection rates must sum to 1"):
        MtpMetrics(acceptance_rate=0.8, rejection_rate=0.3)

    assert MtpMetrics(acceptance_rate=0.8, rejection_rate=0.2)
    assert MtpMetrics(acceptance_rate=0.8)


def _calibration() -> CalibrationManifest:
    return CalibrationManifest(
        model=ModelIdentity(model_id="Qwen/Qwen3.6-27B", revision=_SOURCE_REVISION),
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
    kind: Literal["uniform-4bit", "uniform-6bit", "candidate"],
    weight_bytes: int,
) -> ArtifactSizeEvidence:
    model = (
        ModelIdentity(model_id="org/candidate", revision=_CANDIDATE_REVISION)
        if kind == "candidate"
        else ModelIdentity(model_id="org/uniform-4bit", revision=_BASELINE_REVISION)
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
            revision=_CANDIDATE_REVISION if candidate else _BASELINE_REVISION,
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
            "runtime_env": {},
        },
    )


def test_size_evidence_rejects_inconsistent_measured_bpw() -> None:
    with pytest.raises(ValueError, match="does not match byte accounting"):
        ArtifactSizeEvidence(
            kind="candidate",
            model=ModelIdentity(model_id="org/candidate", revision=_CANDIDATE_REVISION),
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


def test_validation_uses_bound_weighted_mtp_ab_and_absolute_exactness() -> None:
    profile = ProfileName.AGENT_CODING
    reference = _evaluation(mode="reference")
    assert reference.mtp is not None
    reference.mtp = reference.mtp.model_copy(update={"divergence_rate": None})
    direct = _evaluation(mode="direct")
    candidate = _evaluation(mode="mtp")
    assert candidate.mtp is not None
    candidate.mtp = candidate.mtp.model_copy(update={"divergence_rate": 0.0})
    candidate.hardware = candidate.hardware.model_copy(
        update={"mtp_effective_tokens_per_second": 112.0}
    )
    mtp_ab = MtpAbComparison(
        profile_name="benchmark-ab",
        model=candidate.model,
        runtime=RuntimeName.AX_ENGINE,
        workload=candidate.workload,
        dataset_sha256=candidate.dataset_sha256,
        random_seed=candidate.random_seed,
        generation_controls={
            key: candidate.benchmark_metadata[key]
            for key in (
                "prompt_count",
                "warmup_trials",
                "measured_trials",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "draft_depth",
                "power_mode",
                "quantizer",
                "quantizer_version",
            )
        },
        runtime_env={},
        draft_depth=3,
        exactness_pass=True,
        divergent_trial_count=0,
        measured_trial_count=5,
        failed_trial_count=0,
        direct_tokens_per_second_p50=100.0,
        mtp_tokens_per_second_p50=112.0,
        direct_token_weighted_decode_tps=100.0,
        mtp_token_weighted_decode_tps=125.0,
        prompt_median_speedup=1.12,
        token_weighted_decode_speedup=1.25,
        speedup_metric="token-weighted-decode-tps",
        speedup=1.25,
        minimum_speedup=1.20,
        minimum_prompt_median_speedup=1.10,
        prompt_median_speedup_pass=True,
        speedup_pass=True,
        release_ready=True,
        ax_engine_version="6.11.1",
        runtime_chip="M4 Max",
        software_versions=_versions(),
    )

    report = validate_evaluations(
        reference,
        direct,
        candidate,
        profile=profile,
        thresholds=thresholds_for(profile),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
        mtp_ab=mtp_ab,
    )

    assert report.passed is True
    assert report.comparisons["hardware.effective_speedup"] == 1.25
    assert report.comparisons["hardware.token_weighted_decode_speedup"] == 1.25
    assert report.comparisons["hardware.prompt_median_speedup"] == 1.12
    assert report.comparisons["mtp.divergence_rate"] == 0.0


def test_six_bit_validation_uses_uniform6_size_policy() -> None:
    profile = ProfileName.AGENT_CODING
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=profile,
        thresholds=thresholds_for(profile),
        target_class="6bit",
        calibration=_calibration(),
        size_reference=_size("uniform-6bit", 1000),
        candidate_size=_size("candidate", 1050),
    )

    assert report.passed is True
    assert report.target_class == "6bit"
    assert report.comparisons["artifact.size_reference_kind"] == "uniform-6bit"
    assert report.comparisons["artifact.uniform6_measured_bpw"] == 8.0


@pytest.mark.parametrize(
    ("target_class", "reference_kind"),
    [("4bit", "uniform-6bit"), ("6bit", "uniform-4bit")],
)
def test_validation_rejects_size_reference_from_another_target_class(
    target_class: Literal["4bit", "6bit"],
    reference_kind: Literal["uniform-4bit", "uniform-6bit"],
) -> None:
    profile = ProfileName.AGENT_CODING
    report = validate_evaluations(
        _evaluation(mode="reference"),
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=profile,
        thresholds=thresholds_for(profile),
        target_class=target_class,
        calibration=_calibration(),
        size_reference=_size(reference_kind, 1000),
        candidate_size=_size("candidate", 1050),
    )

    assert report.passed is False
    assert any(issue.metric == "artifact.size_reference.kind" for issue in report.issues)


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


def test_validation_requires_mtp_token_accuracy_horizons() -> None:
    reference = _evaluation(mode="reference")
    candidate = _evaluation(mode="mtp")
    assert reference.mtp is not None
    assert candidate.mtp is not None
    reference.mtp.token_accuracy = {}
    candidate.mtp.token_accuracy = {}
    report = validate_evaluations(
        reference,
        _evaluation(mode="direct"),
        candidate,
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )
    assert report.passed is False
    assert any(issue.metric == "mtp.token_accuracy" for issue in report.issues)


def test_validation_rejects_size_evidence_for_another_candidate() -> None:
    candidate_size = _size("candidate", 1050).model_copy(
        update={"model": ModelIdentity(model_id="org/another", revision="d" * 40)}
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


def test_validation_preserves_zero_mtp_throughput() -> None:
    candidate = _evaluation(mode="mtp")
    candidate.hardware.mtp_effective_tokens_per_second = 0.0
    candidate.hardware.decode_tokens_per_second = 1_000.0

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
    assert report.comparisons["hardware.effective_speedup"] == 0.0
    assert any(issue.metric == "hardware.effective_speedup" for issue in report.issues)


def test_validation_rejects_nonpositive_reference_peak_memory() -> None:
    reference = _evaluation(mode="reference")
    reference.hardware.peak_memory_bytes = 0

    report = validate_evaluations(
        reference,
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )

    assert report.passed is False
    assert "hardware.peak_memory_ratio" not in report.comparisons
    assert any(
        issue.metric == "hardware.peak_memory_ratio" and "must be positive" in issue.message
        for issue in report.issues
    )


def test_validation_rejects_empty_or_inconsistent_benchmark_metadata() -> None:
    candidate = _evaluation(mode="mtp")
    candidate.benchmark_metadata["quantizer"] = ""
    candidate.benchmark_metadata["successful_measured_trials"] = 1

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
    issue_metrics = {issue.metric for issue in report.issues}
    assert "candidate.benchmark_metadata.quantizer" in issue_metrics
    assert "candidate.benchmark_metadata.successful_measured_trials" in issue_metrics


def test_validation_binds_runtime_environment() -> None:
    candidate = _evaluation(mode="mtp")
    candidate.benchmark_metadata["runtime_env"] = {"AX_ENGINE_FAST_FFN_MATVEC": "1"}

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
    assert any(
        issue.metric == "candidate.benchmark_metadata.runtime_env"
        and "controls differ" in issue.message
        for issue in report.issues
    )


def test_validation_binds_reference_benchmark_controls() -> None:
    reference = _evaluation(mode="reference")
    reference.benchmark_metadata["runtime_env"] = {"AX_ENGINE_FAST_FFN_MATVEC": "0"}
    report = validate_evaluations(
        reference,
        _evaluation(mode="direct"),
        _evaluation(mode="mtp"),
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )
    assert report.passed is False
    assert any(
        issue.metric == "reference.benchmark_metadata.runtime_env"
        and "controls differ" in issue.message
        for issue in report.issues
    )


def test_validation_requires_matched_reference_seed_runtime_and_baseline_kinds() -> None:
    reference = _evaluation(mode="reference")
    reference.random_seed = 8
    reference.runtime = RuntimeName.MLX_LM
    reference.baseline_kind = "bf16"
    candidate_direct = _evaluation(mode="direct")
    candidate_direct.baseline_kind = "candidate"
    candidate = _evaluation(mode="mtp")
    candidate.baseline_kind = "candidate"

    report = validate_evaluations(
        reference,
        candidate_direct,
        candidate,
        profile=ProfileName.AGENT_CODING,
        thresholds=thresholds_for(ProfileName.AGENT_CODING),
        calibration=_calibration(),
        size_reference=_size("uniform-4bit", 1000),
        candidate_size=_size("candidate", 1050),
    )

    assert report.passed is False
    issue_metrics = {issue.metric for issue in report.issues}
    assert {
        "reference.random_seed",
        "reference.runtime",
        "reference.baseline_kind",
        "candidate_direct.baseline_kind",
        "candidate.baseline_kind",
    }.issubset(issue_metrics)
