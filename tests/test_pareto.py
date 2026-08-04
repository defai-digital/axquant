from __future__ import annotations

import pytest

from axquant.errors import RefinementError
from axquant.pareto import build_pareto_report
from axquant.schema import (
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    ModelIdentity,
    ProfileName,
    RefinementMeasurementSet,
)


def _measurement(
    candidate_id: str,
    *,
    measurement_id: str | None = None,
    bpw: float,
    quality: float,
    acceptance: float,
    speedup: float,
    memory: float,
    passed: bool = True,
    profile: ProfileName = ProfileName.AGENT_CODING,
    chip: str = "M4 Max",
) -> CompleteCandidateMeasurement:
    return CompleteCandidateMeasurement(
        candidate_id=candidate_id,
        measurement_id=measurement_id or candidate_id,
        candidate_model=ModelIdentity(
            model_id=f"AutomatosX/{candidate_id}",
            revision=f"{candidate_id}-revision",
        ),
        profile=profile,
        plan_sha256=f"{candidate_id}-plan",
        artifact_manifest_sha256=f"{candidate_id}-artifact",
        quality_comparison_sha256=f"{candidate_id}-quality",
        validation_sha256=f"{candidate_id}-validation",
        measured_bpw=bpw,
        objective_loss=max(0.0, 1.0 - quality),
        quality_retention=quality,
        mtp_acceptance_retention=acceptance,
        mtp_speedup=speedup,
        peak_memory_ratio=memory,
        hardware=CompleteCandidateHardware(
            device_name="Test Mac",
            chip=chip,
            unified_memory_bytes=128 * 1024**3,
            os_version="macOS",
            ax_engine_version="6.11.1",
            mlx_version="0.32.0",
            mlx_lm_version="0.31.0",
            power_mode="AC power",
            kernel_fallbacks=0,
        ),
        validation_passed=passed,
    )


def test_pareto_report_marks_dominance_and_tradeoffs() -> None:
    measurements = RefinementMeasurementSet(
        refinement_sha256="refinement",
        evaluator_version="test",
        measurements=[
            _measurement(
                "balanced",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.3,
                memory=0.8,
            ),
            _measurement(
                "dominated",
                bpw=5.2,
                quality=0.99,
                acceptance=0.97,
                speedup=1.2,
                memory=0.9,
            ),
            _measurement(
                "smaller",
                bpw=4.9,
                quality=0.99,
                acceptance=0.97,
                speedup=1.2,
                memory=0.85,
            ),
            _measurement(
                "failed",
                bpw=4.8,
                quality=1.0,
                acceptance=0.99,
                speedup=1.4,
                memory=0.7,
                passed=False,
            ),
        ],
    )
    report = build_pareto_report(measurements)
    assert report.frontier_candidate_ids == ["balanced", "smaller"]
    points = {point.candidate_id: point for point in report.points}
    assert points["dominated"].dominated_by == ["balanced", "smaller"]
    assert points["failed"].frontier is False
    assert report.measurement_set_sha256


def test_pareto_report_rejects_mixed_profiles() -> None:
    measurements = RefinementMeasurementSet(
        refinement_sha256="refinement",
        evaluator_version="test",
        measurements=[
            _measurement(
                "agent",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.3,
                memory=0.8,
            ),
            _measurement(
                "coding",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.3,
                memory=0.8,
                profile=ProfileName.CODING,
            ),
        ],
    )
    with pytest.raises(RefinementError, match="one workload profile"):
        build_pareto_report(measurements)


def test_pareto_report_tracks_multiple_hosts_for_one_candidate() -> None:
    measurements = RefinementMeasurementSet(
        refinement_sha256="refinement",
        evaluator_version="test",
        measurements=[
            _measurement(
                "candidate",
                measurement_id="candidate-m3-max",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.3,
                memory=0.8,
            ),
            _measurement(
                "candidate",
                measurement_id="candidate-m4-max",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.4,
                memory=0.8,
            ),
        ],
    )

    report = build_pareto_report(measurements)

    assert report.frontier_candidate_ids == ["candidate"]
    points = {point.measurement_id: point for point in report.points}
    assert points["candidate-m4-max"].frontier
    assert points["candidate-m3-max"].dominated_by == ["candidate-m4-max"]


def test_pareto_report_does_not_compare_different_hardware() -> None:
    measurements = RefinementMeasurementSet(
        refinement_sha256="refinement",
        evaluator_version="test",
        measurements=[
            _measurement(
                "candidate",
                measurement_id="candidate-m3",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.2,
                memory=0.8,
                chip="M3 Max",
            ),
            _measurement(
                "candidate",
                measurement_id="candidate-m4",
                bpw=5.0,
                quality=1.0,
                acceptance=0.98,
                speedup=1.4,
                memory=0.8,
                chip="M4 Max",
            ),
        ],
    )

    report = build_pareto_report(measurements)
    points = {point.measurement_id: point for point in report.points}

    assert points["candidate-m3"].frontier
    assert points["candidate-m4"].frontier
    assert not points["candidate-m3"].dominated_by
    assert not points["candidate-m4"].dominated_by
