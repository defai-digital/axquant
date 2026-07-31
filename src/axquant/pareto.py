from __future__ import annotations

from axquant.errors import RefinementError
from axquant.schema import (
    CompleteCandidateMeasurement,
    ParetoPoint,
    ParetoReport,
    RefinementMeasurementSet,
)
from axquant.serde import stable_sha256


def _dominates(
    left: CompleteCandidateMeasurement,
    right: CompleteCandidateMeasurement,
) -> bool:
    no_worse = (
        left.measured_bpw <= right.measured_bpw
        and left.quality_retention >= right.quality_retention
        and left.mtp_acceptance_retention >= right.mtp_acceptance_retention
        and left.mtp_speedup >= right.mtp_speedup
        and left.peak_memory_ratio <= right.peak_memory_ratio
    )
    strictly_better = (
        left.measured_bpw < right.measured_bpw
        or left.quality_retention > right.quality_retention
        or left.mtp_acceptance_retention > right.mtp_acceptance_retention
        or left.mtp_speedup > right.mtp_speedup
        or left.peak_memory_ratio < right.peak_memory_ratio
    )
    return no_worse and strictly_better


def build_pareto_report(measurements: RefinementMeasurementSet) -> ParetoReport:
    profiles = {measurement.profile for measurement in measurements.measurements}
    if len(profiles) != 1:
        raise RefinementError("Pareto measurements must use one workload profile")
    passing: list[CompleteCandidateMeasurement] = []
    for measurement in measurements.measurements:
        if measurement.validation_passed:
            passing.append(measurement)
    points: list[ParetoPoint] = []
    for measurement in sorted(measurements.measurements, key=lambda item: item.measurement_id):
        dominators = sorted(
            candidate.measurement_id
            for candidate in passing
            if candidate.measurement_id != measurement.measurement_id
            and _dominates(candidate, measurement)
        )
        frontier = measurement.validation_passed and not dominators
        points.append(
            ParetoPoint(
                candidate_id=measurement.candidate_id,
                measurement_id=measurement.measurement_id,
                candidate_model=measurement.candidate_model,
                plan_sha256=measurement.plan_sha256,
                measured_bpw=measurement.measured_bpw,
                quality_retention=measurement.quality_retention,
                mtp_acceptance_retention=measurement.mtp_acceptance_retention,
                mtp_speedup=measurement.mtp_speedup,
                peak_memory_ratio=measurement.peak_memory_ratio,
                hardware=measurement.hardware,
                validation_passed=measurement.validation_passed,
                frontier=frontier,
                dominated_by=dominators,
            )
        )
    frontier_candidate_ids = sorted({point.candidate_id for point in points if point.frontier})
    return ParetoReport(
        profile=next(iter(profiles)),
        measurement_set_sha256=stable_sha256(measurements),
        points=points,
        frontier_candidate_ids=frontier_candidate_ids,
    )
