from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from axquant.benchmark import result_to_evaluation_bundle, validate_ab_invariant
from axquant.errors import ArtifactError, BenchmarkError, RefinementError
from axquant.identity import same_model_identity
from axquant.refinement import build_complete_candidate_measurement
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    Allocation,
    ArtifactManifest,
    BenchmarkResult,
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    EvaluationBundle,
    HardwareKernelCoverage,
    HardwareMeasurementProtocol,
    HardwareProfileRegistry,
    HardwareRegistryEntry,
    HardwareRegistryRequest,
    QualityComparisonReport,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantMethod,
    RefinementMeasurementSet,
    SensitivityReport,
    TrialResult,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _resolved(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _required_file(base: Path, value: str, label: str) -> Path:
    path = _resolved(base, value)
    if not path.is_file():
        raise ArtifactError(f"{label} does not exist: {path}")
    return path


def _required_text(value: str | None, label: str) -> str:
    if not value or not value.strip():
        raise RefinementError(f"hardware registry evidence is missing {label}")
    return value


def _required_positive_int(value: int | None, label: str) -> int:
    if value is None or value <= 0:
        raise RefinementError(f"hardware registry evidence is missing {label}")
    return value


def _measured_trials(result: BenchmarkResult) -> list[TrialResult]:
    return [trial for trial in result.trials if not trial.is_warmup and trial.success]


def _commands(result: BenchmarkResult) -> list[list[str]]:
    successful = [trial for trial in result.trials if trial.success]
    if any(
        not trial.command or any(not argument.strip() for argument in trial.command)
        for trial in successful
    ):
        raise RefinementError(
            "hardware registry benchmark result contains a non-executable command"
        )
    commands = [trial.command for trial in successful]
    if not commands:
        raise RefinementError("hardware registry benchmark result has no executable commands")
    return commands


def _kernel_fallbacks(result: BenchmarkResult) -> int:
    return sum(trial.kernel_fallbacks or 0 for trial in _measured_trials(result))


def _same_finite_number(left: object, right: object) -> bool:
    if (
        not isinstance(left, (int, float))
        or isinstance(left, bool)
        or not isinstance(right, (int, float))
        or isinstance(right, bool)
    ):
        return False
    left_value = float(left)
    right_value = float(right)
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)
    )


def _check_result_bundle(
    *,
    label: str,
    result: BenchmarkResult,
    bundle: EvaluationBundle,
    mtp_enabled: bool,
    issues: list[str],
    kernel_issues: list[str],
) -> None:
    config = result.config
    if not same_model_identity(config.model, bundle.model):
        issues.append(f"{label} raw result and evaluation identify different checkpoints")
    if config.runtime != bundle.runtime:
        issues.append(f"{label} raw result and evaluation use different runtimes")
    if config.mtp_enabled != mtp_enabled or bundle.mtp_enabled != mtp_enabled:
        issues.append(f"{label} MTP mode is inconsistent")
    expected_kind = "axquant-mtp-on" if mtp_enabled else "axquant-mtp-off"
    if config.baseline_kind != expected_kind or bundle.baseline_kind != expected_kind:
        issues.append(f"{label} baseline kind is inconsistent")
    if config.workload != bundle.workload:
        issues.append(f"{label} raw result and evaluation use different workloads")
    if config.dataset_sha256 != bundle.dataset_sha256:
        issues.append(f"{label} raw result and evaluation use different datasets")
    if config.random_seed != bundle.random_seed:
        issues.append(f"{label} raw result and evaluation use different random seeds")
    if result.failed_count or result.timed_out_count:
        message = f"{label} raw benchmark contains failed or timed-out trials"
        issues.append(message)
        kernel_issues.append(message)
    actual_failed = sum(not trial.success for trial in result.trials)
    if actual_failed != result.failed_count:
        message = f"{label} raw benchmark failed-trial count is inconsistent"
        issues.append(message)
        kernel_issues.append(message)
    if result.measured_count != config.measured_trials:
        message = f"{label} raw benchmark did not complete every measured trial"
        issues.append(message)
        kernel_issues.append(message)
    measured = _measured_trials(result)
    warmups = [trial for trial in result.trials if trial.is_warmup]
    if len(measured) != result.measured_count:
        message = f"{label} raw benchmark measured count is inconsistent"
        issues.append(message)
        kernel_issues.append(message)
    if len(warmups) != config.warmup_trials:
        message = f"{label} raw benchmark warmup count is inconsistent"
        issues.append(message)
        kernel_issues.append(message)
    if len(result.trials) != config.warmup_trials + config.measured_trials:
        message = f"{label} raw benchmark trial count is inconsistent"
        issues.append(message)
        kernel_issues.append(message)
    trial_indices = [trial.trial_index for trial in result.trials]
    if len(trial_indices) != len(set(trial_indices)):
        message = f"{label} raw benchmark contains duplicate trial indices"
        issues.append(message)
        kernel_issues.append(message)
    for trial in measured:
        if not trial.command or any(not argument.strip() for argument in trial.command):
            message = f"{label} measured trial has no executable command"
            issues.append(message)
            kernel_issues.append(message)
        if (
            trial.prompt_tokens <= 0
            or trial.tokens_generated <= 0
            or trial.latency_seconds <= 0.0
            or trial.tokens_per_second <= 0.0
        ):
            message = f"{label} measured trial contains vacuous throughput evidence"
            issues.append(message)
            kernel_issues.append(message)
        if trial.kernel_fallbacks is None:
            message = f"{label} measured trial is missing kernel fallback telemetry"
            issues.append(message)
            kernel_issues.append(message)
        if trial.peak_memory_bytes is None or trial.peak_memory_bytes <= 0:
            message = f"{label} measured trial is missing positive peak-memory telemetry"
            issues.append(message)
            kernel_issues.append(message)
        trial_identity = {
            "device_name": trial.runtime_device_name,
            "chip": trial.runtime_chip,
            "unified_memory_bytes": trial.unified_memory_bytes,
            "os_version": trial.os_version,
        }
        result_identity = {
            "device_name": result.runtime_device_name,
            "chip": result.runtime_chip,
            "unified_memory_bytes": result.unified_memory_bytes,
            "os_version": result.os_version,
        }
        if trial_identity != result_identity:
            message = f"{label} measured trial hardware identity differs from its result"
            issues.append(message)
            kernel_issues.append(message)
        if mtp_enabled:
            if (
                trial.mtp_active is not True
                or trial.mtp_proposed_tokens is None
                or trial.mtp_proposed_tokens <= 0
                or trial.mtp_accepted_tokens is None
                or trial.mtp_accepted_tokens > trial.mtp_proposed_tokens
                or trial.mtp_decode_steps is None
                or trial.mtp_decode_steps <= 0
            ):
                message = f"{label} measured trial has incomplete MTP telemetry"
                issues.append(message)
                kernel_issues.append(message)
        elif any(
            value is not None
            for value in (
                trial.mtp_accepted_tokens,
                trial.mtp_proposed_tokens,
                trial.mtp_rejected_tokens,
                trial.mtp_decode_steps,
                trial.mtp_active,
            )
        ):
            message = f"{label} direct measured trial contains MTP telemetry"
            issues.append(message)
            kernel_issues.append(message)
    if result.software_versions is None:
        message = f"{label} raw benchmark is missing software-version evidence"
        issues.append(message)
        kernel_issues.append(message)
    elif result.software_versions != bundle.software_versions:
        message = f"{label} raw result and evaluation use different software versions"
        issues.append(message)
        kernel_issues.append(message)

    rebuilt: EvaluationBundle | None
    try:
        rebuilt = result_to_evaluation_bundle(
            result,
            software_versions=bundle.software_versions,
        )
    except BenchmarkError as exc:
        rebuilt = None
        message = f"{label} raw benchmark cannot rebuild evaluation metrics: {exc}"
        issues.append(message)
        kernel_issues.append(message)
    if rebuilt is not None:
        for field_name in (
            "peak_memory_bytes",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "mtp_effective_tokens_per_second",
            "kernel_fallbacks",
            "device_name",
            "chip",
            "unified_memory_bytes",
            "os_version",
        ):
            if getattr(bundle.hardware, field_name) != getattr(rebuilt.hardware, field_name):
                message = f"{label} evaluation {field_name} does not match raw trials"
                issues.append(message)
                kernel_issues.append(message)
        if mtp_enabled:
            if bundle.mtp is None or rebuilt.mtp is None:
                message = f"{label} evaluation is missing measured MTP metrics"
                issues.append(message)
                kernel_issues.append(message)
            else:
                for field_name in (
                    "token_accuracy",
                    "average_accepted_tokens",
                    "acceptance_rate",
                    "rejection_rate",
                    "effective_tokens_per_forward",
                    "repetition_rate",
                ):
                    if getattr(bundle.mtp, field_name) != getattr(rebuilt.mtp, field_name):
                        message = f"{label} evaluation MTP {field_name} does not match raw trials"
                        issues.append(message)
                        kernel_issues.append(message)
    if not mtp_enabled and bundle.mtp is not None:
        message = f"{label} direct evaluation unexpectedly contains MTP metrics"
        issues.append(message)
        kernel_issues.append(message)

    expected_fallbacks = _kernel_fallbacks(result)
    if bundle.hardware.kernel_fallbacks != expected_fallbacks:
        message = f"{label} evaluation kernel fallback count does not match raw trials"
        issues.append(message)
        kernel_issues.append(message)
    expected_peak = max(
        (trial.peak_memory_bytes for trial in measured if trial.peak_memory_bytes is not None),
        default=None,
    )
    if bundle.hardware.peak_memory_bytes != expected_peak:
        issues.append(f"{label} evaluation peak memory does not match raw trials")
    identity_fields = {
        "device_name": result.runtime_device_name,
        "chip": result.runtime_chip,
        "unified_memory_bytes": result.unified_memory_bytes,
        "os_version": result.os_version,
    }
    for field_name, expected in identity_fields.items():
        if getattr(bundle.hardware, field_name) != expected:
            message = f"{label} evaluation {field_name} does not match raw trials"
            issues.append(message)
            kernel_issues.append(message)
    metadata_fields: dict[str, object] = {
        "prompt_count": config.prompt_count,
        "warmup_trials": config.warmup_trials,
        "measured_trials": config.measured_trials,
        "successful_measured_trials": result.measured_count,
        "failed_trials": result.failed_count,
        "timed_out_trials": result.timed_out_count,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "draft_depth": config.draft_depth,
        "power_mode": config.power_mode,
        "quantizer": config.quantizer,
        "quantizer_version": config.quantizer_version,
        "ax_engine_version": result.ax_engine_version,
    }
    for field_name, expected_metadata in metadata_fields.items():
        if bundle.benchmark_metadata.get(field_name) != expected_metadata:
            issues.append(f"{label} evaluation metadata {field_name} does not match raw result")
    if bundle.benchmark_metadata.get("runtime_env") != dict(config.runtime_env):
        issues.append(f"{label} evaluation metadata runtime_env does not match raw result")


def _coverage(
    *,
    plan: QuantizationPlan,
    sensitivity: SensitivityReport,
    execution: QuantizerExecutionManifest,
    kernel_evidence: Literal["measured", "unmeasured"],
    issues: list[str],
    kernel_issues: list[str],
) -> tuple[list[HardwareKernelCoverage], int]:
    if stable_sha256(sensitivity) != plan.analysis_sha256:
        message = "candidate plan does not bind the supplied sensitivity report"
        issues.append(message)
        kernel_issues.append(message)
    if (
        not same_model_identity(sensitivity.model, plan.source_model)
        or sensitivity.profile != plan.profile
    ):
        message = "sensitivity report identity/profile does not match the candidate plan"
        issues.append(message)
        kernel_issues.append(message)
    normalized_plan_profile = plan.architecture_profile.model_copy(
        update={"support_tier": sensitivity.architecture_profile.support_tier}
    )
    if normalized_plan_profile != sensitivity.architecture_profile:
        message = "sensitivity architecture profile does not match the candidate plan"
        issues.append(message)
        kernel_issues.append(message)
    if sensitivity.calibration != plan.calibration:
        message = "sensitivity calibration evidence does not match the candidate plan"
        issues.append(message)
        kernel_issues.append(message)
    if not sensitivity.evidence_kind.release_quality or not plan.evidence_kind.release_quality:
        message = "hardware registry requires release-quality sensitivity and plan evidence"
        issues.append(message)
        kernel_issues.append(message)

    tensors = {entry.tensor.name: entry.tensor for entry in sensitivity.entries}
    sensitivity_entries = {entry.tensor.name: entry for entry in sensitivity.entries}
    if len(tensors) != len(sensitivity.entries):
        message = "sensitivity report contains duplicate tensor names"
        issues.append(message)
        kernel_issues.append(message)
    plan_tensors = {allocation.tensor for allocation in plan.assignments}
    if plan_tensors != set(tensors):
        message = "candidate plan tensor coverage does not match sensitivity evidence"
        issues.append(message)
        kernel_issues.append(message)

    expected_records = {
        allocation.module_path: allocation
        for allocation in plan.assignments
        if allocation.bits < 16
    }
    if len(expected_records) != sum(allocation.bits < 16 for allocation in plan.assignments):
        message = "quantized plan assignments do not have unique module paths"
        issues.append(message)
        kernel_issues.append(message)
    actual_records = {record.module_path: record for record in execution.records}
    if len(actual_records) != len(execution.records):
        message = "quantizer execution manifest contains duplicate module paths"
        issues.append(message)
        kernel_issues.append(message)
    if execution.plan_sha256 != stable_sha256(plan):
        message = "quantizer execution manifest does not bind the candidate plan"
        issues.append(message)
        kernel_issues.append(message)
    if set(expected_records) != set(actual_records):
        message = "quantizer execution coverage does not match quantized plan modules"
        issues.append(message)
        kernel_issues.append(message)
    for module_path in sorted(set(expected_records) & set(actual_records)):
        allocation = expected_records[module_path]
        record = actual_records[module_path]
        if (
            record.bits != allocation.bits
            or record.group_size != allocation.group_size
            or record.method != allocation.method
        ):
            message = f"quantizer execution settings differ for {module_path}"
            issues.append(message)
            kernel_issues.append(message)
        if not record.success or record.fallback:
            message = f"quantizer execution failed or fell back for {module_path}"
            issues.append(message)
            kernel_issues.append(message)

    grouped: dict[
        tuple[int, int | None, QuantMethod],
        list[tuple[Allocation, tuple[int, ...]]],
    ] = defaultdict(list)
    all_shapes: set[tuple[int, ...]] = set()
    for allocation in plan.assignments:
        tensor = tensors.get(allocation.tensor)
        if tensor is None:
            message = f"plan tensor is absent from sensitivity evidence: {allocation.tensor}"
            issues.append(message)
            kernel_issues.append(message)
            continue
        if (
            tensor.module_path != allocation.module_path
            or tensor.role != allocation.role
            or tensor.parameters != allocation.parameters
        ):
            message = f"plan tensor metadata differs from sensitivity evidence: {allocation.tensor}"
            issues.append(message)
            kernel_issues.append(message)
        matching_candidates = [
            candidate
            for candidate in sensitivity_entries[allocation.tensor].candidates
            if candidate.bits == allocation.bits
            and candidate.method == allocation.method
            and candidate.group_size == allocation.group_size
            and candidate.supported
        ]
        if len(matching_candidates) != 1:
            message = (
                "plan allocation has no unique supported sensitivity candidate: "
                f"{allocation.tensor}"
            )
            issues.append(message)
            kernel_issues.append(message)
        elif matching_candidates[0].metrics != allocation.metrics:
            message = f"plan metrics differ from sensitivity evidence: {allocation.tensor}"
            issues.append(message)
            kernel_issues.append(message)
        grouped[(allocation.bits, allocation.group_size, allocation.method)].append(
            (allocation, tensor.shape)
        )
        all_shapes.add(tensor.shape)

    coverage: list[HardwareKernelCoverage] = []
    for (bits, group_size, method), records in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1] or 0,
            str(item[0][2]),
        ),
    ):
        module_paths = {allocation.module_path for allocation, _shape in records}
        roles = sorted(
            {allocation.role for allocation, _shape in records},
            key=lambda role: role.value,
        )
        shapes = sorted({shape for _allocation, shape in records})
        conversion_count = sum(module_path in actual_records for module_path in module_paths)
        coverage.append(
            HardwareKernelCoverage(
                bits=bits,
                group_size=group_size,
                method=method,
                roles=roles,
                shapes=shapes,
                module_count=len(module_paths),
                parameter_count=sum(allocation.parameters for allocation, _shape in records),
                quantizer_execution_records=conversion_count,
                kernel_evidence=kernel_evidence,
            )
        )
    if not coverage:
        raise RefinementError("candidate plan produced no hardware coverage records")
    return coverage, len(all_shapes)


def build_hardware_profile_registry(
    request_path: str | Path,
) -> HardwareProfileRegistry:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, HardwareRegistryRequest)
    base = request_source.parent
    measurement_path = _required_file(
        base,
        request.measurement_set_file,
        "hardware registry measurement set",
    )
    measurements = load_model(measurement_path, RefinementMeasurementSet)
    if _SHA256.fullmatch(measurements.refinement_sha256) is None:
        raise RefinementError("hardware registry measurement set has an invalid refinement digest")
    if not measurements.evaluator_version.strip():
        raise RefinementError("hardware registry measurement set has no evaluator version")
    measurements_by_id = {
        measurement.measurement_id: measurement for measurement in measurements.measurements
    }
    entries: list[HardwareRegistryEntry] = []
    registry_issues: list[str] = []

    for candidate_input in request.candidates:
        measurement: CompleteCandidateMeasurement | None
        measurement_id = candidate_input.measurement_id
        if measurement_id is None:
            candidate_measurements = [
                measurement
                for measurement in measurements.measurements
                if measurement.candidate_id == candidate_input.candidate_id
            ]
            if len(candidate_measurements) != 1:
                raise RefinementError(
                    f"hardware registry candidate {candidate_input.candidate_id!r} has "
                    f"{len(candidate_measurements)} measurements; specify measurement_id"
                )
            measurement = candidate_measurements[0]
        else:
            measurement = measurements_by_id.get(measurement_id)
        if measurement is None:
            raise RefinementError(
                f"hardware registry candidate has no complete measurement: {measurement_id}"
            )
        if measurement.candidate_id != candidate_input.candidate_id:
            raise RefinementError(
                f"hardware registry measurement {measurement.measurement_id!r} belongs to "
                f"another candidate"
            )
        paths = {
            "plan": _required_file(base, candidate_input.plan_file, "candidate plan"),
            "artifact": _required_file(
                base,
                candidate_input.artifact_manifest_file,
                "candidate artifact manifest",
            ),
            "sensitivity": _required_file(
                base, candidate_input.sensitivity_file, "sensitivity report"
            ),
            "quality": _required_file(
                base,
                candidate_input.quality_comparison_file,
                "quality comparison",
            ),
            "validation": _required_file(
                base, candidate_input.validation_file, "validation report"
            ),
            "direct_evaluation": _required_file(
                base, candidate_input.direct_evaluation_file, "direct evaluation"
            ),
            "mtp_evaluation": _required_file(
                base, candidate_input.mtp_evaluation_file, "MTP evaluation"
            ),
            "direct_result": _required_file(
                base,
                candidate_input.direct_benchmark_result_file,
                "direct benchmark result",
            ),
            "mtp_result": _required_file(
                base,
                candidate_input.mtp_benchmark_result_file,
                "MTP benchmark result",
            ),
            "execution": _required_file(
                base,
                candidate_input.quantizer_execution_file,
                "quantizer execution manifest",
            ),
        }
        plan = load_model(paths["plan"], QuantizationPlan)
        artifact = load_model(paths["artifact"], ArtifactManifest)
        sensitivity = load_model(paths["sensitivity"], SensitivityReport)
        quality = load_model(paths["quality"], QualityComparisonReport)
        validation = load_model(paths["validation"], ValidationReport)
        direct_evaluation = load_model(paths["direct_evaluation"], EvaluationBundle)
        mtp_evaluation = load_model(paths["mtp_evaluation"], EvaluationBundle)
        direct_result = load_model(paths["direct_result"], BenchmarkResult)
        mtp_result = load_model(paths["mtp_result"], BenchmarkResult)
        execution = load_model(paths["execution"], QuantizerExecutionManifest)

        issues: list[str] = []
        kernel_issues: list[str] = []
        plan_sha256 = stable_sha256(plan)
        if plan_sha256 != measurement.plan_sha256:
            message = "complete measurement does not bind the supplied candidate plan"
            issues.append(message)
            kernel_issues.append(message)
        artifact_sha256 = file_sha256(paths["artifact"])
        if artifact_sha256 != measurement.artifact_manifest_sha256:
            issues.append("complete measurement does not bind the supplied artifact manifest")
        quality_sha256 = file_sha256(paths["quality"])
        if quality_sha256 != measurement.quality_comparison_sha256:
            issues.append("complete measurement does not bind the supplied quality comparison")
        validation_sha256 = file_sha256(paths["validation"])
        if validation_sha256 != measurement.validation_sha256:
            message = "complete measurement does not bind the supplied validation report"
            issues.append(message)
            kernel_issues.append(message)
        if not same_model_identity(validation.candidate_model, measurement.candidate_model):
            issues.append("validation and complete measurement candidate identities differ")
        if not is_immutable_revision(measurement.candidate_model.revision):
            message = "complete measurement candidate revision is not immutable"
            issues.append(message)
            kernel_issues.append(message)
        if validation.profile != measurement.profile or plan.profile != measurement.profile:
            issues.append("plan, validation, and complete measurement profiles differ")
        if not validation.passed:
            issues.append("candidate validation did not pass")
        if validation.passed != measurement.validation_passed:
            issues.append("validation status differs from the complete measurement")
        try:
            rebuilt_measurement = build_complete_candidate_measurement(
                candidate_id=measurement.candidate_id,
                measurement_id=measurement.measurement_id,
                plan=plan,
                artifact=artifact,
                artifact_sha256=artifact_sha256,
                quality=quality,
                quality_sha256=quality_sha256,
                validation=validation,
                validation_sha256=validation_sha256,
            )
        except RefinementError as exc:
            issues.append(f"complete measurement evidence cannot be rebuilt: {exc}")
        else:
            if rebuilt_measurement != measurement:
                issues.append("complete measurement objective differs from its bound evidence")

        try:
            validate_ab_invariant(direct_result.config, mtp_result.config)
        except Exception as exc:
            message = f"raw A/B benchmark invariant failed: {exc}"
            issues.append(message)
            kernel_issues.append(message)
        _check_result_bundle(
            label="direct",
            result=direct_result,
            bundle=direct_evaluation,
            mtp_enabled=False,
            issues=issues,
            kernel_issues=kernel_issues,
        )
        _check_result_bundle(
            label="MTP",
            result=mtp_result,
            bundle=mtp_evaluation,
            mtp_enabled=True,
            issues=issues,
            kernel_issues=kernel_issues,
        )
        if not same_model_identity(direct_evaluation.model, mtp_evaluation.model):
            message = "direct and MTP evaluations use different candidate checkpoints"
            issues.append(message)
            kernel_issues.append(message)
        if not same_model_identity(mtp_evaluation.model, measurement.candidate_model):
            issues.append("evaluation and complete measurement candidate identities differ")
        for field_name in ("device_name", "chip", "unified_memory_bytes", "os_version"):
            if getattr(direct_evaluation.hardware, field_name) != getattr(
                mtp_evaluation.hardware, field_name
            ):
                message = f"direct and MTP hardware identity differs for {field_name}"
                issues.append(message)
                kernel_issues.append(message)
        if direct_evaluation.software_versions != mtp_evaluation.software_versions:
            message = "direct and MTP evaluations use different software versions"
            issues.append(message)
            kernel_issues.append(message)

        direct_speed = direct_evaluation.hardware.decode_tokens_per_second
        mtp_speed = (
            mtp_evaluation.hardware.mtp_effective_tokens_per_second
            if mtp_evaluation.hardware.mtp_effective_tokens_per_second is not None
            else mtp_evaluation.hardware.decode_tokens_per_second
        )
        recorded_speedup = validation.comparisons.get("hardware.effective_speedup")
        if (
            direct_speed is None
            or direct_speed <= 0.0
            or mtp_speed is None
            or mtp_speed <= 0.0
            or not _same_finite_number(recorded_speedup, mtp_speed / direct_speed)
        ):
            message = "validation effective speedup does not match candidate A/B evidence"
            issues.append(message)
            kernel_issues.append(message)

        direct_fallbacks = _kernel_fallbacks(direct_result)
        mtp_fallbacks = _kernel_fallbacks(mtp_result)
        if direct_fallbacks or mtp_fallbacks:
            message = (
                f"runtime kernel fallbacks are nonzero "
                f"(direct={direct_fallbacks}, mtp={mtp_fallbacks})"
            )
            issues.append(message)
            kernel_issues.append(message)
        if measurement.hardware.kernel_fallbacks != mtp_fallbacks:
            message = "complete measurement kernel fallback count differs from MTP raw result"
            issues.append(message)
            kernel_issues.append(message)

        versions = mtp_evaluation.software_versions
        power_mode_value = mtp_evaluation.benchmark_metadata.get("power_mode")
        power_mode = _required_text(
            power_mode_value if isinstance(power_mode_value, str) else None,
            "power mode",
        )
        if direct_evaluation.benchmark_metadata.get("power_mode") != power_mode:
            message = "direct and MTP evaluations use different power modes"
            issues.append(message)
            kernel_issues.append(message)
        hardware = CompleteCandidateHardware(
            device_name=_required_text(mtp_evaluation.hardware.device_name, "device name"),
            chip=_required_text(mtp_evaluation.hardware.chip, "chip"),
            unified_memory_bytes=_required_positive_int(
                mtp_evaluation.hardware.unified_memory_bytes,
                "unified memory",
            ),
            os_version=_required_text(mtp_evaluation.hardware.os_version, "OS version"),
            ax_engine_version=_required_text(versions.ax_engine, "AX Engine version"),
            mlx_version=_required_text(versions.mlx, "MLX version"),
            mlx_lm_version=_required_text(versions.mlx_lm, "MLX-LM version"),
            power_mode=power_mode,
            kernel_fallbacks=mtp_fallbacks,
        )
        if hardware != measurement.hardware:
            issues.append("complete measurement hardware does not match benchmark evidence")

        kernel_evidence: Literal["measured", "unmeasured"] = (
            "measured" if not kernel_issues else "unmeasured"
        )
        coverage, unique_shapes = _coverage(
            plan=plan,
            sensitivity=sensitivity,
            execution=execution,
            kernel_evidence=kernel_evidence,
            issues=issues,
            kernel_issues=kernel_issues,
        )
        # Coverage checks can add kernel issues, so normalize evidence after they run.
        kernel_evidence = "measured" if not kernel_issues else "unmeasured"
        if any(item.kernel_evidence != kernel_evidence for item in coverage):
            coverage = [
                item.model_copy(update={"kernel_evidence": kernel_evidence}) for item in coverage
            ]

        backend_version = _required_text(versions.ax_engine, "AX Engine version")
        protocol_fingerprint = {
            "backend": "ax-engine-bench",
            "backend_version": backend_version,
            "direct_config": direct_result.config.model_dump(mode="json"),
            "mtp_config": mtp_result.config.model_dump(mode="json"),
            "deterministic_tolerance": request.deterministic_tolerance,
        }
        protocol = HardwareMeasurementProtocol(
            protocol_id=f"ax-engine-ab-v1-{stable_sha256(protocol_fingerprint)[:16]}",
            backend_version=backend_version,
            dataset_sha256=mtp_result.config.dataset_sha256,
            random_seed=mtp_result.config.random_seed,
            prompt_count=mtp_result.config.prompt_count,
            warmup_trials=mtp_result.config.warmup_trials,
            measured_trials=mtp_result.config.measured_trials,
            power_mode=power_mode,
            deterministic_tolerance=request.deterministic_tolerance,
            direct_commands=_commands(direct_result),
            mtp_commands=_commands(mtp_result),
        )
        release_ready = validation.passed and kernel_evidence == "measured" and not issues
        entry = HardwareRegistryEntry(
            entry_id=candidate_input.entry_id,
            candidate_id=candidate_input.candidate_id,
            measurement_id=measurement.measurement_id,
            candidate_model=measurement.candidate_model,
            profile=measurement.profile,
            plan_file=str(paths["plan"]),
            plan_file_sha256=file_sha256(paths["plan"]),
            plan_sha256=plan_sha256,
            artifact_manifest_file=str(paths["artifact"]),
            artifact_manifest_sha256=artifact_sha256,
            sensitivity_file=str(paths["sensitivity"]),
            sensitivity_sha256=file_sha256(paths["sensitivity"]),
            quality_comparison_file=str(paths["quality"]),
            quality_comparison_sha256=quality_sha256,
            validation_file=str(paths["validation"]),
            validation_sha256=validation_sha256,
            direct_evaluation_file=str(paths["direct_evaluation"]),
            direct_evaluation_sha256=file_sha256(paths["direct_evaluation"]),
            mtp_evaluation_file=str(paths["mtp_evaluation"]),
            mtp_evaluation_sha256=file_sha256(paths["mtp_evaluation"]),
            direct_benchmark_result_file=str(paths["direct_result"]),
            direct_benchmark_result_sha256=file_sha256(paths["direct_result"]),
            mtp_benchmark_result_file=str(paths["mtp_result"]),
            mtp_benchmark_result_sha256=file_sha256(paths["mtp_result"]),
            quantizer_execution_file=str(paths["execution"]),
            quantizer_execution_sha256=file_sha256(paths["execution"]),
            hardware=hardware,
            protocol=protocol,
            coverage=coverage,
            total_modules=len({allocation.module_path for allocation in plan.assignments}),
            unique_shapes=unique_shapes,
            kernel_evidence=kernel_evidence,
            validation_passed=validation.passed,
            release_ready=release_ready,
            issues=issues,
        )
        entries.append(entry)
        registry_issues.extend(f"{candidate_input.entry_id}: {message}" for message in entry.issues)

    distinct_hosts = len(
        {
            (
                entry.hardware.device_name,
                entry.hardware.chip,
                entry.hardware.unified_memory_bytes,
                entry.hardware.os_version,
            )
            for entry in entries
            if entry.release_ready
        }
    )
    release_ready = bool(distinct_hosts) and all(entry.release_ready for entry in entries)
    return HardwareProfileRegistry(
        registry_id=request.registry_id,
        measurement_set_sha256=stable_sha256(measurements),
        measurement_set_file=str(measurement_path),
        measurement_set_file_sha256=file_sha256(measurement_path),
        entries=entries,
        distinct_named_hosts=distinct_hosts,
        release_ready=release_ready and not registry_issues,
        issues=registry_issues,
    )
