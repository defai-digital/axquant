from __future__ import annotations

from collections.abc import Callable, Mapping

from axquant.revisions import is_immutable_revision
from axquant.schema import (
    ArtifactSizeEvidence,
    CalibrationManifest,
    EvaluationBundle,
    ProfileName,
    RuntimeName,
    ValidationIssue,
    ValidationReport,
    ValidationThresholds,
)


def _metadata_value_missing(metadata: Mapping[str, object], name: str) -> bool:
    if name not in metadata or metadata[name] is None:
        return True
    value = metadata[name]
    if isinstance(value, str):
        return not value.strip()
    if name == "runtime_env":
        # No caller-supplied overrides is a valid, explicitly recorded
        # environment.
        return not isinstance(value, dict)
    if isinstance(value, (dict, list)):
        return not value
    return False


def validate_evaluations(
    reference: EvaluationBundle,
    candidate_direct: EvaluationBundle,
    candidate: EvaluationBundle,
    *,
    profile: ProfileName,
    thresholds: ValidationThresholds,
    calibration: CalibrationManifest | None = None,
    size_reference: ArtifactSizeEvidence | None = None,
    candidate_size: ArtifactSizeEvidence | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    comparisons: dict[str, float | int | str | bool | None] = {}

    def issue(metric: str, message: str, severity: str = "error") -> None:
        issues.append(
            ValidationIssue(
                severity="warning" if severity == "warning" else "error",
                metric=metric,
                message=message,
            )
        )

    def require_pair(
        metric: str,
        reference_value: float | int | None,
        candidate_value: float | int | None,
        check: Callable[[float, float], None],
    ) -> None:
        if reference_value is None or candidate_value is None:
            severity = "error" if thresholds.require_complete_metrics else "warning"
            issue(metric, "reference and candidate values are required", severity)
            return
        check(float(reference_value), float(candidate_value))

    if size_reference is None or candidate_size is None:
        if thresholds.require_artifact_size:
            issue(
                "artifact.weight_size",
                "uniform-4-bit and candidate artifact-size evidence are required",
            )
    else:
        if size_reference.kind != "uniform-4bit":
            issue("artifact.size_reference.kind", "size reference must be uniform 4-bit")
        if candidate_size.kind != "candidate":
            issue("artifact.candidate_size.kind", "candidate size evidence has the wrong kind")
        if candidate_size.model != candidate.model:
            issue(
                "artifact.candidate_size.model",
                "candidate size evidence and benchmark evaluation identify different checkpoints",
            )
        if candidate_size.logical_parameters != size_reference.logical_parameters:
            issue(
                "artifact.logical_parameters",
                "candidate and uniform-4-bit logical parameter counts differ",
            )
        size_ratio = candidate_size.weight_bytes / size_reference.weight_bytes
        comparisons["artifact.weight_size_ratio"] = size_ratio
        comparisons["artifact.candidate_measured_bpw"] = candidate_size.measured_bpw
        comparisons["artifact.uniform4_measured_bpw"] = size_reference.measured_bpw
        comparisons["artifact.candidate_weight_bytes"] = candidate_size.weight_bytes
        comparisons["artifact.uniform4_weight_bytes"] = size_reference.weight_bytes
        comparisons["artifact.logical_parameters"] = candidate_size.logical_parameters
        comparisons["artifact.candidate_source_sha256"] = candidate_size.source_sha256
        comparisons["artifact.uniform4_source_sha256"] = size_reference.source_sha256
        if size_ratio > thresholds.max_weight_size_ratio:
            issue(
                "artifact.weight_size_ratio",
                f"ratio {size_ratio:.4f} exceeds {thresholds.max_weight_size_ratio:.4f}",
            )

    if calibration is None:
        severity = "error" if thresholds.require_complete_metrics else "warning"
        issue(
            "calibration",
            "calibration manifest is required to verify evaluation separation",
            severity,
        )
    else:
        comparisons["calibration.dataset_sha256"] = calibration.dataset_sha256
        if calibration.profile != profile:
            issue("calibration.profile", "calibration profile does not match validation profile")
        if not is_immutable_revision(calibration.model.revision):
            issue("calibration.model.revision", "calibration source revision is not pinned")
        if not calibration.calibration_evaluation_separation_attested:
            issue(
                "calibration.separation_attested",
                "calibration/evaluation separation is not attested",
            )
        quality_dataset_sha256 = candidate.benchmark_metadata.get("quality_dataset_sha256")
        if calibration.dataset_sha256 == quality_dataset_sha256:
            issue(
                "calibration.quality_dataset_sha256",
                "calibration and quality evaluation datasets are identical",
            )
        if calibration.dataset_sha256 == candidate.dataset_sha256:
            issue(
                "calibration.benchmark_dataset_sha256",
                "calibration and benchmark prompt datasets are identical",
            )

    if reference.dataset_sha256 != candidate.dataset_sha256:
        issue("dataset_sha256", "reference and candidate used different evaluation datasets")
    if reference.workload != candidate.workload:
        issue("workload", "reference and candidate used different workloads")
    if candidate_direct.dataset_sha256 != candidate.dataset_sha256:
        issue(
            "candidate_direct.dataset_sha256",
            "MTP-off and MTP-on candidates used different evaluation datasets",
        )
    if candidate_direct.workload != candidate.workload:
        issue(
            "candidate_direct.workload",
            "MTP-off and MTP-on candidates used different workloads",
        )
    if candidate_direct.model != candidate.model:
        issue(
            "candidate_direct.model",
            "MTP speedup requires the identical AXQuant checkpoint",
        )
    if candidate_direct.runtime != candidate.runtime:
        issue(
            "candidate_direct.runtime",
            "MTP-off and MTP-on candidates used different runtimes",
        )
    if candidate.runtime is not RuntimeName.AX_ENGINE:
        issue("runtime", "production MTP validation must run on AX Engine")
    if reference.runtime is not RuntimeName.AX_ENGINE:
        issue("reference.runtime", "reference validation must run on AX Engine")
    elif reference.runtime != candidate.runtime:
        issue("reference.runtime", "reference and candidate used different runtimes")
    if reference.random_seed != candidate.random_seed:
        issue("reference.random_seed", "reference and candidate used different seeds")
    if reference.baseline_kind != "uniform-6bit":
        issue("reference.baseline_kind", "quality reference must be the uniform-6bit baseline")
    if candidate_direct.baseline_kind != "axquant-mtp-off":
        issue(
            "candidate_direct.baseline_kind",
            "direct candidate must use the axquant-mtp-off baseline kind",
        )
    if candidate.baseline_kind != "axquant-mtp-on":
        issue(
            "candidate.baseline_kind",
            "MTP candidate must use the axquant-mtp-on baseline kind",
        )
    if candidate_direct.mtp_enabled:
        issue("candidate_direct.mtp_enabled", "direct candidate must have MTP disabled")
    if not candidate.mtp_enabled:
        issue("candidate.mtp_enabled", "MTP candidate must have MTP enabled")
    if candidate_direct.random_seed != candidate.random_seed:
        issue("candidate.random_seed", "MTP-off and MTP-on candidates used different seeds")
    if candidate_direct.hardware.chip != candidate.hardware.chip:
        issue("candidate.hardware.chip", "MTP-off and MTP-on candidates used different chips")
    if candidate_direct.hardware.os_version != candidate.hardware.os_version:
        issue(
            "candidate.hardware.os_version",
            "MTP-off and MTP-on candidates used different operating systems",
        )
    if candidate_direct.hardware.device_name != candidate.hardware.device_name:
        issue(
            "candidate.hardware.device_name",
            "MTP-off and MTP-on candidates used different devices",
        )
    if candidate_direct.hardware.unified_memory_bytes != candidate.hardware.unified_memory_bytes:
        issue(
            "candidate.hardware.unified_memory_bytes",
            "MTP-off and MTP-on candidates used different memory configurations",
        )
    if candidate_direct.software_versions.ax_engine != candidate.software_versions.ax_engine:
        issue(
            "candidate.software.ax_engine",
            "MTP-off and MTP-on candidates used different AX Engine versions",
        )
    if candidate_direct.software_versions.mlx != candidate.software_versions.mlx:
        issue(
            "candidate.software.mlx",
            "MTP-off and MTP-on candidates used different MLX versions",
        )
    if candidate_direct.software_versions.mlx_lm != candidate.software_versions.mlx_lm:
        issue(
            "candidate.software.mlx_lm",
            "MTP-off and MTP-on candidates used different MLX-LM versions",
        )
    for hardware_name in (
        "device_name",
        "chip",
        "unified_memory_bytes",
        "os_version",
    ):
        if getattr(reference.hardware, hardware_name) != getattr(candidate.hardware, hardware_name):
            issue(
                f"reference.hardware.{hardware_name}",
                "reference and candidate used different hardware",
            )
    for version_name in ("ax_engine", "mlx", "mlx_lm"):
        if getattr(reference.software_versions, version_name) != getattr(
            candidate.software_versions, version_name
        ):
            issue(
                f"reference.software.{version_name}",
                "reference and candidate used different runtime versions",
            )
    for bundle_name, bundle in (
        ("reference", reference),
        ("candidate_direct", candidate_direct),
        ("candidate", candidate),
    ):
        hardware_identity = {
            "device_name": bundle.hardware.device_name,
            "chip": bundle.hardware.chip,
            "unified_memory_bytes": bundle.hardware.unified_memory_bytes,
            "os_version": bundle.hardware.os_version,
        }
        if thresholds.require_complete_metrics:
            for hardware_name, value in hardware_identity.items():
                if value is None or value == "":
                    issue(
                        f"{bundle_name}.hardware.{hardware_name}",
                        "named hardware evidence is required",
                    )
        versions = bundle.software_versions
        required_versions = {
            "axquant": versions.axquant,
            "python": versions.python,
            "mlx": versions.mlx,
            "mlx_lm": versions.mlx_lm,
            "ax_engine": versions.ax_engine,
            "safetensors": versions.safetensors,
            "pydantic": versions.pydantic,
        }
        for version_name, value in required_versions.items():
            if not value:
                issue(
                    f"{bundle_name}.software_versions.{version_name}",
                    "required software version is missing",
                )
        metadata = bundle.benchmark_metadata
        if thresholds.require_complete_metrics:
            required_metadata = (
                "prompt_count",
                "warmup_trials",
                "measured_trials",
                "successful_measured_trials",
                "failed_trials",
                "timed_out_trials",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "draft_depth",
                "power_mode",
                "quantizer",
                "quantizer_version",
                "ax_engine_version",
                "quality_dataset_sha256",
                "runtime_env",
            )

            missing_metadata = {
                metadata_name
                for metadata_name in required_metadata
                if _metadata_value_missing(metadata, metadata_name)
            }
            for metadata_name in sorted(missing_metadata):
                issue(
                    f"{bundle_name}.benchmark_metadata.{metadata_name}",
                    "required benchmark metadata is missing or empty",
                )

            count_minima = {
                "prompt_count": 1,
                "warmup_trials": 0,
                "measured_trials": 1,
                "successful_measured_trials": 0,
                "failed_trials": 0,
                "timed_out_trials": 0,
            }
            counts: dict[str, int] = {}
            for metadata_name, minimum in count_minima.items():
                if metadata_name in missing_metadata:
                    continue
                metadata_count_value = metadata.get(metadata_name)
                if (
                    isinstance(metadata_count_value, bool)
                    or not isinstance(metadata_count_value, int)
                    or metadata_count_value < minimum
                ):
                    issue(
                        f"{bundle_name}.benchmark_metadata.{metadata_name}",
                        f"benchmark count must be an integer >= {minimum}",
                    )
                    continue
                counts[metadata_name] = metadata_count_value

            required_counts = set(count_minima)
            if required_counts.issubset(counts):
                warmups = counts["warmup_trials"]
                measured = counts["measured_trials"]
                successful = counts["successful_measured_trials"]
                failed = counts["failed_trials"]
                timed_out = counts["timed_out_trials"]
                if successful > measured:
                    issue(
                        f"{bundle_name}.benchmark_metadata.successful_measured_trials",
                        "successful measured trials exceed configured measured trials",
                    )
                if timed_out > failed:
                    issue(
                        f"{bundle_name}.benchmark_metadata.timed_out_trials",
                        "timed-out trials exceed total failed trials",
                    )
                if failed > measured + warmups:
                    issue(
                        f"{bundle_name}.benchmark_metadata.failed_trials",
                        "failed trials exceed configured warmup plus measured trials",
                    )
                if successful + failed < measured:
                    issue(
                        f"{bundle_name}.benchmark_metadata.successful_measured_trials",
                        "successful measured plus failed trials cannot account for "
                        "configured measured trials",
                    )

            runtime_env = metadata.get("runtime_env")
            if isinstance(runtime_env, dict) and any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in runtime_env.items()
            ):
                issue(
                    f"{bundle_name}.benchmark_metadata.runtime_env",
                    "runtime environment entries must use non-empty string keys and values",
                )

        failed_trials = metadata.get("failed_trials")
        if isinstance(failed_trials, int) and not isinstance(failed_trials, bool) and failed_trials:
            issue(f"{bundle_name}.benchmark.failed_trials", "benchmark contains failed trials")
        timed_out_trials = metadata.get("timed_out_trials")
        if (
            isinstance(timed_out_trials, int)
            and not isinstance(timed_out_trials, bool)
            and timed_out_trials
        ):
            issue(
                f"{bundle_name}.benchmark.timed_out_trials",
                "benchmark contains timed-out trials",
            )
        if thresholds.require_complete_metrics and bundle.hardware.kernel_fallbacks is None:
            issue(
                f"{bundle_name}.hardware.kernel_fallbacks",
                "kernel fallback count is required",
            )
        elif bundle.hardware.kernel_fallbacks:
            issue(
                f"{bundle_name}.hardware.kernel_fallbacks",
                f"benchmark used {bundle.hardware.kernel_fallbacks} kernel fallbacks",
            )

    candidate_power_mode_value = candidate.benchmark_metadata.get("power_mode")
    candidate_power_mode = (
        candidate_power_mode_value if isinstance(candidate_power_mode_value, str) else None
    )
    comparisons.update(
        {
            "hardware.device_name": candidate.hardware.device_name,
            "hardware.chip": candidate.hardware.chip,
            "hardware.unified_memory_bytes": candidate.hardware.unified_memory_bytes,
            "hardware.os_version": candidate.hardware.os_version,
            "software.ax_engine": candidate.software_versions.ax_engine,
            "software.mlx": candidate.software_versions.mlx,
            "software.mlx_lm": candidate.software_versions.mlx_lm,
            "hardware.power_mode": candidate_power_mode,
            "hardware.kernel_fallbacks": candidate.hardware.kernel_fallbacks,
        }
    )

    invariant_fields = (
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
        "ax_engine_version",
        "quality_dataset_sha256",
        "runtime_env",
    )
    for field_name in invariant_fields:
        if reference.benchmark_metadata.get(field_name) != candidate.benchmark_metadata.get(
            field_name
        ):
            issue(
                f"reference.benchmark_metadata.{field_name}",
                "reference and candidate benchmark controls differ",
            )
        if candidate_direct.benchmark_metadata.get(field_name) != candidate.benchmark_metadata.get(
            field_name
        ):
            issue(
                f"candidate.benchmark_metadata.{field_name}",
                "MTP-off and MTP-on benchmark controls differ",
            )
    if reference.benchmark_metadata.get(
        "quality_dataset_sha256"
    ) != candidate.benchmark_metadata.get("quality_dataset_sha256"):
        issue(
            "quality_dataset_sha256",
            "reference and candidate used different quality evaluation datasets",
        )

    def check_perplexity(reference_value: float, candidate_value: float) -> None:
        increase = candidate_value / reference_value - 1.0
        comparisons["perplexity_relative_increase"] = increase
        if increase > thresholds.max_perplexity_relative_increase:
            issue(
                "perplexity",
                f"relative increase {increase:.4f} exceeds "
                f"{thresholds.max_perplexity_relative_increase:.4f}",
            )

    require_pair(
        "perplexity",
        reference.quality.perplexity,
        candidate.quality.perplexity,
        check_perplexity,
    )

    reference_tasks = reference.quality.task_scores
    candidate_tasks = candidate.quality.task_scores
    if thresholds.require_complete_metrics and (not reference_tasks or not candidate_tasks):
        issue("task_scores", "reference and candidate task scores are required")
    required_tasks = set(thresholds.required_task_scores)
    for bundle_name, task_scores in (
        ("reference", reference_tasks),
        ("candidate", candidate_tasks),
    ):
        missing_required = sorted(required_tasks - set(task_scores))
        if missing_required:
            issue(
                f"{bundle_name}.task_scores",
                f"required critical task scores are missing: {missing_required}",
            )
    missing_tasks = sorted(set(reference_tasks) - set(candidate_tasks))
    if missing_tasks:
        severity = "error" if thresholds.require_complete_metrics else "warning"
        issue("task_scores", f"candidate is missing task scores {missing_tasks}", severity)
    for task in sorted(set(reference_tasks) & set(candidate_tasks)):
        drop = reference_tasks[task] - candidate_tasks[task]
        comparisons[f"task.{task}.drop"] = drop
        if drop > thresholds.max_task_score_drop:
            issue(
                f"task.{task}",
                f"score drop {drop:.4f} exceeds {thresholds.max_task_score_drop:.4f}",
            )
    retention_values = [
        candidate_tasks[task] / reference_tasks[task]
        for task in sorted(set(reference_tasks) & set(candidate_tasks))
        if reference_tasks[task] > 0
    ]
    if retention_values:
        aggregate_retention = sum(retention_values) / len(retention_values)
        comparisons["quality.aggregate_retention"] = aggregate_retention
        if aggregate_retention < thresholds.minimum_aggregate_quality_retention:
            issue(
                "quality.aggregate_retention",
                f"retention {aggregate_retention:.4f} is below "
                f"{thresholds.minimum_aggregate_quality_retention:.4f}",
            )
    elif thresholds.require_complete_metrics:
        issue("quality.aggregate_retention", "cannot compute aggregate quality retention")

    for metric, reference_value, candidate_value in (
        (
            "json_valid_rate",
            reference.quality.json_valid_rate,
            candidate.quality.json_valid_rate,
        ),
        (
            "syntax_valid_rate",
            reference.quality.syntax_valid_rate,
            candidate.quality.syntax_valid_rate,
        ),
    ):

        def check_structured(
            baseline: float,
            measured: float,
            *,
            metric_name: str = metric,
        ) -> None:
            drop = baseline - measured
            comparisons[f"quality.{metric_name}.drop"] = drop
            if drop > thresholds.max_structured_output_drop:
                issue(
                    f"quality.{metric_name}",
                    f"drop {drop:.4f} exceeds {thresholds.max_structured_output_drop:.4f}",
                )

        require_pair(metric, reference_value, candidate_value, check_structured)

    if reference.mtp is None or candidate.mtp is None:
        severity = "error" if thresholds.require_complete_metrics else "warning"
        issue("mtp", "reference and candidate MTP measurements are required", severity)
    else:

        def check_acceptance(reference_value: float, candidate_value: float) -> None:
            drop = reference_value - candidate_value
            retention = candidate_value / reference_value if reference_value else 0.0
            comparisons["mtp.acceptance_rate_drop"] = drop
            comparisons["mtp.acceptance_retention"] = retention
            if drop > thresholds.max_mtp_acceptance_drop:
                issue(
                    "mtp.acceptance_rate",
                    f"drop {drop:.4f} exceeds {thresholds.max_mtp_acceptance_drop:.4f}",
                )
            if retention < thresholds.minimum_mtp_acceptance_retention:
                issue(
                    "mtp.acceptance_retention",
                    f"retention {retention:.4f} is below "
                    f"{thresholds.minimum_mtp_acceptance_retention:.4f}",
                )

        require_pair(
            "mtp.acceptance_rate",
            reference.mtp.acceptance_rate,
            candidate.mtp.acceptance_rate,
            check_acceptance,
        )
        if thresholds.require_complete_metrics and (
            not reference.mtp.token_accuracy or not candidate.mtp.token_accuracy
        ):
            issue(
                "mtp.token_accuracy",
                "reference and candidate token-accuracy horizons are required",
            )
        missing_horizons = sorted(
            set(reference.mtp.token_accuracy) - set(candidate.mtp.token_accuracy)
        )
        if missing_horizons:
            severity = "error" if thresholds.require_complete_metrics else "warning"
            issue(
                "mtp.token_accuracy",
                f"candidate is missing horizons {missing_horizons}",
                severity,
            )
        for horizon in sorted(
            set(reference.mtp.token_accuracy) & set(candidate.mtp.token_accuracy)
        ):
            drop = reference.mtp.token_accuracy[horizon] - candidate.mtp.token_accuracy[horizon]
            comparisons[f"mtp.token_accuracy.{horizon}.drop"] = drop
            if drop > thresholds.max_mtp_token_accuracy_drop:
                issue(
                    f"mtp.token_accuracy.{horizon}",
                    f"drop {drop:.4f} exceeds {thresholds.max_mtp_token_accuracy_drop:.4f}",
                )

        def check_repetition(reference_value: float, candidate_value: float) -> None:
            increase = candidate_value - reference_value
            comparisons["mtp.repetition_rate_increase"] = increase
            if increase > thresholds.max_repetition_increase:
                issue(
                    "mtp.repetition_rate",
                    f"increase {increase:.4f} exceeds {thresholds.max_repetition_increase:.4f}",
                )

        def check_divergence(reference_value: float, candidate_value: float) -> None:
            increase = candidate_value - reference_value
            comparisons["mtp.divergence_rate_increase"] = increase
            if increase > thresholds.max_divergence_increase:
                issue(
                    "mtp.divergence_rate",
                    f"increase {increase:.4f} exceeds {thresholds.max_divergence_increase:.4f}",
                )

        require_pair(
            "mtp.repetition_rate",
            reference.mtp.repetition_rate,
            candidate.mtp.repetition_rate,
            check_repetition,
        )
        require_pair(
            "mtp.divergence_rate",
            reference.mtp.divergence_rate,
            candidate.mtp.divergence_rate,
            check_divergence,
        )

    reference_speed = candidate_direct.hardware.decode_tokens_per_second
    candidate_speed = (
        candidate.hardware.mtp_effective_tokens_per_second
        if candidate.hardware.mtp_effective_tokens_per_second is not None
        else candidate.hardware.decode_tokens_per_second
    )

    def check_speed(reference_value: float, candidate_value: float) -> None:
        if reference_value <= 0.0:
            issue(
                "hardware.effective_speedup",
                "direct candidate decode throughput must be positive",
            )
            return
        speedup = candidate_value / reference_value
        comparisons["hardware.effective_speedup"] = speedup
        if speedup < thresholds.min_effective_speedup:
            issue(
                "hardware.effective_speedup",
                f"speedup {speedup:.4f} is below {thresholds.min_effective_speedup:.4f}",
            )

    require_pair("hardware.effective_speedup", reference_speed, candidate_speed, check_speed)

    def check_memory(reference_value: float, candidate_value: float) -> None:
        if reference_value <= 0.0:
            issue(
                "hardware.peak_memory_ratio",
                "reference peak memory must be positive",
            )
            return
        ratio = candidate_value / reference_value
        comparisons["hardware.peak_memory_ratio"] = ratio
        if ratio > thresholds.max_peak_memory_ratio:
            issue(
                "hardware.peak_memory_ratio",
                f"ratio {ratio:.4f} exceeds {thresholds.max_peak_memory_ratio:.4f}",
            )

    require_pair(
        "hardware.peak_memory_ratio",
        reference.hardware.peak_memory_bytes,
        candidate.hardware.peak_memory_bytes,
        check_memory,
    )

    integrity = candidate.integrity
    required_integrity = {
        "safetensors_valid": integrity.safetensors_valid,
        "index_complete": integrity.index_complete,
        "config_valid": integrity.config_valid,
        "source_revision_pinned": integrity.source_revision_pinned,
    }
    if candidate.mtp is not None:
        required_integrity["mtp_layout_valid"] = integrity.mtp_layout_valid is True
    for metric, integrity_value in required_integrity.items():
        comparisons[f"integrity.{metric}"] = integrity_value
        if not integrity_value:
            issue(f"integrity.{metric}", "integrity check did not pass")

    return ValidationReport(
        reference_model=reference.model,
        candidate_model=candidate.model,
        profile=profile,
        passed=not any(item.severity == "error" for item in issues),
        thresholds=thresholds,
        issues=issues,
        comparisons=comparisons,
    )
