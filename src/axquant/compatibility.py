from __future__ import annotations

from pathlib import Path

from axquant.errors import ArtifactError, ValidationGateError
from axquant.inspector import inspect_model
from axquant.release_exceptions import release_exception_allows_size
from axquant.schema import (
    ArchitectureSupportLevel,
    ArtifactManifest,
    CheckpointCompatibility,
    CompatibilityCandidateInput,
    CompatibilityMatrix,
    CompatibilityMatrixRequest,
    QuantizationPlan,
    RuntimeCheck,
    RuntimeName,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256


def _resolved(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_artifact_file(directory: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactError(f"unsafe artifact manifest path: {relative_name}")
    return directory / relative


def _validate_artifact_files(
    directory: Path,
    manifest: ArtifactManifest,
    issues: list[str],
) -> None:
    for record in manifest.files:
        path = _safe_artifact_file(directory, record.path)
        if not path.is_file():
            issues.append(f"artifact manifest file is missing: {record.path}")
        elif path.stat().st_size != record.size_bytes:
            issues.append(f"artifact manifest file size changed: {record.path}")
        elif file_sha256(path) != record.sha256:
            issues.append(f"artifact manifest file checksum changed: {record.path}")


def _runtime_targets_artifact(check: RuntimeCheck, artifact: Path) -> bool:
    option = "--mlx-model-artifacts-dir" if check.runtime == RuntimeName.AX_ENGINE else "--model"
    try:
        target = check.command[check.command.index(option) + 1]
    except (ValueError, IndexError):
        return False
    return Path(target).expanduser().resolve() == artifact


def _check_runtime(
    *,
    check: RuntimeCheck,
    runtime: RuntimeName,
    artifact: Path,
    expected_model_id: str,
    expected_revision: str | None,
    issues: list[str],
) -> bool:
    expected_kind = "doctor" if runtime == RuntimeName.AX_ENGINE else "generation-smoke"
    if check.runtime != runtime:
        issues.append(f"{runtime.value} evidence declares the wrong runtime")
    if check.check_kind != expected_kind:
        issues.append(f"{runtime.value} release evidence must use {expected_kind}")
    if not check.available or not check.passed:
        issues.append(f"{runtime.value} runtime check did not pass")
    if not _runtime_targets_artifact(check, artifact):
        issues.append(f"{runtime.value} runtime check does not target the candidate artifact")
    if check.model.model_id != expected_model_id or check.model.revision != expected_revision:
        issues.append(f"{runtime.value} runtime check identifies a different candidate model")
    check_local_path = check.model.local_path
    if check_local_path is None or Path(check_local_path).expanduser().resolve() != artifact:
        issues.append(f"{runtime.value} runtime check model path differs from the artifact")
    return (
        check.runtime == runtime
        and check.check_kind == expected_kind
        and check.available
        and check.passed
        and _runtime_targets_artifact(check, artifact)
        and check.model.model_id == expected_model_id
        and check.model.revision == expected_revision
        and check_local_path is not None
        and Path(check_local_path).expanduser().resolve() == artifact
    )


def _candidate_entry(
    *,
    base: Path,
    candidate: CompatibilityCandidateInput,
) -> CheckpointCompatibility:
    artifact = _resolved(base, candidate.artifact_directory)
    if not artifact.is_dir():
        raise ArtifactError(f"compatibility artifact directory does not exist: {artifact}")
    manifest_path = artifact / "axquant_manifest.json"
    plan_path = artifact / "axquant_plan.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    plan = load_model(plan_path, QuantizationPlan)
    ax_path = _resolved(base, candidate.ax_engine_check)
    mlx_path = _resolved(base, candidate.mlx_lm_check)
    validation_path = _resolved(base, candidate.validation_report)
    ax_check = load_model(ax_path, RuntimeCheck)
    mlx_check = load_model(mlx_path, RuntimeCheck)
    validation = load_model(validation_path, ValidationReport)
    issues: list[str] = []

    _validate_artifact_files(artifact, manifest, issues)
    if stable_sha256(plan) != manifest.plan_sha256:
        issues.append("artifact plan checksum does not match axquant_plan.json")
    if (
        plan.source_model.model_id != manifest.source_model.model_id
        or plan.source_model.revision != manifest.source_model.revision
    ):
        issues.append("artifact plan and manifest source identities differ")

    inventory = inspect_model(
        artifact,
        model_id=manifest.source_model.model_id,
        revision=manifest.source_model.revision,
        allow_quantized=True,
    )
    architecture = inventory.architecture_profile
    if architecture.support_level != ArchitectureSupportLevel.SUPPORTED:
        issues.append("candidate architecture is not supported for conversion")
    if architecture.product_family != "qwen3.6" or architecture.dense is not True:
        issues.append("candidate is not a dense Qwen 3.6 checkpoint")

    ax_passed = _check_runtime(
        check=ax_check,
        runtime=RuntimeName.AX_ENGINE,
        artifact=artifact,
        expected_model_id=validation.candidate_model.model_id,
        expected_revision=validation.candidate_model.revision,
        issues=issues,
    )
    mlx_passed = _check_runtime(
        check=mlx_check,
        runtime=RuntimeName.MLX_LM,
        artifact=artifact,
        expected_model_id=validation.candidate_model.model_id,
        expected_revision=validation.candidate_model.revision,
        issues=issues,
    )

    candidate_local_path = validation.candidate_model.local_path
    if (
        candidate_local_path is None
        or Path(candidate_local_path).expanduser().resolve() != artifact
    ):
        issues.append("validation report does not identify the candidate artifact directory")
    if not validation.candidate_model.revision:
        issues.append("validated candidate revision is not immutable")
    if not validation.passed:
        issues.append("candidate release validation did not pass")
    candidate_weight_bytes = validation.comparisons.get("artifact.candidate_weight_bytes")
    if candidate_weight_bytes != manifest.weight_file_size_bytes:
        issues.append("validation weight bytes do not match the artifact manifest")
    numeric_comparisons = {
        "artifact.weight_size_ratio": (
            float("-inf"),
            validation.thresholds.max_weight_size_ratio,
        ),
        "quality.aggregate_retention": (
            validation.thresholds.minimum_aggregate_quality_retention,
            float("inf"),
        ),
        "mtp.acceptance_retention": (
            validation.thresholds.minimum_mtp_acceptance_retention,
            float("inf"),
        ),
        "hardware.effective_speedup": (
            validation.thresholds.min_effective_speedup,
            float("inf"),
        ),
        "hardware.peak_memory_ratio": (
            float("-inf"),
            validation.thresholds.max_peak_memory_ratio,
        ),
        "hardware.kernel_fallbacks": (0.0, 0.0),
    }
    named_hardware = {
        "hardware.device_name",
        "hardware.chip",
        "hardware.unified_memory_bytes",
        "hardware.os_version",
        "hardware.power_mode",
        "software.mlx_lm",
    }
    required_comparisons = set(numeric_comparisons) | named_hardware
    missing_comparisons = sorted(
        key for key in required_comparisons if validation.comparisons.get(key) in {None, ""}
    )
    if missing_comparisons:
        issues.append(f"validation report is missing release evidence: {missing_comparisons}")
    for metric, (minimum, maximum) in numeric_comparisons.items():
        value = validation.comparisons.get(metric)
        if isinstance(value, (int, float)) and not minimum <= float(value) <= maximum:
            if metric == "artifact.weight_size_ratio":
                try:
                    release_exception_allows_size(validation, plan=plan)
                    continue
                except ValidationGateError:
                    pass
            issues.append(f"validation comparison violates its release threshold: {metric}")
    if any(issue.severity == "error" for issue in validation.issues):
        issues.append("validation report contains release-blocking issues")

    supported_bits = sorted({assignment.bits for assignment in plan.assignments})
    return CheckpointCompatibility(
        candidate_model=validation.candidate_model,
        source_model=manifest.source_model,
        profile=validation.profile,
        artifact_path=str(artifact),
        artifact_manifest_sha256=file_sha256(manifest_path),
        plan_sha256=manifest.plan_sha256,
        adapter_id=architecture.adapter_id,
        dense=architecture.dense is True,
        text_layer_count=architecture.text_layer_count,
        measured_total_bpw=manifest.measured_total_bpw,
        mtp_present=manifest.mtp_present,
        supported_bits=supported_bits,
        ax_engine_check_sha256=file_sha256(ax_path),
        ax_engine_passed=ax_passed,
        mlx_lm_check_sha256=file_sha256(mlx_path),
        mlx_lm_passed=mlx_passed,
        validation_sha256=file_sha256(validation_path),
        validation_passed=validation.passed,
        compatible=not issues,
        issues=issues,
    )


def build_compatibility_matrix(request_path: str | Path) -> CompatibilityMatrix:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, CompatibilityMatrixRequest)
    entries = [
        _candidate_entry(base=request_source.parent, candidate=candidate)
        for candidate in request.candidates
    ]
    source_keys = {
        (entry.source_model.model_id, entry.source_model.revision)
        for entry in entries
        if entry.dense
    }
    issues: list[str] = []
    required_model_ids = {requirement.model_id for requirement in request.required_dense_models}
    observed_model_ids = {entry.source_model.model_id for entry in entries if entry.dense}
    missing_models = sorted(required_model_ids - observed_model_ids)
    unexpected_models = sorted(observed_model_ids - required_model_ids)
    if missing_models:
        issues.append(f"official dense Qwen 3.6 models are missing: {missing_models}")
    if unexpected_models:
        issues.append(
            f"compatibility request includes undeclared dense Qwen 3.6 models: {unexpected_models}"
        )
    required_profiles = set(request.required_profiles)
    for requirement in request.required_dense_models:
        model_entries = [
            entry
            for entry in entries
            if entry.dense and entry.source_model.model_id == requirement.model_id
        ]
        revisions = {entry.source_model.revision for entry in model_entries}
        if None in revisions or len(revisions) != 1:
            issues.append(
                f"{requirement.model_id} must use one immutable source revision across profiles"
            )
        compatible_profiles = {entry.profile for entry in model_entries if entry.compatible}
        missing_profiles = sorted(
            profile.value for profile in required_profiles - compatible_profiles
        )
        if missing_profiles:
            issues.append(
                f"{requirement.model_id} is missing compatible profiles: {missing_profiles}"
            )
        artifact_paths = {entry.artifact_path for entry in model_entries}
        candidate_models = {
            (
                entry.candidate_model.model_id,
                entry.candidate_model.revision,
                entry.candidate_model.local_path,
            )
            for entry in model_entries
        }
        plan_sha256 = {entry.plan_sha256 for entry in model_entries}
        if len(artifact_paths) > 1 or len(candidate_models) > 1 or len(plan_sha256) > 1:
            issues.append(
                f"{requirement.model_id} profile evidence does not bind one candidate artifact"
            )
    if any(not entry.compatible for entry in entries):
        issues.append("one or more compatibility entries failed")
    candidate_keys = {
        (
            entry.candidate_model.model_id,
            entry.candidate_model.revision,
            entry.profile,
        )
        for entry in entries
    }
    if len(candidate_keys) != len(entries):
        issues.append("compatibility matrix candidate/profile identities must be unique")
    return CompatibilityMatrix(
        scope_policy=request.scope_policy,
        official_catalog_url=request.official_catalog_url,
        catalog_verified_at=request.catalog_verified_at,
        required_dense_models=request.required_dense_models,
        required_profiles=request.required_profiles,
        required_dense_checkpoints=len(request.required_dense_models),
        entries=entries,
        distinct_dense_source_checkpoints=len(source_keys),
        release_ready=not issues,
        issues=issues,
    )
