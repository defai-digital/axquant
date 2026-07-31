from __future__ import annotations

import shutil
from pathlib import Path

from axquant.errors import ArtifactError, ValidationGateError
from axquant.planner import allocate_kv_cache_measured
from axquant.recipes import RECIPE_BUNDLE_FILE, export_recipe_bundle, load_recipe_bundle
from axquant.release_exceptions import release_exception_allows_size
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    CalibrationManifest,
    EvaluationBundle,
    HardwareProfileRegistry,
    KvSensitivityReport,
    MtpSidecarLayout,
    ParetoReport,
    PreparedMtpSidecarManifest,
    ProfileName,
    QuantizationPlan,
    RefinementMeasurementSet,
    ReleaseValidationIndex,
    ReproductionCommand,
    ReproductionRecipe,
    RuntimeName,
    SupportTier,
    ValidationReport,
)
from axquant.serde import (
    file_sha256,
    load_model,
    read_data,
    stable_sha256,
    write_data,
    write_text,
)


def _validate_manifest_files(directory: Path, manifest: ArtifactManifest) -> None:
    for record in manifest.files:
        relative = Path(record.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationGateError(f"unsafe manifest path: {record.path}")
        path = directory / relative
        if not path.is_file():
            raise ValidationGateError(f"manifest file is missing: {record.path}")
        if path.stat().st_size != record.size_bytes:
            raise ValidationGateError(f"manifest size changed: {record.path}")
        if file_sha256(path) != record.sha256:
            raise ValidationGateError(f"manifest checksum changed: {record.path}")


def _artifact_files(directory: Path) -> list[ArtifactFile]:
    return [
        ArtifactFile(
            path=path.relative_to(directory).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
        if path.name != "axquant_manifest.json"
    ]


def _resolved_evidence_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


_HARDWARE_EVIDENCE_FIELDS = (
    ("plan_file", "plan_file_sha256", "plan.json"),
    (
        "artifact_manifest_file",
        "artifact_manifest_sha256",
        "artifact_manifest.json",
    ),
    ("sensitivity_file", "sensitivity_sha256", "sensitivity.json"),
    (
        "quality_comparison_file",
        "quality_comparison_sha256",
        "quality_comparison.json",
    ),
    ("validation_file", "validation_sha256", "validation.json"),
    ("direct_evaluation_file", "direct_evaluation_sha256", "evaluation_mtp_off.json"),
    ("mtp_evaluation_file", "mtp_evaluation_sha256", "evaluation_mtp_on.json"),
    (
        "direct_benchmark_result_file",
        "direct_benchmark_result_sha256",
        "benchmark_mtp_off.json",
    ),
    ("mtp_benchmark_result_file", "mtp_benchmark_result_sha256", "benchmark_mtp_on.json"),
    ("quantizer_execution_file", "quantizer_execution_sha256", "quantizer_execution.json"),
)


def _hardware_evidence_sources(
    *,
    registry_source: Path,
    registry: HardwareProfileRegistry,
) -> dict[str, dict[str, Path]]:
    measurement_path = _resolved_evidence_path(
        registry_source.parent,
        registry.measurement_set_file,
    )
    if (
        not measurement_path.is_file()
        or file_sha256(measurement_path) != registry.measurement_set_file_sha256
    ):
        raise ValidationGateError("hardware registry measurement set checksum mismatch")
    measurements = load_model(measurement_path, RefinementMeasurementSet)
    if stable_sha256(measurements) != registry.measurement_set_sha256:
        raise ValidationGateError("hardware registry measurement set semantic digest mismatch")
    measurements_by_id = {
        measurement.measurement_id: measurement for measurement in measurements.measurements
    }
    sources: dict[str, dict[str, Path]] = {}
    for entry in registry.entries:
        measurement = measurements_by_id.get(entry.measurement_id)
        if (
            measurement is None
            or measurement.candidate_id != entry.candidate_id
            or measurement.candidate_model != entry.candidate_model
            or measurement.profile != entry.profile
            or measurement.plan_sha256 != entry.plan_sha256
            or measurement.artifact_manifest_sha256 != entry.artifact_manifest_sha256
            or measurement.quality_comparison_sha256 != entry.quality_comparison_sha256
            or measurement.validation_sha256 != entry.validation_sha256
            or measurement.hardware != entry.hardware
            or measurement.validation_passed != entry.validation_passed
        ):
            raise ValidationGateError(f"hardware registry measurement differs for {entry.entry_id}")
        entry_sources: dict[str, Path] = {}
        for file_field, sha_field, _destination_name in _HARDWARE_EVIDENCE_FIELDS:
            path = _resolved_evidence_path(
                registry_source.parent,
                str(getattr(entry, file_field)),
            )
            if not path.is_file() or file_sha256(path) != getattr(entry, sha_field):
                raise ValidationGateError(
                    f"hardware registry evidence checksum mismatch: {entry.entry_id}/{file_field}"
                )
            entry_sources[file_field] = path
        sources[entry.entry_id] = entry_sources
    return sources


def _package_hardware_evidence(
    *,
    directory: Path,
    registry: HardwareProfileRegistry,
    sources: dict[str, dict[str, Path]],
    measurement_source: Path,
) -> Path:
    packaged = registry.model_copy(deep=True)
    packaged_measurements = directory / "refinement_measurements.json"
    if measurement_source != packaged_measurements:
        shutil.copy2(measurement_source, packaged_measurements)
    packaged.measurement_set_file = packaged_measurements.relative_to(directory).as_posix()
    packaged_entries = {entry.entry_id: entry for entry in packaged.entries}
    for entry_id, entry_sources in sources.items():
        target_directory = directory / "hardware_evidence" / entry_id
        target_directory.mkdir(parents=True, exist_ok=True)
        packaged_entry = packaged_entries[entry_id]
        for file_field, _sha_field, destination_name in _HARDWARE_EVIDENCE_FIELDS:
            source = entry_sources[file_field]
            destination = target_directory / destination_name
            if source != destination:
                shutil.copy2(source, destination)
            setattr(
                packaged_entry,
                file_field,
                destination.relative_to(directory).as_posix(),
            )
    output = directory / "hardware_profile_registry.json"
    write_data(output, packaged)
    return output


def _benchmark_evidence_sources(
    *,
    index_source: Path,
    index: BenchmarkEvidenceIndex,
    validation: ValidationReport,
) -> list[tuple[BenchmarkEvidenceEntry, Path]]:
    if not index.release_ready:
        raise ValidationGateError("publication requires a release-ready benchmark evidence index")
    if index.profile != validation.profile:
        raise ValidationGateError("benchmark evidence profile does not match validation")
    benchmark_entries = {entry.kind: entry for entry in index.entries}
    if len(benchmark_entries) != len(index.entries) or set(benchmark_entries) != set(
        BenchmarkEvidenceKind
    ):
        raise ValidationGateError("benchmark evidence index does not list every baseline once")
    reference_entry = benchmark_entries[BenchmarkEvidenceKind.UNIFORM_6BIT]
    direct_entry = benchmark_entries[BenchmarkEvidenceKind.AXQUANT_MTP_OFF]
    mtp_entry = benchmark_entries[BenchmarkEvidenceKind.AXQUANT_MTP_ON]
    required_entries = (
        benchmark_entries[BenchmarkEvidenceKind.BF16],
        benchmark_entries[BenchmarkEvidenceKind.UNIFORM_4BIT],
        reference_entry,
        direct_entry,
        mtp_entry,
    )
    for entry in required_entries:
        if entry.status != "available" or entry.model is None:
            raise ValidationGateError(f"required benchmark evidence is unavailable: {entry.kind}")
    assert reference_entry.model is not None
    assert direct_entry.model is not None
    assert mtp_entry.model is not None
    if reference_entry.model != validation.reference_model:
        raise ValidationGateError("uniform-6 benchmark evidence does not match validation")
    for entry in (direct_entry, mtp_entry):
        if entry.model != validation.candidate_model:
            raise ValidationGateError("AXQuant benchmark evidence does not match validation")

    sources: list[tuple[BenchmarkEvidenceEntry, Path]] = []
    for entry in index.entries:
        if entry.status != "available":
            continue
        assert entry.evaluation_file is not None
        assert entry.evaluation_sha256 is not None
        assert entry.model is not None
        source = _resolved_evidence_path(index_source.parent, entry.evaluation_file)
        if not source.is_file() or file_sha256(source) != entry.evaluation_sha256:
            raise ValidationGateError(f"benchmark evidence checksum mismatch: {entry.kind}")
        bundle = load_model(source, EvaluationBundle)
        if (
            bundle.baseline_kind != entry.kind.value
            or bundle.model != entry.model
            or bundle.workload != index.profile.value
            or bundle.dataset_sha256 != index.dataset_sha256
            or bundle.random_seed != index.random_seed
        ):
            raise ValidationGateError(f"benchmark evidence contents are inconsistent: {entry.kind}")
        sources.append((entry, source))
    return sources


def _package_benchmark_evidence(
    *,
    directory: Path,
    index: BenchmarkEvidenceIndex,
    sources: list[tuple[BenchmarkEvidenceEntry, Path]],
    index_name: str,
    evidence_directory_name: str,
) -> Path:
    packaged = index.model_copy(deep=True)
    packaged_entries = {entry.kind: entry for entry in packaged.entries}
    evidence_directory = directory / evidence_directory_name
    evidence_directory.mkdir(exist_ok=True)
    for source_entry, source in sources:
        entry = packaged_entries[source_entry.kind]
        assert entry.evaluation_sha256 is not None
        destination = evidence_directory / f"{entry.kind.value}.json"
        if destination.exists():
            if file_sha256(destination) != entry.evaluation_sha256:
                raise ValidationGateError(f"packaged benchmark evidence changed: {entry.kind}")
        else:
            shutil.copy2(source, destination)
        entry.evaluation_file = destination.relative_to(directory).as_posix()
    packaged_path = directory / index_name
    write_data(packaged_path, packaged)
    return packaged_path


def plan_markdown(plan: QuantizationPlan) -> str:
    distribution = "\n".join(
        f"| {precision} | {share.parameters:,} | {share.fraction:.2%} |"
        for precision, share in plan.weight_distribution.items()
    )
    mtp_distribution = "\n".join(
        f"| {precision} | {share.parameters:,} | {share.fraction:.2%} |"
        for precision, share in plan.mtp_distribution.items()
    )
    if not mtp_distribution:
        mtp_distribution = "| none | 0 | 0.00% |"
    return f"""# AXQuant Plan Report

| Property | Value |
| --- | --- |
| Source | `{plan.source_model.model_id}` |
| Revision | `{plan.source_model.revision or "unrecorded"}` |
| Profile | `{plan.profile.value}` |
| Family | `{plan.architecture_profile.product_family}` |
| Support tier | `{plan.architecture_profile.support_tier.value}` |
| Target class | `{plan.target_class}` |
| Target BPW | {plan.target_bpw:.4f} |
| Nominal BPW | {plan.nominal_bpw:.4f} |
| Storage-adjusted BPW | {plan.effective_bpw:.4f} |
| Evidence | `{plan.evidence_kind.value}` |
| MTP policy | `{plan.mtp.mode}` |
| Global validation required | `{plan.global_validation_required}` |

## Precision distribution

| Precision | Parameters | Share |
| --- | ---: | ---: |
{distribution}

## MTP precision distribution

| Precision | Parameters | Share |
| --- | ---: | ---: |
{mtp_distribution}

## Release constraints

| Constraint | Value |
| --- | ---: |
| Maximum effective BPW | {plan.constraints.effective_bpw_limit:.4f} |
| Maximum size ratio to uniform 4-bit | {plan.constraints.max_model_size_ratio_to_uniform4:.4f} |
| Minimum quality retention | {plan.constraints.minimum_quality_retention:.4f} |
| Minimum MTP acceptance retention | {plan.constraints.minimum_mtp_acceptance_retention:.4f} |
| Minimum MTP speedup | {plan.constraints.minimum_mtp_speedup:.4f} |

Proxy planning does not satisfy the quality, MTP, or speed release gates. Those constraints require
complete-model validation evidence.
"""


def validation_markdown(report: ValidationReport) -> str:
    issues = "\n".join(
        f"- **{item.severity.upper()}** `{item.metric}`: {item.message}" for item in report.issues
    )
    if not issues:
        issues = "- No validation issues."
    comparisons = "\n".join(
        f"| `{metric}` | {value} |" for metric, value in sorted(report.comparisons.items())
    )
    exceptions = "\n".join(
        f"- `{exception.exception_id}` — approved by {exception.approved_by}; "
        f"owner {exception.owner}; expires {exception.expires_at.isoformat()}"
        for exception in report.release_exceptions
    )
    if not exceptions:
        exceptions = "- None."
    return f"""# AXQuant Benchmark Report

| Property | Value |
| --- | --- |
| Reference | `{report.reference_model.model_id}` |
| Candidate | `{report.candidate_model.model_id}` |
| Profile | `{report.profile.value}` |
| Release gates | `{"PASS" if report.passed else "FAIL"}` |

## Comparisons

| Metric | Value |
| --- | ---: |
{comparisons}

## Issues

{issues}

## Governed release exceptions

{exceptions}
"""


def _verify_measured_kv_plan(directory: Path, plan: QuantizationPlan) -> None:
    """Prove a measured KV plan reproduces from its packaged report (AXQ-025)."""
    kv = plan.kv_cache
    if kv is None or kv.allocation_basis != "measured":
        return
    report_path = directory / "kv_sensitivity.json"
    if not report_path.is_file():
        raise ValidationGateError(
            "a measured KV-cache plan requires the packaged kv_sensitivity.json report"
        )
    report = load_model(report_path, KvSensitivityReport)
    if stable_sha256(report) != kv.sensitivity_sha256:
        raise ValidationGateError("packaged KV sensitivity report does not match the plan binding")
    if kv.max_output_kl is None:
        raise ValidationGateError("a measured KV-cache plan must record its selection budget")
    reproduced = allocate_kv_cache_measured(
        report,
        max_output_kl=kv.max_output_kl,
        min_bits=kv.min_bits,
    )
    if reproduced.layers != kv.layers:
        raise ValidationGateError(
            "the measured KV-cache allocation cannot be reproduced from its packaged report"
        )


def _package_recipe_bundle(
    *,
    directory: Path,
    plan_file: Path,
    repo_id: str,
    lineage: dict[str, str],
) -> Path:
    """Package the release plan as a recipe bundle (AXQ-020); idempotent on re-prepare."""
    recipe_dir = directory / "recipe"
    bundle_record = recipe_dir / RECIPE_BUNDLE_FILE
    if bundle_record.exists():
        record, _ = load_recipe_bundle(bundle_record)
        if file_sha256(plan_file) != record.payload_sha256:
            raise ValidationGateError("packaged recipe bundle does not match the packaged plan")
        return bundle_record
    return export_recipe_bundle(
        plan=plan_file,
        output_dir=recipe_dir,
        bundle_id=f"{repo_id.rsplit('/', 1)[-1]}-recipe",
        lineage=lineage,
        notes=["Exported from prepared release evidence (AXQ-020)."],
    )


def prepare_publication(
    *,
    model_dir: str | Path,
    repo_id: str,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
) -> list[Path]:
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ArtifactError(f"model directory does not exist: {directory}")
    manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
    plan = load_model(directory / "axquant_plan.json", QuantizationPlan)
    if plan.architecture_profile.support_tier is SupportTier.INSPECT_ONLY:
        raise ValidationGateError(
            "publication requires at least the convertible support tier; the packaged plan "
            "records an inspect-only family (AXQ-017)"
        )
    _verify_measured_kv_plan(directory, plan)
    validation_index_source = Path(validation_index_path).expanduser().resolve()
    validation_index = load_model(validation_index_source, ReleaseValidationIndex)
    if not validation_index.release_ready:
        raise ValidationGateError("publication requires a release-ready validation index")
    evidence: dict[
        ProfileName,
        tuple[
            ValidationReport,
            BenchmarkEvidenceIndex,
            list[tuple[BenchmarkEvidenceEntry, Path]],
        ],
    ] = {}
    for entry in validation_index.entries:
        validation_path = _resolved_evidence_path(
            validation_index_source.parent,
            entry.validation_file,
        )
        benchmark_index_source = _resolved_evidence_path(
            validation_index_source.parent,
            entry.benchmark_index_file,
        )
        if not validation_path.is_file() or file_sha256(validation_path) != entry.validation_sha256:
            raise ValidationGateError(f"{entry.profile.value} validation checksum mismatch")
        if (
            not benchmark_index_source.is_file()
            or file_sha256(benchmark_index_source) != entry.benchmark_index_sha256
        ):
            raise ValidationGateError(f"{entry.profile.value} benchmark index checksum mismatch")
        validation_report = load_model(validation_path, ValidationReport)
        benchmark_index = load_model(benchmark_index_source, BenchmarkEvidenceIndex)
        if (
            validation_report.profile != entry.profile
            or benchmark_index.profile != entry.profile
            or validation_report.reference_model != entry.reference_model
            or validation_report.candidate_model != entry.candidate_model
            or not validation_report.passed
            or not entry.passed
        ):
            raise ValidationGateError(
                f"{entry.profile.value} release validation evidence is inconsistent"
            )
        benchmark_sources = _benchmark_evidence_sources(
            index_source=benchmark_index_source,
            index=benchmark_index,
            validation=validation_report,
        )
        evidence[entry.profile] = (
            validation_report,
            benchmark_index,
            benchmark_sources,
        )
    if set(evidence) != {ProfileName.AGENT_CODING, ProfileName.GENERAL}:
        raise ValidationGateError("publication requires agent-coding and general validation")
    validation, benchmark_index, benchmark_sources = evidence[ProfileName.AGENT_CODING]
    general_validation, general_benchmark_index, general_sources = evidence[ProfileName.GENERAL]
    if (
        validation.candidate_model != general_validation.candidate_model
        or validation.reference_model != general_validation.reference_model
        or benchmark_index.dataset_sha256 == general_benchmark_index.dataset_sha256
    ):
        raise ValidationGateError("required validation profiles are not matched and disjoint")
    hardware_registry_source = Path(hardware_registry_path).expanduser().resolve()
    pareto_report_source = Path(pareto_report_path).expanduser().resolve()
    hardware_registry = load_model(hardware_registry_source, HardwareProfileRegistry)
    pareto_report = load_model(pareto_report_source, ParetoReport)
    if not hardware_registry.release_ready:
        raise ValidationGateError("publication requires a release-ready hardware registry")
    if (
        pareto_report.profile != ProfileName.AGENT_CODING
        or not pareto_report.frontier_candidate_ids
    ):
        raise ValidationGateError("publication requires an agent-coding Pareto frontier")
    if pareto_report.measurement_set_sha256 != hardware_registry.measurement_set_sha256:
        raise ValidationGateError("hardware registry and Pareto report use different measurements")
    hardware_sources = _hardware_evidence_sources(
        registry_source=hardware_registry_source,
        registry=hardware_registry,
    )
    conversion_manifest_path = directory / "axquant_conversion_manifest.json"
    conversion_manifest = (
        load_model(conversion_manifest_path, ArtifactManifest)
        if conversion_manifest_path.is_file()
        else manifest
    )
    if (
        conversion_manifest.plan_sha256 != manifest.plan_sha256
        or conversion_manifest.logical_parameters != manifest.logical_parameters
        or conversion_manifest.weight_file_size_bytes != manifest.weight_file_size_bytes
    ):
        raise ValidationGateError("immutable conversion manifest does not match candidate artifact")
    if not validation.passed:
        raise ValidationGateError("publication preparation requires passing validation")
    if validation.candidate_model.model_id != repo_id:
        raise ValidationGateError("validated candidate model does not match publication repository")
    candidate_revision = validation.candidate_model.revision
    if not candidate_revision:
        raise ValidationGateError("publication requires an immutable candidate revision")
    approved_exception = None
    for label, report in (
        ("agent-coding", validation),
        ("general", general_validation),
    ):
        size_ratio = report.comparisons.get("artifact.weight_size_ratio")
        if not isinstance(size_ratio, (int, float)):
            raise ValidationGateError(
                f"publication requires {label} measured artifact size comparison"
            )
        if float(size_ratio) > report.thresholds.max_weight_size_ratio:
            exception = release_exception_allows_size(report, plan=plan)
            if approved_exception is not None and stable_sha256(
                approved_exception
            ) != stable_sha256(exception):
                raise ValidationGateError("publication profiles use different release exceptions")
            approved_exception = exception
        elif report.release_exceptions:
            raise ValidationGateError(
                f"{label} validation contains an unnecessary release exception"
            )
    candidate_weight_bytes = validation.comparisons.get("artifact.candidate_weight_bytes")
    if candidate_weight_bytes != manifest.weight_file_size_bytes:
        raise ValidationGateError("validation size evidence does not match candidate weight bytes")
    candidate_size_source = validation.comparisons.get("artifact.candidate_source_sha256")
    source_manifest_path = (
        conversion_manifest_path
        if conversion_manifest_path.is_file()
        else directory / "axquant_manifest.json"
    )
    if candidate_size_source != file_sha256(source_manifest_path):
        raise ValidationGateError("validation size evidence does not bind the candidate manifest")
    if not plan.evidence_kind.release_quality or plan.calibration is None:
        raise ValidationGateError("publication requires measured calibration evidence")
    calibration_path = directory / "calibration_manifest.json"
    calibration_sha256 = plan.calibration.metadata.get("calibration_manifest_sha256")
    if not calibration_path.is_file() or not isinstance(calibration_sha256, str):
        raise ValidationGateError("publication requires the bound calibration manifest")
    if file_sha256(calibration_path) != calibration_sha256:
        raise ValidationGateError("packaged calibration manifest does not match the plan")
    calibration = load_model(calibration_path, CalibrationManifest)
    if (
        calibration.model != plan.source_model
        or calibration.profile != plan.profile
        or calibration.dataset_sha256 != plan.calibration.dataset_sha256
        or not calibration.calibration_evaluation_separation_attested
    ):
        raise ValidationGateError("packaged calibration provenance is inconsistent")
    if not plan.source_model.revision:
        raise ValidationGateError("publication requires an immutable source revision")
    if stable_sha256(plan) != manifest.plan_sha256:
        raise ValidationGateError("manifest plan hash does not match axquant_plan.json")
    matching_hardware = [
        entry
        for entry in hardware_registry.entries
        if entry.candidate_model == validation.candidate_model
        and entry.profile == ProfileName.AGENT_CODING
        and entry.plan_sha256 == manifest.plan_sha256
        and entry.release_ready
    ]
    if not matching_hardware:
        raise ValidationGateError("hardware registry does not certify the publication candidate")
    matching_frontier = [
        point
        for point in pareto_report.points
        if point.candidate_model == validation.candidate_model
        and point.plan_sha256 == manifest.plan_sha256
        and point.measurement_id in {entry.measurement_id for entry in matching_hardware}
        and point.frontier
        and point.candidate_id in pareto_report.frontier_candidate_ids
    ]
    if not matching_frontier:
        raise ValidationGateError("publication candidate is not on the measured Pareto frontier")
    _validate_manifest_files(directory, manifest)
    if manifest.runtime.primary_runtime.name != RuntimeName.AX_ENGINE:
        raise ValidationGateError("AX Engine must be the primary runtime")
    if not (directory / "model-manifest.json").is_file():
        raise ValidationGateError("AX Engine model-manifest.json is required")
    if not (directory / "axquant_runtime.json").is_file():
        raise ValidationGateError("axquant_runtime.json is required")
    if not conversion_manifest_path.exists():
        shutil.copy2(directory / "axquant_manifest.json", conversion_manifest_path)

    # Hardware evidence must remain the exact byte snapshot that was measured and
    # checksum-bound by the registry.  Publication updates the live artifact manifest
    # below with formal MTP metrics and the final packaged-file inventory, so copy all
    # registry evidence before mutating that source file.
    packaged_hardware_registry = _package_hardware_evidence(
        directory=directory,
        registry=hardware_registry,
        sources=hardware_sources,
        measurement_source=_resolved_evidence_path(
            hardware_registry_source.parent,
            hardware_registry.measurement_set_file,
        ),
    )

    acceptance = validation.comparisons.get("mtp.acceptance_retention")
    speedup = validation.comparisons.get("hardware.effective_speedup")
    if isinstance(acceptance, (int, float)):
        manifest.mtp_acceptance_retention = float(acceptance)
        manifest.runtime.mtp.acceptance_retention = float(acceptance)
    if isinstance(speedup, (int, float)):
        manifest.mtp_measured_speedup = float(speedup)
        manifest.runtime.mtp.measured_speedup = float(speedup)
    manifest.runtime.mtp.optimized = manifest.mtp_present and validation.passed
    write_data(directory / "axquant_manifest.json", manifest)
    write_data(directory / "axquant_runtime.json", manifest.runtime)

    benchmark_json = directory / "benchmark_report.json"
    benchmark_markdown = directory / "benchmark_report.md"
    general_benchmark_json = directory / "general_benchmark_report.json"
    general_benchmark_markdown = directory / "general_benchmark_report.md"
    write_data(benchmark_json, validation)
    write_text(benchmark_markdown, validation_markdown(validation))
    write_data(general_benchmark_json, general_validation)
    write_text(general_benchmark_markdown, validation_markdown(general_validation))
    packaged_release_exception = directory / "release_exception.json"
    if approved_exception is not None:
        write_data(packaged_release_exception, approved_exception)
    elif packaged_release_exception.exists():
        raise ValidationGateError("publication artifact contains a stale release_exception.json")
    packaged_benchmark_index = _package_benchmark_evidence(
        directory=directory,
        index=benchmark_index,
        sources=benchmark_sources,
        index_name="benchmark_evidence_index.json",
        evidence_directory_name="benchmark_evidence",
    )
    packaged_general_benchmark_index = _package_benchmark_evidence(
        directory=directory,
        index=general_benchmark_index,
        sources=general_sources,
        index_name="general_benchmark_evidence_index.json",
        evidence_directory_name="general_benchmark_evidence",
    )
    packaged_validation_index_model = validation_index.model_copy(deep=True)
    packaged_validation_entries = {
        entry.profile: entry for entry in packaged_validation_index_model.entries
    }
    primary_release_entry = packaged_validation_entries[ProfileName.AGENT_CODING]
    primary_release_entry.validation_file = benchmark_json.name
    primary_release_entry.validation_sha256 = file_sha256(benchmark_json)
    primary_release_entry.benchmark_index_file = packaged_benchmark_index.name
    primary_release_entry.benchmark_index_sha256 = file_sha256(packaged_benchmark_index)
    general_release_entry = packaged_validation_entries[ProfileName.GENERAL]
    general_release_entry.validation_file = general_benchmark_json.name
    general_release_entry.validation_sha256 = file_sha256(general_benchmark_json)
    general_release_entry.benchmark_index_file = packaged_general_benchmark_index.name
    general_release_entry.benchmark_index_sha256 = file_sha256(packaged_general_benchmark_index)
    packaged_validation_index = directory / "release_validation_index.json"
    write_data(packaged_validation_index, packaged_validation_index_model)
    packaged_pareto_report = directory / "pareto_report.json"
    write_data(packaged_pareto_report, pareto_report)
    quantization_plan = directory / "quantization_plan.json"
    reproduction_recipe = directory / "reproduction_recipe.yaml"
    write_data(quantization_plan, plan)
    _package_recipe_bundle(
        directory=directory,
        plan_file=quantization_plan,
        repo_id=repo_id,
        lineage={
            "plan": manifest.plan_sha256,
            "calibration_manifest": file_sha256(calibration_path),
            "agent_coding_validation": file_sha256(benchmark_json),
            "general_validation": file_sha256(general_benchmark_json),
        },
    )
    source_revision = plan.source_model.revision
    assert source_revision is not None
    mtp_sidecar_path = directory / "mtp.safetensors"
    mtp_sidecar_file = mtp_sidecar_path.name if mtp_sidecar_path.is_file() else None
    mtp_companion_files: list[ArtifactFile] = []
    mtp_layout = MtpSidecarLayout.BYTE_PRESERVED
    mtp_provenance_path = directory / "ax_mtp_sidecar_manifest.json"
    for companion_name in ("ax_mtp_sidecar_manifest.json", "mtplx_runtime.json"):
        companion_path = directory / companion_name
        if companion_path.is_file():
            mtp_companion_files.append(
                ArtifactFile(
                    path=companion_name,
                    size_bytes=companion_path.stat().st_size,
                    sha256=file_sha256(companion_path),
                )
            )
    if mtp_sidecar_file is not None and mtp_provenance_path.is_file():
        mtp_provenance = read_data(mtp_provenance_path)
        transform = mtp_provenance.get("transform") if isinstance(mtp_provenance, dict) else None
        transform_mode = transform.get("mode") if isinstance(transform, dict) else None
        if transform_mode == MtpSidecarLayout.AX_ENGINE_QWEN36_V1.value:
            load_model(mtp_provenance_path, PreparedMtpSidecarManifest)
            mtp_layout = MtpSidecarLayout.AX_ENGINE_QWEN36_V1
            required_companions = {
                "ax_mtp_sidecar_manifest.json",
                "mtplx_runtime.json",
            }
            if {record.path for record in mtp_companion_files} != required_companions:
                raise ValidationGateError(
                    "prepared MTP reproduction requires provenance and runtime companions"
                )
    convert_command = [
        "axquant",
        "convert",
        "--model",
        "source-model",
        "--revision",
        source_revision,
        "--plan",
        quantization_plan.name,
        "--calibration-manifest",
        calibration_path.name,
    ]
    if mtp_sidecar_file is not None:
        convert_command.extend(
            [
                "--mtp-sidecar",
                mtp_sidecar_file,
                "--mtp-layout",
                mtp_layout.value,
            ]
        )
    convert_command.extend(
        [
            "--ax-engine-manifest",
            "required",
            "--output",
            "regenerated-model",
        ]
    )
    reproduction_commands = [
        ReproductionCommand(
            step_id="download-source",
            description="Download the immutable source checkpoint.",
            argv=[
                "hf",
                "download",
                plan.source_model.model_id,
                "--revision",
                source_revision,
                "--local-dir",
                "source-model",
            ],
            expected_outputs=["source-model/config.json"],
        ),
        ReproductionCommand(
            step_id="convert",
            description="Regenerate the AXQuant checkpoint from the bound plan and evidence.",
            argv=convert_command,
            expected_outputs=["regenerated-model/axquant_manifest.json"],
        ),
        ReproductionCommand(
            step_id="verify-ax-engine",
            description="Run the AX Engine readiness contract.",
            argv=[
                "axquant",
                "runtime-check",
                "--model",
                "regenerated-model",
                "--model-id",
                repo_id,
                "--revision",
                candidate_revision,
                "--runtime",
                "ax-engine",
                "--output",
                "runtime-check-ax-engine.json",
            ],
            expected_outputs=["runtime-check-ax-engine.json"],
        ),
        ReproductionCommand(
            step_id="verify-mlx-lm",
            description="Run the MLX-LM standard-generation contract.",
            argv=[
                "axquant",
                "runtime-check",
                "--model",
                "regenerated-model",
                "--model-id",
                repo_id,
                "--revision",
                candidate_revision,
                "--runtime",
                "mlx-lm",
                "--output",
                "runtime-check-mlx-lm.json",
            ],
            expected_outputs=["runtime-check-mlx-lm.json"],
        ),
        ReproductionCommand(
            step_id="verify-reproduction",
            description="Verify regenerated weight bytes and provenance against this release.",
            argv=[
                "axquant",
                "verify-reproduction",
                "--recipe",
                reproduction_recipe.name,
                "--artifact",
                "regenerated-model",
                "--output",
                "reproduction-verification.json",
            ],
            expected_outputs=["reproduction-verification.json"],
        ),
    ]
    expected_weight_files = [
        record
        for record in conversion_manifest.files
        if Path(record.path).suffix.lower() == ".safetensors"
    ]
    if not expected_weight_files:
        raise ValidationGateError("conversion manifest has no reproducible weight files")
    write_data(
        reproduction_recipe,
        ReproductionRecipe(
            source_model=plan.source_model,
            calibration=plan.calibration,
            axquant_version=manifest.axquant_version,
            software_versions=manifest.software_versions,
            random_seed=plan.random_seed,
            profile=plan.profile,
            primary_runtime=manifest.runtime.primary_runtime.name,
            plan_sha256=manifest.plan_sha256,
            output_repository=repo_id,
            plan_file_sha256=file_sha256(quantization_plan),
            calibration_file_sha256=file_sha256(calibration_path),
            conversion_manifest_sha256=file_sha256(conversion_manifest_path),
            mtp_sidecar_file=mtp_sidecar_file,
            mtp_sidecar_sha256=(
                file_sha256(mtp_sidecar_path) if mtp_sidecar_file is not None else None
            ),
            mtp_companion_files=mtp_companion_files,
            expected_logical_parameters=conversion_manifest.logical_parameters,
            expected_weight_file_size_bytes=conversion_manifest.weight_file_size_bytes,
            expected_weight_files=expected_weight_files,
            commands=reproduction_commands,
        ),
    )

    readme = directory / "README.md"
    if readme.exists() and not (directory / "UPSTREAM_README.md").exists():
        shutil.copy2(readme, directory / "UPSTREAM_README.md")
    mtp_status = "included and validated" if manifest.mtp_present else "not included"
    write_text(
        readme,
        f"""---
library_name: mlx
tags:
  - mlx
  - apple-silicon
  - quantized
  - axquant
---

# {repo_id.split("/", 1)[-1]}

This is an MLX checkpoint produced by AXQuant for Apple Silicon. It is optimized for AX Engine
and retains a standard MLX weight layout for MLX-LM compatibility.

| Property | Value |
| --- | --- |
| Base model | `{manifest.source_model.model_id}` |
| Source revision | `{manifest.source_model.revision or "unrecorded"}` |
| Quantizer | AXQuant {manifest.axquant_version} |
| Target class | `{manifest.target_class}` |
| Planned storage-adjusted BPW | {manifest.effective_bpw:.4f} |
| Measured main-model BPW | {manifest.measured_main_bpw:.4f} |
| Measured total BPW (including MTP) | {manifest.measured_total_bpw:.4f} |
| Logical parameters | {manifest.logical_parameters:,} |
| Safetensors weight bytes | {manifest.weight_file_size_bytes:,} |
| Profile | `{manifest.profile.value}` |
| Primary runtime | `AX Engine` (Compatibility Level A) |
| Compatible runtime | `MLX-LM` standard inference (Compatibility Level B) |
| MTP | {mtp_status} |
| Validation | PASS |

AX Engine is the authority for MTP acceleration and runtime-specific performance claims. MLX-LM
compatibility covers standard backbone inference; MTP support is runtime-dependent and AXQuant
metadata may be ignored.

See `model-manifest.json`, `axquant_runtime.json`, `axquant_manifest.json`,
`axquant_conversion_manifest.json`, `quantization_plan.json`, `release_validation_index.json`,
`hardware_profile_registry.json`, `refinement_measurements.json`, `pareto_report.json`, both
benchmark reports and evidence indexes, and `reproduction_recipe.yaml` for auditable evidence.
`UPSTREAM_README.md` preserves the source model card when one was present.
""",
    )
    manifest.files = _artifact_files(directory)
    write_data(directory / "axquant_manifest.json", manifest)
    prepared = [
        benchmark_json,
        benchmark_markdown,
        packaged_benchmark_index,
        general_benchmark_json,
        general_benchmark_markdown,
        packaged_general_benchmark_index,
        packaged_validation_index,
        packaged_hardware_registry,
        directory / "refinement_measurements.json",
        packaged_pareto_report,
        quantization_plan,
        reproduction_recipe,
        readme,
    ]
    if approved_exception is not None:
        prepared.append(packaged_release_exception)
    return prepared
