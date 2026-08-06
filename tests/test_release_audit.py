from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from _capture_helpers import load_test_activation_capture
from safetensors.numpy import save_file

from axquant.calibration import calibration_manifest_sha256
from axquant.capture_binding import activation_capture_metadata
from axquant.cli import main
from axquant.pareto import build_pareto_report
from axquant.planner import plan_quantization
from axquant.profiles import thresholds_for
from axquant.publisher import _rerun_release_audit
from axquant.refinement import (
    COMPLETE_OBJECTIVE_VERSION,
    build_complete_candidate_measurement,
)
from axquant.release_audit import (
    _activation_capture_artifact_issues,
    _artifact_issues,
    _sensitivity_lineage_issues,
    _sensitivity_measurement_issues,
    _wheel_identity,
    build_release_audit,
)
from axquant.release_exceptions import apply_release_exception
from axquant.reproduction import verify_reproduction
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ActivationCaptureManifest,
    ArchitectureProfile,
    ArchitectureSupportLevel,
    ArtifactFile,
    ArtifactIntegrity,
    ArtifactManifest,
    ArtifactSizeEvidence,
    BaselineAudit,
    BaselineKind,
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    CalibrationEvidence,
    CalibrationManifest,
    CandidateEntry,
    CandidateMeasurement,
    CheckpointCompatibility,
    CompatibilityCandidateInput,
    CompatibilityMatrix,
    CompatibilityMatrixRequest,
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    EvaluationBundle,
    EvidenceKind,
    FeasibilityReport,
    HardwareKernelCoverage,
    HardwareMeasurementProtocol,
    HardwareMetrics,
    HardwareProfile,
    HardwareProfileRegistry,
    HardwareRegistryEntry,
    IntegrityMetrics,
    MetricVector,
    ModelIdentity,
    MtpMetrics,
    OfficialDenseCheckpointRequirement,
    OptimizationScope,
    ParetoReport,
    PlanRequest,
    ProfileName,
    QualityComparisonReport,
    QualityScoreComparison,
    QuantizationPlan,
    QuantMethod,
    RefinementConfig,
    RefinementMeasurementSet,
    RefinementResult,
    ReleaseAudit,
    ReleaseAuditRequest,
    ReleaseException,
    ReleaseExceptionTarget,
    ReleaseValidationEntry,
    ReleaseValidationIndex,
    ReproductionCommand,
    ReproductionRecipe,
    RuntimeCheck,
    RuntimeName,
    SensitivityReport,
    SoftwareVersions,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
    ValidationIssue,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

_SOURCE_REVISION = "a" * 40
_BASELINE_REVISION = "b" * 40
_CANDIDATE_REVISION = "c" * 40
_OTHER_REVISION = "d" * 40


def _sensitivity(
    *,
    source_model_id: str = "Qwen/Qwen3.6-test",
    source_revision: str = _SOURCE_REVISION,
) -> SensitivityReport:
    source = ModelIdentity(model_id=source_model_id, revision=source_revision)
    calibration = CalibrationEvidence(
        dataset_id="calibration",
        dataset_sha256="a" * 64,
        samples=8,
        domains=["coding"],
        sequence_length=128,
        backend="mlx-probe-v1",
        reference="calibration_manifest.json",
    )
    tensors = [
        TensorSpec(
            name="model.layers.0.mlp.down_proj.weight",
            module_path="model.layers.0.mlp.down_proj",
            shape=(16, 64),
            dtype="bfloat16",
            parameters=1024,
            role=TensorRole.MLP,
            quantizable=True,
            file="model.safetensors",
            current_precision="bfloat16",
        ),
        TensorSpec(
            name="model.layers.0.input_layernorm.weight",
            module_path="model.layers.0.input_layernorm",
            shape=(64,),
            dtype="bfloat16",
            parameters=64,
            role=TensorRole.NORM,
            quantizable=False,
            file="model.safetensors",
            current_precision="bfloat16",
        ),
    ]
    entries: list[TensorSensitivity] = []
    for tensor in tensors:
        if tensor.quantizable:
            candidates = [
                CandidateMeasurement(
                    bits=bits,
                    method=QuantMethod.DWQ if bits < 16 else QuantMethod.BF16,
                    group_size=64 if bits < 16 else None,
                    metrics=MetricVector(output_kl=(16 - bits) / 100),
                    measured_tokens=8192,
                )
                for bits in (4, 6, 8, 16)
            ]
        else:
            candidates = [
                CandidateMeasurement(
                    bits=bits,
                    method=QuantMethod.DWQ if bits < 16 else QuantMethod.BF16,
                    group_size=64 if bits < 16 else None,
                    metrics=MetricVector(),
                    supported=bits == 16,
                    evidence_scope="preserved",
                    measured_tokens=8192,
                )
                for bits in (4, 6, 8, 16)
            ]
        entries.append(TensorSensitivity(tensor=tensor, candidates=candidates))
    return SensitivityReport(
        model=source,
        architecture_profile=ArchitectureProfile(
            adapter_id="qwen36-v1",
            product_family="qwen3.6",
            support_level=ArchitectureSupportLevel.SUPPORTED,
            optimization_scope=OptimizationScope.TEXT_PATH,
            dense=True,
            text_layer_count=1,
        ),
        profile=ProfileName.AGENT_CODING,
        evidence_kind=EvidenceKind.MEASURED,
        inventory_sha256="inventory",
        calibration=calibration,
        entries=entries,
    )


def _plan(sensitivity: SensitivityReport):
    plan = plan_quantization(
        sensitivity,
        PlanRequest(
            profile=sensitivity.profile,
            target_bpw=8.0,
            candidate_bits=(4, 6, 8, 16),
            group_size=64,
            hardware=HardwareProfile(
                supported_methods=(QuantMethod.DWQ, QuantMethod.BF16),
            ),
        ),
    )
    versions = plan.software_versions.model_copy(update={"axquant": "1.0.0"})
    return plan.model_copy(update={"software_versions": versions})


def _write_wheel(path: Path, version: str, *, production_classifier: bool = True) -> None:
    dist_info = f"axquant-{version}.dist-info"
    classifier = (
        "Classifier: Development Status :: 5 - Production/Stable\n"
        if production_classifier
        else "Classifier: Development Status :: 3 - Alpha\n"
    )
    metadata = (
        f"Name: axquant\n"
        f"Version: {version}\n"
        f"{classifier}"
        "License: MIT\n"
        "Requires-Python: >=3.11\n"
        "Requires-Dist: huggingface-hub>=0.24\n"
        "Requires-Dist: pydantic<3,>=2.8\n"
        "Requires-Dist: pyyaml>=6.0\n"
        "Requires-Dist: safetensors>=0.4.5\n"
        "Requires-Dist: structlog>=24.2\n"
    )
    members = {
        f"{dist_info}/METADATA": metadata.encode(),
        f"{dist_info}/WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
        f"{dist_info}/entry_points.txt": (b"[console_scripts]\naxquant = axquant.cli:entrypoint\n"),
        f"{dist_info}/licenses/LICENSE": b"test license\n",
        "axquant/__init__.py": f'__version__ = "{version}"\n'.encode(),
    }
    for module in (
        "cli/__init__.py",
        "schema/__init__.py",
        "release_audit.py",
        "release_exceptions.py",
        "hardware_registry.py",
        "reporting.py",
    ):
        members[f"axquant/{module}"] = b"\n"
    record_name = f"{dist_info}/RECORD"
    record_buffer = io.StringIO()
    record_writer = csv.writer(record_buffer, lineterminator="\n")
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record_writer.writerow((name, f"sha256={digest}", len(data)))
    record_writer.writerow((record_name, "", ""))

    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in members.items():
            wheel.writestr(name, data)
        wheel.writestr(record_name, record_buffer.getvalue())


def _baseline(kind: BaselineKind, model: ModelIdentity) -> BaselineAudit:
    runtime_checks = [
        RuntimeCheck(
            model=model,
            runtime=RuntimeName.MLX_LM,
            check_kind="static-compatibility",
            available=True,
            passed=True,
        )
    ]
    if kind != BaselineKind.BF16_SOURCE:
        runtime_checks.insert(
            0,
            RuntimeCheck(
                model=model,
                runtime=RuntimeName.AX_ENGINE,
                check_kind="doctor",
                available=True,
                passed=True,
            ),
        )
    return BaselineAudit(
        kind=kind,
        model=model,
        inspected=True,
        adapter_id="qwen36-v1",
        optimization_scope=OptimizationScope.TEXT_PATH,
        quantized=kind != BaselineKind.BF16_SOURCE,
        logical_parameters=1088,
        mtp_logical_parameters=64,
        weight_bytes=544,
        main_weight_bytes=544,
        mtp_weight_bytes=0,
        effective_bpw=4.0,
        main_effective_bpw=4.0,
        precision_parameters={"4": 1088},
        precision_fractions={"4": 1.0},
        integrity=ArtifactIntegrity(
            config_valid=True,
            safetensors_present=True,
            index_present=True,
            index_complete=True,
            native_manifest_present=True,
            native_manifest_valid=True,
            tokenizer_present=True,
            mtp_sidecar_present=False,
            mtp_runtime_present=True,
            mtp_runtime_valid=True,
            mtp_provenance_present=True,
            mtp_provenance_valid=True,
        ),
        runtime_checks=runtime_checks,
        complete=True,
    )


def _benchmark_index(
    *,
    tmp_path: Path,
    candidate: ModelIdentity,
    reference: ModelIdentity,
    profile: ProfileName,
    dataset: str,
) -> BenchmarkEvidenceIndex:
    entries: list[BenchmarkEvidenceEntry] = []
    versions = SoftwareVersions(
        axquant="1.0.0",
        python="3.13",
        mlx="0.32",
        mlx_lm="0.31",
        ax_engine="6.11.1",
        safetensors="0.6",
        pydantic="2.11",
    )
    for kind in BenchmarkEvidenceKind:
        if kind == BenchmarkEvidenceKind.UNIFORM_6BIT:
            model = reference
        elif kind in {
            BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
            BenchmarkEvidenceKind.AXQUANT_MTP_ON,
        }:
            model = candidate
        else:
            model = ModelIdentity(
                model_id=f"fixture/{kind.value}",
                revision=_BASELINE_REVISION,
            )
        evaluation_path = tmp_path / f"{profile.value}-{kind.value}.json"
        write_data(
            evaluation_path,
            EvaluationBundle(
                model=model,
                mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
                baseline_kind=kind.value,
                # RM-20 admissibility requires the MTP-on half of the pair to
                # carry real acceptance metrics and effective throughput.
                mtp=(
                    MtpMetrics(
                        token_accuracy={"1": 0.8},
                        average_accepted_tokens=0.8,
                        acceptance_rate=0.8,
                        rejection_rate=0.2,
                        effective_tokens_per_forward=1.8,
                        repetition_rate=0.01,
                        divergence_rate=0.0,
                    )
                    if kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON
                    else None
                ),
                integrity=IntegrityMetrics(
                    safetensors_valid=True,
                    index_complete=True,
                    config_valid=True,
                    mtp_layout_valid=True,
                    source_revision_pinned=True,
                ),
                hardware=HardwareMetrics(
                    kernel_fallbacks=0,
                    device_name="Mac15,9",
                    chip="Apple M3 Max",
                    unified_memory_bytes=128 * 1024**3,
                    os_version="macOS-test",
                    mtp_effective_tokens_per_second=(
                        12.0 if kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON else None
                    ),
                ),
                workload=profile.value,
                dataset_sha256=dataset,
                software_versions=versions,
                random_seed=7,
                benchmark_metadata={
                    "prompt_count": 1,
                    "warmup_trials": 1,
                    "measured_trials": 1,
                    "successful_measured_trials": 1,
                    "failed_trials": 0,
                    "timed_out_trials": 0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_tokens": 16,
                    "power_mode": "AC power",
                    "quantizer": kind.value,
                    "quantizer_version": "test",
                    "quality_dataset_sha256": dataset,
                },
            ),
        )
        entries.append(
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
    return BenchmarkEvidenceIndex(
        profile=profile,
        dataset_sha256=dataset,
        random_seed=7,
        entries=entries,
        release_ready=True,
        issues=[],
    )


def _inputs(
    tmp_path: Path,
    *,
    wheel_version: str = "1.0.0",
    source_model_id: str = "Qwen/Qwen3.6-test",
    source_revision: str = _SOURCE_REVISION,
    candidate_model_id: str = "AutomatosX/AXQuant-test",
    target_class_override: str | None = None,
) -> Path:
    sensitivity = _sensitivity(
        source_model_id=source_model_id,
        source_revision=source_revision,
    )
    plan = _plan(sensitivity)
    if target_class_override is not None:
        plan = plan.model_copy(update={"target_class": target_class_override})
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    save_file(
        {"model.layers.0.mlp.down_proj.weight": np.zeros((1,), dtype=np.float32)},
        artifact / "model.safetensors",
    )
    save_file(
        {"mtp.fc.weight": np.zeros((1,), dtype=np.float32)},
        artifact / "mtp.safetensors",
    )
    (artifact / "model-manifest.json").write_text("{}\n", encoding="utf-8")
    (artifact / "README.md").write_text("# release\n", encoding="utf-8")
    calibration = CalibrationManifest(
        model=plan.source_model,
        profile=plan.profile,
        dataset_id=plan.calibration.dataset_id if plan.calibration else "missing",
        dataset_sha256=plan.calibration.dataset_sha256 if plan.calibration else "missing",
        samples=8,
        domains=["coding"],
        sequence_length=128,
        random_seed=plan.random_seed,
        calibration_evaluation_separation_attested=True,
    )
    write_data(artifact / "calibration_manifest.json", calibration)
    assert plan.calibration is not None
    calibration_evidence = plan.calibration.model_copy(
        update={
            "metadata": {
                **plan.calibration.metadata,
                "calibration_manifest_sha256": calibration_manifest_sha256(calibration),
                "calibration_random_seed": calibration.random_seed,
            }
        }
    )
    sensitivity = sensitivity.model_copy(update={"calibration": calibration_evidence})
    plan = plan.model_copy(
        update={
            "calibration": calibration_evidence,
            "analysis_sha256": stable_sha256(sensitivity),
        }
    )
    write_data(artifact / "axquant_plan.json", plan)
    write_data(artifact / "quantization_plan.json", plan)
    runtime = build_runtime_metadata(plan, artifact)
    write_data(artifact / "axquant_runtime.json", runtime)
    main_weight_bytes = (artifact / "model.safetensors").stat().st_size
    mtp_weight_bytes = (artifact / "mtp.safetensors").stat().st_size
    total_weight_bytes = main_weight_bytes + mtp_weight_bytes
    manifest = ArtifactManifest(
        axquant_version="1.0.0",
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        calibration=plan.calibration,
        profile=plan.profile,
        target_class=plan.target_class,
        effective_bpw=plan.effective_bpw,
        logical_parameters=1088,
        main_logical_parameters=1088,
        weight_file_size_bytes=total_weight_bytes,
        main_weight_file_size_bytes=main_weight_bytes,
        mtp_weight_file_size_bytes=mtp_weight_bytes,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=total_weight_bytes * 8 / 1088,
        measured_main_bpw=main_weight_bytes * 8 / 1088,
        weight_distribution=plan.weight_distribution,
        mtp_distribution=plan.mtp_distribution,
        mtp_present=True,
        mtp_policy=plan.mtp,
        runtime=runtime,
        software_versions=plan.software_versions,
        files=[
            ArtifactFile(
                path=name,
                size_bytes=(artifact / name).stat().st_size,
                sha256=file_sha256(artifact / name),
            )
            for name in ("model.safetensors", "mtp.safetensors")
        ],
    )
    write_data(artifact / "axquant_manifest.json", manifest)
    write_data(artifact / "axquant_conversion_manifest.json", manifest)

    candidate = ModelIdentity(
        model_id=candidate_model_id,
        revision=_CANDIDATE_REVISION,
        local_path=str(artifact.resolve()),
    )
    reference = ModelIdentity(model_id="fixture/uniform6", revision=_BASELINE_REVISION)
    validation_entries: list[ReleaseValidationEntry] = []
    for profile, dataset in (
        (ProfileName.AGENT_CODING, "b" * 64),
        (ProfileName.GENERAL, "c" * 64),
    ):
        validation = ValidationReport(
            reference_model=reference,
            candidate_model=candidate,
            profile=profile,
            passed=True,
            thresholds=thresholds_for(profile),
            issues=[],
            comparisons={
                "mtp.acceptance_retention": 0.97,
                "hardware.effective_speedup": 1.25,
                "artifact.candidate_weight_bytes": manifest.weight_file_size_bytes,
            },
        )
        benchmark = _benchmark_index(
            tmp_path=tmp_path,
            candidate=candidate,
            reference=reference,
            profile=profile,
            dataset=dataset,
        )
        validation_path = tmp_path / f"{profile.value}-validation.json"
        benchmark_path = tmp_path / f"{profile.value}-benchmark.json"
        write_data(validation_path, validation)
        write_data(benchmark_path, benchmark)
        validation_entries.append(
            ReleaseValidationEntry(
                profile=profile,
                validation_file=str(validation_path),
                validation_sha256=file_sha256(validation_path),
                benchmark_index_file=str(benchmark_path),
                benchmark_index_sha256=file_sha256(benchmark_path),
                reference_model=reference,
                candidate_model=candidate,
                dataset_sha256=dataset,
                passed=True,
            )
        )
    validation_index = ReleaseValidationIndex(
        entries=validation_entries,
        release_ready=True,
        issues=[],
    )
    validation_index_path = tmp_path / "release-validation-index.json"
    write_data(validation_index_path, validation_index)
    write_data(artifact / "release_validation_index.json", validation_index)

    parent_plan = plan_quantization(
        sensitivity,
        PlanRequest(
            profile=sensitivity.profile,
            target_bpw=5.2,
            candidate_bits=(4, 6, 8, 16),
            group_size=64,
            hardware=HardwareProfile(
                supported_methods=(QuantMethod.DWQ, QuantMethod.BF16),
            ),
            random_seed=plan.random_seed + 1,
        ),
    ).model_copy(update={"software_versions": plan.software_versions})
    parent_sha = stable_sha256(parent_plan)
    selected_sha = stable_sha256(plan)
    refinement = RefinementResult(
        config=RefinementConfig(top_n=2),
        history=[
            CandidateEntry(
                candidate_id="parent",
                plan_sha256=parent_sha,
                change_description="initial candidate",
                reason="initial",
                predicted_bpw=parent_plan.effective_bpw,
                measured_bpw=4.8,
                predicted_loss=0.2,
                measured_loss=0.2,
                budget_impact=0.0,
                state="rejected",
            ),
            CandidateEntry(
                candidate_id="selected",
                parent_id="parent",
                plan_sha256=selected_sha,
                change_description="DWQ interaction refinement",
                reason="measured improvement",
                predicted_bpw=plan.effective_bpw,
                measured_bpw=4.7,
                predicted_loss=0.1,
                measured_loss=0.1,
                budget_impact=-0.1,
                state="selected",
            ),
        ],
        candidate_plans={"parent": parent_plan, "selected": plan},
        selected_candidate_id="selected",
        selected_plan=plan,
        selected_plan_sha256=selected_sha,
        selection_basis="complete-model",
        iterations_used=1,
        evaluations_used=2,
        converged=True,
    )
    refinement_path = tmp_path / "refinement.json"
    write_data(refinement_path, refinement)
    sensitivity_path = tmp_path / "sensitivity.json"
    write_data(sensitivity_path, sensitivity)

    hardware_evidence = tmp_path / "hardware-evidence"
    hardware_evidence.mkdir()
    common_evidence_paths: dict[str, Path] = {}
    for name in (
        "direct-evaluation",
        "mtp-evaluation",
        "direct-result",
        "mtp-result",
        "execution",
    ):
        path = hardware_evidence / f"{name}.json"
        path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
        common_evidence_paths[name] = path
    sensitivity_evidence_path = hardware_evidence / "sensitivity.json"
    write_data(sensitivity_evidence_path, sensitivity)
    parent_main_weight_bytes = manifest.main_weight_file_size_bytes - 1
    parent_weight_bytes = parent_main_weight_bytes + manifest.mtp_weight_file_size_bytes
    parent_artifact = manifest.model_copy(
        update={
            "plan_sha256": parent_sha,
            "target_class": parent_plan.target_class,
            "effective_bpw": parent_plan.effective_bpw,
            "weight_file_size_bytes": parent_weight_bytes,
            "main_weight_file_size_bytes": parent_main_weight_bytes,
            "measured_total_bpw": parent_weight_bytes * 8 / 1088,
            "measured_main_bpw": parent_main_weight_bytes * 8 / 1088,
            "weight_distribution": parent_plan.weight_distribution,
            "mtp_distribution": parent_plan.mtp_distribution,
            "mtp_policy": parent_plan.mtp,
        }
    )
    candidate_evidence_paths = {
        "parent": {
            "plan": hardware_evidence / "parent-plan.json",
            "artifact": hardware_evidence / "parent-artifact.json",
            "quality": hardware_evidence / "parent-quality.json",
            "validation": hardware_evidence / "parent-validation.json",
        },
        "selected": {
            "plan": hardware_evidence / "selected-plan.json",
            "artifact": artifact / "axquant_manifest.json",
            "quality": hardware_evidence / "selected-quality.json",
            "validation": hardware_evidence / "selected-validation.json",
        },
    }
    write_data(candidate_evidence_paths["parent"]["plan"], parent_plan)
    write_data(candidate_evidence_paths["parent"]["artifact"], parent_artifact)
    write_data(candidate_evidence_paths["selected"]["plan"], plan)

    def quality_report(*, retention: float, perplexity_ratio: float) -> QualityComparisonReport:
        aggregate = QualityScoreComparison(
            reference=1.0,
            candidate=retention,
            delta=retention - 1.0,
            retention=retention,
        )
        return QualityComparisonReport(
            reference_model=reference,
            candidate_model=candidate,
            dataset_sha256="quality-dataset",
            random_seed=7,
            aggregate=aggregate,
            categories={"coding": aggregate},
            perplexity_ratio=perplexity_ratio,
            tasks=[],
            reference_errors=0,
            candidate_errors=0,
        )

    parent_quality = quality_report(retention=0.98, perplexity_ratio=1.02)
    selected_quality = quality_report(retention=0.99, perplexity_ratio=0.9)
    write_data(candidate_evidence_paths["parent"]["quality"], parent_quality)
    write_data(candidate_evidence_paths["selected"]["quality"], selected_quality)

    def complete_validation(
        *,
        artifact_path: Path,
        artifact_manifest: ArtifactManifest,
        acceptance: float,
        speedup: float,
        peak_memory_ratio: float,
        quality_retention: float,
        perplexity_ratio: float,
    ) -> ValidationReport:
        return ValidationReport(
            reference_model=reference,
            candidate_model=candidate,
            profile=plan.profile,
            passed=True,
            thresholds=thresholds_for(plan.profile),
            issues=[],
            comparisons={
                "artifact.candidate_source_sha256": file_sha256(artifact_path),
                "artifact.candidate_weight_bytes": (artifact_manifest.weight_file_size_bytes),
                "artifact.weight_size_ratio": 1.05,
                "quality.aggregate_retention": quality_retention,
                "perplexity_relative_increase": perplexity_ratio - 1.0,
                "mtp.acceptance_retention": acceptance,
                "hardware.effective_speedup": speedup,
                "hardware.peak_memory_ratio": peak_memory_ratio,
                "hardware.device_name": "Mac15,9",
                "hardware.chip": "Apple M3 Max",
                "hardware.unified_memory_bytes": 128 * 1024**3,
                "hardware.os_version": "macOS-test",
                "hardware.power_mode": "AC power",
                "hardware.kernel_fallbacks": 0,
                "software.ax_engine": "6.11.1",
                "software.mlx": "0.32",
                "software.mlx_lm": "0.31",
            },
        )

    parent_validation = complete_validation(
        artifact_path=candidate_evidence_paths["parent"]["artifact"],
        artifact_manifest=parent_artifact,
        acceptance=0.96,
        speedup=1.21,
        peak_memory_ratio=0.9,
        quality_retention=0.98,
        perplexity_ratio=1.02,
    )
    selected_validation = complete_validation(
        artifact_path=candidate_evidence_paths["selected"]["artifact"],
        artifact_manifest=manifest,
        acceptance=0.97,
        speedup=1.25,
        peak_memory_ratio=0.8,
        quality_retention=0.99,
        perplexity_ratio=0.9,
    )
    write_data(candidate_evidence_paths["parent"]["validation"], parent_validation)
    write_data(candidate_evidence_paths["selected"]["validation"], selected_validation)
    complete_hardware = CompleteCandidateHardware(
        device_name="Mac15,9",
        chip="Apple M3 Max",
        unified_memory_bytes=128 * 1024**3,
        os_version="macOS-test",
        ax_engine_version="6.11.1",
        mlx_version="0.32",
        mlx_lm_version="0.31",
        power_mode="AC power",
        kernel_fallbacks=0,
    )
    parent_measurement = build_complete_candidate_measurement(
        candidate_id="parent",
        plan=parent_plan,
        artifact=parent_artifact,
        artifact_sha256=file_sha256(candidate_evidence_paths["parent"]["artifact"]),
        quality=parent_quality,
        quality_sha256=file_sha256(candidate_evidence_paths["parent"]["quality"]),
        validation=parent_validation,
        validation_sha256=file_sha256(candidate_evidence_paths["parent"]["validation"]),
    )
    selected_measurement = build_complete_candidate_measurement(
        candidate_id="selected",
        plan=plan,
        artifact=manifest,
        artifact_sha256=file_sha256(candidate_evidence_paths["selected"]["artifact"]),
        quality=selected_quality,
        quality_sha256=file_sha256(candidate_evidence_paths["selected"]["quality"]),
        validation=selected_validation,
        validation_sha256=file_sha256(candidate_evidence_paths["selected"]["validation"]),
    )
    refinement = refinement.model_copy(
        update={
            "history": [
                refinement.history[0].model_copy(
                    update={
                        "measured_bpw": parent_measurement.measured_bpw,
                        "measured_loss": parent_measurement.objective_loss,
                    }
                ),
                refinement.history[1].model_copy(
                    update={
                        "measured_bpw": selected_measurement.measured_bpw,
                        "measured_loss": selected_measurement.objective_loss,
                    }
                ),
            ]
        }
    )
    write_data(refinement_path, refinement)
    measurements = RefinementMeasurementSet(
        refinement_sha256=stable_sha256(refinement),
        evaluator_version=f"1.0.0:{COMPLETE_OBJECTIVE_VERSION}",
        measurements=[parent_measurement, selected_measurement],
    )
    measurement_path = tmp_path / "refinement-measurements.json"
    write_data(measurement_path, measurements)
    write_data(artifact / "refinement_measurements.json", measurements)
    measurement_sha = stable_sha256(measurements)
    protocol = HardwareMeasurementProtocol(
        protocol_id="test",
        backend_version="6.11.1",
        dataset_sha256="f" * 64,
        random_seed=7,
        prompt_count=1,
        warmup_trials=1,
        measured_trials=1,
        power_mode="AC power",
        deterministic_tolerance=0.0,
        direct_commands=[["ax-engine-bench", "generate"]],
        mtp_commands=[["ax-engine-bench", "generate"]],
    )

    def registry_entry(
        *,
        candidate_id: str,
        candidate_plan: QuantizationPlan,
        measurement: CompleteCandidateMeasurement,
    ) -> HardwareRegistryEntry:
        paths = candidate_evidence_paths[candidate_id]
        allocation = next(
            assignment for assignment in candidate_plan.assignments if assignment.bits < 16
        )
        return HardwareRegistryEntry(
            entry_id=f"{candidate_id}-m3-max",
            candidate_id=candidate_id,
            measurement_id=measurement.measurement_id,
            candidate_model=candidate,
            profile=ProfileName.AGENT_CODING,
            plan_file=str(paths["plan"]),
            plan_file_sha256=file_sha256(paths["plan"]),
            plan_sha256=stable_sha256(candidate_plan),
            artifact_manifest_file=str(paths["artifact"]),
            artifact_manifest_sha256=file_sha256(paths["artifact"]),
            sensitivity_file=str(sensitivity_evidence_path),
            sensitivity_sha256=file_sha256(sensitivity_evidence_path),
            quality_comparison_file=str(paths["quality"]),
            quality_comparison_sha256=file_sha256(paths["quality"]),
            validation_file=str(paths["validation"]),
            validation_sha256=file_sha256(paths["validation"]),
            direct_evaluation_file=str(common_evidence_paths["direct-evaluation"]),
            direct_evaluation_sha256=file_sha256(common_evidence_paths["direct-evaluation"]),
            mtp_evaluation_file=str(common_evidence_paths["mtp-evaluation"]),
            mtp_evaluation_sha256=file_sha256(common_evidence_paths["mtp-evaluation"]),
            direct_benchmark_result_file=str(common_evidence_paths["direct-result"]),
            direct_benchmark_result_sha256=file_sha256(common_evidence_paths["direct-result"]),
            mtp_benchmark_result_file=str(common_evidence_paths["mtp-result"]),
            mtp_benchmark_result_sha256=file_sha256(common_evidence_paths["mtp-result"]),
            quantizer_execution_file=str(common_evidence_paths["execution"]),
            quantizer_execution_sha256=file_sha256(common_evidence_paths["execution"]),
            hardware=complete_hardware,
            protocol=protocol,
            coverage=[
                HardwareKernelCoverage(
                    bits=allocation.bits,
                    group_size=allocation.group_size,
                    method=allocation.method,
                    roles=[allocation.role],
                    shapes=[(16, 64)],
                    module_count=1,
                    parameter_count=allocation.parameters,
                    quantizer_execution_records=1,
                    kernel_evidence="measured",
                )
            ],
            total_modules=len(candidate_plan.assignments),
            unique_shapes=2,
            kernel_evidence="measured",
            validation_passed=True,
            release_ready=True,
        )

    hardware_registry = HardwareProfileRegistry(
        registry_id="release-test",
        measurement_set_sha256=measurement_sha,
        measurement_set_file=str(measurement_path),
        measurement_set_file_sha256=file_sha256(measurement_path),
        entries=[
            registry_entry(
                candidate_id="parent",
                candidate_plan=parent_plan,
                measurement=parent_measurement,
            ),
            registry_entry(
                candidate_id="selected",
                candidate_plan=plan,
                measurement=selected_measurement,
            ),
        ],
        distinct_named_hosts=1,
        release_ready=True,
        issues=[],
    )
    hardware_path = tmp_path / "hardware-registry.json"
    write_data(hardware_path, hardware_registry)
    write_data(artifact / "hardware_profile_registry.json", hardware_registry)
    pareto = build_pareto_report(measurements)
    pareto_path = tmp_path / "pareto.json"
    write_data(pareto_path, pareto)
    write_data(artifact / "pareto_report.json", pareto)

    compatibility = CompatibilityMatrix(
        catalog_verified_at=datetime(2026, 7, 30, tzinfo=UTC),
        required_dense_models=[
            OfficialDenseCheckpointRequirement(
                model_id=plan.source_model.model_id,
                parameter_size="27B",
            )
        ],
        required_profiles=[ProfileName.AGENT_CODING, ProfileName.GENERAL],
        required_dense_checkpoints=1,
        entries=[
            CheckpointCompatibility(
                candidate_model=candidate,
                source_model=plan.source_model,
                profile=plan.profile,
                artifact_path=str(artifact.resolve()),
                artifact_manifest_sha256=file_sha256(artifact / "axquant_manifest.json"),
                plan_sha256=selected_sha,
                adapter_id="qwen36-v1",
                dense=True,
                text_layer_count=1,
                measured_total_bpw=4.7,
                mtp_present=True,
                supported_bits=[4, 16],
                ax_engine_check_sha256="ax",
                ax_engine_passed=True,
                mlx_lm_check_sha256="mlx",
                mlx_lm_passed=True,
                validation_sha256="validation",
                validation_passed=True,
                compatible=True,
            ),
            CheckpointCompatibility(
                candidate_model=candidate,
                source_model=plan.source_model,
                profile=ProfileName.GENERAL,
                artifact_path=str(artifact.resolve()),
                artifact_manifest_sha256=file_sha256(artifact / "axquant_manifest.json"),
                plan_sha256=selected_sha,
                adapter_id="qwen36-v1",
                dense=True,
                text_layer_count=1,
                measured_total_bpw=4.7,
                mtp_present=True,
                supported_bits=[4, 16],
                ax_engine_check_sha256="ax",
                ax_engine_passed=True,
                mlx_lm_check_sha256="mlx",
                mlx_lm_passed=True,
                validation_sha256="validation-general",
                validation_passed=True,
                compatible=True,
            ),
        ],
        distinct_dense_source_checkpoints=1,
        release_ready=True,
        issues=[],
    )
    compatibility_path = tmp_path / "compatibility.json"
    write_data(compatibility_path, compatibility)

    feasibility = FeasibilityReport(
        status="ready-for-conversion",
        source=_baseline(BaselineKind.BF16_SOURCE, plan.source_model),
        baselines=[
            _baseline(
                BaselineKind.UNIFORM_4BIT,
                ModelIdentity(model_id="fixture/uniform4", revision=_BASELINE_REVISION),
            ),
            _baseline(BaselineKind.UNIFORM_6BIT, reference),
            _baseline(
                BaselineKind.MIXED_PRECISION,
                ModelIdentity(model_id="fixture/mixed", revision=_BASELINE_REVISION),
            ),
        ],
        runtime_checks_requested=True,
        checks={
            "required_baselines_complete": True,
            "logical_parameter_counts_match": True,
            "architecture_profiles_match": True,
            "mtp_tensors_present": True,
            "revisions_pinned": True,
            "source_bf16_available": True,
            "source_bf16_complete": True,
            "ax_engine_runtime_ready": True,
            "mlx_lm_static_compatible": True,
        },
    )
    feasibility_path = tmp_path / "feasibility.json"
    write_data(feasibility_path, feasibility)

    for runtime_name, output_name in (
        (RuntimeName.AX_ENGINE, "ax-engine-check.json"),
        (RuntimeName.MLX_LM, "mlx-lm-check.json"),
    ):
        option = "--mlx-model-artifacts-dir" if runtime_name == RuntimeName.AX_ENGINE else "--model"
        check = RuntimeCheck(
            model=candidate,
            runtime=runtime_name,
            check_kind="doctor" if runtime_name == RuntimeName.AX_ENGINE else "generation-smoke",
            available=True,
            passed=True,
            command=[runtime_name.value, option, str(artifact.resolve())],
            exit_code=0,
        )
        write_data(tmp_path / output_name, check)

    agent_validation_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.AGENT_CODING
    )
    agent_validation_path = Path(agent_validation_entry.validation_file)
    bound_compatibility_entry = compatibility.entries[0].model_copy(
        update={
            "artifact_manifest_sha256": file_sha256(artifact / "axquant_manifest.json"),
            "ax_engine_check_sha256": file_sha256(tmp_path / "ax-engine-check.json"),
            "mlx_lm_check_sha256": file_sha256(tmp_path / "mlx-lm-check.json"),
            "validation_sha256": agent_validation_entry.validation_sha256,
            "adapter_id": plan.architecture_profile.adapter_id,
            "dense": plan.architecture_profile.dense is True,
            "text_layer_count": plan.architecture_profile.text_layer_count,
            "measured_total_bpw": manifest.measured_total_bpw,
            "mtp_present": manifest.mtp_present,
            "supported_bits": sorted({assignment.bits for assignment in plan.assignments}),
        }
    )

    general_validation_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.GENERAL
    )
    general_validation_path = Path(general_validation_entry.validation_file)
    bound_general_compatibility_entry = compatibility.entries[1].model_copy(
        update={
            "artifact_manifest_sha256": file_sha256(artifact / "axquant_manifest.json"),
            "ax_engine_check_sha256": file_sha256(tmp_path / "ax-engine-check.json"),
            "mlx_lm_check_sha256": file_sha256(tmp_path / "mlx-lm-check.json"),
            "validation_sha256": general_validation_entry.validation_sha256,
            "adapter_id": plan.architecture_profile.adapter_id,
            "dense": plan.architecture_profile.dense is True,
            "text_layer_count": plan.architecture_profile.text_layer_count,
            "measured_total_bpw": manifest.measured_total_bpw,
            "mtp_present": manifest.mtp_present,
            "supported_bits": sorted({assignment.bits for assignment in plan.assignments}),
        }
    )
    compatibility = compatibility.model_copy(
        update={"entries": [bound_compatibility_entry, bound_general_compatibility_entry]}
    )
    write_data(compatibility_path, compatibility)
    compatibility_general_validation_path = tmp_path / "compatibility-general-validation.json"
    write_data(
        compatibility_general_validation_path,
        load_model(general_validation_path, ValidationReport),
    )
    assert file_sha256(compatibility_general_validation_path) == (
        general_validation_entry.validation_sha256
    )
    compatibility_request_path = tmp_path / "compatibility-request.json"
    write_data(
        compatibility_request_path,
        CompatibilityMatrixRequest(
            catalog_verified_at=datetime(2026, 7, 30, tzinfo=UTC),
            required_dense_models=[
                OfficialDenseCheckpointRequirement(
                    model_id=plan.source_model.model_id,
                    parameter_size="27B",
                )
            ],
            candidates=[
                CompatibilityCandidateInput(
                    artifact_directory=str(artifact),
                    ax_engine_check=str(tmp_path / "ax-engine-check.json"),
                    mlx_lm_check=str(tmp_path / "mlx-lm-check.json"),
                    validation_report=str(agent_validation_path),
                ),
                CompatibilityCandidateInput(
                    artifact_directory=str(artifact),
                    ax_engine_check=str(tmp_path / "ax-engine-check.json"),
                    mlx_lm_check=str(tmp_path / "mlx-lm-check.json"),
                    validation_report=str(compatibility_general_validation_path),
                ),
            ],
        ),
    )

    weight_records = [
        ArtifactFile(
            path=name,
            size_bytes=(artifact / name).stat().st_size,
            sha256=file_sha256(artifact / name),
        )
        for name in ("model.safetensors", "mtp.safetensors")
    ]
    recipe = ReproductionRecipe(
        source_model=plan.source_model,
        calibration=plan.calibration,
        axquant_version="1.0.0",
        software_versions=plan.software_versions,
        random_seed=plan.random_seed,
        profile=plan.profile,
        primary_runtime=RuntimeName.AX_ENGINE,
        plan_sha256=selected_sha,
        output_repository=candidate.model_id,
        plan_file_sha256=file_sha256(artifact / "quantization_plan.json"),
        calibration_file_sha256=file_sha256(artifact / "calibration_manifest.json"),
        conversion_manifest_sha256=file_sha256(artifact / "axquant_conversion_manifest.json"),
        expected_logical_parameters=manifest.logical_parameters,
        expected_weight_file_size_bytes=manifest.weight_file_size_bytes,
        expected_weight_files=weight_records,
        commands=[
            ReproductionCommand(
                step_id="download-source",
                description="download",
                argv=["hf", "download"],
            ),
            ReproductionCommand(
                step_id="convert",
                description="convert",
                argv=["axquant", "convert"],
            ),
            ReproductionCommand(
                step_id="verify-reproduction",
                description="verify",
                argv=["axquant", "verify-reproduction"],
            ),
        ],
    )
    recipe_path = artifact / "reproduction_recipe.yaml"
    write_data(recipe_path, recipe)
    verification = verify_reproduction(recipe_path=recipe_path, artifact_dir=artifact)
    verification_path = tmp_path / "reproduction-verification.json"
    write_data(verification_path, verification)

    wheel_path = tmp_path / f"axquant-{wheel_version}-py3-none-any.whl"
    _write_wheel(wheel_path, wheel_version)
    request_path = tmp_path / "release-audit-request.json"
    write_data(
        request_path,
        ReleaseAuditRequest(
            artifact_directory=str(artifact),
            feasibility_report=str(feasibility_path),
            sensitivity_report=str(sensitivity_path),
            refinement_result=str(refinement_path),
            release_validation_index=str(validation_index_path),
            hardware_registry=str(hardware_path),
            pareto_report=str(pareto_path),
            compatibility_matrix=str(compatibility_path),
            compatibility_request=str(compatibility_request_path),
            reproduction_recipe=str(recipe_path),
            reproduction_verification=str(verification_path),
            ax_engine_check=str(tmp_path / "ax-engine-check.json"),
            mlx_lm_check=str(tmp_path / "mlx-lm-check.json"),
            toolkit_wheel=str(wheel_path),
        ),
    )
    return request_path


def test_release_audit_proves_every_milestone(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    audit = build_release_audit(request_path)

    assert audit.release_ready
    assert audit.schema_version == "axquant.release-audit.v4"
    assert audit.request_sha256 == file_sha256(request_path)
    assert [check.gate_id for check in audit.checks] == [
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
    ]
    assert not audit.blockers
    assert audit.toolkit_version == "1.0.0"


def test_release_audit_v4_semantics_are_golden(tmp_path: Path) -> None:
    audit = build_release_audit(_inputs(tmp_path))
    semantic_projection = {
        "schema_version": audit.schema_version,
        "release_ready": audit.release_ready,
        "checks": [
            {
                "gate_id": check.gate_id,
                "passed": check.passed,
                "issues": check.issues,
            }
            for check in audit.checks
        ],
        "blockers": audit.blockers,
    }

    assert stable_sha256(semantic_projection) == (
        "7f0665b36e8f97d6cb1fe7cc062066c430b6aa79aae22f32cf8a4d901b688688"
    )


def test_release_audit_v4_still_rejects_a_no_mtp_artifact_at_m1(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    manifest_path = Path(request.artifact_directory) / "axquant_manifest.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    write_data(manifest_path, manifest.model_copy(update={"mtp_present": False}))

    audit = build_release_audit(request_path)
    m1 = next(check for check in audit.checks if check.gate_id == "M1")

    assert not m1.passed
    assert "release artifact does not contain declared MTP weights" in m1.issues


def test_release_audit_rejects_an_unapproved_embedded_exception(
    tmp_path: Path,
) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    validation_index_path = Path(request.release_validation_index)
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    agent_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.AGENT_CODING
    )
    validation_path = Path(agent_entry.validation_file)
    validation = load_model(validation_path, ValidationReport)
    artifact = Path(request.artifact_directory)
    plan = load_model(artifact / "axquant_plan.json", QuantizationPlan)
    approved_at = datetime.now(UTC) - timedelta(days=1)
    validation.release_exceptions = [
        ReleaseException(
            exception_id="AXQ-UNAPPROVED",
            candidate_model=validation.candidate_model,
            plan_sha256=stable_sha256(plan),
            targets=[
                ReleaseExceptionTarget(
                    metric="artifact.weight_size_ratio",
                    observed_value=1.2,
                    required_maximum=1.1,
                    requirement="candidate size must remain within the release limit",
                ),
                ReleaseExceptionTarget(
                    metric="artifact.candidate_measured_bpw",
                    observed_value=5.8,
                    required_minimum=4.3,
                    required_maximum=4.8,
                    requirement="candidate BPW must remain within the target range",
                ),
            ],
            measured_tradeoff="Unapproved test-only tradeoff claim.",
            owner="test owner",
            approved_by="test approver",
            approval_reference="test-reference",
            approved_at=approved_at,
            expires_at=approved_at + timedelta(days=30),
            evidence_sha256={
                "plan": "1" * 64,
                "candidate_size": "2" * 64,
                "size_reference": "3" * 64,
                "tradeoff": "4" * 64,
            },
        )
    ]
    write_data(validation_path, validation)
    agent_entry.validation_sha256 = file_sha256(validation_path)
    write_data(validation_index_path, validation_index)

    audit = build_release_audit(request_path)
    m4 = next(check for check in audit.checks if check.gate_id == "M4")

    assert not m4.passed
    assert "agent-coding validation contains an unapproved release exception" in m4.issues


def test_release_audit_reverifies_an_approved_size_exception(
    tmp_path: Path,
) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    artifact = Path(request.artifact_directory)
    plan_path = artifact / "axquant_plan.json"
    plan = load_model(plan_path, QuantizationPlan)
    validation_index_path = Path(request.release_validation_index)
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    candidate_model = validation_index.entries[0].candidate_model
    candidate_size = ArtifactSizeEvidence(
        kind="candidate",
        model=candidate_model,
        logical_parameters=1088,
        weight_bytes=10,
        measured_bpw=10 * 8 / 1088,
        source_sha256=file_sha256(artifact / "axquant_conversion_manifest.json"),
    )
    size_reference = ArtifactSizeEvidence(
        kind="uniform-4bit",
        model=ModelIdentity(
            model_id="fixture/uniform4",
            revision=_BASELINE_REVISION,
        ),
        logical_parameters=1088,
        weight_bytes=8,
        measured_bpw=8 * 8 / 1088,
        source_sha256="f" * 64,
    )
    candidate_size_path = tmp_path / "candidate-size.json"
    size_reference_path = tmp_path / "size-reference.json"
    tradeoff_path = tmp_path / "tradeoff.json"
    write_data(candidate_size_path, candidate_size)
    write_data(size_reference_path, size_reference)
    write_data(
        tradeoff_path,
        {"quality_retention": 0.99, "peak_memory_ratio": 0.8},
    )
    evidence_paths = {
        "plan": plan_path,
        "candidate_size": candidate_size_path,
        "size_reference": size_reference_path,
        "tradeoff": tradeoff_path,
    }
    approved_at = datetime.now(UTC) - timedelta(days=1)
    exception = ReleaseException(
        exception_id="AXQ-AUDIT-SIZE",
        candidate_model=candidate_model,
        plan_sha256=stable_sha256(plan),
        targets=[
            ReleaseExceptionTarget(
                metric="artifact.weight_size_ratio",
                observed_value=1.25,
                required_maximum=1.1,
                requirement="candidate size must remain within the release limit",
            ),
            ReleaseExceptionTarget(
                metric="artifact.candidate_measured_bpw",
                observed_value=candidate_size.measured_bpw,
                required_minimum=4.3,
                required_maximum=4.8,
                requirement="candidate BPW must remain within the target range",
            ),
        ],
        measured_tradeoff="Quality retention is 99% and peak-memory ratio is 80%.",
        owner="test release owner",
        approved_by="test release authority",
        approval_reference="test-approved-decision",
        approved_at=approved_at,
        expires_at=approved_at + timedelta(days=30),
        evidence_sha256={name: file_sha256(path) for name, path in evidence_paths.items()},
    )
    exception_path = tmp_path / "release-exception.json"
    write_data(exception_path, exception)
    write_data(artifact / "release_exception.json", exception)

    for entry in validation_index.entries:
        validation_path = Path(entry.validation_file)
        validation = load_model(validation_path, ValidationReport)
        validation = ValidationReport.model_validate(
            {
                **validation.model_dump(mode="json"),
                "passed": False,
                "issues": [
                    ValidationIssue(
                        severity="error",
                        metric="artifact.weight_size_ratio",
                        message="ratio 1.2500 exceeds 1.1000",
                    )
                ],
            }
        )
        validation.comparisons.update(
            {
                "artifact.weight_size_ratio": 1.25,
                "artifact.candidate_measured_bpw": candidate_size.measured_bpw,
                "artifact.candidate_source_sha256": candidate_size.source_sha256,
                "artifact.uniform4_source_sha256": size_reference.source_sha256,
                "artifact.candidate_weight_bytes": candidate_size.weight_bytes,
                "artifact.uniform4_weight_bytes": size_reference.weight_bytes,
                "artifact.logical_parameters": candidate_size.logical_parameters,
            }
        )
        validation = apply_release_exception(
            validation,
            exception,
            plan=plan,
            evidence_files=evidence_paths,
        )
        write_data(validation_path, validation)
        entry.validation_sha256 = file_sha256(validation_path)
    write_data(validation_index_path, validation_index)
    write_data(artifact / "release_validation_index.json", validation_index)

    compatibility_path = Path(request.compatibility_matrix)
    compatibility = load_model(compatibility_path, CompatibilityMatrix)
    agent_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.AGENT_CODING
    )
    compatibility.entries[0].validation_sha256 = agent_entry.validation_sha256
    write_data(compatibility_path, compatibility)
    request = ReleaseAuditRequest.model_validate(
        {
            **request.model_dump(mode="json"),
            "release_exceptions": [str(exception_path)],
            "release_exception_evidence": {
                name: str(path) for name, path in evidence_paths.items()
            },
        }
    )
    write_data(request_path, request)

    audit = build_release_audit(request_path)
    m4 = next(check for check in audit.checks if check.gate_id == "M4")

    assert m4.passed, m4.issues
    assert m4.evidence_sha256["release_exception"] == file_sha256(exception_path)
    assert m4.evidence_sha256["release_exception_tradeoff"] == file_sha256(tradeoff_path)


def test_release_audit_rejects_packaged_evidence_drift(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    artifact = Path(request.artifact_directory)
    packaged_pareto_path = artifact / "pareto_report.json"
    packaged_pareto = load_model(packaged_pareto_path, ParetoReport)
    changed_point = packaged_pareto.points[0].model_copy(
        update={"measured_bpw": packaged_pareto.points[0].measured_bpw + 0.1}
    )
    write_data(
        packaged_pareto_path,
        packaged_pareto.model_copy(update={"points": [changed_point, *packaged_pareto.points[1:]]}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M8: packaged Pareto report differs from the audited report" in audit.blockers


def test_release_audit_recomputes_feasibility_instead_of_trusting_status(
    tmp_path: Path,
) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    feasibility_path = Path(request.feasibility_report)
    feasibility = load_model(feasibility_path, FeasibilityReport)
    assert feasibility.source is not None
    write_data(
        feasibility_path,
        feasibility.model_copy(
            update={"source": feasibility.source.model_copy(update={"complete": False})}
        ),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M0: feasibility BF16 source is incomplete or has the wrong kind" in audit.blockers


def test_release_audit_rejects_duplicate_artifact_file_records(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    manifest_path = Path(request.artifact_directory) / "axquant_manifest.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    write_data(
        manifest_path,
        manifest.model_copy(update={"files": [*manifest.files, manifest.files[0]]}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M1: artifact manifest contains duplicate file records" in audit.blockers


def test_release_audit_rejects_unrecorded_artifact_weight_file(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    (Path(request.artifact_directory) / "unrecorded.safetensors").write_bytes(b"extra")

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any(
        blocker.startswith("M1: artifact manifest Safetensors coverage differs")
        for blocker in audit.blockers
    )


def test_release_audit_recomputes_benchmark_trials(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    validation_index_path = Path(request.release_validation_index)
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    validation_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.AGENT_CODING
    )
    benchmark_path = Path(validation_entry.benchmark_index_file)
    benchmark = load_model(benchmark_path, BenchmarkEvidenceIndex)
    benchmark_entry = next(
        entry for entry in benchmark.entries if entry.kind == BenchmarkEvidenceKind.BF16
    )
    assert benchmark_entry.evaluation_file is not None
    evaluation_path = Path(benchmark_entry.evaluation_file)
    evaluation = load_model(evaluation_path, EvaluationBundle)
    write_data(
        evaluation_path,
        evaluation.model_copy(
            update={
                "benchmark_metadata": {
                    **evaluation.benchmark_metadata,
                    "failed_trials": 1,
                }
            }
        ),
    )
    changed_benchmark_entry = benchmark_entry.model_copy(
        update={"evaluation_sha256": file_sha256(evaluation_path)}
    )
    write_data(
        benchmark_path,
        benchmark.model_copy(
            update={
                "entries": [
                    changed_benchmark_entry if entry.kind == BenchmarkEvidenceKind.BF16 else entry
                    for entry in benchmark.entries
                ]
            }
        ),
    )
    changed_validation_entry = validation_entry.model_copy(
        update={"benchmark_index_sha256": file_sha256(benchmark_path)}
    )
    write_data(
        validation_index_path,
        validation_index.model_copy(
            update={
                "entries": [
                    changed_validation_entry if entry.profile == ProfileName.AGENT_CODING else entry
                    for entry in validation_index.entries
                ]
            }
        ),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M2: agent-coding: bf16 benchmark trials are incomplete" in audit.blockers


def test_release_audit_rejects_cross_profile_candidate_mismatch(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    validation_index_path = Path(request.release_validation_index)
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    general_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.GENERAL
    )
    changed_entry = general_entry.model_copy(
        update={
            "candidate_model": general_entry.candidate_model.model_copy(
                update={"revision": _OTHER_REVISION}
            )
        }
    )
    write_data(
        validation_index_path,
        validation_index.model_copy(
            update={
                "entries": [
                    changed_entry if entry.profile == ProfileName.GENERAL else entry
                    for entry in validation_index.entries
                ]
            }
        ),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M2: general release validation entry is inconsistent" in audit.blockers


def test_release_audit_binds_calibration_manifest(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    calibration_path = Path(request.artifact_directory) / "calibration_manifest.json"
    calibration = load_model(calibration_path, CalibrationManifest)
    write_data(
        calibration_path,
        calibration.model_copy(update={"calibration_evaluation_separation_attested": False}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M3: calibration manifest is missing or checksum-mismatched" in audit.blockers


def test_artifact_audit_rejects_symlinks_and_underreported_weight_bytes(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    artifact = Path(request.artifact_directory)
    manifest = load_model(artifact / "axquant_manifest.json", ArtifactManifest)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (artifact / "leak.bin").symlink_to(outside)

    issues = _artifact_issues(artifact, manifest)

    assert any("artifact tree contains symlinks" in issue for issue in issues)

    (artifact / "leak.bin").unlink()
    extra = artifact / "extra.safetensors"
    extra.write_bytes(b"extra")
    manifest.files.append(
        ArtifactFile(
            path=extra.name,
            size_bytes=extra.stat().st_size,
            sha256=file_sha256(extra),
        )
    )

    issues = _artifact_issues(artifact, manifest)

    assert "artifact manifest Safetensors bytes do not match measured weight bytes" in issues


def test_release_audit_requires_measured_parent_for_refinement_gain(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    registry_path = Path(request.hardware_registry)
    registry = load_model(registry_path, HardwareProfileRegistry)
    measurement_path = Path(registry.measurement_set_file)
    measurements = load_model(measurement_path, RefinementMeasurementSet)
    selected_only = measurements.model_copy(
        update={
            "measurements": [
                measurement
                for measurement in measurements.measurements
                if measurement.candidate_id == "selected"
            ]
        }
    )
    write_data(measurement_path, selected_only)
    write_data(
        registry_path,
        registry.model_copy(
            update={
                "measurement_set_sha256": stable_sha256(selected_only),
                "measurement_set_file_sha256": file_sha256(measurement_path),
            }
        ),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert (
        "M6: no measurement-bound interaction refinement improves its parent candidate"
        in audit.blockers
    )


def test_release_audit_rebuilds_pareto_frontier(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    pareto_path = Path(request.pareto_report)
    pareto = load_model(pareto_path, ParetoReport)
    changed_point = pareto.points[0].model_copy(
        update={"measured_bpw": pareto.points[0].measured_bpw + 0.1}
    )
    write_data(
        pareto_path,
        pareto.model_copy(update={"points": [changed_point, *pareto.points[1:]]}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M7: Pareto report cannot be rebuilt from the bound measurements" in audit.blockers


def test_release_audit_rejects_unbound_compatibility_entry(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    compatibility_path = Path(request.compatibility_matrix)
    compatibility = load_model(compatibility_path, CompatibilityMatrix)
    changed_entry = compatibility.entries[0].model_copy(
        update={"ax_engine_check_sha256": "different-runtime-check"}
    )
    write_data(
        compatibility_path,
        compatibility.model_copy(update={"entries": [changed_entry, *compatibility.entries[1:]]}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert (
        "M5: compatibility matrix does not certify the release candidate in every "
        "required profile" in audit.blockers
    )


def test_release_audit_rejects_widened_validation_thresholds(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    validation_index_path = Path(request.release_validation_index)
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    entry = next(e for e in validation_index.entries if e.profile == ProfileName.GENERAL)
    validation_path = Path(entry.validation_file)
    validation = load_model(validation_path, ValidationReport)

    # Widen a threshold well past the authoritative profile policy while leaving
    # `comparisons`/`passed` self-consistent, simulating a tampered or stale
    # validation report that would otherwise sail through as a real pass.
    widened_thresholds = validation.thresholds.model_copy(update={"min_effective_speedup": 0.0})
    tampered = validation.model_copy(update={"thresholds": widened_thresholds})
    write_data(validation_path, tampered)
    new_entries = [
        e.model_copy(update={"validation_sha256": file_sha256(validation_path)})
        if e.profile == ProfileName.GENERAL
        else e
        for e in validation_index.entries
    ]
    write_data(validation_index_path, validation_index.model_copy(update={"entries": new_entries}))

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert (
        "M2: general validation does not use the authoritative profile thresholds" in audit.blockers
    )


def test_release_audit_rejects_stale_profile_evidence(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    compatibility_request = load_model(
        Path(request.compatibility_request),
        CompatibilityMatrixRequest,
    )
    second_validation_path = Path(compatibility_request.candidates[1].validation_report)
    second_validation = load_model(second_validation_path, ValidationReport)
    write_data(
        second_validation_path,
        second_validation.model_copy(
            update={
                "comparisons": {
                    **second_validation.comparisons,
                    "hardware.effective_speedup": 1.30,
                }
            }
        ),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any(
        blocker.startswith("M5: compatibility validation checksum changed")
        for blocker in audit.blockers
    )


def test_release_audit_rejects_compatibility_scope_tampering(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    compatibility_path = Path(request.compatibility_matrix)
    compatibility = load_model(compatibility_path, CompatibilityMatrix)
    changed_scope = [
        OfficialDenseCheckpointRequirement(
            model_id="Qwen/Qwen3.6-32B",
            parameter_size="32B",
        )
    ]
    write_data(
        compatibility_path,
        compatibility.model_copy(update={"required_dense_models": changed_scope}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M5: compatibility matrix official scope binding changed" in audit.blockers
    assert (
        "M5: compatibility matrix release scope differs from its original request" in audit.blockers
    )


def test_release_audit_recomputes_dense_checkpoint_count(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    compatibility_path = Path(request.compatibility_matrix)
    compatibility = load_model(compatibility_path, CompatibilityMatrix)
    write_data(
        compatibility_path,
        compatibility.model_copy(update={"distinct_dense_source_checkpoints": 3}),
    )

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert "M5: compatibility matrix dense-checkpoint count is inconsistent" in audit.blockers


def test_release_audit_is_deterministic_for_publication_rerun(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    audit = build_release_audit(request_path)

    _rerun_release_audit(audit=audit, request_path=request_path)


def test_release_audit_accepts_policy_floored_tensor_measurements() -> None:
    sensitivity = _sensitivity()
    plan = _plan(sensitivity)
    first = sensitivity.entries[0]
    embedding = first.tensor.model_copy(update={"role": TensorRole.EMBEDDING})
    floored = first.model_copy(
        update={
            "tensor": embedding,
            "candidates": [
                candidate for candidate in first.candidates if candidate.bits in {8, 16}
            ],
        }
    )
    report = sensitivity.model_copy(update={"entries": [floored, *sensitivity.entries[1:]]})

    assert _sensitivity_measurement_issues(report, plan) == []


def test_release_audit_requires_measurement_even_when_candidate_is_dominated() -> None:
    sensitivity = _sensitivity()
    plan = _plan(sensitivity)
    first = sensitivity.entries[0]
    dominated = first.model_copy(
        update={
            "candidates": [
                candidate.model_copy(update={"supported": False})
                if candidate.bits == 4
                else candidate
                for candidate in first.candidates
            ]
        }
    )
    report = sensitivity.model_copy(update={"entries": [dominated, *sensitivity.entries[1:]]})

    assert _sensitivity_measurement_issues(report, plan) == []

    failed = dominated.model_copy(
        update={
            "candidates": [
                candidate.model_copy(update={"measured_tokens": 0})
                if candidate.bits == 4
                else candidate
                for candidate in dominated.candidates
            ]
        }
    )
    failed_report = sensitivity.model_copy(update={"entries": [failed, *sensitivity.entries[1:]]})
    assert _sensitivity_measurement_issues(failed_report, plan) == [
        "model.layers.0.mlp.down_proj.weight lacks complete measured candidates at bits [4]"
    ]


def test_release_audit_binds_packaged_awq_capture_manifest(tmp_path: Path) -> None:
    sensitivity = _sensitivity()
    plan = _plan(sensitivity)
    target = next(assignment for assignment in plan.assignments if assignment.bits < 16)
    plan.hardware = plan.hardware.model_copy(
        update={
            "supported_methods": (
                *plan.hardware.supported_methods,
                QuantMethod.AWQ,
            )
        }
    )
    target.method = QuantMethod.AWQ
    assert sensitivity.calibration is not None
    assert plan.calibration is not None
    plan.calibration = plan.calibration.model_copy(deep=True)
    capture_manifest = ActivationCaptureManifest(
        model=plan.source_model.model_id,
        revision=plan.source_model.revision,
        tokenized_cache_manifest_sha256="b" * 64,
        cache_key_sha256="c" * 64,
        calibration_dataset_id=plan.calibration.dataset_id,
        max_rows=4,
    )
    capture = load_test_activation_capture(
        tmp_path,
        manifest=capture_manifest,
        activations={},
    )
    capture_manifest = capture.manifest
    binding = activation_capture_metadata(capture)
    sensitivity.calibration.metadata.update(binding)
    plan.calibration.metadata.update(binding)

    assert _activation_capture_artifact_issues(tmp_path, sensitivity, plan) == []

    plan.calibration.metadata["activation_capture_manifest_sha256"] = "0" * 64
    issues = _activation_capture_artifact_issues(tmp_path, sensitivity, plan)
    assert any("bindings differ" in issue for issue in issues)
    assert any("does not match the activation capture" in issue for issue in issues)


def test_release_audit_cli_fails_non_v1_wheel(tmp_path: Path) -> None:
    request = _inputs(tmp_path, wheel_version="0.9.0")
    output = tmp_path / "audit.json"

    assert (
        main(
            [
                "release-audit",
                "--request",
                str(request),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    audit = ReleaseAudit.model_validate_json(output.read_text(encoding="utf-8"))
    assert not audit.release_ready
    assert any("toolkit version '0.9.0' is not '1.0.0'" in issue for issue in audit.blockers)


def test_release_audit_rejects_tampered_refinement_measurements(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    registry = load_model(Path(request.hardware_registry), HardwareProfileRegistry)
    Path(registry.measurement_set_file).write_text("{}\n", encoding="utf-8")

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any(
        "hardware measurement set checksum does not match the registry" in issue
        for issue in audit.blockers
    )


def test_release_audit_rejects_renamed_wheel(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    wheel = Path(request.toolkit_wheel)
    renamed = wheel.with_name("renamed.whl")
    wheel.rename(renamed)
    write_data(request_path, request.model_copy(update={"toolkit_wheel": str(renamed)}))

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any("toolkit wheel filename 'renamed.whl'" in issue for issue in audit.blockers)


def test_release_audit_rejects_wheel_member_with_stale_record_hash(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    wheel_path = Path(request.toolkit_wheel)
    with zipfile.ZipFile(wheel_path) as wheel:
        members = {name: wheel.read(name) for name in wheel.namelist()}
    members["axquant/schema/__init__.py"] = b"# tampered after RECORD generation\n"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for name, data in members.items():
            wheel.writestr(name, data)

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any(
        "toolkit wheel RECORD hash mismatch for 'axquant/schema/__init__.py'" in issue
        for issue in audit.blockers
    )


def test_release_audit_rejects_unrecorded_wheel_member(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    with zipfile.ZipFile(Path(request.toolkit_wheel), "a") as wheel:
        wheel.writestr("axquant/unrecorded.py", b"# not covered by RECORD\n")

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any("toolkit wheel has unrecorded members" in issue for issue in audit.blockers)


def test_release_audit_rejects_nonproduction_wheel_classifier(tmp_path: Path) -> None:
    wheel_path = tmp_path / "axquant-1.0.0-py3-none-any.whl"
    _write_wheel(wheel_path, "1.0.0", production_classifier=False)

    version, issues = _wheel_identity(wheel_path)

    assert version == "1.0.0"
    assert "toolkit wheel is not classified as production/stable" in issues


def test_release_audit_rejects_wheel_missing_runtime_dependency(tmp_path: Path) -> None:
    wheel_path = tmp_path / "axquant-1.0.0-py3-none-any.whl"
    _write_wheel(wheel_path, "1.0.0")
    with zipfile.ZipFile(wheel_path) as wheel:
        members = {name: wheel.read(name) for name in wheel.namelist()}
    metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
    members[metadata_name] = members[metadata_name].replace(
        b"Requires-Dist: safetensors>=0.4.5\n",
        b"",
    )
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for name, data in members.items():
            wheel.writestr(name, data)

    version, issues = _wheel_identity(wheel_path)

    assert version == "1.0.0"
    assert "toolkit wheel is missing runtime dependencies: ['safetensors']" in issues


def test_release_audit_rejects_artifact_from_another_toolkit_version(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    manifest_path = Path(request.artifact_directory) / "axquant_manifest.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    write_data(manifest_path, manifest.model_copy(update={"axquant_version": "0.9.0"}))

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any(
        "artifact manifest AXQuant version '0.9.0' differs from toolkit '1.0.0'" in issue
        for issue in audit.blockers
    )


def test_release_audit_rejects_artifact_with_different_plan_provenance(tmp_path: Path) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    manifest_path = Path(request.artifact_directory) / "axquant_manifest.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    write_data(manifest_path, manifest.model_copy(update={"profile": ProfileName.GENERAL}))

    audit = build_release_audit(request_path)

    assert not audit.release_ready
    assert any(
        "artifact and plan profile/calibration provenance differ" in issue
        for issue in audit.blockers
    )


def test_sensitivity_lineage_verifies_base_candidates_and_provenance() -> None:
    base = _sensitivity()
    assert base.calibration is not None
    base_entry = base.entries[0]
    affine = CandidateMeasurement(
        bits=4,
        method=QuantMethod.AFFINE,
        group_size=64,
        metrics=MetricVector(output_kl=0.05),
        measured_tokens=8192,
    )
    refined_entry = base_entry.model_copy(update={"candidates": [*base_entry.candidates, affine]})
    calibration = base.calibration.model_copy(
        update={
            "backend": "mlx-probe-v2",
            "reference": "measured-forward-probe-refinement",
            "metadata": {
                **base.calibration.metadata,
                "base_sensitivity_sha256": stable_sha256(base),
                "base_inventory_sha256": base.inventory_sha256,
                "base_probe_backend": base.calibration.backend,
                "refinement_probe_backend": "mlx-probe-v2",
                "candidate_methods": "affine",
                "target_tensor_count": 1,
            },
        }
    )
    refined = base.model_copy(
        update={
            "calibration": calibration,
            "inventory_sha256": "refined-inventory",
            "entries": [refined_entry, *base.entries[1:]],
        }
    )

    assert _sensitivity_lineage_issues(refined, [base]) == []
    assert any("parent is missing" in issue for issue in _sensitivity_lineage_issues(refined, []))

    changed_base_candidate = base_entry.candidates[0].model_copy(
        update={"metrics": MetricVector(output_kl=0.9)}
    )
    tampered_entry = refined_entry.model_copy(
        update={
            "candidates": [
                changed_base_candidate,
                *refined_entry.candidates[1:],
            ]
        }
    )
    tampered = refined.model_copy(update={"entries": [tampered_entry, *refined.entries[1:]]})
    assert any(
        "changed base candidates" in issue
        for issue in _sensitivity_lineage_issues(tampered, [base])
    )


def test_release_audit_recomputes_complete_objective_from_quality_file(
    tmp_path: Path,
) -> None:
    request_path = _inputs(tmp_path)
    request = load_model(request_path, ReleaseAuditRequest)
    registry_path = Path(request.hardware_registry)
    registry = load_model(registry_path, HardwareProfileRegistry)
    selected_entry = next(entry for entry in registry.entries if entry.candidate_id == "selected")
    quality_path = Path(selected_entry.quality_comparison_file)
    quality = load_model(quality_path, QualityComparisonReport)
    write_data(quality_path, quality.model_copy(update={"perplexity_ratio": 0.5}))
    quality_sha256 = file_sha256(quality_path)

    measurement_path = Path(registry.measurement_set_file)
    measurements = load_model(measurement_path, RefinementMeasurementSet)
    updated_measurements = measurements.model_copy(
        update={
            "measurements": [
                measurement.model_copy(update={"quality_comparison_sha256": quality_sha256})
                if measurement.measurement_id == selected_entry.measurement_id
                else measurement
                for measurement in measurements.measurements
            ]
        }
    )
    write_data(measurement_path, updated_measurements)
    updated_entries = [
        entry.model_copy(update={"quality_comparison_sha256": quality_sha256})
        if entry.measurement_id == selected_entry.measurement_id
        else entry
        for entry in registry.entries
    ]
    write_data(
        registry_path,
        registry.model_copy(
            update={
                "measurement_set_sha256": stable_sha256(updated_measurements),
                "measurement_set_file_sha256": file_sha256(measurement_path),
                "entries": updated_entries,
            }
        ),
    )

    audit = build_release_audit(request_path)
    m6 = next(check for check in audit.checks if check.gate_id == "M6")

    assert not m6.passed
    assert any("complete-model measurement evidence is invalid" in issue for issue in m6.issues)
