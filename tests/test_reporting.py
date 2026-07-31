from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.benchmark_evidence import build_benchmark_evidence_index
from axquant.cli import main
from axquant.errors import ValidationGateError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.release_validation import build_release_validation_index
from axquant.reporting import prepare_publication
from axquant.reproduction import verify_reproduction
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceInput,
    BenchmarkEvidenceKind,
    BenchmarkEvidenceRequest,
    CalibrationEvidence,
    CalibrationManifest,
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    EvaluationBundle,
    EvidenceKind,
    HardwareKernelCoverage,
    HardwareMeasurementProtocol,
    HardwareMetrics,
    HardwareProfileRegistry,
    HardwareRegistryEntry,
    IntegrityMetrics,
    ModelIdentity,
    ParetoPoint,
    ParetoReport,
    PlanRequest,
    ProfileName,
    QuantMethod,
    RefinementMeasurementSet,
    ReleaseException,
    ReleaseExceptionTarget,
    ReleaseValidationIndex,
    ReleaseValidationInput,
    ReleaseValidationRequest,
    ReproductionRecipe,
    ReproductionVerification,
    SoftwareVersions,
    TensorRole,
    ValidationIssue,
    ValidationReport,
    ValidationThresholds,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


def _plan(model_dir: Path):
    inventory = inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
    )
    return plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )


def _validation(
    candidate_id: str,
    candidate_dir: Path,
    *,
    profile: ProfileName = ProfileName.AGENT_CODING,
) -> ValidationReport:
    return ValidationReport(
        reference_model=ModelIdentity(
            model_id="Qwen/Qwen3.6-27B-MLX-6bit",
            revision="baseline-revision",
        ),
        candidate_model=ModelIdentity(
            model_id=candidate_id,
            revision="candidate-revision",
        ),
        profile=profile,
        passed=True,
        thresholds=ValidationThresholds(min_effective_speedup=1.2),
        issues=[],
        comparisons={
            "artifact.weight_size_ratio": 1.05,
            "artifact.candidate_weight_bytes": 10,
            "artifact.candidate_source_sha256": file_sha256(
                candidate_dir / "axquant_manifest.json"
            ),
            "mtp.acceptance_retention": 0.97,
            "hardware.effective_speedup": 1.25,
        },
    )


def _benchmark_index(
    tmp_path: Path,
    *,
    candidate_id: str,
    profile: ProfileName,
    dataset_sha256: str,
) -> Path:
    evidence_directory = tmp_path / f"{profile.value}-benchmark-index-inputs"
    evidence_directory.mkdir(exist_ok=True)
    versions = SoftwareVersions(
        axquant="0.1.0a0",
        python="3.13",
        mlx="0.32",
        mlx_lm="0.31",
        ax_engine="6.11.1",
        safetensors="0.6",
        pydantic="2.11",
    )
    available = {
        BenchmarkEvidenceKind.BF16,
        BenchmarkEvidenceKind.UNIFORM_4BIT,
        BenchmarkEvidenceKind.UNIFORM_6BIT,
        BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
        BenchmarkEvidenceKind.AXQUANT_MTP_ON,
    }
    entries: list[BenchmarkEvidenceInput] = []
    for kind in BenchmarkEvidenceKind:
        if kind not in available:
            entries.append(
                BenchmarkEvidenceInput(
                    kind=kind,
                    status="unavailable",
                    unavailable_reason=f"{kind.value} is unavailable in this fixture",
                )
            )
            continue
        if kind == BenchmarkEvidenceKind.UNIFORM_6BIT:
            model = ModelIdentity(
                model_id="Qwen/Qwen3.6-27B-MLX-6bit",
                revision="baseline-revision",
            )
        elif kind in {
            BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
            BenchmarkEvidenceKind.AXQUANT_MTP_ON,
        }:
            model = ModelIdentity(model_id=candidate_id, revision="candidate-revision")
        else:
            model = ModelIdentity(
                model_id=f"Qwen/Qwen3.6-27B-{kind.value}",
                revision=f"{kind.value}-revision",
            )
        path = evidence_directory / f"{kind.value}.json"
        write_data(
            path,
            EvaluationBundle(
                model=model,
                mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
                baseline_kind=kind.value,
                hardware=HardwareMetrics(
                    kernel_fallbacks=0,
                    device_name="Mac15,9",
                    chip="Apple M3 Max",
                    unified_memory_bytes=128 * 1024**3,
                    os_version="macOS-test",
                ),
                integrity=IntegrityMetrics(
                    safetensors_valid=True,
                    index_complete=True,
                    config_valid=True,
                    mtp_layout_valid=True,
                    source_revision_pinned=True,
                ),
                workload=profile.value,
                dataset_sha256=dataset_sha256,
                software_versions=versions,
                random_seed=11,
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
                    "power_mode": "AC power",
                    "quantizer": kind.value,
                    "quantizer_version": "fixture-v1",
                    "quality_dataset_sha256": f"{profile.value}-quality",
                    "ax_engine_version": "6.11.1",
                },
            ),
        )
        entries.append(
            BenchmarkEvidenceInput(
                kind=kind,
                status="available",
                evaluation_file=path.name,
            )
        )
    request_path = evidence_directory / "request.json"
    write_data(
        request_path,
        BenchmarkEvidenceRequest(
            profile=profile,
            entries=entries,
        ),
    )
    index = build_benchmark_evidence_index(request_path)
    assert index.release_ready
    index_path = evidence_directory / "index.json"
    write_data(index_path, index)
    return index_path


def _release_validation_index(
    tmp_path: Path,
    *,
    candidate_id: str,
    primary_validation: Path,
    general_validation_report: ValidationReport | None = None,
) -> Path:
    general_validation = tmp_path / "general-validation.json"
    write_data(
        general_validation,
        general_validation_report
        or _validation(
            candidate_id,
            tmp_path / "candidate",
            profile=ProfileName.GENERAL,
        ),
    )
    request_path = tmp_path / "release-validation-request.json"
    write_data(
        request_path,
        ReleaseValidationRequest(
            entries=[
                ReleaseValidationInput(
                    profile=ProfileName.AGENT_CODING,
                    validation_file=str(primary_validation),
                    benchmark_index_file=str(
                        _benchmark_index(
                            tmp_path,
                            candidate_id=candidate_id,
                            profile=ProfileName.AGENT_CODING,
                            dataset_sha256="b" * 64,
                        )
                    ),
                ),
                ReleaseValidationInput(
                    profile=ProfileName.GENERAL,
                    validation_file=str(general_validation),
                    benchmark_index_file=str(
                        _benchmark_index(
                            tmp_path,
                            candidate_id=candidate_id,
                            profile=ProfileName.GENERAL,
                            dataset_sha256="c" * 64,
                        )
                    ),
                ),
            ]
        ),
    )
    index = build_release_validation_index(request_path)
    index_path = tmp_path / "release-validation-index.json"
    write_data(index_path, index)
    return index_path


def _write_candidate(directory: Path, plan) -> None:
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "mtp.safetensors").write_bytes(b"mtp")
    (directory / "model-manifest.json").write_text("{}", encoding="utf-8")
    write_data(directory / "axquant_plan.json", plan)
    if plan.calibration is not None:
        calibration_manifest = CalibrationManifest(
            model=plan.source_model,
            profile=plan.profile,
            dataset_id=plan.calibration.dataset_id,
            dataset_sha256=plan.calibration.dataset_sha256,
            samples=plan.calibration.samples,
            domains=plan.calibration.domains,
            sequence_length=plan.calibration.sequence_length,
            random_seed=plan.random_seed,
            calibration_evaluation_separation_attested=True,
        )
        write_data(directory / "calibration_manifest.json", calibration_manifest)
        plan.calibration.metadata["calibration_manifest_sha256"] = file_sha256(
            directory / "calibration_manifest.json"
        )
        write_data(directory / "axquant_plan.json", plan)
    runtime = build_runtime_metadata(plan, directory)
    write_data(directory / "axquant_runtime.json", runtime)
    files = [
        ArtifactFile(
            path=path.relative_to(directory).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]
    manifest = ArtifactManifest(
        axquant_version=plan.software_versions.axquant,
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        calibration=plan.calibration,
        profile=plan.profile,
        target_class=plan.target_class,
        effective_bpw=plan.effective_bpw,
        logical_parameters=1,
        main_logical_parameters=1,
        weight_file_size_bytes=10,
        main_weight_file_size_bytes=7,
        mtp_weight_file_size_bytes=3,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=80.0,
        measured_main_bpw=56.0,
        weight_distribution=plan.weight_distribution,
        mtp_distribution=plan.mtp_distribution,
        mtp_present=True,
        mtp_policy=plan.mtp,
        runtime=runtime,
        software_versions=plan.software_versions,
        files=files,
    )
    write_data(directory / "axquant_manifest.json", manifest)


def _m7_evidence(
    tmp_path: Path,
    *,
    candidate_id: str,
    plan,
    artifact_manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    candidate_model = ModelIdentity(
        model_id=candidate_id,
        revision="candidate-revision",
    )
    hardware = CompleteCandidateHardware(
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
    evidence_directory = tmp_path / f"{candidate_id.split('/')[-1]}-hardware-inputs"
    evidence_directory.mkdir(exist_ok=True)
    evidence_paths = {
        "plan": evidence_directory / "plan.json",
        "artifact": artifact_manifest_path or evidence_directory / "artifact_manifest.json",
        "sensitivity": evidence_directory / "sensitivity.json",
        "quality": evidence_directory / "quality_comparison.json",
        "validation": evidence_directory / "validation.json",
        "direct_evaluation": evidence_directory / "evaluation_mtp_off.json",
        "mtp_evaluation": evidence_directory / "evaluation_mtp_on.json",
        "direct_result": evidence_directory / "benchmark_mtp_off.json",
        "mtp_result": evidence_directory / "benchmark_mtp_on.json",
        "execution": evidence_directory / "quantizer_execution.json",
    }
    write_data(evidence_paths["plan"], plan)
    for label, path in evidence_paths.items():
        if label != "plan" and not (label == "artifact" and artifact_manifest_path is not None):
            write_data(path, {"fixture": label})
    measurements = RefinementMeasurementSet(
        refinement_sha256="refinement",
        evaluator_version="test",
        measurements=[
            CompleteCandidateMeasurement(
                candidate_id="release-candidate",
                candidate_model=candidate_model,
                profile=ProfileName.AGENT_CODING,
                plan_sha256=stable_sha256(plan),
                artifact_manifest_sha256=file_sha256(evidence_paths["artifact"]),
                quality_comparison_sha256=file_sha256(evidence_paths["quality"]),
                validation_sha256=file_sha256(evidence_paths["validation"]),
                measured_bpw=4.7,
                objective_loss=0.1,
                quality_retention=0.99,
                mtp_acceptance_retention=0.97,
                mtp_speedup=1.25,
                peak_memory_ratio=0.8,
                hardware=hardware,
                validation_passed=True,
            )
        ],
    )
    measurement_path = evidence_directory / "refinement_measurements.json"
    write_data(measurement_path, measurements)
    measurement_sha256 = stable_sha256(measurements)
    coverage = HardwareKernelCoverage(
        bits=16,
        method=QuantMethod.BF16,
        roles=[TensorRole.NORM],
        shapes=[(1,)],
        module_count=1,
        parameter_count=1,
        quantizer_execution_records=0,
        kernel_evidence="measured",
    )
    registry = HardwareProfileRegistry(
        registry_id="publication-test",
        measurement_set_sha256=measurement_sha256,
        measurement_set_file=str(measurement_path),
        measurement_set_file_sha256=file_sha256(measurement_path),
        entries=[
            HardwareRegistryEntry(
                entry_id="candidate-m3-max",
                candidate_id="release-candidate",
                measurement_id="release-candidate",
                candidate_model=candidate_model,
                profile=ProfileName.AGENT_CODING,
                plan_file=str(evidence_paths["plan"]),
                plan_file_sha256=file_sha256(evidence_paths["plan"]),
                plan_sha256=stable_sha256(plan),
                artifact_manifest_file=str(evidence_paths["artifact"]),
                artifact_manifest_sha256=file_sha256(evidence_paths["artifact"]),
                sensitivity_file=str(evidence_paths["sensitivity"]),
                sensitivity_sha256=file_sha256(evidence_paths["sensitivity"]),
                quality_comparison_file=str(evidence_paths["quality"]),
                quality_comparison_sha256=file_sha256(evidence_paths["quality"]),
                validation_file=str(evidence_paths["validation"]),
                validation_sha256=file_sha256(evidence_paths["validation"]),
                direct_evaluation_file=str(evidence_paths["direct_evaluation"]),
                direct_evaluation_sha256=file_sha256(evidence_paths["direct_evaluation"]),
                mtp_evaluation_file=str(evidence_paths["mtp_evaluation"]),
                mtp_evaluation_sha256=file_sha256(evidence_paths["mtp_evaluation"]),
                direct_benchmark_result_file=str(evidence_paths["direct_result"]),
                direct_benchmark_result_sha256=file_sha256(evidence_paths["direct_result"]),
                mtp_benchmark_result_file=str(evidence_paths["mtp_result"]),
                mtp_benchmark_result_sha256=file_sha256(evidence_paths["mtp_result"]),
                quantizer_execution_file=str(evidence_paths["execution"]),
                quantizer_execution_sha256=file_sha256(evidence_paths["execution"]),
                hardware=hardware,
                protocol=HardwareMeasurementProtocol(
                    protocol_id="ax-engine-ab-v1-test",
                    backend_version="6.11.1",
                    dataset_sha256="8" * 64,
                    random_seed=7,
                    prompt_count=1,
                    warmup_trials=1,
                    measured_trials=1,
                    power_mode="AC power",
                    deterministic_tolerance=0.0,
                    direct_commands=[["ax-engine-bench", "generate"]],
                    mtp_commands=[["ax-engine-bench", "generate"]],
                ),
                coverage=[coverage],
                total_modules=1,
                unique_shapes=1,
                kernel_evidence="measured",
                validation_passed=True,
                release_ready=True,
            )
        ],
        distinct_named_hosts=1,
        release_ready=True,
        issues=[],
    )
    pareto = ParetoReport(
        profile=ProfileName.AGENT_CODING,
        measurement_set_sha256=measurement_sha256,
        points=[
            ParetoPoint(
                candidate_id="release-candidate",
                candidate_model=candidate_model,
                plan_sha256=stable_sha256(plan),
                measured_bpw=4.7,
                quality_retention=0.99,
                mtp_acceptance_retention=0.97,
                mtp_speedup=1.25,
                peak_memory_ratio=0.8,
                hardware=hardware,
                validation_passed=True,
                frontier=True,
            )
        ],
        frontier_candidate_ids=["release-candidate"],
    )
    registry_path = tmp_path / f"{candidate_id.split('/')[-1]}-hardware-registry.json"
    pareto_path = tmp_path / f"{candidate_id.split('/')[-1]}-pareto.json"
    write_data(registry_path, registry)
    write_data(pareto_path, pareto)
    return registry_path, pareto_path


def test_publication_rejects_architecture_prior_evidence(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    _write_candidate(candidate, plan)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )
    with pytest.raises(ValidationGateError, match="measured calibration evidence"):
        prepare_publication(
            model_dir=candidate,
            repo_id="AutomatosX/candidate",
            validation_index_path=_release_validation_index(
                tmp_path,
                candidate_id="AutomatosX/candidate",
                primary_validation=validation,
            ),
            hardware_registry_path=hardware_registry,
            pareto_report_path=pareto,
        )


def test_publication_rejects_validation_for_another_repository(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    _write_candidate(candidate, plan)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/another-candidate", candidate))
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/another-candidate",
        plan=plan,
    )
    with pytest.raises(ValidationGateError, match="publication repository"):
        prepare_publication(
            model_dir=candidate,
            repo_id="AutomatosX/candidate",
            validation_index_path=_release_validation_index(
                tmp_path,
                candidate_id="AutomatosX/another-candidate",
                primary_validation=validation,
            ),
            hardware_registry_path=hardware_registry,
            pareto_report_path=pareto,
        )


def test_publication_rejects_unpinned_candidate_revision(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    _write_candidate(candidate, plan)
    validation_report = _validation("AutomatosX/candidate", candidate)
    validation_report.candidate_model.revision = None
    validation = tmp_path / "validation.json"
    write_data(validation, validation_report)
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )

    with pytest.raises(ValidationGateError, match="release-ready validation index"):
        prepare_publication(
            model_dir=candidate,
            repo_id="AutomatosX/candidate",
            validation_index_path=_release_validation_index(
                tmp_path,
                candidate_id="AutomatosX/candidate",
                primary_validation=validation,
            ),
            hardware_registry_path=hardware_registry,
            pareto_report_path=pareto,
        )


def test_publication_materializes_runtime_and_reproduction_evidence(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    validation_index = _release_validation_index(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        primary_validation=validation,
    )
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )
    files = prepare_publication(
        model_dir=candidate,
        repo_id="AutomatosX/candidate",
        validation_index_path=validation_index,
        hardware_registry_path=hardware_registry,
        pareto_report_path=pareto,
    )
    assert candidate / "README.md" in files
    readme = (candidate / "README.md").read_text(encoding="utf-8")
    assert "Compatibility Level A" in readme
    assert "Compatibility Level B" in readme
    manifest = load_model(candidate / "axquant_manifest.json", ArtifactManifest)
    assert manifest.mtp_acceptance_retention == 0.97
    assert manifest.mtp_measured_speedup == 1.25
    assert manifest.runtime.mtp.optimized is True
    assert any(record.path == "benchmark_report.json" for record in manifest.files)
    packaged_index = load_model(
        candidate / "benchmark_evidence_index.json",
        BenchmarkEvidenceIndex,
    )
    assert packaged_index.release_ready
    assert all(
        entry.evaluation_file is None or entry.evaluation_file.startswith("benchmark_evidence/")
        for entry in packaged_index.entries
    )
    assert len(list((candidate / "benchmark_evidence").glob("*.json"))) == 5
    assert len(list((candidate / "general_benchmark_evidence").glob("*.json"))) == 5
    assert len(list((candidate / "hardware_evidence").rglob("*.json"))) == 10
    assert list((candidate / "hardware_evidence").rglob("artifact_manifest.json"))
    assert list((candidate / "hardware_evidence").rglob("quality_comparison.json"))
    packaged_hardware = load_model(
        candidate / "hardware_profile_registry.json",
        HardwareProfileRegistry,
    )
    assert packaged_hardware.release_ready
    assert packaged_hardware.measurement_set_file == "refinement_measurements.json"
    assert (candidate / packaged_hardware.measurement_set_file).is_file()
    assert packaged_hardware.entries[0].plan_file.startswith("hardware_evidence/")
    packaged_pareto = load_model(candidate / "pareto_report.json", ParetoReport)
    assert packaged_pareto.frontier_candidate_ids == ["release-candidate"]
    packaged_validation_index = load_model(
        candidate / "release_validation_index.json",
        ReleaseValidationIndex,
    )
    assert packaged_validation_index.release_ready
    assert {entry.profile for entry in packaged_validation_index.entries} == {
        ProfileName.AGENT_CODING,
        ProfileName.GENERAL,
    }
    recipe = load_model(candidate / "reproduction_recipe.yaml", ReproductionRecipe)
    assert recipe.plan_sha256 == manifest.plan_sha256
    assert recipe.source_model.revision == plan.source_model.revision
    assert [command.step_id for command in recipe.commands] == [
        "download-source",
        "convert",
        "verify-ax-engine",
        "verify-mlx-lm",
        "verify-reproduction",
    ]
    assert recipe.plan_file_sha256 == file_sha256(candidate / "quantization_plan.json")
    assert recipe.calibration_file_sha256 == file_sha256(candidate / "calibration_manifest.json")
    assert recipe.mtp_sidecar_sha256 == file_sha256(candidate / "mtp.safetensors")
    convert_command = next(command for command in recipe.commands if command.step_id == "convert")
    assert convert_command.argv[convert_command.argv.index("--mtp-layout") + 1] == "byte-preserved"
    verification = verify_reproduction(
        recipe_path=candidate / "reproduction_recipe.yaml",
        artifact_dir=candidate,
    )
    assert verification.passed
    assert sorted(verification.verified_weight_files) == [
        "model.safetensors",
        "mtp.safetensors",
    ]
    assert (candidate / "axquant_conversion_manifest.json").is_file()
    repeated_files = prepare_publication(
        model_dir=candidate,
        repo_id="AutomatosX/candidate",
        validation_index_path=validation_index,
        hardware_registry_path=hardware_registry,
        pareto_report_path=pareto,
    )
    assert candidate / "README.md" in repeated_files


def test_publication_snapshots_hardware_manifest_before_runtime_updates(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    measured_manifest = candidate / "axquant_manifest.json"
    measured_manifest_sha256 = file_sha256(measured_manifest)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
        artifact_manifest_path=measured_manifest,
    )

    prepare_publication(
        model_dir=candidate,
        repo_id="AutomatosX/candidate",
        validation_index_path=_release_validation_index(
            tmp_path,
            candidate_id="AutomatosX/candidate",
            primary_validation=validation,
        ),
        hardware_registry_path=hardware_registry,
        pareto_report_path=pareto,
    )

    packaged_registry = load_model(
        candidate / "hardware_profile_registry.json",
        HardwareProfileRegistry,
    )
    packaged_entry = packaged_registry.entries[0]
    packaged_manifest = candidate / packaged_entry.artifact_manifest_file
    assert packaged_entry.artifact_manifest_sha256 == measured_manifest_sha256
    assert file_sha256(packaged_manifest) == measured_manifest_sha256
    assert file_sha256(candidate / "axquant_manifest.json") != measured_manifest_sha256


def test_prepared_mtp_reproduction_binds_required_companions(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    runtime_path = candidate / "mtplx_runtime.json"
    write_data(runtime_path, {"layout": "ax-engine-qwen36-v1"})
    sidecar_path = candidate / "mtp.safetensors"
    write_data(
        candidate / "ax_mtp_sidecar_manifest.json",
        {
            "schema_version": "axquant.mtp-sidecar-provenance.v3",
            "generated_by": "axquant",
            "source": {
                "model": {
                    "model_id": plan.source_model.model_id,
                    "revision": plan.source_model.revision,
                },
                "path": "source",
                "index_sha256": "1" * 64,
                "shards": [
                    {
                        "name": "source.safetensors",
                        "size_bytes": 1,
                        "sha256": "2" * 64,
                    }
                ],
            },
            "input": {
                "manifest": {
                    "path": "raw-manifest.json",
                    "size_bytes": 1,
                    "sha256": "3" * 64,
                },
                "mtp": {
                    "path": "raw-mtp.safetensors",
                    "size_bytes": 1,
                    "sha256": "4" * 64,
                },
            },
            "output": {
                "mtp": {
                    "path": sidecar_path.name,
                    "size_bytes": sidecar_path.stat().st_size,
                    "sha256": file_sha256(sidecar_path),
                },
                "runtime": {
                    "path": runtime_path.name,
                    "size_bytes": runtime_path.stat().st_size,
                    "sha256": file_sha256(runtime_path),
                },
            },
            "transform": {
                "mode": "ax-engine-qwen36-v1",
                "implementation": "axquant",
                "operation": "add_one_to_qwen36_mtp_norms_bf16",
                "transformed_tensors": ["mtp.norm.weight"],
                "unchanged_tensors": ["mtp.fc.weight"],
            },
            "tensor_count": 1,
            "tensor_payloads": [
                {
                    "name": "mtp.norm.weight",
                    "dtype": "BF16",
                    "shape": [1],
                    "byte_count": sidecar_path.stat().st_size,
                    "sha256": file_sha256(sidecar_path),
                    "source_sha256": "5" * 64,
                    "operation": "add_one_bf16",
                }
            ],
            "total_payload_bytes": sidecar_path.stat().st_size,
        },
    )
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )

    prepare_publication(
        model_dir=candidate,
        repo_id="AutomatosX/candidate",
        validation_index_path=_release_validation_index(
            tmp_path,
            candidate_id="AutomatosX/candidate",
            primary_validation=validation,
        ),
        hardware_registry_path=hardware_registry,
        pareto_report_path=pareto,
    )

    recipe_path = candidate / "reproduction_recipe.yaml"
    recipe = load_model(recipe_path, ReproductionRecipe)
    assert recipe.schema_version == "axquant.reproduction.v3"
    assert {record.path for record in recipe.mtp_companion_files} == {
        "ax_mtp_sidecar_manifest.json",
        "mtplx_runtime.json",
    }
    convert = next(command for command in recipe.commands if command.step_id == "convert")
    assert convert.argv[convert.argv.index("--mtp-layout") + 1] == "ax-engine-qwen36-v1"
    assert verify_reproduction(recipe_path=recipe_path, artifact_dir=candidate).passed

    write_data(runtime_path, {"layout": "tampered"})
    verification = verify_reproduction(recipe_path=recipe_path, artifact_dir=candidate)
    assert verification.passed is False
    assert any("MTP companion checksum" in issue for issue in verification.issues)


def test_publication_packages_a_governed_size_exception(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    candidate_model = ModelIdentity(
        model_id="AutomatosX/candidate",
        revision="candidate-revision",
    )
    approved_at = datetime.now(UTC) - timedelta(days=1)
    exception = ReleaseException(
        exception_id="AXQ-SIZE-PUBLISH",
        candidate_model=candidate_model,
        plan_sha256=stable_sha256(plan),
        targets=[
            ReleaseExceptionTarget(
                metric="artifact.weight_size_ratio",
                observed_value=1.2,
                required_maximum=1.1,
                requirement="candidate must be no more than 110% of uniform 4-bit",
            ),
            ReleaseExceptionTarget(
                metric="artifact.candidate_measured_bpw",
                observed_value=5.76,
                required_minimum=4.3,
                required_maximum=4.8,
                requirement="candidate must remain within the target BPW range",
            ),
        ],
        measured_tradeoff="Measured quality is retained while peak memory is reduced.",
        owner="AutomatosX release owner",
        approved_by="Release authority",
        approval_reference="decision-publish",
        approved_at=approved_at,
        expires_at=approved_at + timedelta(days=30),
        evidence_sha256={
            "plan": "1" * 64,
            "candidate_size": "2" * 64,
            "size_reference": "3" * 64,
            "tradeoff": "4" * 64,
        },
    )

    def excepted(profile: ProfileName) -> ValidationReport:
        report = _validation("AutomatosX/candidate", candidate, profile=profile)
        report.comparisons["artifact.weight_size_ratio"] = 1.2
        report.comparisons["artifact.candidate_measured_bpw"] = 5.76
        report.issues = [
            ValidationIssue(
                severity="warning",
                metric="artifact.weight_size_ratio",
                message=("ratio 1.2000 exceeds 1.1000; governed exception AXQ-SIZE-PUBLISH"),
            )
        ]
        report.release_exceptions = [exception]
        return report

    agent_validation_path = tmp_path / "validation.json"
    write_data(agent_validation_path, excepted(ProfileName.AGENT_CODING))
    validation_index = _release_validation_index(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        primary_validation=agent_validation_path,
        general_validation_report=excepted(ProfileName.GENERAL),
    )
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )

    files = prepare_publication(
        model_dir=candidate,
        repo_id="AutomatosX/candidate",
        validation_index_path=validation_index,
        hardware_registry_path=hardware_registry,
        pareto_report_path=pareto,
    )

    exception_path = candidate / "release_exception.json"
    assert exception_path in files
    assert load_model(exception_path, ReleaseException) == exception
    manifest = load_model(candidate / "axquant_manifest.json", ArtifactManifest)
    assert any(record.path == "release_exception.json" for record in manifest.files)


def test_reproduction_verification_detects_changed_weight_bytes(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )
    prepare_publication(
        model_dir=candidate,
        repo_id="AutomatosX/candidate",
        validation_index_path=_release_validation_index(
            tmp_path,
            candidate_id="AutomatosX/candidate",
            primary_validation=validation,
        ),
        hardware_registry_path=hardware_registry,
        pareto_report_path=pareto,
    )
    (candidate / "model.safetensors").write_bytes(b"changed")

    verification = verify_reproduction(
        recipe_path=candidate / "reproduction_recipe.yaml",
        artifact_dir=candidate,
    )

    assert not verification.passed
    assert "reproduced weight file checksum changed: model.safetensors" in verification.issues
    output = tmp_path / "reproduction-verification.json"
    assert (
        main(
            [
                "verify-reproduction",
                "--recipe",
                str(candidate / "reproduction_recipe.yaml"),
                "--artifact",
                str(candidate),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    cli_verification = load_model(output, ReproductionVerification)
    assert not cli_verification.passed


def test_publication_rejects_unmeasured_hardware_registry(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    validation_index = _release_validation_index(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        primary_validation=validation,
    )
    hardware_registry_path, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )
    registry = load_model(hardware_registry_path, HardwareProfileRegistry)
    entry = registry.entries[0]
    failed_entry = entry.model_copy(
        update={
            "coverage": [
                item.model_copy(update={"kernel_evidence": "unmeasured"}) for item in entry.coverage
            ],
            "kernel_evidence": "unmeasured",
            "release_ready": False,
            "issues": ["runtime kernel fallback"],
        }
    )
    failed_registry = registry.model_copy(
        update={
            "entries": [failed_entry],
            "distinct_named_hosts": 0,
            "release_ready": False,
            "issues": ["candidate-m3-max: runtime kernel fallback"],
        }
    )
    write_data(hardware_registry_path, failed_registry)

    with pytest.raises(ValidationGateError, match="release-ready hardware registry"):
        prepare_publication(
            model_dir=candidate,
            repo_id="AutomatosX/candidate",
            validation_index_path=validation_index,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto,
        )


def test_publication_rejects_tampered_general_validation(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    validation = tmp_path / "validation.json"
    write_data(validation, _validation("AutomatosX/candidate", candidate))
    validation_index_path = _release_validation_index(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        primary_validation=validation,
    )
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    general_entry = next(
        entry for entry in validation_index.entries if entry.profile == ProfileName.GENERAL
    )
    Path(general_entry.validation_file).write_text("{}\n", encoding="utf-8")
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )

    with pytest.raises(ValidationGateError, match="general validation checksum mismatch"):
        prepare_publication(
            model_dir=candidate,
            repo_id="AutomatosX/candidate",
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry,
            pareto_report_path=pareto,
        )


@pytest.mark.parametrize(
    ("comparison", "value", "message"),
    [
        (
            "artifact.candidate_weight_bytes",
            11,
            "validation size evidence does not match candidate weight bytes",
        ),
        (
            "artifact.candidate_source_sha256",
            "0" * 64,
            "validation size evidence does not bind the candidate manifest",
        ),
    ],
)
def test_publication_rejects_size_evidence_for_another_artifact(
    qwen36_model_dir: Path,
    tmp_path: Path,
    comparison: str,
    value: int | str,
    message: str,
) -> None:
    candidate = tmp_path / "candidate"
    plan = _plan(qwen36_model_dir)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id="internal/agent-coding-calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "tool-use"],
        sequence_length=2048,
        backend="mlx",
        reference="calibration-manifest.json",
    )
    _write_candidate(candidate, plan)
    report = _validation("AutomatosX/candidate", candidate)
    report.comparisons[comparison] = value
    validation = tmp_path / "validation.json"
    write_data(validation, report)
    hardware_registry, pareto = _m7_evidence(
        tmp_path,
        candidate_id="AutomatosX/candidate",
        plan=plan,
    )
    with pytest.raises(ValidationGateError, match=message):
        prepare_publication(
            model_dir=candidate,
            repo_id="AutomatosX/candidate",
            validation_index_path=_release_validation_index(
                tmp_path,
                candidate_id="AutomatosX/candidate",
                primary_validation=validation,
            ),
            hardware_registry_path=hardware_registry,
            pareto_report_path=pareto,
        )
