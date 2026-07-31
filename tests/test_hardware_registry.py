from __future__ import annotations

from pathlib import Path

from axquant.cli import main
from axquant.hardware_registry import build_hardware_profile_registry
from axquant.planner import plan_quantization
from axquant.profiles import thresholds_for
from axquant.refinement import build_complete_candidate_measurement
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    ArtifactManifest,
    BenchmarkConfig,
    BenchmarkResult,
    CalibrationEvidence,
    CandidateMeasurement,
    EvaluationBundle,
    EvidenceKind,
    HardwareMetrics,
    HardwareRegistryCandidateInput,
    HardwareRegistryRequest,
    IntegrityMetrics,
    MetricVector,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QualityComparisonReport,
    QualityScoreComparison,
    QuantizerExecutionManifest,
    QuantizerExecutionRecord,
    QuantMethod,
    RefinementMeasurementSet,
    SensitivityReport,
    SoftwareVersions,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
    TrialResult,
    ValidationIssue,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


def _sensitivity() -> SensitivityReport:
    model = ModelIdentity(model_id="Qwen/Qwen3.6-test", revision="source-revision")
    mlp = TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(16, 64),
        dtype="bfloat16",
        parameters=1024,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bfloat16",
    )
    norm = TensorSpec(
        name="model.layers.0.input_layernorm.weight",
        module_path="model.layers.0.input_layernorm",
        shape=(64,),
        dtype="bfloat16",
        parameters=64,
        role=TensorRole.NORM,
        quantizable=False,
        file="model.safetensors",
        current_precision="bfloat16",
    )
    return SensitivityReport(
        model=model,
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
        calibration=CalibrationEvidence(
            dataset_id="calibration",
            dataset_sha256="calibration-sha",
            samples=8,
            domains=["coding"],
            sequence_length=128,
            backend="test-probe",
            reference="fixture",
        ),
        entries=[
            TensorSensitivity(
                tensor=mlp,
                candidates=[
                    CandidateMeasurement(
                        bits=4,
                        method=QuantMethod.AFFINE,
                        group_size=64,
                        metrics=MetricVector(output_kl=0.1),
                        measured_tokens=1024,
                    ),
                    CandidateMeasurement(
                        bits=16,
                        method=QuantMethod.BF16,
                        metrics=MetricVector(),
                        measured_tokens=1024,
                    ),
                ],
            ),
            TensorSensitivity(
                tensor=norm,
                candidates=[
                    CandidateMeasurement(
                        bits=16,
                        method=QuantMethod.BF16,
                        metrics=MetricVector(),
                        evidence_scope="preserved",
                    )
                ],
            ),
        ],
    )


def _versions() -> SoftwareVersions:
    return SoftwareVersions(
        axquant="0.1.0a0",
        python="3.13",
        mlx="0.32.0",
        mlx_lm="0.31.0",
        ax_engine="6.11.1",
        safetensors="0.6",
        pydantic="2.11",
    )


def _benchmark_result(
    *,
    model: ModelIdentity,
    mtp: bool,
    fallback_count: int = 0,
) -> BenchmarkResult:
    config = BenchmarkConfig(
        model=model,
        mtp_enabled=mtp,
        baseline_kind="axquant-mtp-on" if mtp else "axquant-mtp-off",
        workload=ProfileName.AGENT_CODING.value,
        dataset_sha256="benchmark-sha",
        prompt_count=1,
        warmup_trials=0,
        measured_trials=1,
        max_tokens=32,
        draft_depth=1,
        power_mode="AC power; automatic mode",
        quantizer="axquant",
        quantizer_version="0.1.0a0",
        random_seed=7,
    )
    trial = TrialResult(
        trial_index=0,
        success=True,
        command=["ax-engine-bench", "generate", "--json"],
        prompt_tokens=8,
        tokens_generated=32,
        latency_seconds=1.0,
        decode_seconds=1.0,
        tokens_per_second=125.0 if mtp else 100.0,
        mtp_accepted_tokens=24 if mtp else None,
        mtp_proposed_tokens=32 if mtp else None,
        mtp_decode_steps=16 if mtp else None,
        mtp_active=True if mtp else None,
        kernel_fallbacks=fallback_count,
        peak_memory_bytes=800 if mtp else 850,
        runtime_device_name="Mac15,9",
        runtime_chip="Apple M3 Max",
        unified_memory_bytes=128 * 1024**3,
        os_version="macOS-test",
    )
    return BenchmarkResult(
        config=config,
        trials=[trial],
        measured_count=1,
        runtime_device_name="Mac15,9",
        runtime_chip="Apple M3 Max",
        unified_memory_bytes=128 * 1024**3,
        os_version="macOS-test",
        ax_engine_version="6.11.1",
    )


def _evaluation(result: BenchmarkResult) -> EvaluationBundle:
    config = result.config
    trial = result.trials[0]
    return EvaluationBundle(
        model=config.model,
        mtp_enabled=config.mtp_enabled,
        baseline_kind=config.baseline_kind,
        hardware=HardwareMetrics(
            peak_memory_bytes=trial.peak_memory_bytes,
            decode_tokens_per_second=trial.tokens_per_second,
            mtp_effective_tokens_per_second=trial.tokens_per_second if config.mtp_enabled else None,
            kernel_fallbacks=trial.kernel_fallbacks,
            device_name=result.runtime_device_name,
            chip=result.runtime_chip,
            unified_memory_bytes=result.unified_memory_bytes,
            os_version=result.os_version,
        ),
        integrity=IntegrityMetrics(
            safetensors_valid=True,
            index_complete=True,
            config_valid=True,
            mtp_layout_valid=True if config.mtp_enabled else None,
            source_revision_pinned=True,
        ),
        workload=config.workload,
        dataset_sha256=config.dataset_sha256,
        software_versions=_versions(),
        random_seed=config.random_seed,
        benchmark_metadata={
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
        },
    )


def _request(tmp_path: Path, *, fallback_count: int = 0) -> Path:
    sensitivity = _sensitivity()
    plan = plan_quantization(
        sensitivity,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=8.0,
            candidate_bits=(4, 16),
            group_size=64,
        ),
    )
    candidate_id = "candidate-a"
    candidate_model = ModelIdentity(
        model_id="AutomatosX/AX-Qwen3.6-test",
        revision="candidate-revision",
    )
    direct_result = _benchmark_result(model=candidate_model, mtp=False)
    mtp_result = _benchmark_result(
        model=candidate_model,
        mtp=True,
        fallback_count=fallback_count,
    )
    direct_evaluation = _evaluation(direct_result)
    mtp_evaluation = _evaluation(mtp_result)
    paths = {
        "plan": tmp_path / "plan.json",
        "artifact": tmp_path / "artifact-manifest.json",
        "sensitivity": tmp_path / "sensitivity.json",
        "quality": tmp_path / "quality-comparison.json",
        "validation": tmp_path / "validation.json",
        "direct_evaluation": tmp_path / "evaluation-mtp-off.json",
        "mtp_evaluation": tmp_path / "evaluation-mtp-on.json",
        "direct_result": tmp_path / "benchmark-mtp-off.json",
        "mtp_result": tmp_path / "benchmark-mtp-on.json",
        "execution": tmp_path / "quantizer-execution.json",
    }
    logical_parameters = sum(allocation.parameters for allocation in plan.assignments)
    weight_bytes = 640
    artifact = ArtifactManifest(
        axquant_version="1.0.0",
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        calibration=plan.calibration,
        profile=plan.profile,
        target_class=plan.target_class,
        effective_bpw=plan.effective_bpw,
        logical_parameters=logical_parameters,
        main_logical_parameters=logical_parameters,
        weight_file_size_bytes=weight_bytes,
        main_weight_file_size_bytes=weight_bytes,
        mtp_weight_file_size_bytes=0,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=weight_bytes * 8 / logical_parameters,
        measured_main_bpw=weight_bytes * 8 / logical_parameters,
        weight_distribution=plan.weight_distribution,
        mtp_distribution=plan.mtp_distribution,
        mtp_present=False,
        mtp_policy=plan.mtp,
        runtime=build_runtime_metadata(plan, tmp_path),
        software_versions=plan.software_versions,
        files=[],
    )
    quality_score = QualityScoreComparison(
        reference=1.0,
        candidate=0.99,
        delta=-0.01,
        retention=0.99,
    )
    quality = QualityComparisonReport(
        reference_model=ModelIdentity(model_id="baseline", revision="baseline-revision"),
        candidate_model=candidate_model,
        dataset_sha256="quality-dataset",
        random_seed=7,
        aggregate=quality_score,
        categories={"coding": quality_score},
        perplexity_ratio=0.95,
        tasks=[],
        reference_errors=0,
        candidate_errors=0,
    )
    write_data(paths["plan"], plan)
    write_data(paths["artifact"], artifact)
    write_data(paths["sensitivity"], sensitivity)
    write_data(paths["quality"], quality)
    write_data(paths["direct_evaluation"], direct_evaluation)
    write_data(paths["mtp_evaluation"], mtp_evaluation)
    write_data(paths["direct_result"], direct_result)
    write_data(paths["mtp_result"], mtp_result)
    validation = ValidationReport(
        reference_model=quality.reference_model,
        candidate_model=candidate_model,
        profile=plan.profile,
        passed=fallback_count == 0,
        thresholds=thresholds_for(plan.profile),
        issues=(
            []
            if fallback_count == 0
            else [
                ValidationIssue(
                    severity="error",
                    metric="hardware.kernel_fallbacks",
                    message="runtime kernel fallbacks are nonzero",
                )
            ]
        ),
        comparisons={
            "artifact.candidate_source_sha256": file_sha256(paths["artifact"]),
            "artifact.candidate_weight_bytes": weight_bytes,
            "artifact.weight_size_ratio": 1.05,
            "mtp.acceptance_retention": 0.97,
            "hardware.effective_speedup": 1.25,
            "hardware.peak_memory_ratio": 0.8,
            "hardware.device_name": "Mac15,9",
            "hardware.chip": "Apple M3 Max",
            "hardware.unified_memory_bytes": 128 * 1024**3,
            "hardware.os_version": "macOS-test",
            "hardware.power_mode": "AC power; automatic mode",
            "hardware.kernel_fallbacks": fallback_count,
            "software.ax_engine": "6.11.1",
            "software.mlx": "0.32.0",
            "software.mlx_lm": "0.31.0",
        },
    )
    write_data(paths["validation"], validation)
    quantized = [allocation for allocation in plan.assignments if allocation.bits < 16]
    write_data(
        paths["execution"],
        QuantizerExecutionManifest(
            plan_sha256=stable_sha256(plan),
            records=[
                QuantizerExecutionRecord(
                    method=allocation.method,
                    module_path=allocation.module_path,
                    bits=allocation.bits,
                    group_size=allocation.group_size,
                    success=True,
                )
                for allocation in quantized
            ],
        ),
    )
    measurement = build_complete_candidate_measurement(
        candidate_id=candidate_id,
        plan=plan,
        artifact=artifact,
        artifact_sha256=file_sha256(paths["artifact"]),
        quality=quality,
        quality_sha256=file_sha256(paths["quality"]),
        validation=validation,
        validation_sha256=file_sha256(paths["validation"]),
    )
    measurement_set = RefinementMeasurementSet(
        refinement_sha256="refinement",
        evaluator_version="test",
        measurements=[measurement],
    )
    measurement_path = tmp_path / "measurements.json"
    write_data(measurement_path, measurement_set)
    request_path = tmp_path / "hardware-request.json"
    write_data(
        request_path,
        HardwareRegistryRequest(
            registry_id="apple-silicon-test",
            measurement_set_file=str(measurement_path),
            candidates=[
                HardwareRegistryCandidateInput(
                    entry_id="candidate-a-m3-max",
                    candidate_id=candidate_id,
                    plan_file=str(paths["plan"]),
                    artifact_manifest_file=str(paths["artifact"]),
                    sensitivity_file=str(paths["sensitivity"]),
                    quality_comparison_file=str(paths["quality"]),
                    validation_file=str(paths["validation"]),
                    direct_evaluation_file=str(paths["direct_evaluation"]),
                    mtp_evaluation_file=str(paths["mtp_evaluation"]),
                    direct_benchmark_result_file=str(paths["direct_result"]),
                    mtp_benchmark_result_file=str(paths["mtp_result"]),
                    quantizer_execution_file=str(paths["execution"]),
                )
            ],
        ),
    )
    return request_path


def test_hardware_registry_binds_measured_kernel_and_shape_coverage(tmp_path: Path) -> None:
    registry = build_hardware_profile_registry(_request(tmp_path))

    assert registry.release_ready is True
    assert registry.distinct_named_hosts == 1
    entry = registry.entries[0]
    assert entry.kernel_evidence == "measured"
    assert entry.hardware.mlx_lm_version == "0.31.0"
    assert entry.hardware.power_mode == "AC power; automatic mode"
    assert entry.unique_shapes == 2
    assert {item.bits for item in entry.coverage} == {4, 16}
    assert all(item.kernel_evidence == "measured" for item in entry.coverage)
    assert entry.protocol.direct_commands
    assert entry.protocol.mtp_commands


def test_hardware_registry_does_not_certify_kernel_fallbacks(tmp_path: Path) -> None:
    registry = build_hardware_profile_registry(_request(tmp_path, fallback_count=1))

    assert registry.release_ready is False
    assert registry.distinct_named_hosts == 0
    assert registry.entries[0].kernel_evidence == "unmeasured"
    assert any("kernel fallbacks are nonzero" in issue for issue in registry.issues)


def test_hardware_registry_cli_writes_release_ready_artifact(tmp_path: Path) -> None:
    output = tmp_path / "hardware-registry.json"
    assert (
        main(
            [
                "hardware-registry",
                "--request",
                str(_request(tmp_path)),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()


def test_hardware_registry_supports_multiple_hosts_for_one_candidate(tmp_path: Path) -> None:
    request_path = _request(tmp_path)
    request = load_model(request_path, HardwareRegistryRequest)
    measurement_path = Path(request.measurement_set_file)
    measurement_set = load_model(measurement_path, RefinementMeasurementSet)
    first_measurement = measurement_set.measurements[0]
    second_measurement = first_measurement.model_copy(
        update={"measurement_id": f"{first_measurement.candidate_id}-host-2"}
    )
    write_data(
        measurement_path,
        measurement_set.model_copy(
            update={"measurements": [first_measurement, second_measurement]}
        ),
    )
    first_input = request.candidates[0].model_copy(
        update={"measurement_id": first_measurement.measurement_id}
    )
    second_input = first_input.model_copy(
        update={
            "entry_id": f"{first_input.entry_id}-host-2",
            "measurement_id": second_measurement.measurement_id,
        }
    )
    write_data(
        request_path,
        request.model_copy(update={"candidates": [first_input, second_input]}),
    )

    registry = build_hardware_profile_registry(request_path)

    assert registry.release_ready
    assert [entry.candidate_id for entry in registry.entries] == [
        first_measurement.candidate_id,
        first_measurement.candidate_id,
    ]
    assert {entry.measurement_id for entry in registry.entries} == {
        first_measurement.measurement_id,
        second_measurement.measurement_id,
    }
