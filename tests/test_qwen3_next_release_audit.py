from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from axquant import publisher
from axquant.architectures.registry import support_matrix
from axquant.calibration import calibration_manifest_sha256
from axquant.certification.common import (
    architecture_fingerprint,
    build_source_checkpoint_manifest,
)
from axquant.certification.packaging import prepare_direct_publication
from axquant.certification.policy import direct_policy, direct_policy_sha256
from axquant.certification.qwen3_next_direct import (
    build_direct_pareto,
    build_direct_release_validation_index,
    build_qwen3_next_release_audit,
)
from axquant.certification.registry import (
    DIRECT_CERTIFICATION_ALLOWED_CLAIMS,
    append_certified_checkpoint,
)
from axquant.coding_suite import NEAR_DUPLICATE_THRESHOLD, SANDBOX_PROFILE_SHA256
from axquant.errors import ArtifactError, PublishingError
from axquant.publisher import (
    _package_release_audit,
    _require_release_audit,
    _rerun_release_audit,
    publish_model,
)
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    Allocation,
    ArchitectureProfile,
    ArchitectureSupportLevel,
    ArtifactFile,
    ArtifactIntegrity,
    ArtifactManifest,
    BaselineAudit,
    BaselineKind,
    CalibrationEvidence,
    CalibrationManifest,
    CandidateEntry,
    CandidateMeasurement,
    CodingOverlapReport,
    CodingScorer,
    CodingSuiteManifest,
    CodingSuiteSelfTestReport,
    CodingTaskManifest,
    CodingTaskPayload,
    DirectBaselineKind,
    DirectBenchmarkArm,
    DirectBenchmarkEvidenceIndex,
    DirectBenchmarkProfile,
    DirectBenchmarkTrial,
    DirectHardwareProfileRegistry,
    DirectHardwareRegistryEntry,
    DirectQualityEvaluation,
    DirectQualityTaskOutcome,
    DirectRefinementMeasurement,
    DirectRefinementMeasurementSet,
    DirectReleaseValidationIndex,
    DirectReleaseValidationRequest,
    DirectValidationEntry,
    DirectValidationRequestEntry,
    EvidenceArchiveIndex,
    EvidenceArchiveRecord,
    EvidenceKind,
    FeasibilityReport,
    HardwareProfile,
    Inventory,
    MetricVector,
    ModelIdentity,
    MtpPolicy,
    ObjectiveWeights,
    OptimizationScope,
    PlanningConstraints,
    PrecisionShare,
    ProfileName,
    QualityCheck,
    QualityTask,
    QuantizationPlan,
    QuantMethod,
    Qwen3NextCompatibilityMatrix,
    Qwen3NextCompatibilityRequest,
    Qwen3NextReleaseAuditRequest,
    RefinementConfig,
    RefinementResult,
    ReproductionCommand,
    ReproductionRecipe,
    RuntimeCheck,
    RuntimeName,
    ScaleStrategy,
    SensitivityReport,
    SoftwareVersions,
    SourceConversionProvenance,
    SupportTier,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
)
from axquant.serde import file_sha256, stable_sha256, write_data

_SOURCE_REVISION = "b" * 40
_CANDIDATE_REVISION = "c" * 40
_SOURCE_ID = "Qwen/Qwen3-Coder-Next"
_CANDIDATE_ID = "AutomatosX/AX-Qwen3-Coder-Next-4bit"
_HARDWARE_SCOPE = "m2-ultra-192gb"


def _versions() -> SoftwareVersions:
    return SoftwareVersions(
        axquant="1.2.0",
        python="3.13",
        mlx="0.32.0",
        mlx_lm="0.31.3",
        ax_engine="test-direct-v1",
        safetensors="0.6.2",
        pydantic="2.11.0",
    )


def _write_wheel(path: Path) -> None:
    version = "1.2.0"
    dist_info = f"axquant-{version}.dist-info"
    metadata = (
        "Name: axquant\n"
        f"Version: {version}\n"
        "Classifier: Development Status :: 5 - Production/Stable\n"
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
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist_info}/entry_points.txt": b"[console_scripts]\naxquant = axquant.cli:entrypoint\n",
        f"{dist_info}/licenses/LICENSE": b"test license\n",
        "axquant/__init__.py": f'__version__ = "{version}"\n'.encode(),
    }
    for module in (
        "cli/__init__.py",
        "schema/__init__.py",
        "schema/certification.py",
        "schema/coding_suite.py",
        "certification/dispatch.py",
        "certification/qwen3_next_direct.py",
        "certification/registry.py",
        "release_audit.py",
        "release_exceptions.py",
        "hardware_registry.py",
        "reporting.py",
    ):
        members[f"axquant/{module}"] = b"\n"
    record_name = f"{dist_info}/RECORD"
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow((record_name, "", ""))
    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in members.items():
            wheel.writestr(name, data)
        wheel.writestr(record_name, output.getvalue())


def _tensors() -> list[TensorSpec]:
    values = (
        ("model.layers.0.mlp.switch_mlp.weight", TensorRole.EXPERT, (480, 1, 1), 480, True),
        ("model.layers.0.self_attn.q_proj.weight", TensorRole.ATTENTION, (240, 1), 240, True),
        ("model.layers.0.mlp.down_proj.weight", TensorRole.MLP, (240, 1), 240, True),
        ("model.layers.0.mlp.gate.weight", TensorRole.ROUTER, (10, 1), 10, True),
        ("model.embed_tokens.weight", TensorRole.EMBEDDING, (10, 1), 10, True),
        ("model.layers.0.input_layernorm.weight", TensorRole.NORM, (10,), 10, False),
        ("lm_head.weight", TensorRole.LM_HEAD, (10, 1), 10, True),
    )
    return [
        TensorSpec(
            name=name,
            module_path=name.removesuffix(".weight"),
            shape=shape,
            dtype="bfloat16",
            parameters=parameters,
            storage_bytes=parameters * 2,
            role=role,
            quantizable=quantizable,
            file="model.safetensors",
            current_precision="bfloat16",
            protected_recommendation=role
            in {TensorRole.ROUTER, TensorRole.EMBEDDING, TensorRole.NORM, TensorRole.LM_HEAD},
        )
        for name, role, shape, parameters, quantizable in values
    ]


def _profile() -> ArchitectureProfile:
    return ArchitectureProfile(
        adapter_id="qwen3-next-v1",
        product_family="qwen3-next",
        config_model_type="qwen3_next",
        support_level=ArchitectureSupportLevel.SUPPORTED,
        support_tier=SupportTier.CONVERTIBLE,
        optimization_scope=OptimizationScope.FULL_MODEL,
        dense=False,
        text_layer_count=48,
        mtp_declared=False,
        vision_present=False,
    )


def _assignment(tensor: TensorSpec) -> Allocation:
    bits = {
        TensorRole.EXPERT: 4,
        TensorRole.ATTENTION: 6,
        TensorRole.MLP: 4,
        TensorRole.ROUTER: 8,
        TensorRole.EMBEDDING: 8,
        TensorRole.NORM: 16,
        TensorRole.LM_HEAD: 16,
    }[tensor.role]
    return Allocation(
        tensor=tensor.name,
        module_path=tensor.module_path,
        role=tensor.role,
        parameters=tensor.parameters,
        bits=bits,
        method=QuantMethod.BF16 if bits == 16 else QuantMethod.AFFINE,
        group_size=None if bits == 16 else 64,
        predicted_loss=0.0,
        metrics=MetricVector(),
        reason="synthetic measured certification fixture",
        scale_strategy=ScaleStrategy.NONE if bits == 16 else ScaleStrategy.GROUP_AFFINE,
    )


def _integrity(*, bf16: bool) -> ArtifactIntegrity:
    return ArtifactIntegrity(
        config_valid=True,
        safetensors_present=True,
        index_present=False,
        index_complete=True,
        native_manifest_present=not bf16,
        native_manifest_valid=not bf16,
        tokenizer_present=True,
        mtp_sidecar_present=False,
        mtp_runtime_present=False,
        mtp_runtime_valid=True,
        mtp_provenance_present=False,
        mtp_provenance_valid=True,
    )


def _baseline(
    kind: BaselineKind, source: ModelIdentity, *, weight_bytes: int, bpw: float
) -> BaselineAudit:
    return BaselineAudit(
        kind=kind,
        model=source,
        inspected=True,
        inventory_sha256=None,
        adapter_id="qwen3-next-v1",
        optimization_scope=OptimizationScope.FULL_MODEL,
        quantized=kind is not BaselineKind.BF16_SOURCE,
        logical_parameters=1000,
        mtp_logical_parameters=0,
        weight_bytes=weight_bytes,
        main_weight_bytes=weight_bytes,
        mtp_weight_bytes=0,
        effective_bpw=bpw,
        main_effective_bpw=bpw,
        precision_parameters={str(bpw): 1000},
        precision_fractions={str(bpw): 1.0},
        integrity=_integrity(bf16=kind is BaselineKind.BF16_SOURCE),
        complete=True,
    )


def _benchmark_trial(prefix: str, index: int, *, warmup: bool, decode: float, ttft: float):
    return DirectBenchmarkTrial(
        trial_id=f"{prefix}-{index}",
        warmup=warmup,
        success=True,
        decode_tokens_per_second=decode,
        ttft_seconds=ttft,
        peak_memory_bytes=1024,
        output_sha256=hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest(),
    )


def _benchmark_arm(
    kind: DirectBaselineKind,
    *,
    candidate: ModelIdentity,
    manifest_sha256: str,
    plan_sha256: str,
    profile: ProfileName,
) -> DirectBenchmarkArm:
    decode, ttft = {
        DirectBaselineKind.BF16: (10.0, 0.2),
        DirectBaselineKind.UNIFORM_4BIT: (13.0, 0.095),
        DirectBaselineKind.UNIFORM_6BIT: (12.0, 0.11),
        DirectBaselineKind.CANDIDATE: (12.5, 0.1),
    }[kind]
    model = (
        candidate
        if kind is DirectBaselineKind.CANDIDATE
        else ModelIdentity(model_id=f"fixture/{kind.value}", revision=_CANDIDATE_REVISION)
    )
    trials = [
        *[
            _benchmark_trial(
                f"{profile.value}-{kind.value}-warmup", index, warmup=True, decode=decode, ttft=ttft
            )
            for index in range(2)
        ],
        *[
            _benchmark_trial(
                f"{profile.value}-{kind.value}-trial", index, warmup=False, decode=decode, ttft=ttft
            )
            for index in range(5)
        ],
    ]
    return DirectBenchmarkArm(
        kind=kind,
        model=model,
        artifact_manifest_sha256=(
            manifest_sha256 if kind is DirectBaselineKind.CANDIDATE else "d" * 64
        ),
        plan_sha256=plan_sha256 if kind is DirectBaselineKind.CANDIDATE else "e" * 64,
        tokenizer_sha256="f" * 64,
        prompt_sha256="1" * 64,
        ordered_prompt_ids_sha256="2" * 64,
        random_seed=7,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=64,
        runtime_version="test-direct-v1",
        runtime_executable_sha256="3" * 64,
        runtime_environment={"AX_ENGINE_TEST": "1"},
        hardware_scope_id=_HARDWARE_SCOPE,
        os_version="macOS-test",
        power_mode="AC power",
        background_policy="idle-sequential",
        trials=trials,
        mlx_lm_parity_tokens=128,
        mlx_lm_matching_tokens=128,
    )


def _coding_payload(
    task_id: str,
    category: str,
    language: str,
    scorer: CodingScorer,
) -> CodingTaskPayload:
    test_path = "test.txt" if scorer is CodingScorer.UNIT_TEST else None
    return CodingTaskPayload(
        task_id=task_id,
        category=category,
        language=language,
        scorer=scorer,
        prompt=f"prompt-{task_id}",
        reference=f"reference-{task_id}",
        candidate_path="candidate.txt",
        test_path=test_path,
        fixture_files=({"test.txt": "fixture"} if test_path is not None else {}),
        expected_json=(
            {"name": "fixture", "arguments": {}} if scorer is CodingScorer.TOOL_EXACT else None
        ),
        expected_text=(f"reference-{task_id}" if scorer is CodingScorer.TEXT_EXACT else None),
        target_tokens=200,
    )


def _coding_tasks() -> list[CodingTaskManifest]:
    quotas = {
        "python": (24, "python"),
        "javascript-typescript": (20, "typescript"),
        "rust": (16, "rust"),
        "go": (16, "go"),
        "repository-context": (16, "python"),
        "json-tool": (16, "json"),
        "algorithm-reasoning": (12, "python"),
        "long-context": (8, "text"),
    }
    tasks: list[CodingTaskManifest] = []
    for category, (count, language) in quotas.items():
        for index in range(count):
            task_id = f"{category}-{index:03d}"
            scorer = (
                CodingScorer.TOOL_EXACT
                if category == "json-tool"
                else (
                    CodingScorer.TEXT_EXACT
                    if category == "long-context"
                    else CodingScorer.UNIT_TEST
                )
            )
            payload = _coding_payload(task_id, category, language, scorer)
            tasks.append(
                CodingTaskManifest(
                    task_id=task_id,
                    category=category,
                    language=language,
                    prompt_sha256=hashlib.sha256(f"prompt-{task_id}".encode()).hexdigest(),
                    reference_sha256=hashlib.sha256(f"reference-{task_id}".encode()).hexdigest(),
                    payload_sha256=stable_sha256(payload),
                    scorer=scorer,
                    license_id="CC0-1.0",
                    provenance="clean-room-authored fixture",
                    target_tokens=200,
                    timeout_seconds=5.0,
                    cpu_time_seconds=4,
                    memory_limit_bytes=128 * 1024**2,
                    process_limit=8,
                    output_limit_bytes=64 * 1024,
                    file_size_limit_bytes=64 * 1024**2,
                    open_file_limit=128,
                    long_context=category == "long-context",
                )
            )
    return tasks


def _quality_evaluation(
    *,
    profile: ProfileName,
    model: ModelIdentity,
    model_artifact_sha256: str,
    evaluation_manifest_sha256: str,
    dataset_sha256: str,
    task_ids: list[str],
    tokenizer_sha256: str,
    coding_tasks: dict[str, CodingTaskManifest] | None = None,
    toolchains: dict[str, str] | None = None,
    raw_log_prefix: str | None = None,
) -> DirectQualityEvaluation:
    task_lookup = coding_tasks or {}
    toolchain_lookup = toolchains or {}
    toolchain_by_language = {
        "python": "python",
        "javascript": "node",
        "typescript": "typescript",
        "rust": "rust",
        "go": "go",
    }

    def outcome(task_id: str) -> DirectQualityTaskOutcome:
        task = task_lookup.get(task_id)
        scorer = task.scorer if task is not None else None
        executable = scorer in {CodingScorer.UNIT_TEST, CodingScorer.COMPILE}
        toolchain_key = toolchain_by_language.get(task.language) if task is not None else None
        return DirectQualityTaskOutcome(
            task_id=task_id,
            score=1.0,
            scored_tokens=task.target_tokens if task is not None else 200,
            scorer=scorer,
            syntax_valid=(True if executable else None),
            tool_valid=(
                True if scorer in {CodingScorer.TOOL_EXACT, CodingScorer.JSON_SCHEMA} else None
            ),
            unit_tests_passed=(True if scorer is CodingScorer.UNIT_TEST else None),
            output_file=(
                f"{raw_log_prefix}/{task_id}.model-output.txt"
                if raw_log_prefix is not None
                else None
            ),
            output_sha256=hashlib.sha256(f"output-{task_id}".encode()).hexdigest(),
            sandboxed=(True if executable else None),
            network_disabled=(True if executable else None),
            timed_out=(False if executable else None),
            exit_code=(0 if executable else None),
            duration_seconds=(0.1 if executable else None),
            stdout_file=(
                f"{raw_log_prefix}/{task_id}.stdout.txt"
                if executable and raw_log_prefix is not None
                else None
            ),
            stderr_file=(
                f"{raw_log_prefix}/{task_id}.stderr.txt"
                if executable and raw_log_prefix is not None
                else None
            ),
            stdout_sha256=(hashlib.sha256(b"").hexdigest() if executable else None),
            stderr_sha256=(hashlib.sha256(b"").hexdigest() if executable else None),
            toolchain=(toolchain_lookup.get(toolchain_key) if toolchain_key else None),
            sandbox_profile_sha256=(SANDBOX_PROFILE_SHA256 if executable else None),
        )

    return DirectQualityEvaluation(
        profile=profile,
        model=model,
        model_artifact_sha256=model_artifact_sha256,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        dataset_sha256=dataset_sha256,
        tokenizer_sha256=tokenizer_sha256,
        generation={
            "prompt_format": "raw",
            "thinking_enabled": False,
            "max_sequence_length": 4096,
            "max_generation_tokens": 200,
        },
        random_seed=7,
        evaluated_tokens=4096,
        software_versions=_versions(),
        perplexity=5.0,
        outcomes=[outcome(task_id) for task_id in task_ids],
    )


def _build_inputs(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    config = {
        "model_type": "qwen3_next",
        "architectures": ["Qwen3NextForCausalLM"],
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "full_attention_interval": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 512,
    }
    write_data(source_dir / "config.json", config)
    write_data(source_dir / "tokenizer.json", {"version": "fixture"})
    (source_dir / "model.safetensors").write_bytes(b"s" * 2000)
    source = ModelIdentity(
        model_id=_SOURCE_ID,
        revision=_SOURCE_REVISION,
        architecture="Qwen3NextForCausalLM",
        local_path=str(source_dir),
    )
    write_data(
        source_dir / "axquant_source.json",
        SourceConversionProvenance(
            source_model=_SOURCE_ID,
            source_revision=_SOURCE_REVISION,
            key_remap_applied=False,
        ),
    )
    tensors = _tensors()
    inventory = Inventory(
        model=source,
        tensors=tensors,
        total_parameters=1000,
        quantizable_parameters=sum(tensor.parameters for tensor in tensors if tensor.quantizable),
        weight_bytes=2000,
        precision_parameters={"bfloat16": 1000},
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        architecture_profile=_profile(),
        config_sha256=file_sha256(source_dir / "config.json"),
    )
    inventory_path = tmp_path / "source-inventory.json"
    write_data(inventory_path, inventory)
    source_manifest = build_source_checkpoint_manifest(source_dir, inventory=inventory)
    source_manifest_path = tmp_path / "source-checkpoint-manifest.json"
    write_data(source_manifest_path, source_manifest)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"q" * 600)
    write_data(artifact / "model-manifest.json", {"schema": "fixture"})

    calibration_manifest = CalibrationManifest(
        model=source,
        profile=ProfileName.AGENT_CODING,
        dataset_id="clean-room-calibration-v1",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "reasoning"],
        sequence_length=2048,
        random_seed=7,
        calibration_evaluation_separation_attested=True,
    )
    calibration_path = artifact / "calibration_manifest.json"
    write_data(calibration_path, calibration_manifest)
    calibration = CalibrationEvidence(
        dataset_id=calibration_manifest.dataset_id,
        dataset_sha256=calibration_manifest.dataset_sha256,
        samples=calibration_manifest.samples,
        domains=calibration_manifest.domains,
        sequence_length=calibration_manifest.sequence_length,
        backend="mlx-probe-formal-v1",
        reference="calibration_manifest.json",
        metadata={"calibration_manifest_sha256": calibration_manifest_sha256(calibration_manifest)},
    )
    assignments = [_assignment(tensor) for tensor in tensors]
    entries = [
        TensorSensitivity(
            tensor=tensor,
            candidates=[
                CandidateMeasurement(
                    bits=assignment.bits,
                    method=assignment.method,
                    group_size=assignment.group_size,
                    metrics=MetricVector(output_kl=0.001),
                    measured_tokens=8192,
                    evidence_scope="preserved" if assignment.bits == 16 else "tensor",
                )
            ],
        )
        for tensor, assignment in zip(tensors, assignments, strict=True)
    ]
    sensitivity = SensitivityReport(
        model=source,
        architecture_profile=_profile(),
        profile=ProfileName.AGENT_CODING,
        evidence_kind=EvidenceKind.MEASURED,
        inventory_sha256=stable_sha256(inventory),
        entries=entries,
        calibration=calibration,
    )
    sensitivity_path = tmp_path / "sensitivity.json"
    write_data(sensitivity_path, sensitivity)
    distribution = {
        "4bit": PrecisionShare(parameters=720, fraction=0.72),
        "6bit": PrecisionShare(parameters=240, fraction=0.24),
        "8bit": PrecisionShare(parameters=20, fraction=0.02),
        "bf16": PrecisionShare(parameters=20, fraction=0.02),
    }
    plan = QuantizationPlan(
        source_model=source,
        architecture_profile=_profile(),
        profile=ProfileName.AGENT_CODING,
        target_class="mixed-4.8bpw",
        target_bpw=4.8,
        nominal_bpw=4.0,
        effective_bpw=4.8,
        candidate_bits=(4, 6, 8, 16),
        group_size=64,
        objective=ObjectiveWeights(
            output_kl=1.0,
            hidden_state_error=0.0,
            cosine_distance=0.0,
            token_disagreement=0.0,
            task_loss_delta=0.0,
            mtp_acceptance_loss=0.0,
            long_context_loss=0.0,
            peak_memory_cost=0.0,
            prefill_latency_cost=0.0,
            decode_latency_cost=0.0,
        ),
        hardware=HardwareProfile(),
        mtp=MtpPolicy(mode="disabled", candidate_bits=(16,), min_bits=16),
        constraints=PlanningConstraints(effective_bpw_limit=4.8),
        target_mode="balanced",
        random_seed=7,
        software_versions=_versions(),
        analysis_sha256=stable_sha256(sensitivity),
        evidence_kind=EvidenceKind.MEASURED,
        calibration=calibration,
        assignments=assignments,
        weight_distribution=distribution,
        mtp_distribution={},
    )
    plan_path = artifact / "axquant_plan.json"
    write_data(plan_path, plan)
    runtime = build_runtime_metadata(plan, artifact)
    manifest = ArtifactManifest(
        axquant_version="1.2.0",
        source_model=source,
        plan_sha256=stable_sha256(plan),
        calibration=calibration,
        profile=plan.profile,
        target_class=plan.target_class,
        effective_bpw=4.8,
        logical_parameters=1000,
        main_logical_parameters=1000,
        weight_file_size_bytes=600,
        main_weight_file_size_bytes=600,
        mtp_weight_file_size_bytes=0,
        protected_weight_file_size_bytes=40,
        measured_total_bpw=4.8,
        measured_main_bpw=4.8,
        weight_distribution=distribution,
        mtp_distribution={},
        mtp_present=False,
        mtp_policy=plan.mtp,
        runtime=runtime,
        software_versions=_versions(),
        files=[
            ArtifactFile(
                path="model.safetensors",
                size_bytes=600,
                sha256=file_sha256(artifact / "model.safetensors"),
            )
        ],
    )
    manifest_path = artifact / "axquant_manifest.json"
    write_data(manifest_path, manifest)

    feasibility = FeasibilityReport(
        status="ready-for-conversion",
        source=_baseline(BaselineKind.BF16_SOURCE, source, weight_bytes=2000, bpw=16.0),
        baselines=[
            _baseline(BaselineKind.UNIFORM_4BIT, source, weight_bytes=500, bpw=4.0),
            _baseline(BaselineKind.UNIFORM_6BIT, source, weight_bytes=750, bpw=6.0),
            _baseline(BaselineKind.MIXED_PRECISION, source, weight_bytes=600, bpw=4.8),
        ],
        checks={"immutable_source": True, "non_mtp": True},
    )
    feasibility_path = tmp_path / "feasibility.json"
    write_data(feasibility_path, feasibility)

    candidate = ModelIdentity(
        model_id=_CANDIDATE_ID,
        revision=_CANDIDATE_REVISION,
        architecture="Qwen3NextForCausalLM",
        local_path=str(artifact),
    )
    runtime_paths: dict[str, Path] = {}
    for name, runtime_name, check_kind in (
        ("ax-manifest", RuntimeName.AX_ENGINE, "manifest"),
        ("ax-doctor", RuntimeName.AX_ENGINE, "doctor"),
        ("ax-runtime", RuntimeName.AX_ENGINE, "generation-smoke"),
        ("mlx-runtime", RuntimeName.MLX_LM, "generation-smoke"),
    ):
        path = tmp_path / f"{name}.json"
        write_data(
            path,
            RuntimeCheck(
                model=candidate,
                runtime=runtime_name,
                check_kind=check_kind,
                available=True,
                passed=True,
                exit_code=0,
                report={"kernel_fallbacks": 0, "bringup_allowed": True},
            ),
        )
        runtime_paths[name] = path

    benchmark = DirectBenchmarkEvidenceIndex(
        profiles=[
            DirectBenchmarkProfile(
                profile=profile,
                arms=[
                    _benchmark_arm(
                        kind,
                        candidate=candidate,
                        manifest_sha256=file_sha256(manifest_path),
                        plan_sha256=stable_sha256(plan),
                        profile=profile,
                    )
                    for kind in DirectBaselineKind
                ],
            )
            for profile in (ProfileName.AGENT_CODING, ProfileName.GENERAL)
        ]
    )
    benchmark_path = tmp_path / "benchmark.json"
    write_data(benchmark_path, benchmark)

    tasks = _coding_tasks()
    payloads = [
        _coding_payload(task.task_id, task.category, task.language, task.scorer) for task in tasks
    ]
    shard_path = tmp_path / "coding-tasks.jsonl"
    shard_path.write_text(
        "\n".join(payload.model_dump_json() for payload in payloads) + "\n",
        encoding="utf-8",
    )
    shard_bindings = {shard_path.name: file_sha256(shard_path)}
    overlap = CodingOverlapReport(
        suite_dataset_sha256=stable_sha256(shard_bindings),
        calibration_dataset_sha256=calibration.dataset_sha256,
        similarity_threshold=NEAR_DUPLICATE_THRESHOLD,
        passed=True,
    )
    overlap_path = tmp_path / "coding-overlap.json"
    write_data(overlap_path, overlap)
    toolchains = {
        "python": "python fixture",
        "node": "node fixture",
        "typescript": "typescript fixture",
        "rust": "rust fixture",
        "go": "go fixture",
        "sandbox": "seatbelt fixture",
    }
    coding = CodingSuiteManifest(
        suite_id="axquant-coding-suite-v2-fixture",
        version="2.0.0",
        dataset_sha256=stable_sha256(shard_bindings),
        tasks=tasks,
        task_shards=shard_bindings,
        calibration_overlap_attested=True,
        calibration_overlap_report=overlap_path.name,
        calibration_overlap_report_sha256=file_sha256(overlap_path),
        toolchains=toolchains,
        sandbox_profile_sha256=SANDBOX_PROFILE_SHA256,
        near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
        random_seed=7,
    )
    coding_path = tmp_path / "coding-suite.json"
    write_data(coding_path, coding)

    coding_tasks = {task.task_id: task for task in tasks}
    oracle_evaluation = _quality_evaluation(
        profile=ProfileName.AGENT_CODING,
        model=source,
        model_artifact_sha256=file_sha256(source_manifest_path),
        evaluation_manifest_sha256=file_sha256(coding_path),
        dataset_sha256=coding.dataset_sha256,
        task_ids=[task.task_id for task in tasks],
        tokenizer_sha256=source_manifest.tokenizer_sha256,
        coding_tasks=coding_tasks,
        toolchains=toolchains,
        raw_log_prefix="coding-self-test-oracle",
    )
    mutant_evaluation = _quality_evaluation(
        profile=ProfileName.AGENT_CODING,
        model=source,
        model_artifact_sha256=file_sha256(source_manifest_path),
        evaluation_manifest_sha256=file_sha256(coding_path),
        dataset_sha256=coding.dataset_sha256,
        task_ids=[task.task_id for task in tasks],
        tokenizer_sha256=source_manifest.tokenizer_sha256,
        coding_tasks=coding_tasks,
        toolchains=toolchains,
        raw_log_prefix="coding-self-test-mutant",
    )
    for outcome in mutant_evaluation.outcomes:
        outcome.score = 0.0
        outcome.syntax_valid = False if outcome.syntax_valid is not None else None
        outcome.tool_valid = False if outcome.tool_valid is not None else None
        outcome.unit_tests_passed = False if outcome.unit_tests_passed is not None else None
        outcome.output_sha256 = hashlib.sha256(b"").hexdigest()
    for evaluation, output_contents in (
        (oracle_evaluation, lambda task_id: f"output-{task_id}"),
        (mutant_evaluation, lambda _task_id: ""),
    ):
        for outcome in evaluation.outcomes:
            assert outcome.output_file is not None
            output_log = tmp_path / outcome.output_file
            output_log.parent.mkdir(parents=True, exist_ok=True)
            output_log.write_text(output_contents(outcome.task_id), encoding="utf-8")
            for relative_log in (outcome.stdout_file, outcome.stderr_file):
                if relative_log is not None:
                    log_path = tmp_path / relative_log
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_bytes(b"")
    coding_self_test = CodingSuiteSelfTestReport(
        suite_manifest_sha256=file_sha256(coding_path),
        toolchains=toolchains,
        sandbox_profile_sha256=SANDBOX_PROFILE_SHA256,
        oracle_outcomes=oracle_evaluation.outcomes,
        empty_mutant_outcomes=mutant_evaluation.outcomes,
        passed=True,
    )
    coding_self_test_path = tmp_path / "coding-suite-self-test.json"
    write_data(coding_self_test_path, coding_self_test)
    coding_self_test_dependency_paths = [
        tmp_path / relative_path
        for outcome in [
            *coding_self_test.oracle_outcomes,
            *coding_self_test.empty_mutant_outcomes,
        ]
        for relative_path in (outcome.output_file, outcome.stdout_file, outcome.stderr_file)
        if relative_path is not None
    ]

    general_manifest_path = tmp_path / "general-suite.jsonl"
    general_tasks = [
        QualityTask(
            task_id=f"general-{index:03d}",
            category="reasoning" if index < 8 else "instruction",
            prompt=f"Return marker {index}.",
            reference=str(index),
            checks=[QualityCheck(kind="exact", value=str(index))],
        )
        for index in range(16)
    ]
    general_manifest_path.write_text(
        "\n".join(task.model_dump_json() for task in general_tasks) + "\n",
        encoding="utf-8",
    )
    general_overlap = CodingOverlapReport(
        suite_dataset_sha256=file_sha256(general_manifest_path),
        calibration_dataset_sha256=calibration.dataset_sha256,
        similarity_threshold=NEAR_DUPLICATE_THRESHOLD,
        passed=True,
    )
    general_overlap_path = tmp_path / "general-overlap.json"
    write_data(general_overlap_path, general_overlap)
    validation_entries: list[DirectValidationEntry] = []
    quality_dependency_paths: list[Path] = [
        general_manifest_path,
        general_overlap_path,
        coding_path,
    ]
    for profile, dataset, task_ids, evaluation_manifest_path in (
        (
            ProfileName.AGENT_CODING,
            coding.dataset_sha256,
            [task.task_id for task in tasks],
            coding_path,
        ),
        (
            ProfileName.GENERAL,
            file_sha256(general_manifest_path),
            [f"general-{index:03d}" for index in range(16)],
            general_manifest_path,
        ),
    ):
        reference_log_prefix = f"{profile.value}-bf16-logs"
        candidate_log_prefix = f"{profile.value}-candidate-logs"
        reference = _quality_evaluation(
            profile=profile,
            model=source,
            model_artifact_sha256=file_sha256(source_manifest_path),
            evaluation_manifest_sha256=file_sha256(evaluation_manifest_path),
            dataset_sha256=dataset,
            task_ids=task_ids,
            tokenizer_sha256=source_manifest.tokenizer_sha256,
            coding_tasks=(coding_tasks if profile is ProfileName.AGENT_CODING else None),
            toolchains=toolchains,
            raw_log_prefix=reference_log_prefix,
        )
        candidate_evaluation = _quality_evaluation(
            profile=profile,
            model=candidate,
            model_artifact_sha256=file_sha256(manifest_path),
            evaluation_manifest_sha256=file_sha256(evaluation_manifest_path),
            dataset_sha256=dataset,
            task_ids=task_ids,
            tokenizer_sha256=source_manifest.tokenizer_sha256,
            coding_tasks=(coding_tasks if profile is ProfileName.AGENT_CODING else None),
            toolchains=toolchains,
            raw_log_prefix=candidate_log_prefix,
        )
        for evaluation in (reference, candidate_evaluation):
            for outcome in evaluation.outcomes:
                if outcome.output_file is not None:
                    output_log = tmp_path / outcome.output_file
                    output_log.parent.mkdir(parents=True, exist_ok=True)
                    output_log.write_text(f"output-{outcome.task_id}", encoding="utf-8")
                for relative_log in (outcome.stdout_file, outcome.stderr_file):
                    if relative_log is not None:
                        log_path = tmp_path / relative_log
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_bytes(b"")
        reference_path = tmp_path / f"{profile.value}-bf16-quality.json"
        candidate_path = tmp_path / f"{profile.value}-candidate-quality.json"
        write_data(reference_path, reference)
        write_data(candidate_path, candidate_evaluation)
        quality_dependency_paths.extend((reference_path, candidate_path))
        quality_dependency_paths.extend(
            tmp_path / relative_path
            for evaluation in (reference, candidate_evaluation)
            for outcome in evaluation.outcomes
            for relative_path in (outcome.output_file, outcome.stdout_file, outcome.stderr_file)
            if relative_path is not None
        )
        validation_entries.append(
            DirectValidationEntry(
                profile=profile,
                evaluation_manifest_file=evaluation_manifest_path.name,
                evaluation_manifest_sha256=file_sha256(evaluation_manifest_path),
                reference_evaluation_file=reference_path.name,
                reference_evaluation_sha256=file_sha256(reference_path),
                candidate_evaluation_file=candidate_path.name,
                candidate_evaluation_sha256=file_sha256(candidate_path),
                passed=True,
            )
        )
    validation = DirectReleaseValidationIndex(
        entries=validation_entries,
        general_calibration_overlap_report_file=general_overlap_path.name,
        general_calibration_overlap_report_sha256=file_sha256(general_overlap_path),
        release_ready=True,
    )
    validation_path = tmp_path / "validation.json"
    write_data(validation_path, validation)

    parent_plan = plan.model_copy(
        update={
            "target_class": "parent-4bit",
            "target_bpw": 4.9,
            "effective_bpw": 4.9,
            "constraints": plan.constraints.model_copy(update={"effective_bpw_limit": 4.9}),
        }
    )
    parent_sha = stable_sha256(parent_plan)
    plan_sha = stable_sha256(plan)
    refinement = RefinementResult(
        config=RefinementConfig(random_seed=7),
        history=[
            CandidateEntry(
                candidate_id="parent",
                plan_sha256=parent_sha,
                change_description="uniform parent",
                reason="matched control",
                predicted_bpw=4.9,
                measured_bpw=4.9,
                predicted_loss=0.2,
                measured_loss=0.2,
                budget_impact=0.0,
                state="rejected",
            ),
            CandidateEntry(
                candidate_id="candidate",
                parent_id="parent",
                plan_sha256=plan_sha,
                change_description="measured mixed precision",
                reason="lower objective loss",
                predicted_bpw=4.8,
                measured_bpw=4.8,
                predicted_loss=0.1,
                measured_loss=0.1,
                budget_impact=-0.1,
                state="selected",
            ),
        ],
        candidate_plans={"parent": parent_plan, "candidate": plan},
        selected_candidate_id="candidate",
        selected_plan=plan,
        selected_plan_sha256=plan_sha,
        selection_basis="complete-model",
        evidence_label="holdout-bound",
        iterations_used=1,
        evaluations_used=2,
        converged=True,
    )
    refinement_path = tmp_path / "refinement.json"
    write_data(refinement_path, refinement)
    measurement_common = {
        "target_class": "4bit",
        "candidate_model": candidate,
        "quality_evidence_sha256": file_sha256(validation_path),
        "benchmark_evidence_sha256": file_sha256(benchmark_path),
    }
    measurements = DirectRefinementMeasurementSet(
        refinement_sha256=stable_sha256(refinement),
        evaluator_version=f"axquant:{direct_policy().policy_id}",
        selected_candidate_id="candidate",
        measurements=[
            DirectRefinementMeasurement(
                measurement_id="parent-measurement",
                candidate_id="parent",
                plan_sha256=parent_sha,
                artifact_manifest_sha256=file_sha256(manifest_path),
                measured_bpw=4.9,
                objective_loss=0.2,
                quality_retention=0.99,
                decode_tokens_per_second=12.0,
                peak_memory_bytes=700,
                validation_passed=True,
                **measurement_common,
            ),
            DirectRefinementMeasurement(
                measurement_id="candidate-measurement",
                candidate_id="candidate",
                parent_candidate_id="parent",
                plan_sha256=plan_sha,
                artifact_manifest_sha256=file_sha256(manifest_path),
                measured_bpw=4.8,
                objective_loss=0.1,
                quality_retention=1.0,
                decode_tokens_per_second=12.5,
                peak_memory_bytes=600,
                validation_passed=True,
                **measurement_common,
            ),
        ],
    )
    measurements_path = tmp_path / "measurements.json"
    write_data(measurements_path, measurements)
    pareto_path = tmp_path / "pareto.json"
    write_data(pareto_path, build_direct_pareto(measurements))

    hardware = DirectHardwareProfileRegistry(
        registry_id="qwen-next-m2-ultra",
        entries=[
            DirectHardwareRegistryEntry(
                entry_id="qwen-next-4bit-m2-ultra",
                hardware_scope_id=_HARDWARE_SCOPE,
                candidate_id="candidate",
                artifact_manifest_sha256=file_sha256(manifest_path),
                benchmark_evidence_sha256=file_sha256(benchmark_path),
                device_name="Mac14,13",
                chip=direct_policy().formal_hardware_chip,
                unified_memory_bytes=direct_policy().formal_hardware_memory_bytes,
                os_version="macOS-test",
                ax_engine_version="test-direct-v1",
                ax_engine_executable_sha256="3" * 64,
                metal_version="32023.98",
                metallib_version="32023.98",
                power_mode="AC power",
                doctor_passed=True,
                kernel_fallbacks=0,
                release_ready=True,
            )
        ],
        release_ready=True,
    )
    hardware_path = tmp_path / "hardware.json"
    write_data(hardware_path, hardware)

    compatibility_request = Qwen3NextCompatibilityRequest(
        source_model=source,
        target_class="4bit",
    )
    compatibility_request_path = tmp_path / "compatibility-request.json"
    write_data(compatibility_request_path, compatibility_request)
    compatibility = Qwen3NextCompatibilityMatrix(
        source_model=source,
        target_class="4bit",
        artifact_manifest_sha256=file_sha256(manifest_path),
        profiles_passed=[ProfileName.AGENT_CODING, ProfileName.GENERAL],
        runtimes_passed=[RuntimeName.AX_ENGINE, RuntimeName.MLX_LM],
        release_ready=True,
    )
    compatibility_path = tmp_path / "compatibility.json"
    write_data(compatibility_path, compatibility)

    recipe = ReproductionRecipe(
        source_model=source,
        calibration=calibration,
        axquant_version="1.2.0",
        software_versions=_versions(),
        random_seed=7,
        profile=plan.profile,
        primary_runtime=RuntimeName.AX_ENGINE,
        plan_sha256=plan_sha,
        output_repository=_CANDIDATE_ID,
        plan_file="artifact/axquant_plan.json",
        plan_file_sha256=file_sha256(plan_path),
        calibration_file="artifact/calibration_manifest.json",
        calibration_file_sha256=file_sha256(calibration_path),
        conversion_manifest_file="artifact/axquant_manifest.json",
        conversion_manifest_sha256=file_sha256(manifest_path),
        expected_logical_parameters=1000,
        expected_weight_file_size_bytes=600,
        expected_weight_files=manifest.files,
        commands=[
            ReproductionCommand(
                step_id=step_id,
                description=step_id,
                argv=["true"],
            )
            for step_id in ("download-source", "convert", "verify-reproduction")
        ],
    )
    recipe_path = tmp_path / "recipe.json"
    write_data(recipe_path, recipe)
    from axquant.reproduction import verify_reproduction

    reproduction_path = tmp_path / "reproduction.json"
    write_data(
        reproduction_path,
        verify_reproduction(recipe_path=recipe_path, artifact_dir=artifact),
    )

    wheel_path = tmp_path / "axquant-1.2.0-py3-none-any.whl"
    _write_wheel(wheel_path)
    fingerprint = architecture_fingerprint(source_dir, inventory=inventory)

    evidence_paths = [
        inventory_path,
        source_manifest_path,
        feasibility_path,
        sensitivity_path,
        refinement_path,
        measurements_path,
        validation_path,
        general_manifest_path,
        benchmark_path,
        coding_path,
        coding_self_test_path,
        hardware_path,
        pareto_path,
        compatibility_path,
        compatibility_request_path,
        recipe_path,
        reproduction_path,
        *runtime_paths.values(),
        manifest_path,
        plan_path,
        overlap_path,
        *[tmp_path / name for name in coding.task_shards],
        *coding_self_test_dependency_paths,
        *quality_dependency_paths,
    ]
    evidence_paths = list(dict.fromkeys(evidence_paths))
    archive = EvidenceArchiveIndex(
        records=[
            EvidenceArchiveRecord(
                logical_name=f"evidence-{index:03d}-{path.name}",
                path=path.relative_to(tmp_path).as_posix(),
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
                durable_uri=f"/Volumes/axquant-evidence/{index:03d}/{path.name}",
            )
            for index, path in enumerate(evidence_paths)
        ],
        complete=True,
    )
    archive_path = tmp_path / "archive.json"
    write_data(archive_path, archive)

    request = Qwen3NextReleaseAuditRequest(
        certification_scope={
            "source_model": source.model_dump(mode="json"),
            "architecture": fingerprint.model_dump(mode="json"),
            "target_class": "4bit",
            "artifact_manifest_sha256": file_sha256(manifest_path),
            "hardware_scope_ids": [_HARDWARE_SCOPE],
        },
        artifact_directory="artifact",
        source_inventory=inventory_path.name,
        source_checkpoint_manifest=source_manifest_path.name,
        feasibility_report=feasibility_path.name,
        sensitivity_report=sensitivity_path.name,
        refinement_result=refinement_path.name,
        refinement_measurements=measurements_path.name,
        release_validation_index=validation_path.name,
        benchmark_evidence_index=benchmark_path.name,
        coding_suite_manifest=coding_path.name,
        coding_suite_self_test=coding_self_test_path.name,
        hardware_registry=hardware_path.name,
        pareto_report=pareto_path.name,
        compatibility_matrix=compatibility_path.name,
        compatibility_request=compatibility_request_path.name,
        reproduction_recipe=recipe_path.name,
        reproduction_verification=reproduction_path.name,
        ax_engine_manifest_check=runtime_paths["ax-manifest"].name,
        ax_engine_doctor_check=runtime_paths["ax-doctor"].name,
        ax_engine_runtime_check=runtime_paths["ax-runtime"].name,
        mlx_lm_runtime_check=runtime_paths["mlx-runtime"].name,
        evidence_archive_index=archive_path.name,
        toolkit_wheel=wheel_path.name,
        required_toolkit_version="1.2.0",
        policy_sha256=direct_policy_sha256(),
    )
    request_path = tmp_path / "request.json"
    write_data(request_path, request)
    prepare_direct_publication(
        model_dir=artifact,
        repo_id=_CANDIDATE_ID,
        request_path=request_path,
    )
    return request_path


def test_complete_synthetic_non_mtp_audit_passes_n0_through_n8(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    audit = build_qwen3_next_release_audit(request_path)

    assert audit.release_ready
    assert [check.gate_id.value for check in audit.checks] == [f"N{index}" for index in range(9)]
    assert audit.blockers == []


def test_direct_validation_builder_recomputes_passing_raw_profiles(tmp_path: Path) -> None:
    release_request_path = _build_inputs(tmp_path)
    release_request = Qwen3NextReleaseAuditRequest.model_validate_json(
        release_request_path.read_text(encoding="utf-8")
    )
    validation = DirectReleaseValidationIndex.model_validate_json(
        (tmp_path / release_request.release_validation_index).read_text(encoding="utf-8")
    )
    sensitivity = SensitivityReport.model_validate_json(
        (tmp_path / release_request.sensitivity_report).read_text(encoding="utf-8")
    )
    assert sensitivity.calibration is not None
    direct_request = DirectReleaseValidationRequest(
        source_checkpoint_manifest=release_request.source_checkpoint_manifest,
        candidate_artifact_manifest=(f"{release_request.artifact_directory}/axquant_manifest.json"),
        calibration_dataset_sha256=sensitivity.calibration.dataset_sha256,
        coding_suite_manifest=release_request.coding_suite_manifest,
        general_calibration_overlap_report=(validation.general_calibration_overlap_report_file),
        required_toolkit_version=release_request.required_toolkit_version,
        policy_sha256=release_request.policy_sha256,
        entries=[
            DirectValidationRequestEntry(
                profile=entry.profile,
                evaluation_manifest_file=entry.evaluation_manifest_file,
                reference_evaluation_file=entry.reference_evaluation_file,
                candidate_evaluation_file=entry.candidate_evaluation_file,
            )
            for entry in validation.entries
        ],
    )
    direct_request_path = tmp_path / "direct-validation-request.json"
    write_data(direct_request_path, direct_request)

    rebuilt = build_direct_release_validation_index(direct_request_path)

    assert rebuilt.release_ready
    assert rebuilt.issues == []
    assert all(entry.passed for entry in rebuilt.entries)


def test_benchmark_fallback_tamper_fails_n2(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    benchmark_path = tmp_path / request.benchmark_evidence_index
    benchmark = DirectBenchmarkEvidenceIndex.model_validate_json(
        benchmark_path.read_text(encoding="utf-8")
    )
    trial = benchmark.profiles[0].arms[0].trials[0]
    trial.kernel_fallbacks = 1
    write_data(benchmark_path, benchmark)

    audit = build_qwen3_next_release_audit(request_path)

    n2 = next(check for check in audit.checks if check.gate_id.value == "N2")
    assert not n2.passed
    assert any("kernel fallback" in issue for issue in n2.issues)


def test_mtp_contradiction_is_track_eligibility_error(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    inventory_path = tmp_path / request.source_inventory
    inventory = Inventory.model_validate_json(inventory_path.read_text(encoding="utf-8"))
    inventory.mtp_present = True
    write_data(inventory_path, inventory)

    with pytest.raises(ArtifactError, match="eligibility failed"):
        build_qwen3_next_release_audit(request_path)


def _gate(audit: object, gate_id: str):
    return next(check for check in audit.checks if check.gate_id.value == gate_id)


def test_source_checksum_tamper_fails_n0(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    (tmp_path / "source" / "model.safetensors").write_bytes(b"changed")

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N0").passed


def test_untracked_source_weight_file_fails_n0(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    (tmp_path / "source" / "untracked.safetensors").write_bytes(b"untracked")

    audit = build_qwen3_next_release_audit(request_path)

    n0 = _gate(audit, "N0")
    assert not n0.passed
    assert any("Safetensors membership differs" in issue for issue in n0.issues)


def test_artifact_checksum_tamper_fails_n1(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    (tmp_path / "artifact" / "model.safetensors").write_bytes(b"changed")

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N1").passed


def test_sensitivity_token_tamper_fails_n3(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    path = tmp_path / request.sensitivity_report
    report = SensitivityReport.model_validate_json(path.read_text(encoding="utf-8"))
    report.entries[0].candidates[0].measured_tokens = 1
    write_data(path, report)

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N3").passed


def test_raw_quality_error_fails_n4(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    index_path = tmp_path / request.release_validation_index
    index = DirectReleaseValidationIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    entry = next(item for item in index.entries if item.profile is ProfileName.AGENT_CODING)
    evaluation_path = tmp_path / entry.candidate_evaluation_file
    evaluation = DirectQualityEvaluation.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    evaluation.outcomes[0].model_error = True
    write_data(evaluation_path, evaluation)
    entry.candidate_evaluation_sha256 = file_sha256(evaluation_path)
    write_data(index_path, index)

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N4").passed


def test_wrong_candidate_artifact_binding_fails_n4(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    index_path = tmp_path / request.release_validation_index
    index = DirectReleaseValidationIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    entry = next(item for item in index.entries if item.profile is ProfileName.AGENT_CODING)
    evaluation_path = tmp_path / entry.candidate_evaluation_file
    evaluation = DirectQualityEvaluation.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    evaluation.model_artifact_sha256 = "f" * 64
    write_data(evaluation_path, evaluation)
    entry.candidate_evaluation_sha256 = file_sha256(evaluation_path)
    write_data(index_path, index)

    audit = build_qwen3_next_release_audit(request_path)

    n4 = _gate(audit, "N4")
    assert not n4.passed
    assert any("wrong candidate artifact" in issue for issue in n4.issues)


def test_tampered_raw_model_output_is_rejected_before_n4(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    index = DirectReleaseValidationIndex.model_validate_json(
        (tmp_path / request.release_validation_index).read_text(encoding="utf-8")
    )
    entry = next(item for item in index.entries if item.profile is ProfileName.AGENT_CODING)
    evaluation = DirectQualityEvaluation.model_validate_json(
        (tmp_path / entry.candidate_evaluation_file).read_text(encoding="utf-8")
    )
    output_file = evaluation.outcomes[0].output_file
    assert output_file is not None
    (tmp_path / output_file).write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"model output.*checksum"):
        build_qwen3_next_release_audit(request_path)


def test_missing_general_raw_model_output_fails_n4(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    index_path = tmp_path / request.release_validation_index
    index = DirectReleaseValidationIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    entry = next(item for item in index.entries if item.profile is ProfileName.GENERAL)
    evaluation_path = tmp_path / entry.candidate_evaluation_file
    evaluation = DirectQualityEvaluation.model_validate_json(
        evaluation_path.read_text(encoding="utf-8")
    )
    evaluation.outcomes[0].output_file = None
    write_data(evaluation_path, evaluation)
    entry.candidate_evaluation_sha256 = file_sha256(evaluation_path)
    write_data(index_path, index)

    audit = build_qwen3_next_release_audit(request_path)

    n4 = _gate(audit, "N4")
    assert not n4.passed
    assert any("candidate general output is not archived" in issue for issue in n4.issues)


def test_unclassified_quantizable_tensor_fails_n5(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    path = tmp_path / request.source_inventory
    inventory = Inventory.model_validate_json(path.read_text(encoding="utf-8"))
    inventory.tensors[0].role = TensorRole.OTHER
    write_data(path, inventory)

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N5").passed


def test_non_improving_selected_candidate_fails_n6(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    path = tmp_path / request.refinement_measurements
    measurements = DirectRefinementMeasurementSet.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    selected = next(item for item in measurements.measurements if item.candidate_id == "candidate")
    selected.objective_loss = 0.3
    write_data(path, measurements)

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N6").passed


def test_wrong_hardware_scope_fails_n7(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    path = tmp_path / request.hardware_registry
    registry = DirectHardwareProfileRegistry.model_validate_json(path.read_text(encoding="utf-8"))
    registry.entries[0].chip = "Apple M3 Ultra"
    write_data(path, registry)

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N7").passed


def test_unarchived_raw_quality_output_fails_n7(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    index = DirectReleaseValidationIndex.model_validate_json(
        (tmp_path / request.release_validation_index).read_text(encoding="utf-8")
    )
    entry = next(item for item in index.entries if item.profile is ProfileName.AGENT_CODING)
    evaluation = DirectQualityEvaluation.model_validate_json(
        (tmp_path / entry.candidate_evaluation_file).read_text(encoding="utf-8")
    )
    output_file = evaluation.outcomes[0].output_file
    assert output_file is not None
    output_sha256 = evaluation.outcomes[0].output_sha256

    archive_path = tmp_path / request.evidence_archive_index
    archive = EvidenceArchiveIndex.model_validate_json(archive_path.read_text(encoding="utf-8"))
    archive.records = [record for record in archive.records if record.sha256 != output_sha256]
    write_data(archive_path, archive)

    audit = build_qwen3_next_release_audit(request_path)

    n7 = _gate(audit, "N7")
    assert not n7.passed
    assert any("absent from the durable archive" in issue for issue in n7.issues)


def test_unsupported_model_card_claim_fails_n8(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    model_card = tmp_path / "artifact" / "README.md"
    model_card.write_text(
        model_card.read_text(encoding="utf-8") + "\nQwen3-Next is certified.\n",
        encoding="utf-8",
    )

    audit = build_qwen3_next_release_audit(request_path)

    assert not _gate(audit, "N8").passed


def test_direct_publisher_rerun_and_registry_bind_exact_audit(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    audit = build_qwen3_next_release_audit(request_path)
    audit_path = tmp_path / "audit.json"
    write_data(audit_path, audit)

    loaded = _require_release_audit(
        audit_path=audit_path,
        model_dir=tmp_path / "artifact",
        repo_id=_CANDIDATE_ID,
        validation_index_path=tmp_path / request.release_validation_index,
        hardware_registry_path=tmp_path / request.hardware_registry,
        pareto_report_path=tmp_path / request.pareto_report,
    )
    _rerun_release_audit(audit=loaded, request_path=request_path)
    packaged = _package_release_audit(audit_path, tmp_path / "artifact")
    assert packaged == tmp_path / "artifact" / "certification" / "audit.json"

    registry_path = tmp_path / "certified-checkpoints.json"
    registry = append_certified_checkpoint(
        registry_path=registry_path,
        audit_path=audit_path,
        artifact_directory=tmp_path / "artifact",
        candidate_id="qwen-next-candidate",
        measured_bpw=4.8,
        allowed_claims=list(DIRECT_CERTIFICATION_ALLOWED_CLAIMS),
    )
    assert len(registry.entries) == 1
    assert registry.entries[0].release_audit_sha256 == file_sha256(audit_path)
    matrix = support_matrix(str(registry_path))
    qwen_next = next(entry for entry in matrix.entries if entry.adapter_id == "qwen3-next-v1")
    assert qwen_next.support_tier is SupportTier.CONVERTIBLE
    assert len(matrix.certified_checkpoints) == 1
    assert matrix.certified_checkpoints[0].candidate_model.model_id == _CANDIDATE_ID

    with pytest.raises(PublishingError, match="differs from the requested append"):
        append_certified_checkpoint(
            registry_path=registry_path,
            audit_path=audit_path,
            artifact_directory=tmp_path / "artifact",
            candidate_id="qwen-next-candidate",
            measured_bpw=4.8,
            allowed_claims=[DIRECT_CERTIFICATION_ALLOWED_CLAIMS[0]],
        )
    with pytest.raises(PublishingError, match="claims exceed"):
        append_certified_checkpoint(
            registry_path=tmp_path / "unsafe-claims-registry.json",
            audit_path=audit_path,
            artifact_directory=tmp_path / "artifact",
            candidate_id="qwen-next-candidate",
            measured_bpw=4.8,
            allowed_claims=["MTP and family-wide certified"],
        )


def test_certification_registry_rechecks_artifact_after_audit(tmp_path: Path) -> None:
    request_path = _build_inputs(tmp_path)
    audit_path = tmp_path / "audit.json"
    write_data(audit_path, build_qwen3_next_release_audit(request_path))
    weight_path = tmp_path / "artifact" / "model.safetensors"
    weight_path.write_bytes(weight_path.read_bytes() + b"drift")

    with pytest.raises(PublishingError, match="artifact integrity"):
        append_certified_checkpoint(
            registry_path=tmp_path / "registry.json",
            audit_path=audit_path,
            artifact_directory=tmp_path / "artifact",
            candidate_id="qwen-next-candidate",
            measured_bpw=4.8,
            allowed_claims=list(DIRECT_CERTIFICATION_ALLOWED_CLAIMS),
        )


def test_direct_executed_publish_uploads_only_after_registry_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _build_inputs(tmp_path)
    request = Qwen3NextReleaseAuditRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    audit_path = tmp_path / "audit.json"
    write_data(audit_path, build_qwen3_next_release_audit(request_path))
    registry_path = tmp_path / "registry.json"
    calls: list[str] = []

    class FakeHub:
        def create_repo(self, **_kwargs: object) -> None:
            calls.append("create")

        def upload_folder(self, **_kwargs: object) -> None:
            calls.append("upload")

    monkeypatch.setattr(publisher, "HfApi", FakeHub)
    publish_model(
        model_dir=tmp_path / "artifact",
        repo_id=_CANDIDATE_ID,
        validation_index_path=tmp_path / request.release_validation_index,
        hardware_registry_path=tmp_path / request.hardware_registry,
        pareto_report_path=tmp_path / request.pareto_report,
        release_audit_path=audit_path,
        release_audit_request_path=request_path,
        certification_registry_path=registry_path,
        execute=True,
    )

    assert calls == ["create", "upload"]
    assert registry_path.is_file()
    assert (tmp_path / "artifact" / "certification" / "audit.json").is_file()
