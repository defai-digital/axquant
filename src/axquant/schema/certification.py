from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.revisions import is_immutable_revision
from axquant.schema._base import SoftwareVersions, StrictModel, utc_now
from axquant.schema.artifacts import QualityGenerationConfig
from axquant.schema.coding_suite import CodingScorer
from axquant.schema.enums import ProfileName, RuntimeName
from axquant.schema.inventory import ModelIdentity

_SHA256 = r"^[0-9a-f]{64}$"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


def _relative_artifact_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if (
        not value
        or normalized.startswith(("/", "~/"))
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
        or "\\" in value
    ):
        raise ValueError("certification artifact paths must be safe relative paths")
    return value


class CertificationTrack(StrEnum):
    QWEN36_MTP_V1 = "qwen36-mtp-v1"
    QWEN36_MTP_V2 = "qwen36-mtp-v2"
    QWEN3_NEXT_DIRECT_V1 = "qwen3-next-direct-v1"


class Qwen3NextTargetClass(StrEnum):
    FOUR_BIT = "4bit"
    SIX_BIT = "6bit"


class DirectBaselineKind(StrEnum):
    BF16 = "bf16"
    UNIFORM_4BIT = "uniform-4bit"
    UNIFORM_6BIT = "uniform-6bit"
    CANDIDATE = "candidate"


class NonMtpGateId(StrEnum):
    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    N4 = "N4"
    N5 = "N5"
    N6 = "N6"
    N7 = "N7"
    N8 = "N8"


class Qwen3NextDirectPolicy(StrictModel):
    schema_version: Literal["axquant.qwen3-next-direct-policy.v1"] = (
        "axquant.qwen3-next-direct-policy.v1"
    )
    policy_id: Literal["axquant.qwen3-next-direct-policy.v1"] = (
        "axquant.qwen3-next-direct-policy.v1"
    )
    calibration_samples_min: int = 128
    sensitivity_tokens_min: int = 8192
    coding_tasks_min: int = 128
    coding_scored_tokens_min: int = 25_000
    benchmark_successful_trials_min: int = 5
    benchmark_warmups_min: int = 2
    model_runtime_errors_max: int = 0
    agent_coding_perplexity_ratio_max: float = 1.02
    general_perplexity_ratio_max: float = 1.03
    aggregate_retention_min: float = 0.99
    syntax_validity_min: float = 0.95
    syntax_validity_delta_min: float = -0.01
    tool_validity_min: float = 0.98
    tool_validity_delta_min: float = -0.01
    runtime_token_agreement_min: float = 1.0
    kernel_fallbacks_max: int = 0
    decode_speedup_vs_bf16_min: float = 1.20
    throughput_retention_vs_uniform_min: float = 0.95
    ttft_ratio_vs_uniform_max: float = 1.10
    measured_plan_bpw_delta_max: float = 0.01
    four_bit_artifact_weight_ratio_max: float = 0.35
    six_bit_artifact_weight_ratio_max: float = 0.45
    formal_hardware_chip: Literal["Apple M2 Ultra"] = "Apple M2 Ultra"
    formal_hardware_memory_bytes: int = 206_158_430_208
    required_profiles: tuple[ProfileName, ProfileName] = (
        ProfileName.AGENT_CODING,
        ProfileName.GENERAL,
    )


class ArchitectureFingerprint(StrictModel):
    model_type: Literal["qwen3_next"]
    architecture: Literal["Qwen3NextForCausalLM"]
    text_layer_count: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    full_attention_interval: int = Field(gt=0)
    expert_count: int = Field(gt=0)
    experts_per_token: int = Field(gt=0)
    expert_intermediate_size: int = Field(gt=0)
    mtp_declared: Literal[False]
    vision_present: bool
    config_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)


class SourceCheckpointFile(StrictModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256)


class SourceCheckpointManifest(StrictModel):
    schema_version: Literal["axquant.source-checkpoint-manifest.v1"] = (
        "axquant.source-checkpoint-manifest.v1"
    )
    source_model: ModelIdentity
    config_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    files: list[SourceCheckpointFile] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def membership_is_safe_and_unique(self) -> SourceCheckpointManifest:
        paths = [record.path for record in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("source checkpoint file paths must be unique")
        for path in paths:
            parts = path.replace("\\", "/").split("/")
            if path.startswith("/") or ".." in parts or "\\" in path:
                raise ValueError(f"unsafe source checkpoint path: {path}")
        return self


class ExactCertificationScope(StrictModel):
    track: Literal[CertificationTrack.QWEN3_NEXT_DIRECT_V1] = (
        CertificationTrack.QWEN3_NEXT_DIRECT_V1
    )
    source_model: ModelIdentity
    architecture: ArchitectureFingerprint
    target_class: Qwen3NextTargetClass
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    hardware_scope_ids: list[str] = Field(min_length=1)
    policy_id: Literal["axquant.qwen3-next-direct-policy.v1"] = (
        "axquant.qwen3-next-direct-policy.v1"
    )

    @model_validator(mode="after")
    def immutable_and_unique(self) -> ExactCertificationScope:
        if not is_immutable_revision(self.source_model.revision):
            raise ValueError("certification source revision must be a full immutable commit SHA")
        if len(self.hardware_scope_ids) != len(set(self.hardware_scope_ids)):
            raise ValueError("hardware scope IDs must be unique")
        return self


class Qwen3NextReleaseAuditRequest(StrictModel):
    schema_version: Literal["axquant.qwen3-next-release-audit-request.v1"] = (
        "axquant.qwen3-next-release-audit-request.v1"
    )
    certification_scope: ExactCertificationScope
    artifact_directory: str = Field(min_length=1)
    source_inventory: str = Field(min_length=1)
    source_checkpoint_manifest: str = Field(min_length=1)
    feasibility_report: str = Field(min_length=1)
    sensitivity_report: str = Field(min_length=1)
    sensitivity_lineage: list[str] = Field(default_factory=list)
    refinement_result: str = Field(min_length=1)
    refinement_measurements: str = Field(min_length=1)
    release_validation_index: str = Field(min_length=1)
    benchmark_evidence_index: str = Field(min_length=1)
    coding_suite_manifest: str = Field(min_length=1)
    coding_suite_self_test: str = Field(min_length=1)
    hardware_registry: str = Field(min_length=1)
    pareto_report: str = Field(min_length=1)
    compatibility_matrix: str = Field(min_length=1)
    compatibility_request: str = Field(min_length=1)
    reproduction_recipe: str = Field(min_length=1)
    reproduction_verification: str = Field(min_length=1)
    ax_engine_manifest_check: str = Field(min_length=1)
    ax_engine_doctor_check: str = Field(min_length=1)
    ax_engine_runtime_check: str = Field(min_length=1)
    mlx_lm_runtime_check: str = Field(min_length=1)
    evidence_archive_index: str = Field(min_length=1)
    toolkit_wheel: str = Field(min_length=1)
    required_toolkit_version: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=_SHA256)
    release_exceptions: list[str] = Field(default_factory=list, max_length=0)

    @field_validator(
        "artifact_directory",
        "source_inventory",
        "source_checkpoint_manifest",
        "feasibility_report",
        "sensitivity_report",
        "refinement_result",
        "refinement_measurements",
        "release_validation_index",
        "benchmark_evidence_index",
        "coding_suite_manifest",
        "coding_suite_self_test",
        "hardware_registry",
        "pareto_report",
        "compatibility_matrix",
        "compatibility_request",
        "reproduction_recipe",
        "reproduction_verification",
        "ax_engine_manifest_check",
        "ax_engine_doctor_check",
        "ax_engine_runtime_check",
        "mlx_lm_runtime_check",
        "evidence_archive_index",
        "toolkit_wheel",
    )
    @classmethod
    def evidence_paths_are_relative(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @field_validator("sensitivity_lineage", "release_exceptions")
    @classmethod
    def evidence_path_lists_are_relative(cls, values: list[str]) -> list[str]:
        return [_relative_artifact_path(value) for value in values]


class Qwen3NextReleaseAuditCheck(StrictModel):
    gate_id: NonMtpGateId
    name: str = Field(min_length=1)
    passed: bool
    evidence_sha256: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_is_consistent(self) -> Qwen3NextReleaseAuditCheck:
        if self.passed != (not self.issues):
            raise ValueError("release audit check status is inconsistent with its issues")
        return self


class Qwen3NextReleaseAudit(StrictModel):
    schema_version: Literal["axquant.qwen3-next-release-audit.v1"] = (
        "axquant.qwen3-next-release-audit.v1"
    )
    certification_scope: ExactCertificationScope
    candidate_model: ModelIdentity
    request_sha256: str = Field(pattern=_SHA256)
    policy_sha256: str = Field(pattern=_SHA256)
    toolkit_version: str | None = None
    wheel_sha256: str = Field(pattern=_SHA256)
    checks: list[Qwen3NextReleaseAuditCheck] = Field(min_length=9, max_length=9)
    blockers: list[str] = Field(default_factory=list)
    release_ready: bool
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def complete_and_consistent(self) -> Qwen3NextReleaseAudit:
        expected = list(NonMtpGateId)
        if [check.gate_id for check in self.checks] != expected:
            raise ValueError("Qwen3-Next release audit must contain N0 through N8 in order")
        expected_blockers = [
            f"{check.gate_id.value}: {issue}" for check in self.checks for issue in check.issues
        ]
        if self.blockers != expected_blockers:
            raise ValueError("release audit blockers are inconsistent with gate checks")
        if self.release_ready != all(check.passed for check in self.checks):
            raise ValueError("release audit readiness is inconsistent with gate checks")
        return self


class DirectBenchmarkTrial(StrictModel):
    trial_id: str = Field(pattern=_IDENTIFIER)
    warmup: bool
    success: bool
    timed_out: bool = False
    kernel_fallbacks: int = Field(default=0, ge=0)
    decode_tokens_per_second: float | None = Field(default=None, gt=0.0)
    ttft_seconds: float | None = Field(default=None, gt=0.0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    output_sha256: str | None = Field(default=None, pattern=_SHA256)
    error: str | None = None

    @model_validator(mode="after")
    def outcome_is_complete(self) -> DirectBenchmarkTrial:
        metrics = (
            self.decode_tokens_per_second,
            self.ttft_seconds,
            self.peak_memory_bytes,
            self.output_sha256,
        )
        if self.success and (
            self.timed_out or self.error or any(value is None for value in metrics)
        ):
            raise ValueError("successful benchmark trials require complete metrics and no error")
        if not self.success and not (self.timed_out or self.error):
            raise ValueError("failed benchmark trials require a timeout or error")
        return self


class DirectBenchmarkArm(StrictModel):
    kind: DirectBaselineKind
    model: ModelIdentity
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    plan_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    prompt_sha256: str = Field(pattern=_SHA256)
    ordered_prompt_ids_sha256: str = Field(pattern=_SHA256)
    random_seed: int = Field(ge=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    max_tokens: int = Field(ge=1)
    runtime: Literal[RuntimeName.AX_ENGINE] = RuntimeName.AX_ENGINE
    runtime_version: str = Field(min_length=1)
    runtime_executable_sha256: str = Field(pattern=_SHA256)
    runtime_environment: dict[str, str] = Field(default_factory=dict)
    hardware_scope_id: str = Field(min_length=1)
    os_version: str = Field(min_length=1)
    power_mode: str = Field(min_length=1)
    background_policy: str = Field(min_length=1)
    trials: list[DirectBenchmarkTrial] = Field(min_length=7)
    mlx_lm_parity_tokens: int = Field(ge=1)
    mlx_lm_matching_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def trial_ids_and_parity_are_valid(self) -> DirectBenchmarkArm:
        ids = [trial.trial_id for trial in self.trials]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark trial IDs must be unique")
        if self.mlx_lm_matching_tokens > self.mlx_lm_parity_tokens:
            raise ValueError("matching parity tokens cannot exceed compared tokens")
        return self


class DirectBenchmarkProfile(StrictModel):
    profile: ProfileName
    arms: list[DirectBenchmarkArm] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def baseline_arms_are_complete(self) -> DirectBenchmarkProfile:
        kinds = [arm.kind for arm in self.arms]
        if len(kinds) != len(set(kinds)) or set(kinds) != set(DirectBaselineKind):
            raise ValueError("direct benchmark profile requires all four baseline arms")
        return self


class DirectBenchmarkEvidenceIndex(StrictModel):
    schema_version: Literal["axquant.direct-benchmark-evidence.v1"] = (
        "axquant.direct-benchmark-evidence.v1"
    )
    profiles: list[DirectBenchmarkProfile] = Field(min_length=2, max_length=2)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def profiles_are_complete(self) -> DirectBenchmarkEvidenceIndex:
        profiles = [profile.profile for profile in self.profiles]
        required = {ProfileName.AGENT_CODING, ProfileName.GENERAL}
        if len(profiles) != len(set(profiles)) or set(profiles) != required:
            raise ValueError("direct benchmark evidence requires agent-coding and general")
        return self


class DirectQualityTaskOutcome(StrictModel):
    task_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    scored_tokens: int = Field(ge=0)
    scorer: CodingScorer | None = None
    syntax_valid: bool | None = None
    tool_valid: bool | None = None
    unit_tests_passed: bool | None = None
    model_error: bool = False
    infrastructure_error: bool = False
    output_file: str | None = None
    output_sha256: str = Field(pattern=_SHA256)
    sandboxed: bool | None = None
    network_disabled: bool | None = None
    timed_out: bool | None = None
    exit_code: int | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    stdout_file: str | None = None
    stderr_file: str | None = None
    stdout_sha256: str | None = Field(default=None, pattern=_SHA256)
    stderr_sha256: str | None = Field(default=None, pattern=_SHA256)
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    toolchain: str | None = None
    sandbox_profile_sha256: str | None = Field(default=None, pattern=_SHA256)

    @field_validator("output_file", "stdout_file", "stderr_file")
    @classmethod
    def raw_log_paths_are_relative(cls, value: str | None) -> str | None:
        return _relative_artifact_path(value) if value is not None else None


class DirectQualityEvaluation(StrictModel):
    schema_version: Literal["axquant.direct-quality-evaluation.v1"] = (
        "axquant.direct-quality-evaluation.v1"
    )
    profile: ProfileName
    model: ModelIdentity
    model_artifact_sha256: str = Field(pattern=_SHA256)
    evaluation_manifest_sha256: str = Field(pattern=_SHA256)
    dataset_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    generation: QualityGenerationConfig
    random_seed: int = Field(ge=0)
    evaluated_tokens: int = Field(ge=1)
    software_versions: SoftwareVersions
    perplexity: float = Field(gt=0.0)
    outcomes: list[DirectQualityTaskOutcome] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> DirectQualityEvaluation:
        ids = [outcome.task_id for outcome in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("quality evaluation task IDs must be unique")
        return self


class DirectGeneralQualityModelOutput(StrictModel):
    task_id: str = Field(min_length=1)
    output: str
    score: float = Field(ge=0.0, le=1.0)
    check_scores: dict[str, float]
    generated_tokens: int = Field(ge=0)
    perplexity_loss: float = Field(ge=0.0)
    perplexity_tokens: int = Field(ge=0)
    model_error: str | None = None


class DirectGeneralQualityState(StrictModel):
    schema_version: Literal["axquant.direct-general-quality-state.v1"] = (
        "axquant.direct-general-quality-state.v1"
    )
    dataset_sha256: str = Field(pattern=_SHA256)
    model: ModelIdentity
    model_artifact_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    generation: QualityGenerationConfig
    random_seed: int = Field(ge=0)
    outputs: list[DirectGeneralQualityModelOutput] = Field(default_factory=list)
    completed: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def output_ids_are_unique(self) -> DirectGeneralQualityState:
        ids = [output.task_id for output in self.outputs]
        if len(ids) != len(set(ids)):
            raise ValueError("direct general-quality output IDs must be unique")
        return self


class CodingSuiteSelfTestReport(StrictModel):
    schema_version: Literal["axquant.coding-suite-self-test.v1"] = (
        "axquant.coding-suite-self-test.v1"
    )
    suite_manifest_sha256: str = Field(pattern=_SHA256)
    toolchains: dict[str, str] = Field(min_length=1)
    sandbox_profile_sha256: str = Field(pattern=_SHA256)
    oracle_outcomes: list[DirectQualityTaskOutcome] = Field(min_length=1)
    empty_mutant_outcomes: list[DirectQualityTaskOutcome] = Field(min_length=1)
    passed: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def outcome_membership_and_status_are_consistent(self) -> CodingSuiteSelfTestReport:
        oracle_ids = [outcome.task_id for outcome in self.oracle_outcomes]
        mutant_ids = [outcome.task_id for outcome in self.empty_mutant_outcomes]
        if len(oracle_ids) != len(set(oracle_ids)) or len(mutant_ids) != len(set(mutant_ids)):
            raise ValueError("coding suite self-test outcome IDs must be unique")
        if oracle_ids != mutant_ids:
            raise ValueError("coding suite oracle and mutant membership must match in order")
        if self.passed != (not self.issues):
            raise ValueError("coding suite self-test status is inconsistent")
        return self


class DirectValidationEntry(StrictModel):
    profile: ProfileName
    evaluation_manifest_file: str = Field(min_length=1)
    evaluation_manifest_sha256: str = Field(pattern=_SHA256)
    reference_evaluation_file: str = Field(min_length=1)
    reference_evaluation_sha256: str = Field(pattern=_SHA256)
    candidate_evaluation_file: str = Field(min_length=1)
    candidate_evaluation_sha256: str = Field(pattern=_SHA256)
    passed: bool
    issues: list[str] = Field(default_factory=list)

    @field_validator(
        "evaluation_manifest_file",
        "reference_evaluation_file",
        "candidate_evaluation_file",
    )
    @classmethod
    def evaluation_paths_are_relative(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def status_is_consistent(self) -> DirectValidationEntry:
        if self.passed != (not self.issues):
            raise ValueError("direct validation entry status is inconsistent")
        return self


class DirectValidationRequestEntry(StrictModel):
    profile: ProfileName
    evaluation_manifest_file: str = Field(min_length=1)
    reference_evaluation_file: str = Field(min_length=1)
    candidate_evaluation_file: str = Field(min_length=1)

    @field_validator(
        "evaluation_manifest_file",
        "reference_evaluation_file",
        "candidate_evaluation_file",
    )
    @classmethod
    def evaluation_paths_are_relative(cls, value: str) -> str:
        return _relative_artifact_path(value)


class DirectReleaseValidationRequest(StrictModel):
    schema_version: Literal["axquant.direct-release-validation-request.v1"] = (
        "axquant.direct-release-validation-request.v1"
    )
    source_checkpoint_manifest: str = Field(min_length=1)
    candidate_artifact_manifest: str = Field(min_length=1)
    calibration_dataset_sha256: str = Field(pattern=_SHA256)
    coding_suite_manifest: str = Field(min_length=1)
    general_calibration_overlap_report: str = Field(min_length=1)
    required_toolkit_version: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=_SHA256)
    entries: list[DirectValidationRequestEntry] = Field(min_length=2, max_length=2)

    @field_validator(
        "source_checkpoint_manifest",
        "candidate_artifact_manifest",
        "coding_suite_manifest",
        "general_calibration_overlap_report",
    )
    @classmethod
    def evidence_paths_are_relative(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def required_profiles_are_complete(self) -> DirectReleaseValidationRequest:
        profiles = [entry.profile for entry in self.entries]
        required = {ProfileName.AGENT_CODING, ProfileName.GENERAL}
        if len(profiles) != len(set(profiles)) or set(profiles) != required:
            raise ValueError("direct validation request requires both release profiles")
        return self


class DirectReleaseValidationIndex(StrictModel):
    schema_version: Literal["axquant.direct-release-validation-index.v1"] = (
        "axquant.direct-release-validation-index.v1"
    )
    entries: list[DirectValidationEntry] = Field(min_length=2, max_length=2)
    general_calibration_overlap_report_file: str = Field(min_length=1)
    general_calibration_overlap_report_sha256: str = Field(pattern=_SHA256)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("general_calibration_overlap_report_file")
    @classmethod
    def overlap_path_is_relative(cls, value: str) -> str:
        return _relative_artifact_path(value)

    @model_validator(mode="after")
    def complete_and_consistent(self) -> DirectReleaseValidationIndex:
        profiles = [entry.profile for entry in self.entries]
        required = {ProfileName.AGENT_CODING, ProfileName.GENERAL}
        if len(profiles) != len(set(profiles)) or set(profiles) != required:
            raise ValueError("direct validation index requires both release profiles")
        expected = all(entry.passed for entry in self.entries) and not self.issues
        if self.release_ready != expected:
            raise ValueError("direct validation index readiness is inconsistent")
        return self


class DirectRefinementMeasurement(StrictModel):
    measurement_id: str = Field(pattern=_IDENTIFIER)
    candidate_id: str = Field(pattern=_IDENTIFIER)
    parent_candidate_id: str | None = Field(default=None, pattern=_IDENTIFIER)
    target_class: Qwen3NextTargetClass
    candidate_model: ModelIdentity
    plan_sha256: str = Field(pattern=_SHA256)
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    quality_evidence_sha256: str = Field(pattern=_SHA256)
    benchmark_evidence_sha256: str = Field(pattern=_SHA256)
    measured_bpw: float = Field(gt=0.0)
    objective_loss: float = Field(ge=0.0)
    quality_retention: float = Field(ge=0.0)
    decode_tokens_per_second: float = Field(gt=0.0)
    peak_memory_bytes: int = Field(gt=0)
    validation_passed: bool


class DirectRefinementMeasurementSet(StrictModel):
    schema_version: Literal["axquant.direct-refinement-measurements.v1"] = (
        "axquant.direct-refinement-measurements.v1"
    )
    refinement_sha256: str = Field(pattern=_SHA256)
    evaluator_version: str = Field(min_length=1)
    selected_candidate_id: str = Field(pattern=_IDENTIFIER)
    measurements: list[DirectRefinementMeasurement] = Field(min_length=2)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def measurements_are_unique_and_selected(self) -> DirectRefinementMeasurementSet:
        ids = [item.measurement_id for item in self.measurements]
        if len(ids) != len(set(ids)):
            raise ValueError("direct refinement measurement IDs must be unique")
        if self.selected_candidate_id not in {item.candidate_id for item in self.measurements}:
            raise ValueError("selected candidate is missing from direct measurements")
        return self


class DirectHardwareRegistryEntry(StrictModel):
    entry_id: str = Field(pattern=_IDENTIFIER)
    hardware_scope_id: str = Field(pattern=_IDENTIFIER)
    candidate_id: str = Field(pattern=_IDENTIFIER)
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    benchmark_evidence_sha256: str = Field(pattern=_SHA256)
    device_name: str = Field(min_length=1)
    chip: str = Field(min_length=1)
    unified_memory_bytes: int = Field(gt=0)
    os_version: str = Field(min_length=1)
    ax_engine_version: str = Field(min_length=1)
    ax_engine_executable_sha256: str = Field(pattern=_SHA256)
    metal_version: str = Field(min_length=1)
    metallib_version: str = Field(min_length=1)
    power_mode: str = Field(min_length=1)
    doctor_passed: bool
    kernel_fallbacks: int = Field(ge=0)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_is_consistent(self) -> DirectHardwareRegistryEntry:
        expected = self.doctor_passed and self.kernel_fallbacks == 0 and not self.issues
        if self.release_ready != expected:
            raise ValueError("direct hardware registry entry readiness is inconsistent")
        return self


class DirectHardwareProfileRegistry(StrictModel):
    schema_version: Literal["axquant.direct-hardware-registry.v1"] = (
        "axquant.direct-hardware-registry.v1"
    )
    registry_id: str = Field(pattern=_IDENTIFIER)
    entries: list[DirectHardwareRegistryEntry] = Field(min_length=1)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def registry_is_consistent(self) -> DirectHardwareProfileRegistry:
        ids = [entry.entry_id for entry in self.entries]
        scopes = [entry.hardware_scope_id for entry in self.entries]
        if len(ids) != len(set(ids)) or len(scopes) != len(set(scopes)):
            raise ValueError("direct hardware entries and scope IDs must be unique")
        expected = all(entry.release_ready for entry in self.entries) and not self.issues
        if self.release_ready != expected:
            raise ValueError("direct hardware registry readiness is inconsistent")
        return self


class DirectParetoPoint(StrictModel):
    candidate_id: str = Field(pattern=_IDENTIFIER)
    measurement_id: str = Field(pattern=_IDENTIFIER)
    target_class: Qwen3NextTargetClass
    measured_bpw: float = Field(gt=0.0)
    quality_retention: float = Field(ge=0.0)
    decode_tokens_per_second: float = Field(gt=0.0)
    peak_memory_bytes: int = Field(gt=0)
    validation_passed: bool
    frontier: bool
    dominated_by: list[str] = Field(default_factory=list)


class DirectParetoReport(StrictModel):
    schema_version: Literal["axquant.direct-pareto.v1"] = "axquant.direct-pareto.v1"
    measurement_set_sha256: str = Field(pattern=_SHA256)
    points: list[DirectParetoPoint] = Field(min_length=1)
    frontier_candidate_ids: list[str]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def frontier_is_consistent(self) -> DirectParetoReport:
        expected = sorted({point.candidate_id for point in self.points if point.frontier})
        if self.frontier_candidate_ids != expected:
            raise ValueError("direct Pareto frontier summary is inconsistent")
        return self


class Qwen3NextCompatibilityRequest(StrictModel):
    schema_version: Literal["axquant.qwen3-next-compatibility-request.v1"] = (
        "axquant.qwen3-next-compatibility-request.v1"
    )
    source_model: ModelIdentity
    target_class: Qwen3NextTargetClass
    required_profiles: tuple[ProfileName, ProfileName] = (
        ProfileName.AGENT_CODING,
        ProfileName.GENERAL,
    )
    required_runtimes: tuple[RuntimeName, RuntimeName] = (
        RuntimeName.AX_ENGINE,
        RuntimeName.MLX_LM,
    )


class Qwen3NextCompatibilityMatrix(StrictModel):
    schema_version: Literal["axquant.qwen3-next-compatibility-matrix.v1"] = (
        "axquant.qwen3-next-compatibility-matrix.v1"
    )
    source_model: ModelIdentity
    target_class: Qwen3NextTargetClass
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    profiles_passed: list[ProfileName] = Field(min_length=2, max_length=2)
    runtimes_passed: list[RuntimeName] = Field(min_length=2, max_length=2)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def coverage_is_complete(self) -> Qwen3NextCompatibilityMatrix:
        expected_profiles = {ProfileName.AGENT_CODING, ProfileName.GENERAL}
        expected_runtimes = {RuntimeName.AX_ENGINE, RuntimeName.MLX_LM}
        complete = (
            set(self.profiles_passed) == expected_profiles
            and set(self.runtimes_passed) == expected_runtimes
            and not self.issues
        )
        if self.release_ready != complete:
            raise ValueError("Qwen3-Next compatibility readiness is inconsistent")
        return self


class EvidenceArchiveRecord(StrictModel):
    logical_name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)
    durable_uri: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def archived_path_is_relative(cls, value: str) -> str:
        return _relative_artifact_path(value)


class EvidenceArchiveIndex(StrictModel):
    schema_version: Literal["axquant.evidence-archive-index.v1"] = (
        "axquant.evidence-archive-index.v1"
    )
    records: list[EvidenceArchiveRecord] = Field(min_length=1)
    complete: bool
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def records_are_unique(self) -> EvidenceArchiveIndex:
        names = [record.logical_name for record in self.records]
        if len(names) != len(set(names)):
            raise ValueError("evidence archive logical names must be unique")
        return self


class CertifiedCheckpointEntry(StrictModel):
    entry_id: str = Field(pattern=_IDENTIFIER)
    certification_scope: ExactCertificationScope
    candidate_model: ModelIdentity
    candidate_id: str = Field(pattern=_IDENTIFIER)
    policy_sha256: str = Field(pattern=_SHA256)
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    release_audit_sha256: str = Field(pattern=_SHA256)
    measured_bpw: float = Field(gt=0.0)
    allowed_claims: list[str] = Field(min_length=1)
    hardware_scope_ids: list[str] = Field(min_length=1)
    certified_at: datetime
    supersedes_entry_id: str | None = Field(default=None, pattern=_IDENTIFIER)

    @model_validator(mode="after")
    def scope_and_claims_are_consistent(self) -> CertifiedCheckpointEntry:
        if self.hardware_scope_ids != self.certification_scope.hardware_scope_ids:
            raise ValueError("registry hardware scope must match the certification scope")
        if len(self.allowed_claims) != len(set(self.allowed_claims)) or any(
            not claim.strip() for claim in self.allowed_claims
        ):
            raise ValueError("registry allowed claims must be non-empty and unique")
        return self


class CertifiedCheckpointRegistry(StrictModel):
    schema_version: Literal["axquant.certified-checkpoint-registry.v1"] = (
        "axquant.certified-checkpoint-registry.v1"
    )
    entries: list[CertifiedCheckpointEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def append_only_identity_is_consistent(self) -> CertifiedCheckpointRegistry:
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("certified checkpoint entry IDs must be unique")
        known: dict[str, tuple[str, str | None, Qwen3NextTargetClass]] = {}
        superseded: set[str] = set()
        active_identities: dict[
            tuple[str, str | None, Qwen3NextTargetClass],
            str,
        ] = {}
        for entry in self.entries:
            identity = (
                entry.certification_scope.source_model.model_id,
                entry.certification_scope.source_model.revision,
                entry.certification_scope.target_class,
            )
            active_entry_id = active_identities.get(identity)
            supersedes = entry.supersedes_entry_id
            if active_entry_id is None:
                if supersedes is not None:
                    if supersedes not in known:
                        raise ValueError("registry supersession must reference an earlier entry")
                    raise ValueError("registry supersession must keep the same exact identity")
            elif supersedes != active_entry_id:
                raise ValueError(
                    "replacing an exact certification requires supersession of its active entry"
                )
            if supersedes is not None:
                if supersedes in superseded:
                    raise ValueError("registry entry has already been superseded")
                if known.get(supersedes) != identity:
                    raise ValueError("registry supersession must keep the same exact identity")
                superseded.add(supersedes)
            active_identities[identity] = entry.entry_id
            known[entry.entry_id] = identity
        return self
