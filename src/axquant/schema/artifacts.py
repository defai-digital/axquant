from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from axquant.revisions import is_immutable_revision
from axquant.schema._base import SoftwareVersions, StrictModel, utc_now
from axquant.schema.enums import (
    BaselineKind,
    BenchmarkEvidenceKind,
    EvidenceKind,
    MtpSidecarLayout,
    OptimizationScope,
    ProbeMode,
    ProfileName,
    QuantMethod,
    RuntimeName,
    RuntimeSupportLevel,
    SupportTier,
    TensorRole,
)
from axquant.schema.inventory import ModelIdentity
from axquant.schema.planning import (
    AX_ENGINE_EXECUTABLE_BITS,
    AX_ENGINE_EXECUTABLE_GROUP_SIZES,
    MtpPolicy,
    PrecisionShare,
    QuantizationPlan,
)
from axquant.schema.sensitivity import CalibrationEvidence


class RuntimeProfile(StrictModel):
    name: RuntimeName
    compatibility_level: Literal["A", "B"]
    support_level: RuntimeSupportLevel
    standard_mlx_weights: bool = True
    standard_inference: bool
    mtp_support: Literal["native", "runtime-dependent", "none"]
    manifest: str | None = None
    notes: list[str] = Field(default_factory=list)


class MtpRuntimeMetadata(StrictModel):
    detected: bool
    sidecar_file: str | None = None
    optimized: bool = False
    enabled_by_default: bool = False
    draft_tokens: int | None = Field(default=None, ge=1)
    verification_mode: str | None = None
    head_precision: str | None = None
    acceptance_retention: float | None = Field(default=None, ge=0.0)
    measured_speedup: float | None = Field(default=None, ge=0.0)
    recommended_temperature_max: float | None = Field(default=None, ge=0.0)


class AxEngineOptimizationMetadata(StrictModel):
    model_manifest: str = "model-manifest.json"
    preferred_group_size: int = Field(default=64, ge=1)
    fused_mtp: bool | None = None
    decode_kernel: str | None = None
    kernel_evidence: Literal["unmeasured", "measured"] = "unmeasured"


class KvCacheRuntimeMetadata(StrictModel):
    """Per-layer KV-cache precision table consumed by AX Engine (AXQ-021).

    The MLX-LM fallback values are advisory operator documentation derived from
    the plan's modal allocation; stock MLX-LM behavior is unchanged.
    """

    allocation_basis: Literal["architecture-prior", "measured"]
    layer_bits: list[int]
    layer_group_sizes: list[int]
    advisory_mlx_lm_kv_bits: int = Field(ge=2, le=16)
    advisory_mlx_lm_kv_group_size: int = Field(ge=1)
    advisory: Literal[True] = True

    @model_validator(mode="after")
    def executable_and_consistent(self) -> KvCacheRuntimeMetadata:
        if not self.layer_bits or len(self.layer_bits) != len(self.layer_group_sizes):
            raise ValueError("KV runtime layer bits and group sizes must be non-empty and aligned")
        unsupported_bits = (
            set(self.layer_bits) | {self.advisory_mlx_lm_kv_bits}
        ) - AX_ENGINE_EXECUTABLE_BITS
        if unsupported_bits:
            raise ValueError(
                f"AX Engine KV runtime does not support bits {sorted(unsupported_bits)}"
            )
        unsupported_groups = (
            set(self.layer_group_sizes) | {self.advisory_mlx_lm_kv_group_size}
        ) - AX_ENGINE_EXECUTABLE_GROUP_SIZES
        if unsupported_groups:
            raise ValueError(
                f"AX Engine KV runtime does not support group sizes {sorted(unsupported_groups)}"
            )
        advisory_pair = (
            self.advisory_mlx_lm_kv_bits,
            self.advisory_mlx_lm_kv_group_size,
        )
        if advisory_pair not in set(zip(self.layer_bits, self.layer_group_sizes, strict=True)):
            raise ValueError("KV runtime advisory precision must be an observed layer allocation")
        return self


class RuntimeMetadata(StrictModel):
    schema_version: Literal["axquant.runtime.v1"] = "axquant.runtime.v1"
    primary_runtime: RuntimeProfile
    compatible_runtimes: list[RuntimeProfile]
    optimization_scope: OptimizationScope
    mtp: MtpRuntimeMetadata
    ax_engine: AxEngineOptimizationMetadata
    kv_cache: KvCacheRuntimeMetadata | None = None
    memory_policy: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RuntimeCheck(StrictModel):
    schema_version: Literal["axquant.runtime-check.v2"] = "axquant.runtime-check.v2"
    model: ModelIdentity
    runtime: RuntimeName
    check_kind: Literal[
        "manifest",
        "doctor",
        "static-compatibility",
        "generation-smoke",
        "transcription-smoke",
        "vision-generation-smoke",
        "kv-layered-generation-smoke",
    ]
    available: bool
    passed: bool
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    report: dict[str, JsonValue] = Field(default_factory=dict)
    stderr: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactIntegrity(StrictModel):
    config_valid: bool
    safetensors_present: bool
    index_present: bool
    index_complete: bool
    native_manifest_present: bool
    native_manifest_valid: bool
    tokenizer_present: bool
    mtp_sidecar_present: bool
    mtp_runtime_present: bool
    mtp_runtime_valid: bool
    mtp_provenance_present: bool
    mtp_provenance_valid: bool


class BaselineAudit(StrictModel):
    kind: BaselineKind
    model: ModelIdentity
    inspected: bool
    inventory_sha256: str | None = None
    adapter_id: str = "unknown"
    optimization_scope: OptimizationScope = OptimizationScope.INVENTORY_ONLY
    quantized: bool | None = None
    logical_parameters: int = Field(ge=0)
    mtp_logical_parameters: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    main_weight_bytes: int = Field(ge=0)
    mtp_weight_bytes: int = Field(ge=0)
    effective_bpw: float = Field(ge=0.0)
    main_effective_bpw: float = Field(ge=0.0)
    precision_parameters: dict[str, int]
    precision_fractions: dict[str, float]
    integrity: ArtifactIntegrity
    runtime_checks: list[RuntimeCheck] = Field(default_factory=list)
    complete: bool
    issues: list[str] = Field(default_factory=list)


class FeasibilityReport(StrictModel):
    schema_version: Literal["axquant.feasibility.v1"] = "axquant.feasibility.v1"
    status: Literal["ready-for-conversion", "baseline-ready", "blocked"]
    source: BaselineAudit | None = None
    baselines: list[BaselineAudit]
    runtime_checks_requested: bool = False
    checks: dict[str, bool]
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class SupportMatrixEntry(StrictModel):
    adapter_id: str
    product_family: str
    support_tier: SupportTier
    # Investment posture from support_policy (additive; defaults keep old clients happy).
    investment_posture: str = "secondary"
    priority: int = Field(default=50, ge=1, le=100)
    cert_track: bool = False
    notes: list[str] = Field(default_factory=list)


class CertifiedCheckpointSummary(StrictModel):
    source_model: ModelIdentity
    candidate_model: ModelIdentity
    target_class: str = Field(min_length=1)
    certification_track: str = Field(min_length=1)
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware_scope_ids: list[str] = Field(min_length=1)


class SupportMatrix(StrictModel):
    """Registry-derived family support matrix (AXQ-017)."""

    schema_version: Literal["axquant.support-matrix.v1"] = "axquant.support-matrix.v1"
    axquant_version: str
    entries: list[SupportMatrixEntry]
    certified_checkpoints: list[CertifiedCheckpointSummary] = Field(default_factory=list)
    policy_version: Literal["axquant.support-policy.v1"] = "axquant.support-policy.v1"
    created_at: datetime = Field(default_factory=utc_now)


class RecipeBundle(StrictModel):
    """Checksummed, publishable planning artifact (AXQ-020).

    A bundle binds a plan or manual recipe to a pinned source model so user
    conversions can reuse published planning evidence without re-measuring. A
    bundle never upgrades the evidence kind of its payload.
    """

    schema_version: Literal["axquant.recipe-bundle.v1"] = "axquant.recipe-bundle.v1"
    bundle_id: str = Field(min_length=1)
    source_model: ModelIdentity
    evidence_kind: EvidenceKind
    payload_kind: Literal["plan", "manual-recipe"]
    payload_file: str = Field(min_length=1)
    payload_sha256: str = Field(min_length=64, max_length=64)
    lineage: dict[str, str] = Field(default_factory=dict)
    axquant_version: str
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def source_revision_is_pinned(self) -> RecipeBundle:
        if not is_immutable_revision(self.source_model.revision):
            raise ValueError("a recipe bundle must pin an immutable source model revision")
        return self


class QuickConversionSummary(StrictModel):
    """Result of the one-command `axquant quantize` development conversion (AXQ-019)."""

    schema_version: Literal["axquant.quantize-summary.v1"] = "axquant.quantize-summary.v1"
    source_model: ModelIdentity
    product_family: str
    support_tier: SupportTier
    evidence_kind: EvidenceKind
    plan_source: Literal["architecture-prior", "recipe-bundle"] = "architecture-prior"
    recipe_bundle_id: str | None = None
    convert_ladder: str | None = None
    profile: ProfileName
    target_bpw: float = Field(gt=0.0, le=16.0)
    measured_total_bpw: float = Field(gt=0.0)
    output_path: str
    runtime_smoke: Literal["none", "mlx-lm", "mlx-audio", "mlx-vlm", "ax-engine"] = "none"
    runtime_smoke_passed: bool | None = None
    development_evidence: bool
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ArtifactFile(StrictModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str


class MtpSidecarSourceModel(StrictModel):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class MtpSidecarSourceShard(StrictModel):
    name: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MtpSidecarSource(StrictModel):
    model: MtpSidecarSourceModel
    path: str = Field(min_length=1)
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shards: list[MtpSidecarSourceShard] = Field(min_length=1)


class MtpSidecarFileBinding(StrictModel):
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MtpSidecarOutputBinding(StrictModel):
    mtp: MtpSidecarFileBinding


class RawMtpTensorPayload(StrictModel):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: list[int] = Field(min_length=1)
    byte_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_data_range: list[int] = Field(min_length=2, max_length=2)
    source_shard: str = Field(min_length=1)


class BytePreservedMtpTransform(StrictModel):
    mode: Literal["byte_preserved"]
    implementation: str = Field(min_length=1)


class BytePreservedMtpSidecarManifest(StrictModel):
    schema_version: Literal["axquant.mtp_sidecar_provenance.v2"]
    generated_by: str = Field(min_length=1)
    source: MtpSidecarSource
    output: MtpSidecarOutputBinding
    transform: BytePreservedMtpTransform
    tensor_count: int = Field(ge=1)
    tensor_payloads: list[RawMtpTensorPayload] = Field(min_length=1)
    total_payload_bytes: int = Field(ge=1)


class QuantizedMtpTensorRecord(StrictModel):
    """Per-tensor packing record inside a quantized MTP sidecar (ADR-0005)."""

    name: str = Field(min_length=1)
    quantized: bool
    bits: int = Field(ge=2, le=16)
    group_size: int | None = Field(default=None, ge=1)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def preserved_is_bf16(self) -> QuantizedMtpTensorRecord:
        if not self.quantized and self.bits != 16:
            raise ValueError("preserved sidecar tensors must record 16 bits")
        if self.quantized and self.group_size is None:
            raise ValueError("quantized sidecar tensors require a group size")
        return self


class AxEngineMtpCapabilityCheck(StrictModel):
    """Recorded AX Engine capability probe for a quantized MTP layout."""

    ok: bool
    mtp_enabled: bool
    layout: str = Field(min_length=1)
    ax_engine_version: str = Field(min_length=1)
    # Optional detail newer probes report (`ax-engine mtp-capability`): the
    # loader's accepted sidecar bit widths and packing label. Empty on older
    # recorded checks; when present, sidecar quantization validates against
    # them.
    supported_bits: list[int] = Field(default_factory=list)
    packing: str = ""


class QuantizedMtpSidecarManifest(StrictModel):
    """Opt-in quantized MTP sidecar artifact (ADR-0005 / RM-40).

    Always emitted *alongside* the byte-preserved default, never replacing it,
    and only after a passing AX Engine capability check for the quantized
    layout. The packing is AX Engine's executable MLX-native layout —
    ``mx.quantize`` uint32-packed codes plus BF16 group scales/biases under
    the engine's key convention (``mlx-affine-packed-u32``).
    """

    schema_version: Literal["axquant.mtp-sidecar-quantized.v1"] = "axquant.mtp-sidecar-quantized.v1"
    generated_by: str = Field(min_length=1)
    packing: Literal["mlx-affine-packed-u32"] = "mlx-affine-packed-u32"
    source_sidecar: MtpSidecarFileBinding
    output: MtpSidecarFileBinding
    default_bits: int = Field(ge=2, le=16)
    group_size: int = Field(ge=1)
    tensors: list[QuantizedMtpTensorRecord] = Field(min_length=1)
    capability: AxEngineMtpCapabilityCheck
    byte_preserved_default_retained: Literal[True] = True

    @model_validator(mode="after")
    def capability_gate_passed(self) -> QuantizedMtpSidecarManifest:
        if not (self.capability.ok and self.capability.mtp_enabled):
            raise ValueError(
                "quantized MTP sidecar requires a passing AX Engine capability "
                "check with MTP enabled"
            )
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("quantized MTP sidecar tensor names must be unique")
        if not any(tensor.quantized for tensor in self.tensors):
            raise ValueError("quantized MTP sidecar must quantize at least one tensor")
        return self


class PreparedMtpInputBinding(StrictModel):
    manifest: MtpSidecarFileBinding
    mtp: MtpSidecarFileBinding


class PreparedMtpOutputBinding(StrictModel):
    mtp: MtpSidecarFileBinding
    runtime: MtpSidecarFileBinding


class PreparedMtpTensorPayload(StrictModel):
    name: str = Field(min_length=1)
    dtype: Literal["BF16"]
    shape: list[int] = Field(min_length=1)
    byte_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: Literal["add_one_bf16", "byte_preserved"]


class PreparedMtpTransform(StrictModel):
    mode: Literal["ax-engine-qwen36-v1"]
    implementation: Literal["axquant"]
    operation: Literal["add_one_to_qwen36_mtp_norms_bf16"]
    transformed_tensors: list[str] = Field(min_length=1)
    unchanged_tensors: list[str] = Field(min_length=1)


class PreparedMtpSidecarManifest(StrictModel):
    schema_version: Literal["axquant.mtp-sidecar-provenance.v3"] = (
        "axquant.mtp-sidecar-provenance.v3"
    )
    generated_by: Literal["axquant"] = "axquant"
    source: MtpSidecarSource
    input: PreparedMtpInputBinding
    output: PreparedMtpOutputBinding
    transform: PreparedMtpTransform
    tensor_count: int = Field(ge=1)
    tensor_payloads: list[PreparedMtpTensorPayload] = Field(min_length=1)
    total_payload_bytes: int = Field(ge=1)


class ArtifactManifest(StrictModel):
    schema_version: Literal["axquant.artifact.v2"] = "axquant.artifact.v2"
    format: Literal["mlx"] = "mlx"
    quantizer: Literal["axquant"] = "axquant"
    axquant_version: str
    source_model: ModelIdentity
    plan_sha256: str
    calibration: CalibrationEvidence | None = None
    profile: ProfileName
    target_class: str
    effective_bpw: float
    logical_parameters: int = Field(ge=1)
    main_logical_parameters: int = Field(ge=1)
    weight_file_size_bytes: int = Field(ge=1)
    main_weight_file_size_bytes: int = Field(ge=1)
    mtp_weight_file_size_bytes: int = Field(ge=0)
    protected_weight_file_size_bytes: int = Field(ge=0)
    measured_total_bpw: float = Field(gt=0.0)
    measured_main_bpw: float = Field(gt=0.0)
    weight_distribution: dict[str, PrecisionShare]
    mtp_distribution: dict[str, PrecisionShare]
    mtp_present: bool
    mtp_policy: MtpPolicy
    mtp_acceptance_retention: float | None = Field(default=None, ge=0.0)
    mtp_measured_speedup: float | None = Field(default=None, ge=0.0)
    runtime: RuntimeMetadata
    software_versions: SoftwareVersions
    files: list[ArtifactFile]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def measured_weight_accounting_is_consistent(self) -> ArtifactManifest:
        if (
            self.main_weight_file_size_bytes + self.mtp_weight_file_size_bytes
            != self.weight_file_size_bytes
        ):
            raise ValueError("main and MTP weight bytes must equal total weight bytes")
        if self.protected_weight_file_size_bytes > self.main_weight_file_size_bytes:
            raise ValueError("protected weight bytes cannot exceed main-model weight bytes")
        expected_total_bpw = 8.0 * self.weight_file_size_bytes / self.logical_parameters
        expected_main_bpw = 8.0 * self.main_weight_file_size_bytes / self.main_logical_parameters
        if abs(self.measured_total_bpw - expected_total_bpw) > 1e-9:
            raise ValueError("measured total BPW does not match weight-byte accounting")
        if abs(self.measured_main_bpw - expected_main_bpw) > 1e-9:
            raise ValueError("measured main BPW does not match weight-byte accounting")
        return self


class ReproductionCommand(StrictModel):
    step_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    argv: list[str] = Field(min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    expected_outputs: list[str] = Field(default_factory=list)

    @field_validator("argv")
    @classmethod
    def nonempty_arguments(cls, value: list[str]) -> list[str]:
        if any(not argument for argument in value):
            raise ValueError("reproduction command arguments must be non-empty")
        return value


class ReproductionRecipe(StrictModel):
    schema_version: Literal["axquant.reproduction.v3"] = "axquant.reproduction.v3"
    source_model: ModelIdentity
    calibration: CalibrationEvidence
    axquant_version: str
    software_versions: SoftwareVersions
    random_seed: int = Field(ge=0)
    profile: ProfileName
    primary_runtime: RuntimeName
    plan_sha256: str
    output_repository: str
    plan_file: str = "quantization_plan.json"
    plan_file_sha256: str
    calibration_file: str = "calibration_manifest.json"
    calibration_file_sha256: str
    conversion_manifest_file: str = "axquant_conversion_manifest.json"
    conversion_manifest_sha256: str
    mtp_sidecar_file: str | None = None
    mtp_sidecar_sha256: str | None = None
    mtp_companion_files: list[ArtifactFile] = Field(default_factory=list, max_length=2)
    expected_logical_parameters: int = Field(gt=0)
    expected_weight_file_size_bytes: int = Field(gt=0)
    expected_weight_files: list[ArtifactFile] = Field(min_length=1)
    commands: list[ReproductionCommand] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_executable_recipe(self) -> ReproductionRecipe:
        if not is_immutable_revision(self.source_model.revision):
            raise ValueError("reproduction source revision must be immutable")
        if (self.mtp_sidecar_file is None) != (self.mtp_sidecar_sha256 is None):
            raise ValueError("MTP sidecar path and checksum must be recorded together")
        companion_paths = [record.path for record in self.mtp_companion_files]
        if len(companion_paths) != len(set(companion_paths)):
            raise ValueError("MTP reproduction companion paths must be unique")
        allowed_companions = {
            "ax_mtp_sidecar_manifest.json",
            "mtplx_runtime.json",
        }
        if not set(companion_paths).issubset(allowed_companions):
            raise ValueError("MTP reproduction contains an unsupported companion path")
        if self.mtp_sidecar_file is None and companion_paths:
            raise ValueError("MTP companion files require an MTP sidecar")
        command_ids = [command.step_id for command in self.commands]
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("reproduction command step IDs must be unique")
        required = {"download-source", "convert", "verify-reproduction"}
        if not required.issubset(command_ids):
            raise ValueError("reproduction recipe is missing a required executable step")
        convert = next(command for command in self.commands if command.step_id == "convert")
        if self.mtp_sidecar_file is None:
            if "--mtp-sidecar" in convert.argv or "--mtp-layout" in convert.argv:
                raise ValueError("reproduction convert command has an undeclared MTP input")
        else:
            try:
                sidecar_index = convert.argv.index("--mtp-sidecar")
                layout_index = convert.argv.index("--mtp-layout")
                command_sidecar = convert.argv[sidecar_index + 1]
                command_layout = convert.argv[layout_index + 1]
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    "reproduction convert command has incomplete MTP arguments"
                ) from exc
            if command_sidecar != self.mtp_sidecar_file:
                raise ValueError("reproduction convert command uses another MTP sidecar")
            if (
                command_layout == MtpSidecarLayout.AX_ENGINE_QWEN36_V1.value
                and set(companion_paths) != allowed_companions
            ):
                raise ValueError(
                    "prepared MTP reproduction must bind its provenance and runtime companions"
                )
        weight_paths = [record.path for record in self.expected_weight_files]
        if len(weight_paths) != len(set(weight_paths)):
            raise ValueError("expected reproduction weight paths must be unique")
        return self


class ReproductionVerification(StrictModel):
    schema_version: Literal["axquant.reproduction-verification.v1"] = (
        "axquant.reproduction-verification.v1"
    )
    recipe_sha256: str
    artifact_path: str
    passed: bool
    issues: list[str]
    verified_weight_files: list[str]
    expected_logical_parameters: int = Field(gt=0)
    actual_logical_parameters: int = Field(ge=0)
    expected_weight_file_size_bytes: int = Field(gt=0)
    actual_weight_file_size_bytes: int = Field(ge=0)


class ProtectedTensorSidecarManifest(StrictModel):
    schema_version: Literal["axquant.protected-tensor-sidecar.v1"] = (
        "axquant.protected-tensor-sidecar.v1"
    )
    source_model: ModelIdentity
    role: Literal["vision", "mtp"]
    tensor_count: int = Field(ge=1)
    parameters: int = Field(ge=1)
    dtypes: tuple[str, ...]
    tensor_names_sha256: str
    source_files: list[ArtifactFile]
    output: ArtifactFile
    created_at: datetime = Field(default_factory=utc_now)


class QualityMetrics(StrictModel):
    perplexity: float | None = Field(default=None, gt=0.0)
    task_scores: dict[str, float] = Field(default_factory=dict)
    json_valid_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    syntax_valid_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("task_scores")
    @classmethod
    def valid_task_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(score < 0.0 or score > 1.0 for score in value.values()):
            raise ValueError("quality task scores must be within [0, 1]")
        return value


class QualityCheck(StrictModel):
    kind: Literal[
        "exact",
        "contains",
        "regex",
        "json-valid",
        "json-keys",
        "python-syntax",
        "token-f1",
    ]
    value: str | list[str] | None = None


class QualityTask(StrictModel):
    task_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference: str | None = None
    perplexity_text: str | None = None
    checks: list[QualityCheck] = Field(min_length=1)


class QualityTaskResult(StrictModel):
    task_id: str
    category: str
    output: str
    score: float = Field(ge=0.0, le=1.0)
    check_scores: dict[str, float]
    error: str | None = None


class QualityGenerationConfig(StrictModel):
    prompt_format: Literal["raw", "chat-template"]
    chat_template_sha256: str | None = None
    thinking_enabled: bool = False
    max_sequence_length: int = Field(ge=1)
    max_generation_tokens: int = Field(ge=1)


class QualityEvaluationResult(StrictModel):
    schema_version: Literal["axquant.quality-evaluation.v2"] = "axquant.quality-evaluation.v2"
    model: ModelIdentity
    dataset_sha256: str
    generation: QualityGenerationConfig
    metrics: QualityMetrics
    task_results: list[QualityTaskResult]
    samples: int = Field(ge=1)
    evaluated_tokens: int = Field(ge=0)
    random_seed: int = Field(ge=0)
    software_versions: SoftwareVersions
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def task_aggregates_are_consistent(self) -> QualityEvaluationResult:
        if self.samples != len(self.task_results):
            raise ValueError("quality sample count must equal task result count")
        task_ids = [task.task_id for task in self.task_results]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("quality task result IDs must be unique")
        category_scores: dict[str, list[float]] = {}
        for task in self.task_results:
            category_scores.setdefault(task.category, []).append(task.score)
        if set(category_scores) != set(self.metrics.task_scores):
            raise ValueError("quality task categories must match aggregate categories")
        for category, scores in category_scores.items():
            expected = sum(scores) / len(scores)
            if abs(self.metrics.task_scores[category] - expected) > 1e-9:
                raise ValueError(f"quality aggregate is inconsistent for category {category}")
        return self


class QualityScoreComparison(StrictModel):
    reference: float = Field(ge=0.0)
    candidate: float = Field(ge=0.0)
    delta: float
    retention: float | None = Field(default=None, ge=0.0)


class QualityTaskComparison(QualityScoreComparison):
    task_id: str
    category: str


class QualityComparisonReport(StrictModel):
    schema_version: Literal["axquant.quality-comparison.v1"] = "axquant.quality-comparison.v1"
    reference_model: ModelIdentity
    candidate_model: ModelIdentity
    dataset_sha256: str
    random_seed: int = Field(ge=0)
    aggregate: QualityScoreComparison
    categories: dict[str, QualityScoreComparison]
    perplexity_ratio: float | None = Field(default=None, ge=0.0)
    json_validity: QualityScoreComparison | None = None
    syntax_validity: QualityScoreComparison | None = None
    tasks: list[QualityTaskComparison]
    reference_errors: int = Field(ge=0)
    candidate_errors: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class BenchmarkSuiteManifest(StrictModel):
    schema_version: Literal["axquant.benchmark-suite.v1"] = "axquant.benchmark-suite.v1"
    suite_id: str
    version: str
    random_seed: int = Field(ge=0)
    files: dict[str, str]
    sha256: dict[str, str]
    samples: dict[str, int]
    calibration_disjoint_by_construction: bool = True
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class MtpMetrics(StrictModel):
    token_accuracy: dict[str, float] = Field(default_factory=dict)
    average_accepted_tokens: float | None = Field(default=None, ge=0.0)
    acceptance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rejection_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_tokens_per_forward: float | None = Field(default=None, ge=0.0)
    repetition_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    divergence_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("token_accuracy")
    @classmethod
    def valid_token_accuracy(cls, value: dict[str, float]) -> dict[str, float]:
        if any(score < 0.0 or score > 1.0 for score in value.values()):
            raise ValueError("MTP token accuracies must be within [0, 1]")
        return value

    @model_validator(mode="after")
    def acceptance_and_rejection_are_complements(self) -> MtpMetrics:
        if (
            self.acceptance_rate is not None
            and self.rejection_rate is not None
            and abs(self.acceptance_rate + self.rejection_rate - 1.0) > 1e-9
        ):
            raise ValueError("MTP acceptance and rejection rates must sum to 1")
        return self


class HardwareMetrics(StrictModel):
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    load_seconds: float | None = Field(default=None, ge=0.0)
    prefill_tokens_per_second: float | None = Field(default=None, ge=0.0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0.0)
    mtp_effective_tokens_per_second: float | None = Field(default=None, ge=0.0)
    energy_joules: float | None = Field(default=None, ge=0.0)
    kernel_fallbacks: int | None = Field(default=None, ge=0)
    device_name: str | None = None
    chip: str | None = None
    unified_memory_bytes: int | None = Field(default=None, ge=0)
    os_version: str | None = None


class IntegrityMetrics(StrictModel):
    safetensors_valid: bool
    index_complete: bool
    config_valid: bool
    mtp_layout_valid: bool | None = None
    source_revision_pinned: bool


class EvaluationBundle(StrictModel):
    schema_version: Literal["axquant.evaluation.v1"] = "axquant.evaluation.v1"
    model: ModelIdentity
    runtime: RuntimeName = RuntimeName.AX_ENGINE
    mtp_enabled: bool = False
    baseline_kind: Literal[
        "bf16",
        "uniform-4bit",
        "uniform-6bit",
        "mixed-precision",
        "awq",
        "dwq",
        "gptq",
        "axquant-mtp-off",
        "axquant-mtp-on",
        "candidate",
    ]
    quality: QualityMetrics = Field(default_factory=QualityMetrics)
    mtp: MtpMetrics | None = None
    hardware: HardwareMetrics = Field(default_factory=HardwareMetrics)
    integrity: IntegrityMetrics
    workload: str
    dataset_sha256: str
    software_versions: SoftwareVersions
    random_seed: int = Field(ge=0)
    benchmark_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ValidationThresholds(StrictModel):
    max_perplexity_relative_increase: float = Field(default=0.03, ge=0.0)
    max_task_score_drop: float = Field(default=0.02, ge=0.0)
    max_mtp_acceptance_drop: float = Field(default=0.03, ge=0.0)
    minimum_mtp_acceptance_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    max_mtp_token_accuracy_drop: float = Field(default=0.04, ge=0.0)
    max_structured_output_drop: float = Field(default=0.01, ge=0.0)
    max_repetition_increase: float = Field(default=0.01, ge=0.0)
    max_divergence_increase: float = Field(default=0.01, ge=0.0)
    min_effective_speedup: float = Field(default=1.0, ge=0.0)
    min_prompt_median_speedup: float = Field(default=1.10, ge=0.0)
    max_peak_memory_ratio: float = Field(default=1.0, gt=0.0)
    max_weight_size_ratio: float = Field(default=1.10, gt=0.0)
    minimum_aggregate_quality_retention: float = Field(default=0.98, ge=0.0, le=1.0)
    required_task_scores: tuple[str, ...] = ()
    require_complete_metrics: bool = True
    require_artifact_size: bool = True


class ArtifactSizeEvidence(StrictModel):
    schema_version: Literal["axquant.artifact-size-evidence.v1"] = (
        "axquant.artifact-size-evidence.v1"
    )
    kind: Literal["uniform-4bit", "uniform-6bit", "candidate"]
    model: ModelIdentity
    logical_parameters: int = Field(gt=0)
    weight_bytes: int = Field(gt=0)
    measured_bpw: float = Field(gt=0.0)
    source_sha256: str

    @model_validator(mode="after")
    def measured_bpw_matches_bytes(self) -> ArtifactSizeEvidence:
        expected = 8.0 * self.weight_bytes / self.logical_parameters
        if abs(self.measured_bpw - expected) > 1e-9:
            raise ValueError("artifact size measured BPW does not match byte accounting")
        return self


class ValidationIssue(StrictModel):
    severity: Literal["error", "warning"]
    metric: str
    message: str


class ReleaseExceptionTarget(StrictModel):
    metric: Literal[
        "artifact.weight_size_ratio",
        "artifact.candidate_measured_bpw",
    ]
    observed_value: float = Field(gt=0.0)
    required_minimum: float | None = Field(default=None, gt=0.0)
    required_maximum: float | None = Field(default=None, gt=0.0)
    requirement: str = Field(min_length=1)

    @model_validator(mode="after")
    def observed_value_violates_requirement(self) -> ReleaseExceptionTarget:
        if self.required_minimum is None and self.required_maximum is None:
            raise ValueError("release exception target requires a minimum or maximum")
        if (
            self.required_minimum is not None
            and self.required_maximum is not None
            and self.required_minimum > self.required_maximum
        ):
            raise ValueError("release exception target minimum exceeds maximum")
        below_minimum = (
            self.required_minimum is not None and self.observed_value < self.required_minimum
        )
        above_maximum = (
            self.required_maximum is not None and self.observed_value > self.required_maximum
        )
        if not below_minimum and not above_maximum:
            raise ValueError("release exception target does not violate its requirement")
        return self


class ReleaseException(StrictModel):
    schema_version: Literal["axquant.release-exception.v1"] = "axquant.release-exception.v1"
    exception_id: str = Field(
        min_length=3,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$",
    )
    candidate_model: ModelIdentity
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets: list[ReleaseExceptionTarget] = Field(min_length=2, max_length=2)
    measured_tradeoff: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approval_reference: str = Field(min_length=1)
    approved_at: datetime
    expires_at: datetime
    evidence_sha256: dict[str, str] = Field(min_length=4)

    @model_validator(mode="after")
    def governed_exception_is_complete(self) -> ReleaseException:
        if not is_immutable_revision(self.candidate_model.revision):
            raise ValueError("release exception candidate revision must be immutable")
        if self.approved_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("release exception timestamps must include a timezone")
        if self.expires_at <= self.approved_at:
            raise ValueError("release exception must expire after approval")
        metrics = {target.metric for target in self.targets}
        required_metrics = {
            "artifact.weight_size_ratio",
            "artifact.candidate_measured_bpw",
        }
        if metrics != required_metrics:
            raise ValueError("release exception must cover both governed size targets exactly")
        required_evidence = {
            "plan",
            "candidate_size",
            "size_reference",
            "tradeoff",
        }
        if not required_evidence.issubset(self.evidence_sha256):
            raise ValueError(
                "release exception must bind plan, candidate size, size reference, "
                "and tradeoff evidence"
            )
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in self.evidence_sha256.values()
        ):
            raise ValueError("release exception evidence contains an invalid SHA-256 digest")
        return self


class ValidationReport(StrictModel):
    schema_version: Literal["axquant.validation.v1"] = "axquant.validation.v1"
    reference_model: ModelIdentity
    candidate_model: ModelIdentity
    profile: ProfileName
    target_class: Literal["4bit", "6bit"] = "4bit"
    passed: bool
    thresholds: ValidationThresholds
    issues: list[ValidationIssue]
    comparisons: dict[str, float | int | str | bool | None]
    release_exceptions: list[ReleaseException] = Field(default_factory=list, max_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def status_is_consistent(self) -> ValidationReport:
        expected_passed = not any(issue.severity == "error" for issue in self.issues)
        if self.passed != expected_passed:
            raise ValueError("validation report status is inconsistent with its issues")
        return self


# ---------------------------------------------------------------------------
# v0.2 Benchmark harness schemas
# ---------------------------------------------------------------------------


# Allowlisted AX Engine diagnostic / control environment keys that may be attached to a
# benchmark config. Managed controls such as AX_NO_SPEC are applied by the harness and
# must not be supplied by the caller.
ALLOWED_BENCHMARK_RUNTIME_ENV_KEYS: frozenset[str] = frozenset(
    {
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL",
        "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL",
        "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS",
        "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL",
        "AX_MLX_QWEN_LINEAR_MTP_EXACT",
        "AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE",
        "AX_MLX_MTP_ASYNC_DRAFT",
        "AX_MLX_MTP_LINEAR_EXACT_REPLAY",
        "AX_MLX_MTP_BYPASS_MIN_SAMPLES",
        "AX_MLX_MTP_BYPASS_THRESHOLD",
        "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE",
        "AX_MLX_MTP_MAX_DEPTH",
        "AX_MLX_MTP_MIN_REMAINING_TOKENS",
        "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS",
        "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT",
        "AX_MLX_SPECULATIVE_SPLIT_FFN",
        # MoE-class MTP controls (AXQ-041). A sparse-expert verify step is
        # dominated by fixed per-step cost, not by weight bandwidth, so the
        # axes that decide a MoE Tier 2 outcome are different from the dense
        # ones above and were previously unreachable from a benchmark config.
        # `MLX_MAX_*_PER_BUFFER` are MLX's own Metal command-buffer caps:
        # `gather_qmm` charges the full expert stack against them, which
        # forces a per-layer command-buffer split and turns the engine's
        # `async_eval` submit into a barrier. AX Engine auto-raises them for
        # eligible families and always yields to a caller-set value, so the
        # harness needs them to measure an excluded family such as `qwen3_5`.
        "MLX_MAX_MB_PER_BUFFER",
        "MLX_MAX_OPS_PER_BUFFER",
        "AX_MLX_AUTO_BUFFER_CAPS",
        # Draft-only lm_head requantization. Exactness-neutral by
        # construction: verification re-runs the target lm_head, so these
        # move acceptance rate and draft cost, never the emitted tokens.
        # Material on MoE, where a 248k-entry lm_head is a large share of the
        # per-step active bytes rather than the ~5% it is on a dense sibling.
        "AX_MLX_MTP_DRAFT_LM_HEAD_BITS",
        "AX_MLX_MTP_DRAFT_LM_HEAD_GROUP_SIZE",
        "AX_MLX_MTP_USE_RUNTIME_DRAFT_LM_HEAD",
        "AX_MLX_MOE_LAYER_COMPILE",
        # Layer interval at which the speculative verify build is handed to
        # the GPU, so host graph construction overlaps GPU execution instead
        # of running to completion in front of a blocking eval. Only useful
        # together with the buffer caps above.
        "AX_MLX_MTP_VERIFY_SUBMIT_LAYERS",
        # Layer-boundary residual async_eval hints (`off` / `layer` / `block:N`).
        # Exactness-preserving host/GPU overlap on multi-layer forwards.
        "AX_MLX_PIPELINE_GRANULARITY",
        # Gemma 4 assistant-MTP formal / diagnostic controls (ST2)
        "AX_MLX_GEMMA4_ASSISTANT_MTP",
        "AX_MLX_GEMMA4_ASSISTANT_MTP_MAX_DEPTH",
        "AX_MLX_GEMMA4_ASSISTANT_MTP_REQUIRE_EXACT_PAIR",
        "AX_MLX_GEMMA4_ASSISTANT_MTP_DRAFT_MIN_CONFIDENCE",
        "AX_MLX_GEMMA4_ASSISTANT_MTP_DEEP_DRAFT_MIN_CONFIDENCE",
        "AX_MLX_GEMMA4_ASSISTANT_COMPILE",
        "AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH",
        "AX_MLX_GEMMA4_ASSISTANT_MTP_COALESCED_VERIFY",
        "AX_MLX_GEMMA4_ASSISTANT_MTP_DEBUG",
        # Kill-switch for the greedy sequential production oracle (default ON
        # in AX Engine 6.14.2+). Set 0 to restore multi-token teacher-forced
        # verify for speed experiments; formal Tier 2 must re-check exactness.
        "AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE",
        # MoE long multi-token (dual-edge + qmv-256 identity). Engine product
        # default is fail-closed; formal Gemma assistant-exact profile opts in.
        "AX_MLX_GEMMA4_MOE_LONG_MT",
        # Dense long multi-token bf16 singleton fold (key_len>=512). Engine
        # default ON; formal profile pins it for 12B/31B agent long speed.
        "AX_MLX_DENSE_LONG_MT_BF16_FOLD",
        # Opt-in per-position dense FFN on short multi-token verify (4-bit
        # identity experiments). Default OFF in engine.
        "AX_MLX_GEMMA_MT_PERPOS_FFN",
    }
)


class BenchmarkConfig(StrictModel):
    schema_version: Literal["axquant.benchmark-config.v1"] = "axquant.benchmark-config.v1"
    model: ModelIdentity
    runtime: RuntimeName = RuntimeName.AX_ENGINE
    mtp_enabled: bool = False
    baseline_kind: Literal[
        "bf16",
        "uniform-4bit",
        "uniform-6bit",
        "mixed-precision",
        "awq",
        "dwq",
        "gptq",
        "axquant-mtp-off",
        "axquant-mtp-on",
        "candidate",
    ]
    workload: str
    dataset_sha256: str
    prompt_count: int = Field(ge=1)
    warmup_trials: int = Field(default=2, ge=0)
    measured_trials: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=512, ge=1)
    draft_depth: int | None = Field(default=None, ge=1)
    power_mode: str | None = Field(default=None, min_length=1)
    quantizer: str | None = Field(default=None, min_length=1)
    quantizer_version: str | None = Field(default=None, min_length=1)
    random_seed: int = Field(default=0, ge=0)
    timeout_seconds: float = Field(default=300.0, gt=0.0)
    runtime_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("runtime_env")
    @classmethod
    def _validate_runtime_env(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, raw in value.items():
            name = str(key).strip()
            if not name:
                raise ValueError("runtime_env keys must be non-empty")
            if name == "AX_NO_SPEC":
                raise ValueError("AX_NO_SPEC is managed by the harness from mtp_enabled")
            if name not in ALLOWED_BENCHMARK_RUNTIME_ENV_KEYS:
                allowed = ", ".join(sorted(ALLOWED_BENCHMARK_RUNTIME_ENV_KEYS))
                raise ValueError(f"runtime_env key {name!r} is not allowlisted; allowed: {allowed}")
            text = str(raw).strip()
            if not text:
                raise ValueError(f"runtime_env[{name}] must be a non-empty string")
            if "=" in name or "\n" in name or "\n" in text or "\x00" in text:
                raise ValueError(f"runtime_env entry {name!r} contains invalid characters")
            cleaned[name] = text
        return dict(sorted(cleaned.items()))


class TrialResult(StrictModel):
    trial_index: int = Field(ge=0)
    is_warmup: bool = False
    success: bool = True
    command: list[str] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    tokens_generated: int = Field(default=0, ge=0)
    output_token_ids: list[int] = Field(default_factory=list)
    output_sha256: str | None = None
    latency_seconds: float = Field(default=0.0, ge=0.0)
    time_to_first_token_seconds: float | None = Field(default=None, ge=0.0)
    prefill_seconds: float | None = Field(default=None, ge=0.0)
    decode_seconds: float | None = Field(default=None, ge=0.0)
    tokens_per_second: float = Field(default=0.0, ge=0.0)
    mtp_accepted_tokens: int | None = Field(default=None, ge=0)
    mtp_proposed_tokens: int | None = Field(default=None, ge=0)
    mtp_rejected_tokens: int | None = Field(default=None, ge=0)
    mtp_decode_steps: int | None = Field(default=None, ge=0)
    mtp_active: bool | None = None
    verification_overhead_seconds: float | None = Field(default=None, ge=0.0)
    kernel_fallbacks: int | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    runtime_device_name: str | None = None
    runtime_chip: str | None = None
    unified_memory_bytes: int | None = Field(default=None, ge=0)
    os_version: str | None = None
    terminal_stop_reason: str | None = None
    backend_report: dict[str, JsonValue] = Field(default_factory=dict)
    backend_stderr: str = ""
    error: str | None = None


class BenchmarkResult(StrictModel):
    schema_version: Literal["axquant.benchmark-result.v1"] = "axquant.benchmark-result.v1"
    config: BenchmarkConfig
    trials: list[TrialResult]
    measured_count: int = Field(ge=0)
    failed_count: int = Field(default=0, ge=0)
    timed_out_count: int = Field(default=0, ge=0)
    latency_p50: float | None = Field(default=None, ge=0.0)
    latency_p90: float | None = Field(default=None, ge=0.0)
    latency_p99: float | None = Field(default=None, ge=0.0)
    tokens_per_second_p50: float | None = Field(default=None, ge=0.0)
    tokens_per_second_p90: float | None = Field(default=None, ge=0.0)
    tokens_per_second_p99: float | None = Field(default=None, ge=0.0)
    runtime_device_name: str | None = None
    runtime_chip: str | None = None
    unified_memory_bytes: int | None = Field(default=None, ge=0)
    os_version: str | None = None
    ax_engine_version: str | None = None
    software_versions: SoftwareVersions | None = None
    created_at: datetime = Field(default_factory=utc_now)


class MtpTrialComparison(StrictModel):
    trial_index: int = Field(ge=0)
    outputs_equal: bool
    first_diff_index: int | None = Field(default=None, ge=0)
    direct_output_sha256: str | None = None
    mtp_output_sha256: str | None = None
    direct_token_count: int = Field(default=0, ge=0)
    mtp_token_count: int = Field(default=0, ge=0)
    mtp_proposed_tokens: int | None = Field(default=None, ge=0)
    mtp_accepted_tokens: int | None = Field(default=None, ge=0)
    mtp_rejected_tokens: int | None = Field(default=None, ge=0)
    mtp_active: bool | None = None
    direct_tokens_per_second: float | None = Field(default=None, ge=0.0)
    mtp_tokens_per_second: float | None = Field(default=None, ge=0.0)


class MtpPhaseTimingSummary(StrictModel):
    direct_output_tokens: int = Field(ge=1)
    mtp_output_tokens: int = Field(ge=1)
    direct_generation_wall_us: int = Field(ge=0)
    mtp_generation_wall_us: int = Field(ge=0)
    direct_generation_us_per_output_token: float = Field(ge=0.0)
    mtp_generation_us_per_output_token: float = Field(ge=0.0)
    target_mtp_us_per_output_token: float = Field(ge=0.0)
    required_savings_us_per_output_token: float = Field(ge=0.0)
    draft_wall_us: int = Field(ge=0)
    verify_forward_wall_us: int = Field(ge=0)
    verify_eval_wall_us: int = Field(ge=0)
    rollback_wall_us: int = Field(ge=0)
    cache_clone_wall_us: int = Field(ge=0)
    accept_wall_us: int = Field(ge=0)
    tail_sample_wall_us: int = Field(ge=0)
    phase_accounted_us_per_output_token: float = Field(ge=0.0)
    mtp_decode_steps: int = Field(ge=0)
    direct_fallback_steps: int = Field(ge=0)
    active_step_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    mtp_emitted_tokens: int = Field(ge=0)
    proposed_tokens: int = Field(ge=0)
    accepted_tokens: int = Field(ge=0)
    correctness_mode_conflicts: int = Field(ge=0)


class MtpAbComparison(StrictModel):
    """Soft MTP off/on comparison used for diagnostics and fail-closed release checks."""

    schema_version: Literal["axquant.mtp-ab-comparison.v1"] = "axquant.mtp-ab-comparison.v1"
    profile_name: str
    model: ModelIdentity | None = None
    runtime: RuntimeName | None = None
    workload: str | None = None
    dataset_sha256: str | None = None
    random_seed: int | None = Field(default=None, ge=0)
    generation_controls: dict[str, JsonValue] = Field(default_factory=dict)
    runtime_env: dict[str, str] = Field(default_factory=dict)
    draft_depth: int | None = Field(default=None, ge=1)
    exactness_pass: bool
    divergent_trial_count: int = Field(default=0, ge=0)
    measured_trial_count: int = Field(default=0, ge=0)
    failed_trial_count: int = Field(default=0, ge=0)
    direct_tokens_per_second_p50: float | None = Field(default=None, ge=0.0)
    mtp_tokens_per_second_p50: float | None = Field(default=None, ge=0.0)
    direct_token_weighted_decode_tps: float | None = Field(default=None, ge=0.0)
    mtp_token_weighted_decode_tps: float | None = Field(default=None, ge=0.0)
    prompt_median_speedup: float | None = Field(default=None, ge=0.0)
    token_weighted_decode_speedup: float | None = Field(default=None, ge=0.0)
    speedup_metric: Literal[
        "prompt-median-tps",
        "token-weighted-decode-tps",
    ] = "prompt-median-tps"
    speedup: float | None = Field(default=None, ge=0.0)
    minimum_speedup: float = Field(default=1.20, ge=0.0)
    minimum_prompt_median_speedup: float = Field(default=1.10, ge=0.0)
    prompt_median_speedup_pass: bool = False
    speedup_pass: bool = False
    release_ready: bool = False
    ax_engine_version: str | None = None
    runtime_chip: str | None = None
    software_versions: SoftwareVersions | None = None
    phase_timing: MtpPhaseTimingSummary | None = None
    trial_comparisons: list[MtpTrialComparison] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def release_ready_comparison_is_fully_bound(self) -> MtpAbComparison:
        """Keep legacy v1 comparisons readable but never release-ready when unbound."""
        if not self.release_ready:
            return self
        if not self.exactness_pass or not self.speedup_pass or self.issues:
            raise ValueError("release-ready MTP A/B comparison has failing checks")
        if (
            self.model is None
            or not is_immutable_revision(self.model.revision)
            or self.runtime is None
            or not self.workload
            or not self.dataset_sha256
            or self.random_seed is None
            or not self.generation_controls
            or not self.runtime_chip
            or not self.ax_engine_version
            or self.software_versions is None
        ):
            raise ValueError("release-ready MTP A/B comparison is missing environment bindings")
        if self.software_versions.ax_engine != self.ax_engine_version:
            raise ValueError("MTP A/B comparison AX Engine versions are inconsistent")
        if self.speedup_metric == "token-weighted-decode-tps" and (
            self.direct_token_weighted_decode_tps is None
            or self.direct_token_weighted_decode_tps <= 0.0
            or self.mtp_token_weighted_decode_tps is None
            or self.mtp_token_weighted_decode_tps <= 0.0
            or self.token_weighted_decode_speedup is None
            or not self.prompt_median_speedup_pass
        ):
            raise ValueError("release-ready token-weighted MTP evidence is incomplete")
        return self


class MtpDiagnosticReport(StrictModel):
    """Kill-switch and depth sweep report for M2 exactness/speed diagnosis.

    This is development evidence unless a profile is both exact and meets the
    configured speedup floor under strict (non-optimistic) MTP.
    """

    schema_version: Literal["axquant.mtp-diagnostic.v1"] = "axquant.mtp-diagnostic.v1"
    model: ModelIdentity
    workload: str
    dataset_sha256: str
    minimum_speedup: float = Field(default=1.20, ge=0.0)
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED_DEVELOPMENT
    profiles: list[MtpAbComparison]
    any_exactness_pass: bool = False
    any_release_ready: bool = False
    recommended_next_step: str
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# v0.4 Quantizer plugin and refinement schemas
# ---------------------------------------------------------------------------


class QuantizerExecutionRecord(StrictModel):
    method: QuantMethod
    module_path: str
    bits: int = Field(ge=2, le=16)
    group_size: int | None = Field(default=None, ge=1)
    success: bool
    fallback: bool = False
    note: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class QuantizerExecutionManifest(StrictModel):
    schema_version: Literal["axquant.quantizer-execution.v1"] = "axquant.quantizer-execution.v1"
    plan_sha256: str
    records: list[QuantizerExecutionRecord]
    created_at: datetime = Field(default_factory=utc_now)


class CandidateEntry(StrictModel):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    parent_id: str | None = None
    plan_sha256: str
    change_description: str
    reason: str
    predicted_bpw: float = Field(ge=0.0)
    measured_bpw: float | None = Field(default=None, ge=0.0)
    predicted_loss: float = Field(ge=0.0)
    measured_loss: float | None = Field(default=None, ge=0.0)
    budget_impact: float
    state: Literal["selected", "rejected", "pending"]


class RefinementConfig(StrictModel):
    schema_version: Literal["axquant.refinement-config.v1"] = "axquant.refinement-config.v1"
    top_n: int = Field(default=3, ge=1, le=20)
    max_iterations: int = Field(default=10, ge=1)
    evaluation_budget: int = Field(default=50, ge=1)
    wall_clock_seconds: float = Field(default=86400.0, gt=0.0)
    convergence_threshold: float = Field(default=0.001, ge=0.0)
    swap_radius: int = Field(default=5, ge=1)
    random_seed: int = Field(default=0, ge=0)
    # QP1: optional expected digest of a holdout RefinementMeasurementSet.
    # When set, select_complete_candidate requires a matching measurement set.
    holdout_measurement_set_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class RefinementResult(StrictModel):
    schema_version: Literal["axquant.refinement.v2"] = "axquant.refinement.v2"
    config: RefinementConfig
    history: list[CandidateEntry]
    candidate_plans: dict[str, QuantizationPlan]
    selected_candidate_id: str
    selected_plan: QuantizationPlan
    selected_plan_sha256: str
    selection_basis: Literal["proxy", "complete-model"]
    # QP1: proxy-only runs are always development evidence. ADR-0004:
    # interaction-development marks measured selection on non-holdout roles.
    evidence_label: Literal["proxy-development", "holdout-bound", "interaction-development"] = (
        "proxy-development"
    )
    holdout_measurement_set_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    # ADR-0004: digest of the non-holdout measurement set that drove
    # interaction optimization; never doubles as the holdout binding above.
    interaction_measurement_set_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    warnings: list[str] = Field(default_factory=list)
    iterations_used: int = Field(ge=0)
    evaluations_used: int = Field(ge=0)
    converged: bool
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def selection_is_present_in_history(self) -> RefinementResult:
        selected = self.candidate_plans.get(self.selected_candidate_id)
        if selected is None or selected != self.selected_plan:
            raise ValueError("selected refinement plan is missing from candidate plans")
        history_entry = next(
            (entry for entry in self.history if entry.candidate_id == self.selected_candidate_id),
            None,
        )
        if history_entry is None or history_entry.plan_sha256 != self.selected_plan_sha256:
            raise ValueError("selected refinement plan digest is inconsistent")
        return self


class CompleteCandidateHardware(StrictModel):
    device_name: str = Field(min_length=1)
    chip: str = Field(min_length=1)
    unified_memory_bytes: int = Field(gt=0)
    os_version: str = Field(min_length=1)
    ax_engine_version: str = Field(min_length=1)
    mlx_version: str = Field(min_length=1)
    mlx_lm_version: str = Field(min_length=1)
    power_mode: str = Field(min_length=1)
    kernel_fallbacks: int = Field(ge=0)


class CompleteCandidateMeasurement(StrictModel):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    measurement_id: str = Field(default="", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    candidate_model: ModelIdentity
    profile: ProfileName
    plan_sha256: str
    artifact_manifest_sha256: str
    quality_comparison_sha256: str
    validation_sha256: str
    measured_bpw: float = Field(gt=0.0)
    objective_loss: float = Field(ge=0.0)
    quality_retention: float = Field(ge=0.0)
    perplexity_ratio: float | None = Field(default=None, gt=0.0)
    mtp_acceptance_retention: float = Field(ge=0.0)
    mtp_speedup: float = Field(ge=0.0)
    peak_memory_ratio: float = Field(gt=0.0)
    hardware: CompleteCandidateHardware
    validation_passed: bool
    # ADR-0004 provenance: which campaign dataset role produced this
    # measurement. Empty on legacy artifacts; interaction optimization fails
    # closed on empty or formal-holdout roles.
    dataset_role: str = Field(default="", pattern=r"^$|^[a-z][a-z0-9-]*$")

    @model_validator(mode="after")
    def default_measurement_id(self) -> CompleteCandidateMeasurement:
        if not self.measurement_id:
            object.__setattr__(self, "measurement_id", self.candidate_id)
        return self


class RefinementMeasurementSet(StrictModel):
    schema_version: Literal["axquant.refinement-measurements.v5"] = (
        "axquant.refinement-measurements.v5"
    )
    refinement_sha256: str
    evaluator_version: str
    measurements: list[CompleteCandidateMeasurement] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def measurement_ids_are_unique(self) -> RefinementMeasurementSet:
        measurement_ids = [measurement.measurement_id for measurement in self.measurements]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("complete-candidate measurement IDs must be unique")
        return self


class ParetoPoint(StrictModel):
    candidate_id: str
    measurement_id: str = Field(default="", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    candidate_model: ModelIdentity
    plan_sha256: str
    measured_bpw: float = Field(gt=0.0)
    quality_retention: float = Field(ge=0.0)
    mtp_acceptance_retention: float = Field(ge=0.0)
    mtp_speedup: float = Field(ge=0.0)
    peak_memory_ratio: float = Field(gt=0.0)
    hardware: CompleteCandidateHardware
    validation_passed: bool
    frontier: bool
    dominated_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_measurement_id(self) -> ParetoPoint:
        if not self.measurement_id:
            object.__setattr__(self, "measurement_id", self.candidate_id)
        return self


class ParetoReport(StrictModel):
    schema_version: Literal["axquant.pareto.v3"] = "axquant.pareto.v3"
    profile: ProfileName
    measurement_set_sha256: str
    points: list[ParetoPoint] = Field(min_length=1)
    frontier_candidate_ids: list[str]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def frontier_summary_is_consistent(self) -> ParetoReport:
        expected = sorted({point.candidate_id for point in self.points if point.frontier})
        if self.frontier_candidate_ids != expected:
            raise ValueError("Pareto frontier candidate summary is inconsistent")
        measurement_ids = [point.measurement_id for point in self.points]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("Pareto points must have unique measurement IDs")
        return self


class HardwareRegistryCandidateInput(StrictModel):
    entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    measurement_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    plan_file: str = Field(min_length=1)
    artifact_manifest_file: str = Field(min_length=1)
    sensitivity_file: str = Field(min_length=1)
    quality_comparison_file: str = Field(min_length=1)
    validation_file: str = Field(min_length=1)
    direct_evaluation_file: str = Field(min_length=1)
    mtp_evaluation_file: str = Field(min_length=1)
    direct_benchmark_result_file: str = Field(min_length=1)
    mtp_benchmark_result_file: str = Field(min_length=1)
    quantizer_execution_file: str = Field(min_length=1)


class HardwareRegistryRequest(StrictModel):
    schema_version: Literal["axquant.hardware-registry-request.v3"] = (
        "axquant.hardware-registry-request.v3"
    )
    registry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    measurement_set_file: str = Field(min_length=1)
    deterministic_tolerance: float = Field(default=0.0, ge=0.0)
    candidates: list[HardwareRegistryCandidateInput] = Field(min_length=1)

    @model_validator(mode="after")
    def entry_ids_are_unique(self) -> HardwareRegistryRequest:
        entry_ids = [candidate.entry_id for candidate in self.candidates]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("hardware registry entry IDs must be unique")
        return self


class HardwareMeasurementProtocol(StrictModel):
    protocol_id: str = Field(min_length=1)
    backend: Literal["ax-engine-bench"] = "ax-engine-bench"
    backend_version: str = Field(min_length=1)
    runtime: Literal[RuntimeName.AX_ENGINE] = RuntimeName.AX_ENGINE
    dataset_sha256: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    prompt_count: int = Field(ge=1)
    warmup_trials: int = Field(ge=0)
    measured_trials: int = Field(ge=1)
    power_mode: str = Field(min_length=1)
    deterministic_tolerance: float = Field(ge=0.0)
    direct_commands: list[list[str]] = Field(min_length=1)
    mtp_commands: list[list[str]] = Field(min_length=1)

    @field_validator("direct_commands", "mtp_commands")
    @classmethod
    def commands_are_executable(cls, value: list[list[str]]) -> list[list[str]]:
        if any(not command or any(not argument for argument in command) for command in value):
            raise ValueError("hardware measurement commands must contain non-empty arguments")
        return value


class HardwareKernelCoverage(StrictModel):
    bits: int = Field(ge=2, le=16)
    group_size: int | None = Field(default=None, ge=1)
    method: QuantMethod
    roles: list[TensorRole] = Field(min_length=1)
    shapes: list[tuple[int, ...]] = Field(min_length=1)
    module_count: int = Field(ge=1)
    parameter_count: int = Field(ge=1)
    quantizer_execution_records: int = Field(ge=0)
    kernel_evidence: Literal["measured", "unmeasured"]

    @model_validator(mode="after")
    def precision_fields_are_consistent(self) -> HardwareKernelCoverage:
        if self.bits == 16:
            if self.method != QuantMethod.BF16 or self.group_size is not None:
                raise ValueError("16-bit hardware coverage must be ungrouped BF16")
        elif self.method == QuantMethod.BF16 or self.group_size is None:
            raise ValueError("quantized hardware coverage requires a method and group size")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("hardware coverage roles must be unique")
        if len(self.shapes) != len(set(self.shapes)):
            raise ValueError("hardware coverage shapes must be unique")
        return self


class HardwareRegistryEntry(StrictModel):
    entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    measurement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    candidate_model: ModelIdentity
    profile: ProfileName
    plan_file: str = Field(min_length=1)
    plan_file_sha256: str
    plan_sha256: str
    artifact_manifest_file: str = Field(min_length=1)
    artifact_manifest_sha256: str
    sensitivity_file: str = Field(min_length=1)
    sensitivity_sha256: str
    quality_comparison_file: str = Field(min_length=1)
    quality_comparison_sha256: str
    validation_file: str = Field(min_length=1)
    validation_sha256: str
    direct_evaluation_file: str = Field(min_length=1)
    direct_evaluation_sha256: str
    mtp_evaluation_file: str = Field(min_length=1)
    mtp_evaluation_sha256: str
    direct_benchmark_result_file: str = Field(min_length=1)
    direct_benchmark_result_sha256: str
    mtp_benchmark_result_file: str = Field(min_length=1)
    mtp_benchmark_result_sha256: str
    quantizer_execution_file: str = Field(min_length=1)
    quantizer_execution_sha256: str
    hardware: CompleteCandidateHardware
    protocol: HardwareMeasurementProtocol
    coverage: list[HardwareKernelCoverage] = Field(min_length=1)
    total_modules: int = Field(ge=1)
    unique_shapes: int = Field(ge=1)
    kernel_evidence: Literal["measured", "unmeasured"]
    validation_passed: bool
    release_ready: bool
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def release_status_is_consistent(self) -> HardwareRegistryEntry:
        expected = self.validation_passed and self.kernel_evidence == "measured" and not self.issues
        if self.release_ready != expected:
            raise ValueError("hardware registry entry release status is inconsistent")
        expected_coverage = self.kernel_evidence
        if any(item.kernel_evidence != expected_coverage for item in self.coverage):
            raise ValueError("hardware registry coverage evidence is inconsistent")
        return self


class HardwareProfileRegistry(StrictModel):
    schema_version: Literal["axquant.hardware-registry.v3"] = "axquant.hardware-registry.v3"
    registry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    measurement_set_sha256: str
    measurement_set_file: str = Field(min_length=1)
    measurement_set_file_sha256: str
    entries: list[HardwareRegistryEntry] = Field(min_length=1)
    distinct_named_hosts: int = Field(ge=0)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def registry_status_is_consistent(self) -> HardwareProfileRegistry:
        entry_ids = [entry.entry_id for entry in self.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("hardware registry entries must have unique entry IDs")
        measurement_ids = [entry.measurement_id for entry in self.entries]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("hardware registry entries must have unique measurement IDs")
        expected_hosts = len(
            {
                (
                    entry.hardware.device_name,
                    entry.hardware.chip,
                    entry.hardware.unified_memory_bytes,
                    entry.hardware.os_version,
                )
                for entry in self.entries
                if entry.release_ready
            }
        )
        if self.distinct_named_hosts != expected_hosts:
            raise ValueError("hardware registry named-host count is inconsistent")
        expected_ready = bool(expected_hosts) and all(entry.release_ready for entry in self.entries)
        if self.release_ready != (expected_ready and not self.issues):
            raise ValueError("hardware registry release status is inconsistent")
        return self


class CompatibilityCandidateInput(StrictModel):
    artifact_directory: str = Field(min_length=1)
    ax_engine_check: str = Field(min_length=1)
    mlx_lm_check: str = Field(min_length=1)
    validation_report: str = Field(min_length=1)


class OfficialDenseCheckpointRequirement(StrictModel):
    model_id: str = Field(pattern=r"^Qwen/Qwen3\.6-[A-Za-z0-9._-]+$")
    parameter_size: str = Field(pattern=r"^[1-9][0-9]*(?:\.[0-9]+)?[BMT]$")


class CompatibilityMatrixRequest(StrictModel):
    schema_version: Literal["axquant.compatibility-request.v2"] = "axquant.compatibility-request.v2"
    family: Literal["Qwen 3.6"] = "Qwen 3.6"
    scope_policy: Literal["all-official-dense-sizes-at-release"] = (
        "all-official-dense-sizes-at-release"
    )
    official_catalog_url: Literal["https://huggingface.co/collections/Qwen/qwen36"] = (
        "https://huggingface.co/collections/Qwen/qwen36"
    )
    catalog_verified_at: datetime
    required_dense_models: list[OfficialDenseCheckpointRequirement] = Field(min_length=1)
    required_profiles: list[ProfileName] = Field(
        default_factory=lambda: [
            ProfileName.AGENT_CODING,
            ProfileName.GENERAL,
        ],
        min_length=2,
        max_length=2,
    )
    candidates: list[CompatibilityCandidateInput] = Field(min_length=1)

    @model_validator(mode="after")
    def release_scope_is_complete(self) -> CompatibilityMatrixRequest:
        if self.catalog_verified_at.tzinfo is None:
            raise ValueError("official catalog verification timestamp must include a timezone")
        requirements = [
            (requirement.model_id, requirement.parameter_size)
            for requirement in self.required_dense_models
        ]
        if len({model_id for model_id, _size in requirements}) != len(requirements):
            raise ValueError("official dense model IDs must be unique")
        if len({size for _model_id, size in requirements}) != len(requirements):
            raise ValueError("official dense parameter sizes must be unique")
        if set(self.required_profiles) != {
            ProfileName.AGENT_CODING,
            ProfileName.GENERAL,
        }:
            raise ValueError("compatibility scope requires agent-coding and general profiles")
        evidence_keys = [
            (candidate.artifact_directory, candidate.validation_report)
            for candidate in self.candidates
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("compatibility candidate/profile evidence must be unique")
        return self


class CheckpointCompatibility(StrictModel):
    candidate_model: ModelIdentity
    source_model: ModelIdentity
    profile: ProfileName
    artifact_path: str
    artifact_manifest_sha256: str
    plan_sha256: str
    adapter_id: str
    dense: bool
    text_layer_count: int | None = Field(default=None, ge=1)
    measured_total_bpw: float = Field(gt=0.0)
    mtp_present: bool
    supported_bits: list[int] = Field(min_length=1)
    ax_engine_check_sha256: str
    ax_engine_passed: bool
    mlx_lm_check_sha256: str
    mlx_lm_passed: bool
    validation_sha256: str
    validation_passed: bool
    compatible: bool
    issues: list[str] = Field(default_factory=list)


class CompatibilityMatrix(StrictModel):
    schema_version: Literal["axquant.compatibility-matrix.v2"] = "axquant.compatibility-matrix.v2"
    family: Literal["Qwen 3.6"] = "Qwen 3.6"
    scope_policy: Literal["all-official-dense-sizes-at-release"] = (
        "all-official-dense-sizes-at-release"
    )
    official_catalog_url: Literal["https://huggingface.co/collections/Qwen/qwen36"] = (
        "https://huggingface.co/collections/Qwen/qwen36"
    )
    catalog_verified_at: datetime
    required_dense_models: list[OfficialDenseCheckpointRequirement] = Field(min_length=1)
    required_profiles: list[ProfileName] = Field(
        min_length=2,
        max_length=2,
    )
    required_dense_checkpoints: int = Field(ge=1)
    entries: list[CheckpointCompatibility] = Field(min_length=1)
    distinct_dense_source_checkpoints: int = Field(ge=0)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class BenchmarkEvidenceInput(StrictModel):
    kind: BenchmarkEvidenceKind
    status: Literal["available", "unavailable"]
    evaluation_file: str | None = None
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def availability_fields_are_consistent(self) -> BenchmarkEvidenceInput:
        if self.status == "available":
            if not self.evaluation_file or self.unavailable_reason is not None:
                raise ValueError("available benchmark evidence requires only an evaluation file")
        elif self.evaluation_file is not None or not self.unavailable_reason:
            raise ValueError("unavailable benchmark evidence requires only a reason")
        return self


class BenchmarkEvidenceRequest(StrictModel):
    schema_version: Literal["axquant.benchmark-evidence-request.v1"] = (
        "axquant.benchmark-evidence-request.v1"
    )
    profile: ProfileName
    entries: list[BenchmarkEvidenceInput] = Field(min_length=1)

    @model_validator(mode="after")
    def all_baselines_are_explicit(self) -> BenchmarkEvidenceRequest:
        kinds = [entry.kind for entry in self.entries]
        if len(kinds) != len(set(kinds)):
            raise ValueError("benchmark evidence kinds must be unique")
        missing = sorted(set(BenchmarkEvidenceKind) - set(kinds))
        if missing:
            raise ValueError(
                f"benchmark evidence request must explicitly list every baseline: {missing}"
            )
        return self


class BenchmarkEvidenceEntry(StrictModel):
    kind: BenchmarkEvidenceKind
    status: Literal["available", "unavailable"]
    evaluation_file: str | None = None
    evaluation_sha256: str | None = None
    model: ModelIdentity | None = None
    runtime: RuntimeName | None = None
    mtp_enabled: bool | None = None
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def availability_fields_are_consistent(self) -> BenchmarkEvidenceEntry:
        available_fields = (
            self.evaluation_file,
            self.evaluation_sha256,
            self.model,
            self.runtime,
            self.mtp_enabled,
        )
        if self.status == "available":
            if (
                any(value is None for value in available_fields)
                or self.unavailable_reason is not None
            ):
                raise ValueError("available benchmark index entry is incomplete")
        elif any(value is not None for value in available_fields) or not self.unavailable_reason:
            raise ValueError("unavailable benchmark index entry must contain only a reason")
        return self


class BenchmarkEvidenceIndex(StrictModel):
    schema_version: Literal["axquant.benchmark-evidence-index.v1"] = (
        "axquant.benchmark-evidence-index.v1"
    )
    profile: ProfileName
    dataset_sha256: str | None = None
    random_seed: int | None = Field(default=None, ge=0)
    entries: list[BenchmarkEvidenceEntry] = Field(min_length=1)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def complete_and_consistent(self) -> BenchmarkEvidenceIndex:
        kinds = [entry.kind for entry in self.entries]
        if len(kinds) != len(set(kinds)) or set(kinds) != set(BenchmarkEvidenceKind):
            raise ValueError("benchmark evidence index must list every baseline exactly once")
        if self.release_ready != (not self.issues):
            raise ValueError("benchmark evidence release status is inconsistent with its issues")
        return self


class ReleaseValidationInput(StrictModel):
    profile: ProfileName
    validation_file: str = Field(min_length=1)
    benchmark_index_file: str = Field(min_length=1)


class ReleaseValidationRequest(StrictModel):
    schema_version: Literal["axquant.release-validation-request.v1"] = (
        "axquant.release-validation-request.v1"
    )
    entries: list[ReleaseValidationInput] = Field(min_length=2)

    @model_validator(mode="after")
    def required_profiles_are_explicit(self) -> ReleaseValidationRequest:
        profiles = [entry.profile for entry in self.entries]
        required = {ProfileName.AGENT_CODING, ProfileName.GENERAL}
        if len(profiles) != len(set(profiles)) or set(profiles) != required:
            raise ValueError("release validation requires exactly agent-coding and general")
        return self


class ReleaseValidationEntry(StrictModel):
    profile: ProfileName
    validation_file: str
    validation_sha256: str
    benchmark_index_file: str
    benchmark_index_sha256: str
    reference_model: ModelIdentity
    candidate_model: ModelIdentity
    dataset_sha256: str
    passed: bool


class ReleaseValidationIndex(StrictModel):
    schema_version: Literal["axquant.release-validation-index.v1"] = (
        "axquant.release-validation-index.v1"
    )
    entries: list[ReleaseValidationEntry] = Field(min_length=2)
    release_ready: bool
    issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def complete_and_consistent(self) -> ReleaseValidationIndex:
        profiles = [entry.profile for entry in self.entries]
        required = {ProfileName.AGENT_CODING, ProfileName.GENERAL}
        if len(profiles) != len(set(profiles)) or set(profiles) != required:
            raise ValueError("release validation index must contain both required profiles")
        expected_ready = all(entry.passed for entry in self.entries) and not self.issues
        if self.release_ready != expected_ready:
            raise ValueError(
                "release validation status is inconsistent with its entries and issues"
            )
        return self


class RefinementExecutionRequest(StrictModel):
    schema_version: Literal["axquant.refinement-execution-request.v2"] = (
        "axquant.refinement-execution-request.v2"
    )
    refinement_file: str = Field(min_length=1)
    source_model: str = Field(min_length=1)
    mtp_sidecar: str = Field(min_length=1)
    mtp_layout: MtpSidecarLayout = MtpSidecarLayout.BYTE_PRESERVED
    calibration_manifest: str = Field(min_length=1)
    quality_dataset: str = Field(min_length=1)
    reference_quality: str = Field(min_length=1)
    reference_evaluation: str = Field(min_length=1)
    benchmark_prompts: str = Field(min_length=1)
    size_reference: str = Field(min_length=1)
    candidate_repository_prefix: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    profile: ProfileName = ProfileName.AGENT_CODING
    quality_max_sequence_length: int = Field(default=2048, ge=1)
    quality_max_tokens: int = Field(default=256, ge=1)
    quality_max_samples: int | None = Field(default=None, ge=1)
    benchmark_trials: int = Field(default=5, ge=1)
    benchmark_warmup: int = Field(default=2, ge=0)
    benchmark_max_tokens: int = Field(default=512, ge=1)
    benchmark_draft_depth: int | None = Field(default=None, ge=1)
    benchmark_power_mode: str = Field(min_length=1)
    benchmark_timeout_seconds: float = Field(default=300.0, gt=0.0)
    random_seed: int = Field(default=0, ge=0)
    axquant_executable: str = Field(default="axquant", min_length=1)
    ax_engine_executable: str = Field(default="ax-engine-bench", min_length=1)


class RefinementExecutionStep(StrictModel):
    candidate_id: str | None = None
    step_id: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    expected_outputs: list[str] = Field(default_factory=list)
    acceptable_exit_codes: list[int] = Field(default_factory=lambda: [0], min_length=1)
    state: Literal["pending", "completed", "failed", "skipped"] = "pending"
    exit_code: int | None = None
    gate_passed: bool | None = None
    output_sha256: dict[str, str] = Field(default_factory=dict)
    stderr: str = ""


class RefinementExecutionManifest(StrictModel):
    schema_version: Literal["axquant.refinement-execution.v1"] = "axquant.refinement-execution.v1"
    request_sha256: str
    refinement_sha256: str
    profile: ProfileName
    input_sha256: dict[str, str]
    steps: list[RefinementExecutionStep] = Field(min_length=1)
    measured_candidate_ids: list[str] = Field(default_factory=list)
    failed_candidate_ids: list[str] = Field(default_factory=list)
    selected_result: str | None = None
    pareto_report: str | None = None
    complete: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def step_ids_are_unique(self) -> RefinementExecutionManifest:
        keys = [(step.candidate_id, step.step_id) for step in self.steps]
        if len(keys) != len(set(keys)):
            raise ValueError("refinement execution step IDs must be unique per candidate")
        return self


class ReleaseAuditRequest(StrictModel):
    schema_version: Literal["axquant.release-audit-request.v4"] = "axquant.release-audit-request.v4"
    artifact_directory: str = Field(min_length=1)
    feasibility_report: str = Field(min_length=1)
    sensitivity_report: str = Field(min_length=1)
    sensitivity_lineage: list[str] = Field(default_factory=list)
    refinement_result: str = Field(min_length=1)
    release_validation_index: str = Field(min_length=1)
    hardware_registry: str = Field(min_length=1)
    pareto_report: str = Field(min_length=1)
    compatibility_matrix: str = Field(min_length=1)
    compatibility_request: str = Field(min_length=1)
    reproduction_recipe: str = Field(min_length=1)
    reproduction_verification: str = Field(min_length=1)
    ax_engine_check: str = Field(min_length=1)
    mlx_lm_check: str = Field(min_length=1)
    toolkit_wheel: str = Field(min_length=1)
    required_toolkit_version: str = Field(default="1.0.0", min_length=1)
    required_dense_checkpoints: int = Field(default=1, ge=1)
    release_exceptions: list[str] = Field(default_factory=list, max_length=1)
    release_exception_evidence: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exception_evidence_is_complete(self) -> ReleaseAuditRequest:
        if not self.release_exceptions and self.release_exception_evidence:
            raise ValueError("release exception evidence requires a release exception")
        if self.release_exceptions:
            required_evidence = {
                "plan",
                "candidate_size",
                "size_reference",
                "tradeoff",
            }
            if not required_evidence.issubset(self.release_exception_evidence):
                raise ValueError(
                    "release audit exception evidence must include plan, candidate size, "
                    "size reference, and tradeoff"
                )
        return self


class ReleaseAuditCheck(StrictModel):
    gate_id: Literal["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    name: str = Field(min_length=1)
    passed: bool
    evidence_sha256: dict[str, str] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_is_consistent(self) -> ReleaseAuditCheck:
        if self.passed != (not self.issues):
            raise ValueError("release audit check status is inconsistent with its issues")
        return self


class ReleaseAudit(StrictModel):
    schema_version: Literal["axquant.release-audit.v4"] = "axquant.release-audit.v4"
    request_sha256: str
    candidate_model: ModelIdentity
    source_model: ModelIdentity
    toolkit_version: str | None = None
    wheel_sha256: str
    checks: list[ReleaseAuditCheck] = Field(min_length=9, max_length=9)
    release_ready: bool
    blockers: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def complete_and_consistent(self) -> ReleaseAudit:
        expected = {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"}
        gate_ids = [check.gate_id for check in self.checks]
        if len(gate_ids) != len(set(gate_ids)) or set(gate_ids) != expected:
            raise ValueError("release audit must contain M0 through M8 exactly once")
        expected_blockers = [
            f"{check.gate_id}: {issue}" for check in self.checks for issue in check.issues
        ]
        if self.blockers != expected_blockers:
            raise ValueError("release audit blockers are inconsistent with gate checks")
        if self.release_ready != all(check.passed for check in self.checks):
            raise ValueError("release audit readiness is inconsistent with gate checks")
        return self


# ---------------------------------------------------------------------------
# P0/P1/P2 productization artifacts (probe capacity, scoreboard, unified bind)
# ---------------------------------------------------------------------------


class ProbeCapacityModeAssessment(StrictModel):
    mode: ProbeMode
    feasible: bool
    estimated_bytes: int = Field(ge=0)
    evidence_kind: EvidenceKind
    release_quality_eligible: bool
    reason: str
    notes: list[str] = Field(default_factory=list)


class ProbeCapacityReport(StrictModel):
    schema_version: Literal["axquant.probe-capacity.v1"] = "axquant.probe-capacity.v1"
    model: ModelIdentity | None = None
    parameter_count: int = Field(gt=0)
    available_memory_bytes: int | None = Field(default=None, gt=0)
    headroom_fraction: float = Field(gt=0.0, le=1.0)
    recommended_mode: ProbeMode
    modes: list[ProbeCapacityModeAssessment]
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ScoreboardMetricRow(StrictModel):
    metric_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: Literal["available", "pass", "fail", "unavailable"]
    value: float | str | None = None
    threshold: float | str | None = None
    unit: str | None = None
    owner: str = "axquant"
    reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class ScoreboardReport(StrictModel):
    schema_version: Literal["axquant.scoreboard.v1"] = "axquant.scoreboard.v1"
    certification_tier: Literal["checkpoint", "mtp-acceleration"] = "checkpoint"
    title: str
    plan_profile: ProfileName | None = None
    profile: ProfileName
    source_model: ModelIdentity
    plan_sha256: str = Field(min_length=64, max_length=64)
    evidence_kind: EvidenceKind
    overall_status: Literal["pass", "fail", "incomplete"]
    rows: list[ScoreboardMetricRow]
    missing_mandatory: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class UnifiedSensitivityBinding(StrictModel):
    """Digest binding that ties weight + KV sensitivity into one plan lineage (P1)."""

    schema_version: Literal["axquant.unified-sensitivity.v1"] = "axquant.unified-sensitivity.v1"
    source_model: ModelIdentity
    profile: ProfileName
    weight_sensitivity_sha256: str = Field(min_length=64, max_length=64)
    weight_evidence_kind: EvidenceKind
    kv_sensitivity_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    kv_allocation_basis: Literal["architecture-prior", "measured", "off"] = "off"
    inventory_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class RecoveryTargetRanking(StrictModel):
    """Sensitivity-ordered recovery targets (P2; opt-in, not convert-implied)."""

    schema_version: Literal["axquant.recovery-ranking.v1"] = "axquant.recovery-ranking.v1"
    plan_sha256: str = Field(min_length=64, max_length=64)
    sensitivity_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    targets: list[str]
    scores: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
