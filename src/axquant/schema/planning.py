from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import SoftwareVersions, StrictModel, utc_now
from axquant.schema.enums import (
    EvidenceKind,
    OutlierStrategy,
    ProfileName,
    QuantMethod,
    RuntimeName,
    ScaleStrategy,
    TensorRole,
)
from axquant.schema.inventory import ArchitectureProfile, ModelIdentity
from axquant.schema.sensitivity import (
    CalibrationEvidence,
    CandidateMeasurement,
    MetricVector,
)


class ObjectiveWeights(StrictModel):
    output_kl: float = Field(ge=0.0)
    hidden_state_error: float = Field(ge=0.0)
    cosine_distance: float = Field(ge=0.0)
    token_disagreement: float = Field(ge=0.0)
    task_loss_delta: float = Field(ge=0.0)
    mtp_acceptance_loss: float = Field(ge=0.0)
    long_context_loss: float = Field(ge=0.0)
    peak_memory_cost: float = Field(ge=0.0)
    prefill_latency_cost: float = Field(ge=0.0)
    decode_latency_cost: float = Field(ge=0.0)

    @model_validator(mode="after")
    def nonzero(self) -> ObjectiveWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one objective weight must be positive")
        return self

    def normalized(self) -> dict[str, float]:
        values = {key: float(value) for key, value in self.model_dump().items()}
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}


class MtpPolicy(StrictModel):
    mode: Literal["protected", "adaptive", "disabled"] = "protected"
    candidate_bits: tuple[int, ...] = (8, 16)
    min_bits: int = Field(default=8, ge=2, le=16)
    preserve_external_sidecar: bool = True
    protect_norms: bool = True
    protect_output_head: bool = True
    optimize_for_acceptance: bool = True

    @field_validator("candidate_bits")
    @classmethod
    def valid_candidate_bits(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(value)))
        if not normalized or any(bits < 2 or bits > 16 for bits in normalized):
            raise ValueError("MTP candidate bits must be within [2, 16]")
        return normalized


class HardwareProfile(StrictModel):
    # v2 adds the experimental low-bit range: MLX affine kernels execute 2-
    # and 3-bit natively, and AX Engine admits them behind the documented
    # AX_ENGINE_2BIT_EXPERIMENTAL=1 / AX_ENGINE_3BIT_EXPERIMENTAL=1 gates.
    # Low-bit artifacts are development evidence; release certification
    # still runs the ordinary quality/runtime gates.
    name: str = "ax-engine-apple-silicon-affine-dwq-v2"
    runtime: RuntimeName = RuntimeName.AX_ENGINE
    supported_bits: tuple[int, ...] = (2, 3, 4, 6, 8, 16)
    supported_methods: tuple[QuantMethod, ...] = (
        QuantMethod.AFFINE,
        QuantMethod.AWQ,
        QuantMethod.DWQ,
        QuantMethod.BF16,
    )
    supported_group_sizes: tuple[int, ...] = (32, 64, 128)


class PlanningConstraints(StrictModel):
    effective_bpw_limit: float = Field(default=4.8, gt=0.0, le=16.0)
    max_model_size_ratio_to_uniform4: float = Field(default=1.10, gt=0.0)
    minimum_quality_retention: float = Field(default=0.98, ge=0.0, le=1.0)
    minimum_mtp_acceptance_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_mtp_speedup: float = Field(default=1.20, ge=0.0)
    # AXQ-026: the LM-head weight floor may be lowered from BF16 to 8-bit as
    # the approved size-gate path. The default stays BF16; a lowered floor is
    # recorded here so audits and validation see the governed deviation.
    lm_head_min_bits: Literal[8, 16] = 16


class ManualPrecisionRule(StrictModel):
    rule_id: str = Field(min_length=1)
    bits: int = Field(ge=2, le=16)
    method: QuantMethod
    tensor_glob: str | None = None
    module_glob: str | None = None
    roles: tuple[TensorRole, ...] = ()
    group_size: int | None = Field(default=None, ge=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_rule(self) -> ManualPrecisionRule:
        if not self.tensor_glob and not self.module_glob and not self.roles:
            raise ValueError("manual precision rules require at least one selector")
        if self.bits == 16:
            if self.method != QuantMethod.BF16:
                raise ValueError("16-bit manual rules must use bf16")
            if self.group_size is not None:
                raise ValueError("BF16 manual rules cannot define a group size")
        elif self.method == QuantMethod.BF16:
            raise ValueError("quantized manual rules cannot use bf16")
        return self


class ManualPlanRecipe(StrictModel):
    schema_version: Literal["axquant.manual-recipe.v1"] = "axquant.manual-recipe.v1"
    profile: ProfileName = ProfileName.AGENT_CODING
    target_bpw: float = Field(default=6.0, gt=0.0, le=16.0)
    default_bits: int = Field(default=4, ge=2, le=16)
    default_method: QuantMethod = QuantMethod.AFFINE
    group_size: int = Field(default=64, ge=1)
    rules: list[ManualPrecisionRule] = Field(default_factory=list)
    allow_unmatched_rules: bool = False
    mtp: MtpPolicy = Field(default_factory=MtpPolicy)
    hardware: HardwareProfile = Field(default_factory=HardwareProfile)
    max_model_size_ratio_to_uniform4: float = Field(default=1.10, gt=0.0)
    minimum_quality_retention: float = Field(default=0.98, ge=0.0, le=1.0)
    minimum_mtp_acceptance_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_mtp_speedup: float = Field(default=1.20, ge=0.0)
    # AXQ-026 opt-in: 8 lowers the LM-head weight floor for this recipe only.
    lm_head_min_bits: Literal[8, 16] = 16
    target_mode: Literal["balanced", "quality", "low-memory", "speed"] = "balanced"
    primary_runtime: RuntimeName = RuntimeName.AX_ENGINE
    random_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def valid_recipe(self) -> ManualPlanRecipe:
        if self.default_bits == 16 and self.default_method != QuantMethod.BF16:
            raise ValueError("a 16-bit manual default must use bf16")
        if self.default_bits < 16 and self.default_method == QuantMethod.BF16:
            raise ValueError("a quantized manual default cannot use bf16")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("manual precision rule IDs must be unique")
        if self.primary_runtime != RuntimeName.AX_ENGINE:
            raise ValueError("AX Engine is the only supported primary runtime")
        if self.hardware.runtime != self.primary_runtime:
            raise ValueError("hardware profile runtime does not match the primary runtime")
        return self


class PlanRequest(StrictModel):
    schema_version: Literal["axquant.plan-request.v1"] = "axquant.plan-request.v1"
    profile: ProfileName
    target_bpw: float = Field(gt=0.0, le=16.0)
    candidate_bits: tuple[int, ...] = (4, 6, 8, 16)
    group_size: int = Field(default=64, ge=1)
    # Empty means use ``group_size`` only. Non-empty expands the planner grid (AXQ-028).
    candidate_group_sizes: tuple[int, ...] = ()
    # Empty means no extra method filter beyond hardware support.
    candidate_methods: tuple[QuantMethod, ...] = ()
    allow_unmeasured: bool = False
    candidate_count: int = Field(default=1, ge=1)
    random_seed: int = Field(default=0, ge=0)
    target_mode: Literal["balanced", "quality", "low-memory", "speed"] = "balanced"
    primary_runtime: RuntimeName = RuntimeName.AX_ENGINE
    max_model_size_ratio_to_uniform4: float = Field(default=1.10, gt=0.0)
    minimum_quality_retention: float = Field(default=0.98, ge=0.0, le=1.0)
    minimum_mtp_acceptance_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_mtp_speedup: float = Field(default=1.20, ge=0.0)
    # AXQ-026 opt-in: 8 lowers the LM-head weight floor for this plan only.
    lm_head_min_bits: Literal[8, 16] = 16
    hardware: HardwareProfile = Field(default_factory=HardwareProfile)
    mtp: MtpPolicy = Field(default_factory=MtpPolicy)

    @field_validator("candidate_bits")
    @classmethod
    def valid_candidate_bits(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(value)))
        if not normalized or any(bits < 2 or bits > 16 for bits in normalized):
            raise ValueError("candidate bits must be within [2, 16]")
        return normalized

    @field_validator("candidate_group_sizes")
    @classmethod
    def valid_candidate_group_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            return ()
        normalized = tuple(sorted(set(value)))
        if any(size < 1 for size in normalized):
            raise ValueError("candidate group sizes must be positive")
        return normalized

    @field_validator("candidate_methods")
    @classmethod
    def valid_candidate_methods(cls, value: tuple[QuantMethod, ...]) -> tuple[QuantMethod, ...]:
        if not value:
            return ()
        return tuple(sorted(set(value), key=lambda method: method.value))

    def effective_group_sizes(self) -> tuple[int, ...]:
        return self.candidate_group_sizes or (self.group_size,)

    @model_validator(mode="after")
    def supported_configuration(self) -> PlanRequest:
        if self.primary_runtime != RuntimeName.AX_ENGINE:
            raise ValueError("AX Engine is the only supported primary runtime")
        unsupported = set(self.candidate_bits) - set(self.hardware.supported_bits)
        if unsupported:
            raise ValueError(f"hardware profile does not support bits {sorted(unsupported)}")
        for size in self.effective_group_sizes():
            if size not in self.hardware.supported_group_sizes:
                raise ValueError(f"hardware profile does not support group size {size}")
        if self.candidate_methods:
            unsupported_methods = set(self.candidate_methods) - set(self.hardware.supported_methods)
            # BF16 is always allowed for 16-bit candidates even if omitted from the filter.
            unsupported_methods.discard(QuantMethod.BF16)
            if unsupported_methods:
                raise ValueError(
                    "hardware profile does not support methods "
                    f"{sorted(method.value for method in unsupported_methods)}"
                )
        return self


class KvLayerSensitivity(StrictModel):
    layer_index: int = Field(ge=0)
    candidates: list[CandidateMeasurement]

    @model_validator(mode="after")
    def unique_candidates(self) -> KvLayerSensitivity:
        from axquant.schema.sensitivity import candidate_key

        keys = [
            candidate_key(candidate.bits, candidate.method, candidate.group_size)
            for candidate in self.candidates
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate KV candidates for layer {self.layer_index}")
        return self


class KvSensitivityReport(StrictModel):
    """Measured per-layer KV-cache sensitivity (AXQ-024)."""

    schema_version: Literal["axquant.kv-sensitivity.v1"] = "axquant.kv-sensitivity.v1"
    model: ModelIdentity
    architecture_profile: ArchitectureProfile = Field(default_factory=ArchitectureProfile)
    profile: ProfileName
    evidence_kind: EvidenceKind
    inventory_sha256: str
    probe_backend: str
    group_size: int = Field(ge=1)
    text_layer_count: int = Field(ge=1)
    entries: list[KvLayerSensitivity]
    calibration: CalibrationEvidence | None = None
    created_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_and_provenanced(self) -> KvSensitivityReport:
        indices = sorted(entry.layer_index for entry in self.entries)
        if indices != list(range(self.text_layer_count)):
            raise ValueError("KV sensitivity entries must cover every text layer exactly once")
        if self.evidence_kind != EvidenceKind.ARCHITECTURE_PRIOR and self.calibration is None:
            raise ValueError("measured KV sensitivity requires calibration provenance")
        return self


class KvLayerAllocation(StrictModel):
    layer_index: int = Field(ge=0)
    bits: int = Field(ge=2, le=16)
    group_size: int = Field(ge=1)
    reason: str


class KvCachePlan(StrictModel):
    """Optional per-layer KV-cache precision plan (AXQ-021).

    Absence of this section preserves weight-only planning exactly. The measured
    allocation basis requires the semantic digest of its producing
    KvSensitivityReport (AXQ-024); conversion rejects an unbound measured plan.
    """

    schema_version: Literal["axquant.kv-plan.v1"] = "axquant.kv-plan.v1"
    allocation_basis: Literal["architecture-prior", "measured"]
    min_bits: int = Field(default=4, ge=2, le=16)
    default_bits: int = Field(ge=2, le=16)
    default_group_size: int = Field(default=64, ge=1)
    # Semantic digest of the producing KvSensitivityReport; required for the
    # measured basis (AXQ-024) and absent for architecture priors.
    sensitivity_sha256: str | None = None
    # Selection budget used by the measured allocator so publication can
    # reproduce the allocation from the packaged report (AXQ-025).
    max_output_kl: float | None = Field(default=None, gt=0.0)
    layers: list[KvLayerAllocation]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_and_floored(self) -> KvCachePlan:
        indices = [layer.layer_index for layer in self.layers]
        if not indices:
            raise ValueError("KV-cache plan requires at least one layer")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("KV-cache layers must cover 0..n-1 without gaps or duplicates")
        if any(layer.bits < self.min_bits for layer in self.layers):
            raise ValueError("KV-cache layer bits cannot fall below the policy floor")
        return self


class Allocation(StrictModel):
    tensor: str
    module_path: str
    role: TensorRole
    parameters: int = Field(ge=0)
    bits: int = Field(ge=2, le=16)
    method: QuantMethod
    group_size: int | None = Field(default=None, ge=1)
    predicted_loss: float = Field(ge=0.0)
    metrics: MetricVector
    reason: str
    # AXQ-028: first-class scale / outlier strategy (additive; old plans default).
    scale_strategy: ScaleStrategy = ScaleStrategy.GROUP_AFFINE
    outlier_strategy: OutlierStrategy = OutlierStrategy.NONE
    strategy_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class PrecisionShare(StrictModel):
    parameters: int = Field(ge=0)
    fraction: float = Field(ge=0.0, le=1.0)


class QuantizationPlan(StrictModel):
    schema_version: Literal["axquant.plan.v1"] = "axquant.plan.v1"
    quantizer: Literal["axquant"] = "axquant"
    status: Literal["planned"] = "planned"
    source_model: ModelIdentity
    architecture_profile: ArchitectureProfile = Field(default_factory=ArchitectureProfile)
    profile: ProfileName
    target_class: str
    target_bpw: float
    nominal_bpw: float
    effective_bpw: float
    candidate_bits: tuple[int, ...]
    group_size: int
    # Effective group-size grid used when planning (AXQ-028). Empty on legacy plans.
    candidate_group_sizes: tuple[int, ...] = ()
    objective: ObjectiveWeights
    hardware: HardwareProfile
    mtp: MtpPolicy
    constraints: PlanningConstraints
    target_mode: Literal["balanced", "quality", "low-memory", "speed"]
    primary_runtime: RuntimeName = RuntimeName.AX_ENGINE
    random_seed: int
    software_versions: SoftwareVersions
    global_validation_required: bool = True
    analysis_sha256: str
    evidence_kind: EvidenceKind
    calibration: CalibrationEvidence | None = None
    assignments: list[Allocation]
    weight_distribution: dict[str, PrecisionShare]
    mtp_distribution: dict[str, PrecisionShare]
    kv_cache: KvCachePlan | None = None
    created_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)
