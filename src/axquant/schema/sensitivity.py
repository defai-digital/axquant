from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import SoftwareVersions, StrictModel, utc_now
from axquant.schema.enums import EvidenceKind, ProfileName, QuantMethod
from axquant.schema.inventory import ArchitectureProfile, ModelIdentity, TensorSpec


class MetricVector(StrictModel):
    output_kl: float = Field(default=0.0, ge=0.0)
    hidden_state_error: float = Field(default=0.0, ge=0.0)
    cosine_distance: float = Field(default=0.0, ge=0.0)
    token_disagreement: float = Field(default=0.0, ge=0.0)
    task_loss_delta: float = Field(default=0.0, ge=0.0)
    mtp_acceptance_loss: float = Field(default=0.0, ge=0.0)
    long_context_loss: float = Field(default=0.0, ge=0.0)
    peak_memory_cost: float = Field(default=0.0, ge=0.0)
    prefill_latency_cost: float = Field(default=0.0, ge=0.0)
    decode_latency_cost: float = Field(default=0.0, ge=0.0)


class CandidateMeasurement(StrictModel):
    bits: int = Field(ge=2, le=16)
    method: QuantMethod
    group_size: int | None = Field(default=None, ge=1)
    metrics: MetricVector
    supported: bool = True
    evidence_scope: Literal["tensor", "module-group", "preserved"] = "tensor"
    measured_tokens: int = Field(default=0, ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def bf16_has_no_group(self) -> CandidateMeasurement:
        if self.bits == 16 and self.method != QuantMethod.BF16:
            raise ValueError("16-bit candidates must use the bf16 method")
        if self.method == QuantMethod.BF16 and self.bits != 16:
            raise ValueError("bf16 candidates must use 16 bits")
        if self.bits < 16 and self.group_size is None:
            raise ValueError("quantized candidates require a group size")
        return self


class CalibrationEvidence(StrictModel):
    dataset_id: str
    dataset_sha256: str
    samples: int = Field(ge=1)
    domains: list[str]
    sequence_length: int = Field(ge=1)
    backend: str
    reference: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CalibrationManifest(StrictModel):
    schema_version: Literal["axquant.calibration.v1"] = "axquant.calibration.v1"
    model: ModelIdentity
    profile: ProfileName
    dataset_id: str
    dataset_sha256: str
    samples: int = Field(ge=1)
    domains: list[str]
    sequence_length: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    tokenizer_revision: str | None = None
    calibration_evaluation_separation_attested: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class TensorSensitivity(StrictModel):
    tensor: TensorSpec
    candidates: list[CandidateMeasurement]

    @model_validator(mode="after")
    def unique_candidates(self) -> TensorSensitivity:
        keys = [(candidate.bits, candidate.method) for candidate in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate candidates for {self.tensor.name}")
        return self


class SensitivityReport(StrictModel):
    schema_version: Literal["axquant.sensitivity.v1"] = "axquant.sensitivity.v1"
    model: ModelIdentity
    architecture_profile: ArchitectureProfile = Field(default_factory=ArchitectureProfile)
    profile: ProfileName
    evidence_kind: EvidenceKind
    inventory_sha256: str
    entries: list[TensorSensitivity]
    calibration: CalibrationEvidence | None = None
    created_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def measured_has_calibration(self) -> SensitivityReport:
        if self.evidence_kind != EvidenceKind.ARCHITECTURE_PRIOR and self.calibration is None:
            raise ValueError("measured or imported reports require calibration provenance")
        return self


class ProbeConfig(StrictModel):
    schema_version: Literal["axquant.probe-config.v1"] = "axquant.probe-config.v1"
    model: ModelIdentity
    calibration_cache: str
    profile: ProfileName = ProfileName.AGENT_CODING
    candidate_bits: tuple[int, ...] = (4, 6, 8, 16)
    candidate_methods: tuple[QuantMethod, ...] = (QuantMethod.AFFINE,)
    target_tensors: tuple[str, ...] = ()
    group_size: int = Field(default=64, ge=1)
    token_budget_per_candidate: int = Field(default=2048, ge=1)
    replay_batch_size: int = Field(default=1, ge=1)
    metric_positions_per_sample: int = Field(default=32, ge=1)
    long_context_min_tokens: int = Field(default=1024, ge=2)
    warmup_replays: int = Field(default=1, ge=1)
    module_group_probing: bool = False
    early_termination_factor: float = Field(default=3.0, gt=1.0)
    capture_points: tuple[str, ...] = ("output", "hidden")
    random_seed: int = Field(default=0, ge=0)

    @field_validator("candidate_bits")
    @classmethod
    def valid_probe_bits(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(value)))
        if not normalized or any(bits < 2 or bits > 16 for bits in normalized):
            raise ValueError("probe candidate bits must be within [2, 16]")
        return normalized

    @field_validator("candidate_methods")
    @classmethod
    def valid_probe_methods(
        cls,
        value: tuple[QuantMethod, ...],
    ) -> tuple[QuantMethod, ...]:
        normalized = tuple(sorted(set(value), key=lambda method: method.value))
        supported = {QuantMethod.AFFINE, QuantMethod.DWQ}
        if not normalized or set(normalized) - supported:
            raise ValueError("probe methods must contain only affine and/or dwq")
        return normalized

    @field_validator("target_tensors")
    @classmethod
    def unique_target_tensors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not tensor.strip() for tensor in value):
            raise ValueError("probe target tensor names must be non-empty")
        return tuple(sorted(set(value)))


class TokenizedCacheManifest(StrictModel):
    schema_version: Literal["axquant.tokenized-cache.v1"] = "axquant.tokenized-cache.v1"
    cache_key_sha256: str
    model: ModelIdentity
    dataset_sha256: str
    profile: ProfileName
    domains: list[str] = Field(default_factory=list)
    domain_provenance: Literal["sample-records", "declared"] = "declared"
    sequence_length: int = Field(ge=1)
    samples: int = Field(ge=1)
    shard_count: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    tokenizer_revision: str | None = None
    tokenizer_sha256: str | None = None
    sample_order_sha256: str | None = None
    calibration_manifest_sha256: str | None = None
    calibration_evaluation_separation_attested: bool = False
    backend_version: str = "axquant-tokenizer-v1"
    shard_sha256: dict[str, str] = Field(default_factory=dict)
    software_versions: SoftwareVersions
    complete: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ProbeProgress(StrictModel):
    schema_version: Literal["axquant.probe-progress.v1"] = "axquant.probe-progress.v1"
    inventory_sha256: str
    config_sha256: str
    completed_tensors: dict[str, list[CandidateMeasurement]] = Field(default_factory=dict)
    total_tensors: int = Field(default=0, ge=0)
    complete: bool = False
    created_at: datetime = Field(default_factory=utc_now)
