from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.enums import EvidenceKind, ProfileName, RuntimeName
from axquant.schema.planning import ObjectiveWeights

DeploymentEvidenceKind = Literal[
    "architecture_prior",
    "imported-as-estimate",
    "estimate",
    "measured",
]
WeightBytesBasis = Literal["plan-estimate", "artifact-manifest"]


class MemoryBudgetBreakdown(StrictModel):
    """Pure byte accounting used by the deployment-planning gate."""

    weight_bytes: int = Field(ge=0)
    kv_bytes: int = Field(ge=0)
    reserve_bytes: int = Field(ge=0)
    limit_bytes: int = Field(gt=0)
    remainder_bytes: int
    feasible: bool

    @model_validator(mode="after")
    def accounting_is_exact(self) -> MemoryBudgetBreakdown:
        expected_remainder = self.limit_bytes - (
            self.weight_bytes + self.kv_bytes + self.reserve_bytes
        )
        if self.remainder_bytes != expected_remainder:
            raise ValueError("memory-budget remainder does not match byte accounting")
        if self.feasible != (expected_remainder >= 0):
            raise ValueError("memory-budget feasibility does not match byte accounting")
        return self


class DeploymentPlan(StrictModel):
    """Bound weight + KV deployment decision for one requested memory limit."""

    schema_version: Literal["axquant.deployment-plan.v1"] = "axquant.deployment-plan.v1"
    weight_bytes: int = Field(ge=0)
    kv_bytes: int = Field(ge=0)
    reserve_bytes: int = Field(ge=0)
    limit_bytes: int = Field(gt=0)
    remainder_bytes: int
    feasible: bool
    evidence_kind: DeploymentEvidenceKind
    source_evidence_kind: EvidenceKind
    context_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    profile: ProfileName
    target_class: str = Field(min_length=1)
    runtime: RuntimeName
    mode: Literal["balanced", "quality", "low-memory", "speed"]
    objective: ObjectiveWeights
    minimum_quality_retention: float = Field(ge=0.0, le=1.0)
    weight_bytes_basis: WeightBytesBasis
    measured_main_bpw: float | None = Field(default=None, gt=0.0, le=16.0)
    estimated_main_bpw: float | None = Field(default=None, gt=0.0, le=16.0)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kv_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    notes: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def deployment_contract_is_consistent(self) -> DeploymentPlan:
        expected_remainder = self.limit_bytes - (
            self.weight_bytes + self.kv_bytes + self.reserve_bytes
        )
        if self.remainder_bytes != expected_remainder:
            raise ValueError("deployment remainder does not match byte accounting")
        if self.feasible != (expected_remainder >= 0):
            raise ValueError("deployment feasibility does not match byte accounting")
        if self.runtime is not RuntimeName.AX_ENGINE:
            raise ValueError("AX Engine is the only deployment runtime supported in v1.8")
        if self.weight_bytes_basis == "artifact-manifest":
            if self.measured_main_bpw is None or self.estimated_main_bpw is not None:
                raise ValueError("manifest-backed deployment requires measured main BPW only")
        elif self.estimated_main_bpw is None or self.measured_main_bpw is not None:
            raise ValueError("plan-backed deployment requires estimated main BPW only")
        if (
            self.source_evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR
            and self.evidence_kind != "architecture_prior"
        ):
            raise ValueError("architecture-prior input must remain architecture-prior evidence")
        if (
            self.source_evidence_kind is EvidenceKind.IMPORTED
            and self.evidence_kind != "imported-as-estimate"
        ):
            raise ValueError("imported input must remain imported-as-estimate evidence")
        if self.evidence_kind == "measured":
            if self.weight_bytes_basis != "artifact-manifest":
                raise ValueError("estimated deployment accounting cannot be labeled measured")
            if self.source_evidence_kind is not EvidenceKind.MEASURED:
                raise ValueError("measured deployment evidence requires measured sensitivity")
        return self
