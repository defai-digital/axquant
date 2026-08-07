from __future__ import annotations

import math
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.campaign import BoundFile
from axquant.schema.flagship import CandidateKey

_SHA256 = r"^[0-9a-f]{64}$"


class BoundMetricClaim(StrictModel):
    evidence: BoundFile
    profile: Literal["agent-coding", "general", "hardware"]
    metric_key: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    value: float
    numerator: float | None = None
    denominator: float | None = None
    comparison: Literal["absolute", "higher-is-better", "lower-is-better", "ratio"]

    @model_validator(mode="after")
    def ratio_has_operands(self) -> BoundMetricClaim:
        if self.comparison == "ratio" and (
            self.numerator is None or self.denominator is None or self.denominator == 0
        ):
            raise ValueError("ratio claims require a numerator and nonzero denominator")
        if (
            self.comparison == "ratio"
            and self.numerator is not None
            and self.denominator is not None
            and not math.isclose(
                self.value,
                self.numerator / self.denominator,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("ratio claim value must equal numerator divided by denominator")
        return self


class PublicClaimManifest(StrictModel):
    schema_version: Literal["axquant.public-claim.v1"] = "axquant.public-claim.v1"
    candidate: CandidateKey
    lifecycle_event_sha256: str = Field(pattern=_SHA256)
    audit_sha256: str = Field(pattern=_SHA256)
    public_repository: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    target_class: str = Field(min_length=1)
    measured_main_bpw: float = Field(gt=0, le=16)
    measured_total_bpw: float = Field(gt=0, le=16)
    weight_bytes: int = Field(gt=0)
    hardware_scope_ids: list[str]
    runtime_versions: dict[str, str]
    quality_claims: list[BoundMetricClaim]
    performance_claims: list[BoundMetricClaim]
    limitations: list[str]
    evidence_index: list[BoundFile]
    generated_at: datetime = Field(default_factory=utc_now)

    @field_validator("hardware_scope_ids", "limitations", "evidence_index")
    @classmethod
    def nonempty_lists(cls, value: list[object]) -> list[object]:
        if not value:
            raise ValueError("certified public claims require non-empty evidence and limitations")
        return value

    @model_validator(mode="after")
    def claim_contract(self) -> PublicClaimManifest:
        if "df-macbookpro-m5" not in self.hardware_scope_ids:
            raise ValueError("certified performance scope must include df-macbookpro-m5")
        if not self.runtime_versions or any(
            not key.strip() or not value.strip() for key, value in self.runtime_versions.items()
        ):
            raise ValueError("public claim runtime versions must be complete")
        if not self.quality_claims or not self.performance_claims:
            raise ValueError("certified public claim requires quality and performance evidence")
        return self


class PublicClaimRenderRequest(StrictModel):
    schema_version: Literal["axquant.public-claim-render-request.v1"] = (
        "axquant.public-claim-render-request.v1"
    )
    authorization_audit: str
    lifecycle_registry: str
    artifact_manifest: str
    public_owner: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    quality_claims: list[BoundMetricClaim]
    performance_claims: list[BoundMetricClaim]
    limitations: list[str]
    reviewer: str = Field(min_length=1)

    @field_validator(
        "authorization_audit",
        "lifecycle_registry",
        "artifact_manifest",
    )
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not value
            or value != normalized
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("public claim render paths must be safe relative paths")
        return value
