"""Frozen public certificate records under ``docs/certifications/``.

These models own:

* ``axquant.public-checkpoint-certification.v1``
* ``axquant.public-mtp-acceleration-certification.v1``

Campaign evidence blobs (quality / size / plan details) remain free-form maps so
historical shape variants load without rewriting evidence. The **envelope**
used by documentation generation and Hub claims is strict and frozen under
AXQ-042.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from axquant.revisions import is_immutable_revision
from axquant.schema._base import StrictModel

_SHA256 = r"^[0-9a-f]{64}$"
_REPO_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"

CHECKPOINT_SCHEMA_VERSION: Literal["axquant.public-checkpoint-certification.v1"] = (
    "axquant.public-checkpoint-certification.v1"
)
MTP_SCHEMA_VERSION: Literal["axquant.public-mtp-acceleration-certification.v1"] = (
    "axquant.public-mtp-acceleration-certification.v1"
)

PublicCheckpointStatus = Literal["certified", "not_certified"]
PublicMtpRecordStatus = Literal["certified", "not_certified"]
MtpAccelerationStatus = Literal[
    "not-certified",
    "not-applicable",
    "certified",
    "certified-scoped",
    "certified-see-tier2-record",
]


class PublicIndexMeta(StrictModel):
    """Index metadata required on every checkpoint Tier 1 public record."""

    display_name: str = Field(min_length=1)
    sort_order: int
    edition_label: str = Field(min_length=1)
    listed: bool = True


class PublicCertArtifact(StrictModel):
    """Artifact identity shared by Tier 1 and Tier 2 public certificates."""

    hub_repo_id: str = Field(pattern=_REPO_ID)
    hub_commit: str = Field(min_length=8)
    product_class: str = Field(min_length=1)
    candidate_manifest_sha256: str | None = Field(default=None, pattern=_SHA256)
    source_model_id: str | None = Field(default=None, min_length=1)
    source_revision: str | None = None
    upstream_model_id: str | None = Field(default=None, min_length=1)
    architecture: str | None = Field(default=None, min_length=1)
    artifact_edition: str | None = Field(default=None, min_length=1)
    hub_tag: str | None = Field(default=None, min_length=1)
    marketing_name: str | None = Field(default=None, min_length=1)


class PublicMtpAccelerationBlock(StrictModel):
    """Tier 1 MTP acceleration status block (claim + optional evidence pointers)."""

    status: MtpAccelerationStatus
    reason: str | None = Field(default=None, min_length=1)
    tier2_certificate: str | None = Field(default=None, min_length=1)
    scope: str | None = Field(default=None, min_length=1)
    ax_engine_version: str | None = Field(default=None, min_length=1)
    engine_binary_sha256: str | None = Field(default=None, pattern=_SHA256)
    exactness_required: float | None = None
    prompt_median_speedup_required: float | None = None
    token_weighted_decode_speedup_required: float | None = None


class PublicCheckpointCertification(StrictModel):
    """Published checkpoint Tier 1 certificate (``*-tier1.json``)."""

    schema_version: Literal["axquant.public-checkpoint-certification.v1"] = (
        CHECKPOINT_SCHEMA_VERSION
    )
    status: PublicCheckpointStatus
    certification_tier: Literal["checkpoint"] = "checkpoint"
    host_id: str = Field(min_length=1)
    certified_at: datetime | None = None
    evaluated_at: datetime | None = None
    artifact: PublicCertArtifact
    plan: dict[str, Any]
    size: dict[str, Any]
    quality: dict[str, Any]
    thresholds: dict[str, Any]
    mtp_acceleration: PublicMtpAccelerationBlock
    toolchain: dict[str, Any]
    public_index: PublicIndexMeta
    notes: list[str] | None = None
    evidence_hashes: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    quality_bound_pre_fuse_revision: str | None = Field(default=None, min_length=1)
    datasets: dict[str, Any] | None = None
    evidence_scope: str | None = Field(default=None, min_length=1)
    runtime_default: dict[str, Any] | None = None
    weight_files: dict[str, Any] | None = None

    @model_validator(mode="after")
    def timestamp_present(self) -> PublicCheckpointCertification:
        if self.certified_at is None and self.evaluated_at is None:
            raise ValueError("public checkpoint certificate requires certified_at or evaluated_at")
        return self

    @property
    def event_timestamp(self) -> datetime:
        stamp = self.certified_at or self.evaluated_at
        assert stamp is not None
        return stamp


class PublicMtpAccelerationCertification(StrictModel):
    """Published MTP acceleration Tier 2 certificate (``*-tier2.json``)."""

    schema_version: Literal["axquant.public-mtp-acceleration-certification.v1"] = MTP_SCHEMA_VERSION
    status: PublicMtpRecordStatus
    certification_tier: Literal["mtp-acceleration"] = "mtp-acceleration"
    host_id: str = Field(min_length=1)
    certified_at: datetime | None = None
    artifact: PublicCertArtifact
    thresholds: dict[str, Any]
    mtp_acceleration: dict[str, Any]
    toolchain: dict[str, Any]
    related_certificates: dict[str, Any] | None = None
    evidence_package: str | dict[str, Any] | None = None

    @field_validator("mtp_acceleration")
    @classmethod
    def mtp_status_present(cls, value: dict[str, Any]) -> dict[str, Any]:
        status = value.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("mtp_acceleration.status is required")
        return value


def load_public_checkpoint_certification(
    path: str | Path,
) -> PublicCheckpointCertification:
    payload = Path(path).read_text(encoding="utf-8")
    return PublicCheckpointCertification.model_validate_json(payload)


def load_public_mtp_acceleration_certification(
    path: str | Path,
) -> PublicMtpAccelerationCertification:
    payload = Path(path).read_text(encoding="utf-8")
    return PublicMtpAccelerationCertification.model_validate_json(payload)


def require_immutable_hub_commit(commit: str) -> str:
    if not is_immutable_revision(commit):
        raise ValueError("Hub commit must be a full immutable commit SHA")
    return commit
