from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.artifacts import ReleaseAuditCheck
from axquant.schema.campaign import BoundFile
from axquant.schema.flagship import CandidateKey
from axquant.schema.inventory import ModelIdentity

_SHA256 = r"^[0-9a-f]{64}$"
_PROFILE_KEYS = {"agent-coding", "general"}


def _relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not value
        or value != normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("flagship audit paths must be safe relative paths")
    return value


class FormalHoldoutCompletion(StrictModel):
    schema_version: Literal["axquant.formal-holdout-completion.v1"] = (
        "axquant.formal-holdout-completion.v1"
    )
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate_sha256: str = Field(pattern=_SHA256)
    started_at: datetime
    completed_at: datetime
    unconsumed_at_start: Literal[True] = True
    consumed_at_completion: Literal[True] = True
    dataset_sha256_by_profile: dict[str, str]
    result_file_by_profile: dict[str, BoundFile]
    raw_outputs_archived: Literal[True] = True
    raw_evidence_index: BoundFile
    evaluation_custodian: str = Field(min_length=1)
    custodian_attestation: BoundFile
    verdict: Literal["pass", "fail"]
    gate_issues: list[str]

    @model_validator(mode="after")
    def profiles_and_time_are_complete(self) -> FormalHoldoutCompletion:
        if set(self.dataset_sha256_by_profile) != _PROFILE_KEYS:
            raise ValueError("formal holdout completion requires agent-coding and general datasets")
        if set(self.result_file_by_profile) != _PROFILE_KEYS:
            raise ValueError("formal holdout completion requires both profile result files")
        if self.completed_at <= self.started_at:
            raise ValueError("formal holdout completion must end after it starts")
        if self.verdict == "pass" and self.gate_issues:
            raise ValueError("passing formal holdout completion cannot contain gate issues")
        if self.verdict == "fail" and not self.gate_issues:
            raise ValueError("failing formal holdout completion must explain its gate issues")
        return self


class IndependentReviewRecord(StrictModel):
    schema_version: Literal["axquant.flagship-review.v1"] = "axquant.flagship-review.v1"
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate_sha256: str = Field(pattern=_SHA256)
    legacy_audit_sha256: str = Field(pattern=_SHA256)
    reviewer: str = Field(min_length=1)
    checks_reviewed: list[
        Literal[
            "freeze",
            "candidate-selection",
            "raw-to-summary",
            "hardware-pareto",
            "reproduction",
            "public-claims",
        ]
    ]
    verdict: Literal["pass", "fail"]
    issues: list[str]
    attestation: BoundFile
    reviewed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> IndependentReviewRecord:
        required = {
            "freeze",
            "candidate-selection",
            "raw-to-summary",
            "hardware-pareto",
            "reproduction",
            "public-claims",
        }
        if set(self.checks_reviewed) != required:
            raise ValueError("independent review must cover every flagship review domain")
        if self.verdict == "pass" and self.issues:
            raise ValueError("passing independent review cannot contain issues")
        if self.verdict == "fail" and not self.issues:
            raise ValueError("failing independent review must explain its issues")
        return self


class FinalPublicationReviewRecord(StrictModel):
    schema_version: Literal["axquant.flagship-publication-review.v1"] = (
        "axquant.flagship-publication-review.v1"
    )
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate_sha256: str = Field(pattern=_SHA256)
    authorization_audit_sha256: str = Field(pattern=_SHA256)
    public_claim_sha256: str = Field(pattern=_SHA256)
    model_card_sha256: str = Field(pattern=_SHA256)
    reviewer: str = Field(min_length=1)
    verdict: Literal["pass", "fail"]
    issues: list[str]
    attestation: BoundFile
    reviewed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> FinalPublicationReviewRecord:
        if self.verdict == "pass" and self.issues:
            raise ValueError("passing publication review cannot contain issues")
        if self.verdict == "fail" and not self.issues:
            raise ValueError("failing publication review must explain its issues")
        return self


class ReproductionReviewRecord(StrictModel):
    schema_version: Literal["axquant.flagship-reproduction-review.v1"] = (
        "axquant.flagship-reproduction-review.v1"
    )
    candidate_sha256: str = Field(pattern=_SHA256)
    producing_host_id: Literal["mbp-m5"] = "mbp-m5"
    reproduction_host_id: str = Field(min_length=1)
    path_neutral_identity_verified: Literal[True] = True
    immutable_artifact_verified: Literal[True] = True
    runtime_compatibility_verified: Literal[True] = True
    reproduction_verification: BoundFile
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def host_is_independent(self) -> ReproductionReviewRecord:
        if self.reproduction_host_id == self.producing_host_id:
            raise ValueError("clean reproduction must use a host distinct from mbp-m5")
        return self


class HardwareAuthorizationRecord(StrictModel):
    schema_version: Literal["axquant.flagship-hardware-authorization.v1"] = (
        "axquant.flagship-hardware-authorization.v1"
    )
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate_sha256: str = Field(pattern=_SHA256)
    host_id: Literal["mbp-m5"] = "mbp-m5"
    hardware_id: str = Field(min_length=1)
    hardware_registry: BoundFile
    operator: str = Field(min_length=1)
    attestation: BoundFile
    measured_at: datetime = Field(default_factory=utc_now)


class FlagshipArchiveProof(StrictModel):
    schema_version: Literal["axquant.flagship-archive-proof.v1"] = (
        "axquant.flagship-archive-proof.v1"
    )
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate_sha256: str = Field(pattern=_SHA256)
    durable_evidence_root: str = Field(min_length=1)
    archive_index: BoundFile
    backup_verified: Literal[True] = True
    restore_readback_verified: Literal[True] = True
    verified_at: datetime = Field(default_factory=utc_now)


class FlagshipReleaseAuditRequest(StrictModel):
    schema_version: Literal["axquant.flagship-release-audit-request.v1"] = (
        "axquant.flagship-release-audit-request.v1"
    )
    legacy_release_audit_request: str
    campaign: str
    campaign_preflight: str
    candidate_key: str
    source_checkpoint_manifest: str
    certification_policy: str
    calibration_manifest: str
    activation_capture_or_sentinel: str
    formal_holdout_completion: str
    archive_proof: str
    independent_review: str
    reproduction_review: str
    hardware_authorization: str
    authorization_audit: str | None = None
    lifecycle_registry: str | None = None
    public_claim: str | None = None
    model_card: str | None = None
    final_publication_review: str | None = None

    @field_validator(
        "legacy_release_audit_request",
        "campaign",
        "campaign_preflight",
        "candidate_key",
        "source_checkpoint_manifest",
        "certification_policy",
        "calibration_manifest",
        "activation_capture_or_sentinel",
        "formal_holdout_completion",
        "archive_proof",
        "independent_review",
        "reproduction_review",
        "hardware_authorization",
        "authorization_audit",
        "lifecycle_registry",
        "public_claim",
        "model_card",
        "final_publication_review",
    )
    @classmethod
    def safe_paths(cls, value: str | None) -> str | None:
        return None if value is None else _relative_path(value)

    @model_validator(mode="after")
    def final_claim_inputs_are_all_or_none(self) -> FlagshipReleaseAuditRequest:
        final = (
            self.authorization_audit,
            self.lifecycle_registry,
            self.public_claim,
            self.model_card,
            self.final_publication_review,
        )
        if any(item is not None for item in final) and not all(item is not None for item in final):
            raise ValueError(
                "final flagship audit requires authorization audit, lifecycle, claim, and card"
            )
        return self


class FlagshipReleaseAudit(StrictModel):
    schema_version: Literal["axquant.flagship-release-audit.v1"] = (
        "axquant.flagship-release-audit.v1"
    )
    certification_track: Literal["qwen36-mtp-v2"] = "qwen36-mtp-v2"
    request_sha256: str = Field(pattern=_SHA256)
    legacy_audit_sha256: str = Field(pattern=_SHA256)
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate: CandidateKey
    candidate_model: ModelIdentity
    source_model: ModelIdentity
    toolkit_version: str | None = None
    wheel_sha256: str = Field(pattern=_SHA256)
    checks: list[ReleaseAuditCheck] = Field(min_length=9, max_length=9)
    authorization_ready: bool
    authorization_issues: list[str]
    release_ready: bool
    blockers: list[str]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def complete_and_consistent(self) -> FlagshipReleaseAudit:
        expected = {f"M{index}" for index in range(9)}
        gate_ids = [check.gate_id for check in self.checks]
        if len(gate_ids) != len(set(gate_ids)) or set(gate_ids) != expected:
            raise ValueError("flagship release audit must contain M0 through M8 exactly once")
        expected_blockers = [
            f"{check.gate_id}: {issue}" for check in self.checks for issue in check.issues
        ]
        if self.blockers != expected_blockers:
            raise ValueError("flagship release blockers are inconsistent with checks")
        if self.release_ready != all(check.passed for check in self.checks):
            raise ValueError("flagship release readiness is inconsistent with checks")
        if self.authorization_ready != (not self.authorization_issues):
            raise ValueError("flagship certification authorization status is inconsistent")
        return self
