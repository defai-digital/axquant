from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.campaign import BoundFile
from axquant.schema.flagship import CandidateKey
from axquant.serde import stable_sha256

_SHA256 = r"^[0-9a-f]{64}$"


class ArtifactLifecycleState(StrEnum):
    DEVELOPMENT = "development"
    CANDIDATE = "candidate"
    FROZEN = "frozen"
    CERTIFIED = "certified"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class LifecycleReason(StrEnum):
    CERTIFICATION_PASSED = "certification_passed"
    FORMAL_CYCLE_FAILED = "formal_cycle_failed"
    NEW_CERTIFIED_SUCCESSOR = "new_certified_successor"
    ADAPTER_CLASSIFICATION_CHANGED = "adapter_classification_changed"
    PACKING_SEMANTICS_CHANGED = "packing_semantics_changed"
    SOURCE_OR_TOKENIZER_CHANGED = "source_or_tokenizer_changed"
    RUNTIME_CONTRACT_INVALIDATED = "runtime_contract_invalidated"
    PROVENANCE_ERROR = "provenance_error"
    SECURITY_OR_LICENSE_ISSUE = "security_or_license_issue"


class SemanticImpactScan(StrictModel):
    schema_version: Literal["axquant.semantic-impact-scan.v1"] = "axquant.semantic-impact-scan.v1"
    change_id: str = Field(min_length=1)
    change_kind: Literal[
        "adapter-classification",
        "packing-semantics",
        "source-tokenizer",
        "runtime-contract",
        "provenance",
        "security-license",
    ]
    previous_semantics_sha256: str = Field(pattern=_SHA256)
    current_semantics_sha256: str = Field(pattern=_SHA256)
    affected_candidate_sha256: list[str] = Field(min_length=1)
    outcome: Literal["unaffected", "supersede", "revoke"]
    evidence: BoundFile
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def candidates_are_valid_and_unique(self) -> SemanticImpactScan:
        if len(self.affected_candidate_sha256) != len(set(self.affected_candidate_sha256)):
            raise ValueError("impact scan affected candidates must be unique")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.affected_candidate_sha256
        ):
            raise ValueError("impact scan affected candidate digests must be lowercase SHA-256")
        if self.previous_semantics_sha256 == self.current_semantics_sha256:
            raise ValueError("semantic impact scan requires an actual semantic digest change")
        return self


class ArtifactLifecycleEvent(StrictModel):
    schema_version: Literal["axquant.artifact-lifecycle-event.v1"] = (
        "axquant.artifact-lifecycle-event.v1"
    )
    candidate: CandidateKey
    candidate_sha256: str = Field(pattern=_SHA256)
    previous_state: ArtifactLifecycleState | None
    new_state: ArtifactLifecycleState
    occurred_at: datetime = Field(default_factory=utc_now)
    actor: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reason: LifecycleReason
    narrative: str = Field(min_length=1)
    authorizing_evidence: BoundFile
    replacement_candidate: CandidateKey | None = None
    public_repository: str | None = None
    public_revision: str | None = None
    impact_scan: SemanticImpactScan | None = None

    @model_validator(mode="after")
    def internally_consistent(self) -> ArtifactLifecycleEvent:
        if stable_sha256(self.candidate) != self.candidate_sha256:
            raise ValueError("lifecycle event candidate digest does not match candidate")
        if self.new_state is ArtifactLifecycleState.CERTIFIED:
            if self.actor == self.reviewer:
                raise ValueError("certification transition requires an independent reviewer")
            if self.public_repository is None:
                raise ValueError("certification transition requires a public repository")
            if self.previous_state is ArtifactLifecycleState.FROZEN:
                if self.reason is not LifecycleReason.CERTIFICATION_PASSED:
                    raise ValueError(
                        "initial certification transition requires certification_passed reason"
                    )
                if self.impact_scan is not None:
                    raise ValueError("initial certification cannot carry a semantic impact scan")
            elif self.previous_state is ArtifactLifecycleState.CERTIFIED:
                if (
                    self.reason is LifecycleReason.CERTIFICATION_PASSED
                    or self.impact_scan is None
                    or self.impact_scan.outcome != "unaffected"
                ):
                    raise ValueError(
                        "certification reaffirmation requires an unaffected impact scan"
                    )
            else:
                raise ValueError("certification requires frozen or certified previous state")
        if self.new_state is ArtifactLifecycleState.SUPERSEDED:
            if self.reason is not LifecycleReason.NEW_CERTIFIED_SUCCESSOR:
                raise ValueError("supersession requires new_certified_successor reason")
            if self.replacement_candidate is None:
                raise ValueError("supersession requires a replacement candidate")
            if self.replacement_candidate == self.candidate:
                raise ValueError("supersession replacement must be a different candidate")
        elif self.replacement_candidate is not None:
            raise ValueError("replacement candidate is only valid for supersession")
        if self.new_state is ArtifactLifecycleState.REVOKED and self.reason in {
            LifecycleReason.CERTIFICATION_PASSED,
            LifecycleReason.NEW_CERTIFIED_SUCCESSOR,
            LifecycleReason.FORMAL_CYCLE_FAILED,
        }:
            raise ValueError("revocation requires an invalidation reason")
        if (
            self.reason
            in {
                LifecycleReason.ADAPTER_CLASSIFICATION_CHANGED,
                LifecycleReason.PACKING_SEMANTICS_CHANGED,
            }
            and self.impact_scan is None
        ):
            raise ValueError("semantic invalidation requires an impact scan")
        if self.impact_scan is not None:
            if self.impact_scan.reviewed_by != self.reviewer:
                raise ValueError("impact scan reviewer differs from lifecycle reviewer")
            if self.candidate_sha256 not in self.impact_scan.affected_candidate_sha256:
                raise ValueError("impact scan does not list the lifecycle candidate")
            expected_outcome = {
                ArtifactLifecycleState.CERTIFIED: (
                    "unaffected"
                    if self.previous_state is ArtifactLifecycleState.CERTIFIED
                    else None
                ),
                ArtifactLifecycleState.REVOKED: "revoke",
                ArtifactLifecycleState.SUPERSEDED: "supersede",
            }.get(self.new_state)
            if expected_outcome is not None and self.impact_scan.outcome != expected_outcome:
                raise ValueError("impact scan outcome differs from lifecycle transition")
        if (
            self.new_state
            in {
                ArtifactLifecycleState.SUPERSEDED,
                ArtifactLifecycleState.REVOKED,
            }
            and self.public_repository is None
        ):
            raise ValueError("terminal certified lifecycle transitions require public repository")
        return self


class ArtifactLifecycleRegistry(StrictModel):
    schema_version: Literal["axquant.artifact-lifecycle-registry.v1"] = (
        "axquant.artifact-lifecycle-registry.v1"
    )
    registry_id: str = Field(min_length=1)
    events: list[ArtifactLifecycleEvent]

    @model_validator(mode="after")
    def valid_event_chains(self) -> ArtifactLifecycleRegistry:
        latest: dict[str, ArtifactLifecycleState] = {}
        seen_events: set[str] = set()
        for event in self.events:
            event_sha = stable_sha256(event)
            if event_sha in seen_events:
                raise ValueError("lifecycle registry contains a duplicate event")
            seen_events.add(event_sha)
            previous = latest.get(event.candidate_sha256)
            if event.previous_state != previous:
                raise ValueError("lifecycle event previous state does not match registry history")
            latest[event.candidate_sha256] = event.new_state
        return self
