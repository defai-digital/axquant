from __future__ import annotations

from pathlib import Path

from axquant.errors import ValidationGateError
from axquant.schema import (
    ArtifactLifecycleEvent,
    ArtifactLifecycleRegistry,
    ArtifactLifecycleState,
    BoundFile,
    CandidateKey,
    LifecycleReason,
    SemanticImpactScan,
)
from axquant.serde import stable_sha256, write_data

_ALLOWED_TRANSITIONS = {
    (None, ArtifactLifecycleState.DEVELOPMENT),
    (ArtifactLifecycleState.DEVELOPMENT, ArtifactLifecycleState.CANDIDATE),
    (ArtifactLifecycleState.CANDIDATE, ArtifactLifecycleState.FROZEN),
    (ArtifactLifecycleState.FROZEN, ArtifactLifecycleState.CERTIFIED),
    (ArtifactLifecycleState.CERTIFIED, ArtifactLifecycleState.CERTIFIED),
    (ArtifactLifecycleState.FROZEN, ArtifactLifecycleState.DEVELOPMENT),
    (ArtifactLifecycleState.CERTIFIED, ArtifactLifecycleState.SUPERSEDED),
    (ArtifactLifecycleState.CERTIFIED, ArtifactLifecycleState.REVOKED),
}


def candidate_lifecycle_state(
    registry: ArtifactLifecycleRegistry,
    candidate: CandidateKey,
) -> ArtifactLifecycleState | None:
    digest = stable_sha256(candidate)
    state: ArtifactLifecycleState | None = None
    for event in registry.events:
        if event.candidate_sha256 == digest:
            state = event.new_state
    return state


def require_active_certification(
    registry: ArtifactLifecycleRegistry,
    candidate: CandidateKey,
) -> ArtifactLifecycleEvent:
    digest = stable_sha256(candidate)
    matches = [event for event in registry.events if event.candidate_sha256 == digest]
    if not matches or matches[-1].new_state is not ArtifactLifecycleState.CERTIFIED:
        state = matches[-1].new_state.value if matches else "unregistered"
        raise ValidationGateError(
            f"candidate does not have an active certification lifecycle state: {state}"
        )
    return matches[-1]


def transition_lifecycle(
    *,
    registry: ArtifactLifecycleRegistry,
    candidate: CandidateKey,
    new_state: ArtifactLifecycleState,
    actor: str,
    reviewer: str,
    reason: LifecycleReason,
    narrative: str,
    authorizing_evidence: BoundFile,
    replacement_candidate: CandidateKey | None = None,
    public_repository: str | None = None,
    public_revision: str | None = None,
    impact_scan: SemanticImpactScan | None = None,
    output_path: str | Path | None = None,
) -> ArtifactLifecycleRegistry:
    previous = candidate_lifecycle_state(registry, candidate)
    if (previous, new_state) not in _ALLOWED_TRANSITIONS:
        previous_label = previous.value if previous is not None else "unregistered"
        raise ValidationGateError(
            f"illegal lifecycle transition: {previous_label} -> {new_state.value}"
        )
    if new_state is ArtifactLifecycleState.SUPERSEDED:
        if replacement_candidate is None:
            raise ValidationGateError("supersession requires a replacement candidate")
        try:
            require_active_certification(registry, replacement_candidate)
        except ValidationGateError as exc:
            raise ValidationGateError("supersession replacement is not actively certified") from exc
    event = ArtifactLifecycleEvent(
        candidate=candidate,
        candidate_sha256=stable_sha256(candidate),
        previous_state=previous,
        new_state=new_state,
        actor=actor,
        reviewer=reviewer,
        reason=reason,
        narrative=narrative,
        authorizing_evidence=authorizing_evidence,
        replacement_candidate=replacement_candidate,
        public_repository=public_repository,
        public_revision=public_revision,
        impact_scan=impact_scan,
    )
    updated = registry.model_copy(update={"events": [*registry.events, event]})
    if output_path is not None:
        write_data(output_path, updated)
    return updated
