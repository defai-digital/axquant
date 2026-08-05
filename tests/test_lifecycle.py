from __future__ import annotations

from pathlib import Path

import pytest

from axquant.errors import ValidationGateError
from axquant.lifecycle import require_active_certification, transition_lifecycle
from axquant.schema import (
    ArtifactLifecycleRegistry,
    ArtifactLifecycleState,
    BoundFile,
    CandidateKey,
    CheckpointKey,
    LifecycleReason,
    ModelIdentity,
    SemanticImpactScan,
)
from axquant.serde import file_sha256, stable_sha256


def _candidate(last: str = "a") -> CandidateKey:
    source = CheckpointKey(
        model=ModelIdentity(
            model_id="Qwen/Qwen3.6-27B",
            revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            format="mlx",
        ),
        config_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        weight_index_sha256="3" * 64,
        checkpoint_members_sha256="4" * 64,
    )
    return CandidateKey(
        source=source,
        certification_policy_sha256="5" * 64,
        calibration_sha256="6" * 64,
        activation_capture_sha256="7" * 64,
        sensitivity_sha256="8" * 64,
        plan_sha256="9" * 64,
        artifact_manifest_sha256=last * 64,
        checkpoint_members_sha256="b" * 64,
    )


def _evidence(tmp_path: Path, name: str = "evidence.json") -> BoundFile:
    path = tmp_path / name
    path.write_text("{}", encoding="utf-8")
    return BoundFile(path=name, sha256=file_sha256(path), size_bytes=path.stat().st_size)


def _certified(tmp_path: Path) -> tuple[ArtifactLifecycleRegistry, CandidateKey]:
    candidate = _candidate()
    registry = ArtifactLifecycleRegistry(registry_id="flagship", events=[])
    for state, reason in (
        (ArtifactLifecycleState.DEVELOPMENT, LifecycleReason.PROVENANCE_ERROR),
        (ArtifactLifecycleState.CANDIDATE, LifecycleReason.PROVENANCE_ERROR),
        (ArtifactLifecycleState.FROZEN, LifecycleReason.PROVENANCE_ERROR),
    ):
        registry = transition_lifecycle(
            registry=registry,
            candidate=candidate,
            new_state=state,
            actor="operator",
            reviewer="reviewer",
            reason=reason,
            narrative=f"advance to {state.value}",
            authorizing_evidence=_evidence(tmp_path),
        )
    repository = "owner/AX-Qwen3.6-27B-MLX-AXQ-MP-5p30bpw-MTP"
    registry = transition_lifecycle(
        registry=registry,
        candidate=candidate,
        new_state=ArtifactLifecycleState.CERTIFIED,
        actor="release-manager",
        reviewer="independent-reviewer",
        reason=LifecycleReason.CERTIFICATION_PASSED,
        narrative="M0-M8 passed",
        authorizing_evidence=_evidence(tmp_path),
        public_repository=repository,
    )
    return registry, candidate


def test_certification_requires_legal_append_only_chain(tmp_path: Path) -> None:
    registry, candidate = _certified(tmp_path)

    event = require_active_certification(registry, candidate)

    assert event.new_state is ArtifactLifecycleState.CERTIFIED
    assert len(registry.events) == 4


def test_illegal_transition_fails_closed(tmp_path: Path) -> None:
    registry = ArtifactLifecycleRegistry(registry_id="flagship", events=[])

    with pytest.raises(ValidationGateError, match="illegal lifecycle transition"):
        transition_lifecycle(
            registry=registry,
            candidate=_candidate(),
            new_state=ArtifactLifecycleState.CERTIFIED,
            actor="operator",
            reviewer="reviewer",
            reason=LifecycleReason.CERTIFICATION_PASSED,
            narrative="skip gates",
            authorizing_evidence=_evidence(tmp_path),
            public_repository="owner/repo",
        )


def test_revoked_candidate_is_not_actively_certified(tmp_path: Path) -> None:
    registry, candidate = _certified(tmp_path)
    impact = SemanticImpactScan(
        change_id="qwen-next-expert-fix",
        change_kind="adapter-classification",
        previous_semantics_sha256="c" * 64,
        current_semantics_sha256="d" * 64,
        affected_candidate_sha256=[stable_sha256(candidate)],
        outcome="revoke",
        evidence=_evidence(tmp_path, "impact.json"),
        reviewed_by="independent-reviewer",
    )
    revoked = transition_lifecycle(
        registry=registry,
        candidate=candidate,
        new_state=ArtifactLifecycleState.REVOKED,
        actor="release-manager",
        reviewer="independent-reviewer",
        reason=LifecycleReason.ADAPTER_CLASSIFICATION_CHANGED,
        narrative="expert classification changed candidate semantics",
        authorizing_evidence=_evidence(tmp_path),
        impact_scan=impact,
        public_repository=registry.events[-1].public_repository,
    )

    with pytest.raises(ValidationGateError, match="revoked"):
        require_active_certification(revoked, candidate)


def test_unaffected_impact_scan_reaffirms_active_certification(tmp_path: Path) -> None:
    registry, candidate = _certified(tmp_path)
    impact = SemanticImpactScan(
        change_id="adapter-review",
        change_kind="adapter-classification",
        previous_semantics_sha256="c" * 64,
        current_semantics_sha256="d" * 64,
        affected_candidate_sha256=[stable_sha256(candidate)],
        outcome="unaffected",
        evidence=_evidence(tmp_path, "impact.json"),
        reviewed_by="independent-reviewer",
    )

    reaffirmed = transition_lifecycle(
        registry=registry,
        candidate=candidate,
        new_state=ArtifactLifecycleState.CERTIFIED,
        actor="release-manager",
        reviewer="independent-reviewer",
        reason=LifecycleReason.ADAPTER_CLASSIFICATION_CHANGED,
        narrative="adapter change does not affect this candidate",
        authorizing_evidence=_evidence(tmp_path),
        impact_scan=impact,
        public_repository=registry.events[-1].public_repository,
    )

    active = require_active_certification(reaffirmed, candidate)
    assert active.impact_scan == impact
    assert active.previous_state is ArtifactLifecycleState.CERTIFIED


def test_supersession_requires_an_actively_certified_replacement(tmp_path: Path) -> None:
    registry, candidate = _certified(tmp_path)
    replacement = _candidate("f")

    with pytest.raises(ValidationGateError, match="replacement is not actively certified"):
        transition_lifecycle(
            registry=registry,
            candidate=candidate,
            new_state=ArtifactLifecycleState.SUPERSEDED,
            actor="release-manager",
            reviewer="independent-reviewer",
            reason=LifecycleReason.NEW_CERTIFIED_SUCCESSOR,
            narrative="replace old candidate",
            authorizing_evidence=_evidence(tmp_path),
            replacement_candidate=replacement,
            public_repository=registry.events[-1].public_repository,
        )
