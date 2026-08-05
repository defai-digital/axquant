from __future__ import annotations

from pathlib import Path

import pytest

from axquant.claims import build_public_claim, render_certified_model_card
from axquant.errors import ArtifactError, ValidationGateError
from axquant.lifecycle import transition_lifecycle
from axquant.naming import certified_mixed_precision_name
from axquant.schema import (
    ArtifactLifecycleRegistry,
    ArtifactLifecycleState,
    BoundFile,
    BoundMetricClaim,
    CandidateKey,
    CheckpointKey,
    LifecycleReason,
    ModelIdentity,
)
from axquant.serde import file_sha256


def _candidate() -> CandidateKey:
    return CandidateKey(
        source=CheckpointKey(
            model=ModelIdentity(
                model_id="Qwen/Qwen3.6-27B",
                revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
                format="mlx",
            ),
            config_sha256="1" * 64,
            tokenizer_sha256="2" * 64,
            weight_index_sha256="3" * 64,
            checkpoint_members_sha256="4" * 64,
        ),
        certification_policy_sha256="5" * 64,
        calibration_sha256="6" * 64,
        activation_capture_sha256="7" * 64,
        sensitivity_sha256="8" * 64,
        plan_sha256="9" * 64,
        artifact_manifest_sha256="a" * 64,
        checkpoint_members_sha256="b" * 64,
    )


def _bound(tmp_path: Path) -> BoundFile:
    path = tmp_path / "audit.json"
    path.write_text("{}", encoding="utf-8")
    return BoundFile(path=path.name, sha256=file_sha256(path), size_bytes=path.stat().st_size)


def _certified_registry(tmp_path: Path, candidate: CandidateKey) -> ArtifactLifecycleRegistry:
    registry = ArtifactLifecycleRegistry(registry_id="flagship", events=[])
    for state in (
        ArtifactLifecycleState.DEVELOPMENT,
        ArtifactLifecycleState.CANDIDATE,
        ArtifactLifecycleState.FROZEN,
    ):
        registry = transition_lifecycle(
            registry=registry,
            candidate=candidate,
            new_state=state,
            actor="operator",
            reviewer="reviewer",
            reason=LifecycleReason.PROVENANCE_ERROR,
            narrative="fixture transition",
            authorizing_evidence=_bound(tmp_path),
        )
    return transition_lifecycle(
        registry=registry,
        candidate=candidate,
        new_state=ArtifactLifecycleState.CERTIFIED,
        actor="release-manager",
        reviewer="independent-reviewer",
        reason=LifecycleReason.CERTIFICATION_PASSED,
        narrative="M0-M8 passed",
        authorizing_evidence=_bound(tmp_path),
        public_repository="owner/AX-Qwen3.6-27B-MLX-AXQ-MP-5p30bpw-MTP",
    )


def test_certified_name_uses_decimal_half_up_and_two_places() -> None:
    assert (
        certified_mixed_precision_name("Qwen/Qwen3.6-27B", 5.295, mtp=True)
        == "AX-Qwen3.6-27B-MLX-AXQ-MP-5p30bpw-MTP"
    )
    assert (
        certified_mixed_precision_name("Qwen/Qwen3.6-27B", 7.38, mtp=False)
        == "AX-Qwen3.6-27B-MLX-AXQ-MP-7p38bpw"
    )
    with pytest.raises(ArtifactError):
        certified_mixed_precision_name("Qwen/Qwen3.6-27B", float("nan"), mtp=True)


def test_public_claim_and_card_are_generated_from_bound_metrics(tmp_path: Path) -> None:
    candidate = _candidate()
    registry = _certified_registry(tmp_path, candidate)
    evidence = _bound(tmp_path)
    quality = [
        BoundMetricClaim(
            evidence=evidence,
            profile=profile,  # type: ignore[arg-type]
            metric_key="quality.aggregate_retention",
            unit="ratio",
            value=value,
            comparison="higher-is-better",
        )
        for profile, value in (("agent-coding", 0.99), ("general", 0.98))
    ]
    performance = BoundMetricClaim(
        evidence=evidence,
        profile="hardware",
        metric_key="hardware.effective_speedup",
        unit="x",
        value=1.25,
        numerator=125,
        denominator=100,
        comparison="ratio",
    )

    claim = build_public_claim(
        candidate=candidate,
        lifecycle=registry,
        audit_sha256="c" * 64,
        public_owner="owner",
        base_model="Qwen/Qwen3.6-27B",
        target_class="4bit",
        measured_main_bpw=5.295,
        measured_total_bpw=5.41,
        weight_bytes=1_000_000,
        runtime_versions={"ax-engine": "1.0", "mlx": "1.0", "mlx-lm": "1.0"},
        quality_claims=quality,
        performance_claims=[performance],
        limitations=["Certified only for the exact recorded source and runtime scope."],
        evidence_index=[evidence],
    )
    card = render_certified_model_card(
        claim=claim,
        source_model_id="Qwen/Qwen3.6-27B",
        source_revision=candidate.source.model.revision or "",
        reviewer="independent-reviewer",
    )

    assert claim.display_name.endswith("MP-5p30bpw-MTP")
    assert "mixed-precision" in card
    assert "mbp-m5" in card
    assert claim.audit_sha256 in card


def test_unsupported_numeric_claim_is_rejected(tmp_path: Path) -> None:
    candidate = _candidate()
    registry = _certified_registry(tmp_path, candidate)
    evidence = _bound(tmp_path)
    unsupported = BoundMetricClaim(
        evidence=evidence,
        profile="general",
        metric_key="marketing.best_in_market",
        unit="boolean",
        value=1,
        comparison="absolute",
    )

    with pytest.raises(ValidationGateError, match="unsupported public metric"):
        build_public_claim(
            candidate=candidate,
            lifecycle=registry,
            audit_sha256="c" * 64,
            public_owner="owner",
            base_model="Qwen/Qwen3.6-27B",
            target_class="4bit",
            measured_main_bpw=5.295,
            measured_total_bpw=5.41,
            weight_bytes=1_000_000,
            runtime_versions={"ax-engine": "1.0"},
            quality_claims=[unsupported],
            performance_claims=[unsupported],
            limitations=["Exact scope only."],
            evidence_index=[evidence],
        )
