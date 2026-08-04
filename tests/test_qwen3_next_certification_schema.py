from __future__ import annotations

from datetime import UTC, datetime

import pytest

from axquant.certification.policy import direct_policy, direct_policy_sha256
from axquant.schema import (
    ArchitectureFingerprint,
    CertifiedCheckpointEntry,
    CertifiedCheckpointRegistry,
    DirectReleaseValidationRequest,
    ExactCertificationScope,
    ModelIdentity,
    NonMtpGateId,
    Qwen3NextReleaseAudit,
    Qwen3NextReleaseAuditCheck,
    Qwen3NextReleaseAuditRequest,
)
from axquant.serde import stable_sha256

_SHA = "a" * 64
_REVISION = "b" * 40


def _scope() -> ExactCertificationScope:
    return ExactCertificationScope(
        source_model=ModelIdentity(
            model_id="Qwen/Qwen3-Coder-Next",
            revision=_REVISION,
            architecture="Qwen3NextForCausalLM",
        ),
        architecture=ArchitectureFingerprint(
            model_type="qwen3_next",
            architecture="Qwen3NextForCausalLM",
            text_layer_count=48,
            hidden_size=2048,
            full_attention_interval=4,
            expert_count=512,
            experts_per_token=10,
            expert_intermediate_size=512,
            mtp_declared=False,
            vision_present=False,
            config_sha256=_SHA,
            tokenizer_sha256=_SHA,
        ),
        target_class="4bit",
        artifact_manifest_sha256=_SHA,
        hardware_scope_ids=["m2-ultra-192gb"],
    )


def _audit(*, failed_gate: NonMtpGateId | None = None) -> Qwen3NextReleaseAudit:
    checks = [
        Qwen3NextReleaseAuditCheck(
            gate_id=gate,
            name=f"gate {gate.value}",
            passed=gate is not failed_gate,
            issues=[] if gate is not failed_gate else ["failed"],
        )
        for gate in NonMtpGateId
    ]
    blockers = [] if failed_gate is None else [f"{failed_gate.value}: failed"]
    return Qwen3NextReleaseAudit(
        certification_scope=_scope(),
        candidate_model=ModelIdentity(
            model_id="AutomatosX/AX-Qwen3-Coder-Next-4bit",
            revision=_REVISION,
        ),
        request_sha256=_SHA,
        policy_sha256=direct_policy_sha256(),
        toolkit_version="1.2.0",
        wheel_sha256=_SHA,
        checks=checks,
        blockers=blockers,
        release_ready=failed_gate is None,
    )


def test_direct_policy_is_frozen_and_canonically_hashed() -> None:
    policy = direct_policy()
    assert policy.coding_tasks_min == 128
    assert policy.sensitivity_tokens_min == 8192
    assert policy.formal_hardware_chip == "Apple M2 Ultra"
    assert direct_policy_sha256() == stable_sha256(policy)


def test_audit_requires_n0_through_n8_in_order() -> None:
    audit = _audit()
    payload = audit.model_dump(mode="json")
    payload["checks"] = list(reversed(payload["checks"]))
    with pytest.raises(ValueError, match="N0 through N8 in order"):
        Qwen3NextReleaseAudit.model_validate(payload)


def test_audit_readiness_and_blockers_are_derived_from_checks() -> None:
    audit = _audit(failed_gate=NonMtpGateId.N4)
    assert not audit.release_ready
    assert audit.blockers == ["N4: failed"]
    payload = audit.model_dump(mode="json")
    payload["release_ready"] = True
    with pytest.raises(ValueError, match="readiness is inconsistent"):
        Qwen3NextReleaseAudit.model_validate(payload)


def test_scope_rejects_mutable_source_revision() -> None:
    payload = _scope().model_dump(mode="json")
    payload["source_model"]["revision"] = "main"
    with pytest.raises(ValueError, match="full immutable commit SHA"):
        ExactCertificationScope.model_validate(payload)


def test_direct_audit_request_rejects_path_traversal() -> None:
    request = {
        "certification_scope": _scope().model_dump(mode="json"),
        "artifact_directory": "../artifact",
        "source_inventory": "inventory.json",
        "source_checkpoint_manifest": "source.json",
        "feasibility_report": "feasibility.json",
        "sensitivity_report": "sensitivity.json",
        "refinement_result": "refinement.json",
        "refinement_measurements": "measurements.json",
        "release_validation_index": "validation.json",
        "benchmark_evidence_index": "benchmark.json",
        "coding_suite_manifest": "coding.json",
        "coding_suite_self_test": "coding-self-test.json",
        "hardware_registry": "hardware.json",
        "pareto_report": "pareto.json",
        "compatibility_matrix": "compatibility.json",
        "compatibility_request": "compatibility-request.json",
        "reproduction_recipe": "recipe.json",
        "reproduction_verification": "reproduction.json",
        "ax_engine_manifest_check": "ax-manifest.json",
        "ax_engine_doctor_check": "ax-doctor.json",
        "ax_engine_runtime_check": "ax-runtime.json",
        "mlx_lm_runtime_check": "mlx-runtime.json",
        "evidence_archive_index": "archive.json",
        "toolkit_wheel": "axquant.whl",
        "required_toolkit_version": "1.2.0",
        "policy_sha256": direct_policy_sha256(),
    }

    with pytest.raises(ValueError, match="safe relative paths"):
        Qwen3NextReleaseAuditRequest.model_validate(request)


def test_direct_validation_request_rejects_incomplete_profiles_and_unsafe_paths() -> None:
    request = {
        "source_checkpoint_manifest": "source.json",
        "candidate_artifact_manifest": "artifact/axquant_manifest.json",
        "calibration_dataset_sha256": _SHA,
        "coding_suite_manifest": "coding.json",
        "general_calibration_overlap_report": "general-overlap.json",
        "required_toolkit_version": "1.2.0",
        "policy_sha256": direct_policy_sha256(),
        "entries": [
            {
                "profile": "agent-coding",
                "evaluation_manifest_file": "../coding.json",
                "reference_evaluation_file": "bf16.json",
                "candidate_evaluation_file": "candidate.json",
            },
            {
                "profile": "agent-coding",
                "evaluation_manifest_file": "coding.json",
                "reference_evaluation_file": "bf16-2.json",
                "candidate_evaluation_file": "candidate-2.json",
            },
        ],
    }

    with pytest.raises(ValueError, match=r"safe relative paths|both release profiles"):
        DirectReleaseValidationRequest.model_validate(request)


def test_registry_requires_explicit_supersession() -> None:
    audit = _audit()
    now = datetime.now(UTC)
    entry = CertifiedCheckpointEntry(
        entry_id="qwen-next-4bit-v1",
        certification_scope=_scope(),
        candidate_model=ModelIdentity(
            model_id="AutomatosX/AX-Qwen3-Coder-Next-4bit",
            revision=_REVISION,
        ),
        candidate_id="candidate-4bit",
        policy_sha256=direct_policy_sha256(),
        artifact_manifest_sha256=_SHA,
        release_audit_sha256=stable_sha256(audit),
        measured_bpw=4.8,
        allowed_claims=["exact-checkpoint non-MTP direct decode"],
        hardware_scope_ids=["m2-ultra-192gb"],
        certified_at=now,
    )
    with pytest.raises(ValueError, match="requires supersession"):
        CertifiedCheckpointRegistry(entries=[entry, entry.model_copy(update={"entry_id": "v2"})])

    replacement = entry.model_copy(
        update={"entry_id": "qwen-next-4bit-v2", "supersedes_entry_id": entry.entry_id}
    )
    registry = CertifiedCheckpointRegistry(entries=[entry, replacement])
    assert registry.entries[-1].supersedes_entry_id == entry.entry_id

    stale_replacement = replacement.model_copy(
        update={"entry_id": "qwen-next-4bit-v3", "supersedes_entry_id": entry.entry_id}
    )
    with pytest.raises(ValueError, match="active entry"):
        CertifiedCheckpointRegistry(entries=[entry, replacement, stale_replacement])

    six_bit = entry.model_copy(
        update={
            "entry_id": "qwen-next-6bit-v1",
            "certification_scope": _scope().model_copy(update={"target_class": "6bit"}),
            "candidate_id": "candidate-6bit",
        }
    )
    cross_identity_replacement = six_bit.model_copy(
        update={
            "entry_id": "qwen-next-6bit-v2",
            "supersedes_entry_id": entry.entry_id,
        }
    )
    with pytest.raises(ValueError, match="active entry"):
        CertifiedCheckpointRegistry(entries=[entry, six_bit, cross_identity_replacement])


def test_registry_entry_hardware_scope_must_match_audit_scope() -> None:
    audit = _audit()
    with pytest.raises(ValueError, match="hardware scope"):
        CertifiedCheckpointEntry(
            entry_id="qwen-next-4bit-v1",
            certification_scope=_scope(),
            candidate_model=audit.candidate_model,
            candidate_id="candidate-4bit",
            policy_sha256=direct_policy_sha256(),
            artifact_manifest_sha256=_SHA,
            release_audit_sha256=stable_sha256(audit),
            measured_bpw=4.8,
            allowed_claims=["exact-checkpoint non-MTP direct decode"],
            hardware_scope_ids=["another-host"],
            certified_at=datetime.now(UTC),
        )
