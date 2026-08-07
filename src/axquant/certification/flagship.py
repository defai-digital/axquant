"""Additive Qwen 3.6 flagship certification audit."""

from __future__ import annotations

from pathlib import Path

from axquant.benchmark_evidence import formal_mtp_bundle_issues
from axquant.campaign import (
    FLAGSHIP_SOURCE_MODEL_ID,
    FLAGSHIP_SOURCE_REVISION,
    campaign_bound_files,
    formal_completion_evidence_issues,
    formal_host_scope_evidence_issues,
)
from axquant.claims import render_certified_model_card
from axquant.errors import ArtifactError, ValidationGateError
from axquant.identity import candidate_key_from_artifacts
from axquant.lifecycle import require_active_certification
from axquant.release_audit import build_release_audit
from axquant.schema import (
    ActivationCaptureSentinel,
    ArtifactLifecycleRegistry,
    ArtifactLifecycleState,
    ArtifactManifest,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    CampaignDatasetRole,
    CampaignPreflight,
    CampaignState,
    CandidateKey,
    EvaluationBundle,
    EvidenceArchiveIndex,
    FinalPublicationReviewRecord,
    FlagshipArchiveProof,
    FlagshipCampaign,
    FlagshipReleaseAudit,
    FlagshipReleaseAuditRequest,
    FormalHoldoutCompletion,
    FormalHostScopeEvidence,
    HardwareAuthorizationRecord,
    HardwareProfileRegistry,
    IndependentReviewRecord,
    LifecycleReason,
    PublicClaimManifest,
    QuantizationPlan,
    QuantMethod,
    ReleaseAuditCheck,
    ReleaseAuditRequest,
    ReleaseValidationIndex,
    ReproductionReviewRecord,
    ReproductionVerification,
    SourceCheckpointManifest,
)
from axquant.serde import file_sha256, load_model, stable_sha256

# Precision classes a flagship campaign may certify. The campaign, plan, and
# artifact manifest must all agree on one of these classes; the certified
# identity itself flows from the candidate key and measured BPW, not the label.
_FLAGSHIP_TARGET_CLASSES = frozenset({"4bit", "6bit"})

_REQUIRED_ARCHIVE_NAMES = {
    "activation-capture-or-sentinel",
    "calibration-manifest",
    "campaign",
    "campaign-preflight",
    "candidate-key",
    "certification-policy",
    "formal-agent-coding",
    "formal-general",
    "formal-holdout-completion",
    "formal-raw-evidence-index",
    "formal-custodian-attestation",
    "hardware-authorization",
    "hardware-authorization-attestation",
    "hardware-df-macbookpro-m5",
    "independent-review",
    "independent-review-attestation",
    "reproduction",
    "reproduction-review",
    "source-checkpoint-manifest",
}


def _path(base: Path, relative: str) -> Path:
    normalized_base = base.resolve()
    unresolved = normalized_base / relative
    path = unresolved.resolve()
    try:
        path.relative_to(normalized_base)
    except ValueError as exc:
        raise ArtifactError(f"flagship audit input escapes request root: {relative}") from exc
    if unresolved.is_symlink():
        raise ArtifactError(f"flagship audit input must not be a symlink: {relative}")
    if not path.is_file():
        raise ArtifactError(f"flagship audit input does not exist: {relative}")
    return path


def _bound_file_issues(root: Path, path: str, sha256: str, size_bytes: int) -> list[str]:
    normalized_root = root.resolve()
    unresolved = normalized_root / path
    source = unresolved.resolve()
    try:
        source.relative_to(normalized_root)
    except ValueError:
        return [f"bound file escapes evidence root: {path}"]
    if unresolved.is_symlink():
        return [f"bound file must not be a symlink: {path}"]
    if not source.is_file():
        return [f"bound file is missing: {path}"]
    issues: list[str] = []
    if source.stat().st_size != size_bytes:
        issues.append(f"bound file size changed: {path}")
    if file_sha256(source) != sha256:
        issues.append(f"bound file checksum changed: {path}")
    return issues


def _bind_evidence(check: ReleaseAuditCheck, key: str, path: Path) -> None:
    check.evidence_sha256[key] = file_sha256(path)


def _archive_issues(
    *,
    proof_path: Path,
    proof: FlagshipArchiveProof,
    campaign: FlagshipCampaign,
    candidate: CandidateKey,
    completion_path: Path,
    completion: FormalHoldoutCompletion,
) -> list[str]:
    issues: list[str] = []
    campaign_sha = stable_sha256(campaign)
    candidate_sha = stable_sha256(candidate)
    if proof.campaign_sha256 != campaign_sha or proof.candidate_sha256 != candidate_sha:
        issues.append("archive proof binds another campaign or candidate")
    if (
        Path(proof.durable_evidence_root).resolve()
        != Path(campaign.durable_evidence_root).resolve()
    ):
        issues.append("archive proof durable root differs from the frozen campaign")
    durable_root = Path(campaign.durable_evidence_root).resolve()
    try:
        proof_path.resolve().relative_to(durable_root)
    except ValueError:
        issues.append("archive proof is not stored under the durable campaign root")
    issues.extend(
        _bound_file_issues(
            proof_path.parent,
            proof.archive_index.path,
            proof.archive_index.sha256,
            proof.archive_index.size_bytes,
        )
    )
    index_path = (proof_path.parent / proof.archive_index.path).resolve()
    if not index_path.is_file():
        return issues
    index = load_model(index_path, EvidenceArchiveIndex)
    try:
        index_path.relative_to(durable_root)
    except ValueError:
        issues.append("evidence archive index is not stored under the durable campaign root")
    if not index.complete:
        issues.append("evidence archive index is incomplete")
    names = {record.logical_name for record in index.records}
    missing = sorted(_REQUIRED_ARCHIVE_NAMES - names)
    if missing:
        issues.append(f"evidence archive omits required records: {missing}")
    for record in index.records:
        issues.extend(
            _bound_file_issues(
                index_path.parent,
                record.path,
                record.sha256,
                record.size_bytes,
            )
        )
        normalized_uri = record.durable_uri.replace("\\", "/")
        if "/.internal/tmp" in normalized_uri:
            issues.append(f"archive record is only disposable: {record.logical_name}")
        record_path = (index_path.parent / record.path).resolve()
        try:
            record_path.relative_to(durable_root)
        except ValueError:
            issues.append(
                f"archive record is not under the durable campaign root: {record.logical_name}"
            )
    archived_bindings = {(record.sha256, record.size_bytes) for record in index.records}
    for bound in campaign_bound_files(campaign):
        if (bound.sha256, bound.size_bytes) not in archived_bindings:
            issues.append(f"campaign bound file is absent from durable archive: {bound.path}")
    for bound in (
        *completion.result_file_by_profile.values(),
        completion.raw_evidence_index,
        completion.custodian_attestation,
    ):
        if (bound.sha256, bound.size_bytes) not in archived_bindings:
            issues.append(f"formal bound file is absent from durable archive: {bound.path}")
    raw_index_path = (completion_path.parent / completion.raw_evidence_index.path).resolve()
    if raw_index_path.is_file():
        raw_index = load_model(raw_index_path, EvidenceArchiveIndex)
        for record in raw_index.records:
            if (record.sha256, record.size_bytes) not in archived_bindings:
                issues.append(
                    f"formal raw record is absent from durable archive: {record.logical_name}"
                )
    return issues


def _formal_completion_issues(
    *,
    completion_path: Path,
    completion: FormalHoldoutCompletion,
    campaign: FlagshipCampaign,
    candidate: CandidateKey,
) -> list[str]:
    issues: list[str] = []
    if completion.campaign_sha256 != stable_sha256(campaign):
        issues.append("formal completion binds another campaign")
    if completion.candidate_sha256 != stable_sha256(candidate):
        issues.append("formal completion binds another candidate")
    expected = {
        "agent-coding": next(
            dataset.content_sha256
            for dataset in campaign.datasets
            if dataset.role is CampaignDatasetRole.FORMAL_AGENT_CODING
        ),
        "general": next(
            dataset.content_sha256
            for dataset in campaign.datasets
            if dataset.role is CampaignDatasetRole.FORMAL_GENERAL
        ),
    }
    if completion.dataset_sha256_by_profile != expected:
        issues.append("formal completion dataset digests differ from the frozen holdouts")
    if completion.evaluation_custodian != campaign.roles.evaluation_custodian:
        issues.append("formal completion was not recorded by the frozen evaluation custodian")
    if completion.verdict != "pass" or completion.gate_issues:
        issues.append("formal holdout completion did not pass all frozen gates")
    issues.extend(formal_completion_evidence_issues(completion_path, completion))
    return issues


def _review_issues(
    *,
    review_path: Path,
    review: IndependentReviewRecord,
    campaign: FlagshipCampaign,
    candidate: CandidateKey,
    legacy_audit_sha256: str,
) -> list[str]:
    issues: list[str] = []
    if review.campaign_sha256 != stable_sha256(campaign):
        issues.append("independent review binds another campaign")
    if review.candidate_sha256 != stable_sha256(candidate):
        issues.append("independent review binds another candidate")
    if review.legacy_audit_sha256 != legacy_audit_sha256:
        issues.append("independent review binds another base M0-M8 audit")
    if review.reviewer != campaign.roles.independent_reviewer:
        issues.append("independent review signer differs from the frozen reviewer")
    if review.verdict != "pass":
        issues.append("independent reviewer did not approve the evidence")
    issues.extend(
        _bound_file_issues(
            review_path.parent,
            review.attestation.path,
            review.attestation.sha256,
            review.attestation.size_bytes,
        )
    )
    return issues


def _reproduction_issues(
    *,
    review_path: Path,
    review: ReproductionReviewRecord,
    campaign: FlagshipCampaign,
    candidate: CandidateKey,
) -> list[str]:
    issues: list[str] = []
    if review.candidate_sha256 != stable_sha256(candidate):
        issues.append("clean reproduction review binds another candidate")
    if review.reviewer != campaign.roles.independent_reviewer:
        issues.append("clean reproduction review signer differs from the frozen reviewer")
    bound = review.reproduction_verification
    issues.extend(
        _bound_file_issues(review_path.parent, bound.path, bound.sha256, bound.size_bytes)
    )
    verification_path = (review_path.parent / bound.path).resolve()
    if verification_path.is_file():
        verification = load_model(verification_path, ReproductionVerification)
        if not verification.passed or verification.issues:
            issues.append("clean-host reproduction verification did not pass")
    return issues


def _host_issues(
    *,
    campaign_path: Path,
    campaign: FlagshipCampaign,
    candidate: CandidateKey,
    hardware_registry_path: Path,
    hardware_registry: HardwareProfileRegistry,
    authorization_path: Path,
    authorization: HardwareAuthorizationRecord,
) -> list[str]:
    issues: list[str] = []
    scope_path = (campaign_path.parent / campaign.hardware_scope.path).resolve()
    scope = load_model(scope_path, FormalHostScopeEvidence)
    if scope.contract != campaign.formal_host:
        issues.append("formal host scope differs from the frozen df-macbookpro-m5 contract")
    if scope.evidence != campaign.hardware_scope_evidence:
        issues.append("formal host evidence differs from the frozen campaign bindings")
    issues.extend(formal_host_scope_evidence_issues(campaign_path.parent, scope))
    if authorization.campaign_sha256 != stable_sha256(
        campaign
    ) or authorization.candidate_sha256 != stable_sha256(candidate):
        issues.append("hardware authorization binds another campaign or candidate")
    if authorization.operator != campaign.formal_host.operator:
        issues.append("hardware authorization operator differs from df-macbookpro-m5 contract")
    if authorization.hardware_id != campaign.formal_host.hardware_id:
        issues.append("hardware authorization identity differs from df-macbookpro-m5 contract")
    issues.extend(
        _bound_file_issues(
            authorization_path.parent,
            authorization.hardware_registry.path,
            authorization.hardware_registry.sha256,
            authorization.hardware_registry.size_bytes,
        )
    )
    issues.extend(
        _bound_file_issues(
            authorization_path.parent,
            authorization.attestation.path,
            authorization.attestation.sha256,
            authorization.attestation.size_bytes,
        )
    )
    if authorization.hardware_registry.sha256 != file_sha256(hardware_registry_path):
        issues.append("df-macbookpro-m5 hardware authorization binds another hardware registry")
    for entry in hardware_registry.entries:
        if entry.hardware.os_version != campaign.formal_host.os_version:
            issues.append(f"{entry.entry_id} hardware OS differs from df-macbookpro-m5 contract")
        if entry.hardware.power_mode != campaign.formal_host.power_mode:
            issues.append(f"{entry.entry_id} power mode differs from df-macbookpro-m5 contract")
    return issues


def _formal_mtp_admissibility_issues(
    *,
    legacy_request_path: Path,
    legacy_request: ReleaseAuditRequest,
    campaign: FlagshipCampaign,
    hardware_registry: HardwareProfileRegistry,
) -> list[str]:
    """RM-20: MTP A/B bundles must bind the frozen formal-host contract.

    The authorized device identity comes from the digest-bound hardware
    registry (itself validated against the contract's OS and power mode), so
    a bundle recorded on any other machine — or with drifted controls — is
    named in the audit instead of silently authorizing speed claims.
    """
    identities = {
        (entry.hardware.device_name, entry.hardware.chip) for entry in hardware_registry.entries
    }
    if len(identities) != 1:
        return ["hardware registry entries disagree on the formal device identity"]
    device_name, chip = next(iter(identities))
    validation_path = Path(legacy_request.release_validation_index).expanduser()
    if not validation_path.is_absolute():
        validation_path = (legacy_request_path.parent / validation_path).resolve()
    # Audits run against archived evidence roots, so an unreadable or invalid
    # file must surface as a named gate issue, never crash the whole audit.
    try:
        validation_index = load_model(validation_path, ReleaseValidationIndex)
    except (ArtifactError, OSError, ValueError) as exc:
        return [f"release validation index is unreadable for MTP admissibility: {exc}"]
    issues: list[str] = []
    for entry in validation_index.entries:
        benchmark_path = Path(entry.benchmark_index_file).expanduser()
        if not benchmark_path.is_absolute():
            benchmark_path = (validation_path.parent / benchmark_path).resolve()
        try:
            benchmark = load_model(benchmark_path, BenchmarkEvidenceIndex)
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"{entry.profile.value}: benchmark index is unreadable: {exc}")
            continue
        bundles: dict[BenchmarkEvidenceKind, EvaluationBundle] = {}
        unreadable = False
        for benchmark_entry in benchmark.entries:
            if benchmark_entry.kind not in (
                BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
                BenchmarkEvidenceKind.AXQUANT_MTP_ON,
            ):
                continue
            if benchmark_entry.status != "available" or not benchmark_entry.evaluation_file:
                continue
            bundle_path = Path(benchmark_entry.evaluation_file).expanduser()
            if not bundle_path.is_absolute():
                bundle_path = (benchmark_path.parent / bundle_path).resolve()
            try:
                bundles[benchmark_entry.kind] = load_model(bundle_path, EvaluationBundle)
            except (ArtifactError, OSError, ValueError) as exc:
                issues.append(
                    f"{entry.profile.value}: {benchmark_entry.kind.value} bundle is "
                    f"unreadable: {exc}"
                )
                unreadable = True
        mtp_off = bundles.get(BenchmarkEvidenceKind.AXQUANT_MTP_OFF)
        mtp_on = bundles.get(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
        if mtp_off is None or mtp_on is None:
            if not unreadable:
                issues.append(
                    f"{entry.profile.value}: formal MTP off/on bundle pair is unavailable"
                )
            continue
        issues.extend(
            f"{entry.profile.value}: {issue}"
            for issue in formal_mtp_bundle_issues(
                mtp_off=mtp_off,
                mtp_on=mtp_on,
                contract=campaign.formal_host,
                authorized_device_name=device_name,
                authorized_chip=chip,
            )
        )
    return issues


def _final_claim_issues(
    *,
    request: FlagshipReleaseAuditRequest,
    root: Path,
    campaign: FlagshipCampaign,
    candidate: CandidateKey,
    legacy_audit_sha256: str,
    review: IndependentReviewRecord,
) -> list[str]:
    if request.authorization_audit is None:
        return [
            "final lifecycle, public claim, and generated model card are not supplied; "
            "authorization may proceed but publication may not"
        ]
    assert request.lifecycle_registry is not None
    assert request.public_claim is not None
    assert request.model_card is not None
    assert request.final_publication_review is not None
    issues: list[str] = []
    authorization_path = _path(root, request.authorization_audit)
    authorization = load_model(authorization_path, FlagshipReleaseAudit)
    if (
        not authorization.authorization_ready
        or authorization.candidate != candidate
        or authorization.legacy_audit_sha256 != legacy_audit_sha256
    ):
        issues.append("authorization audit is not valid for this campaign candidate")
    lifecycle_path = _path(root, request.lifecycle_registry)
    lifecycle = load_model(lifecycle_path, ArtifactLifecycleRegistry)
    frozen_lifecycle_path = (
        Path(root / request.campaign).resolve().parent / campaign.lifecycle_registry.path
    ).resolve()
    frozen_lifecycle = load_model(frozen_lifecycle_path, ArtifactLifecycleRegistry)
    if lifecycle.events[: len(frozen_lifecycle.events)] != frozen_lifecycle.events:
        issues.append("final lifecycle registry does not append to the frozen campaign history")
    try:
        event = require_active_certification(lifecycle, candidate)
    except ValidationGateError as exc:
        issues.append(str(exc))
        event = None
    authorization_file_sha = file_sha256(authorization_path)
    initial_certifications = [
        item
        for item in lifecycle.events
        if item.candidate == candidate
        and item.new_state is ArtifactLifecycleState.CERTIFIED
        and item.reason is LifecycleReason.CERTIFICATION_PASSED
    ]
    if (
        len(initial_certifications) != 1
        or initial_certifications[0].authorizing_evidence.sha256 != authorization_file_sha
    ):
        issues.append("initial certified lifecycle event does not bind the authorization audit")
    claim_path = _path(root, request.public_claim)
    claim = load_model(claim_path, PublicClaimManifest)
    if claim.candidate != candidate:
        issues.append("public claim binds another candidate")
    if claim.audit_sha256 != authorization_file_sha:
        issues.append("public claim does not bind the authorization audit")
    if event is not None and claim.lifecycle_event_sha256 != stable_sha256(event):
        issues.append("public claim does not bind the active certified lifecycle event")
    for bound in claim.evidence_index:
        issues.extend(
            _bound_file_issues(claim_path.parent, bound.path, bound.sha256, bound.size_bytes)
        )
    expected_card = render_certified_model_card(
        claim=claim,
        source_model_id=campaign.source.model.model_id,
        source_revision=campaign.source.model.revision or "",
        reviewer=review.reviewer,
    )
    model_card_path = _path(root, request.model_card)
    if model_card_path.read_text(encoding="utf-8") != expected_card:
        issues.append("certified model card is not the deterministic rendering of public claims")
    publication_review_path = _path(root, request.final_publication_review)
    publication_review = load_model(
        publication_review_path,
        FinalPublicationReviewRecord,
    )
    try:
        publication_review_path.relative_to(Path(campaign.durable_evidence_root).resolve())
    except ValueError:
        issues.append("final publication review is not stored under the durable campaign root")
    if publication_review.campaign_sha256 != stable_sha256(
        campaign
    ) or publication_review.candidate_sha256 != stable_sha256(candidate):
        issues.append("final publication review binds another campaign or candidate")
    if publication_review.authorization_audit_sha256 != authorization_file_sha:
        issues.append("final publication review binds another authorization audit")
    if publication_review.public_claim_sha256 != file_sha256(claim_path):
        issues.append("final publication review binds another public claim")
    if publication_review.model_card_sha256 != file_sha256(model_card_path):
        issues.append("final publication review binds another model card")
    if publication_review.reviewer != campaign.roles.independent_reviewer:
        issues.append("final publication review signer differs from the frozen reviewer")
    if publication_review.verdict != "pass":
        issues.append("independent reviewer did not approve final publication claims")
    issues.extend(
        _bound_file_issues(
            publication_review_path.parent,
            publication_review.attestation.path,
            publication_review.attestation.sha256,
            publication_review.attestation.size_bytes,
        )
    )
    return issues


def build_flagship_release_audit(
    request_path: str | Path,
) -> FlagshipReleaseAudit:
    source = Path(request_path).expanduser().resolve()
    root = source.parent
    request = load_model(source, FlagshipReleaseAuditRequest)

    legacy_request_path = _path(root, request.legacy_release_audit_request)
    legacy_request = load_model(legacy_request_path, ReleaseAuditRequest)
    legacy_audit = build_release_audit(legacy_request_path)
    legacy_audit_sha = stable_sha256(legacy_audit)
    checks_by_gate = {check.gate_id: check.model_copy(deep=True) for check in legacy_audit.checks}

    campaign_path = _path(root, request.campaign)
    campaign = load_model(campaign_path, FlagshipCampaign)
    preflight = load_model(_path(root, request.campaign_preflight), CampaignPreflight)
    candidate = load_model(_path(root, request.candidate_key), CandidateKey)
    source_manifest = load_model(
        _path(root, request.source_checkpoint_manifest),
        SourceCheckpointManifest,
    )
    formal_path = _path(root, request.formal_holdout_completion)
    formal = load_model(formal_path, FormalHoldoutCompletion)
    archive_path = _path(root, request.archive_proof)
    archive = load_model(archive_path, FlagshipArchiveProof)
    review_path = _path(root, request.independent_review)
    review = load_model(review_path, IndependentReviewRecord)
    reproduction_path = _path(root, request.reproduction_review)
    reproduction = load_model(reproduction_path, ReproductionReviewRecord)
    hardware_authorization_path = _path(root, request.hardware_authorization)
    hardware_authorization = load_model(
        hardware_authorization_path,
        HardwareAuthorizationRecord,
    )
    ordered_checks = sorted(
        checks_by_gate.values(),
        key=lambda check: int(check.gate_id.removeprefix("M")),
    )

    m0 = checks_by_gate["M0"].issues
    if campaign.state is not CampaignState.FROZEN:
        m0.append("flagship campaign is not frozen")
    if campaign.certification_track != "qwen36-mtp-v2":
        m0.append("campaign does not declare qwen36-mtp-v2")
    if campaign.target_class not in _FLAGSHIP_TARGET_CLASSES:
        m0.append(
            "flagship campaign target class must be one of "
            + ", ".join(sorted(_FLAGSHIP_TARGET_CLASSES))
        )
    if (
        campaign.source.model.model_id != FLAGSHIP_SOURCE_MODEL_ID
        or campaign.source.model.revision != FLAGSHIP_SOURCE_REVISION
    ):
        m0.append("campaign source is not the accepted Qwen 3.6 27B immutable revision")
    if not preflight.passed or preflight.campaign_sha256 != stable_sha256(campaign):
        m0.append("campaign preflight does not pass for the frozen campaign")
    if (
        preflight.host_id != "df-macbookpro-m5"
        or campaign.formal_host.host_id != "df-macbookpro-m5"
    ):
        m0.append("formal campaign host is not df-macbookpro-m5")
    for bound in campaign_bound_files(campaign):
        m0.extend(
            _bound_file_issues(
                campaign_path.parent,
                bound.path,
                bound.sha256,
                bound.size_bytes,
            )
        )
    _bind_evidence(checks_by_gate["M0"], "flagship_campaign", campaign_path)
    _bind_evidence(
        checks_by_gate["M0"],
        "flagship_campaign_preflight",
        _path(root, request.campaign_preflight),
    )

    artifact = Path(legacy_request.artifact_directory).expanduser()
    if not artifact.is_absolute():
        artifact = (legacy_request_path.parent / artifact).resolve()
    plan_path = artifact / "axquant_plan.json"
    manifest_path = artifact / "axquant_manifest.json"
    plan = load_model(plan_path, QuantizationPlan)
    manifest = load_model(manifest_path, ArtifactManifest)
    sensitivity_path = Path(legacy_request.sensitivity_report).expanduser()
    if not sensitivity_path.is_absolute():
        sensitivity_path = (legacy_request_path.parent / sensitivity_path).resolve()
    activation_binding_path = _path(root, request.activation_capture_or_sentinel)
    uses_capture = any(
        assignment.method in {QuantMethod.AWQ, QuantMethod.GPTQ, QuantMethod.GPTQ_ACT}
        for assignment in plan.assignments
    )
    if uses_capture:
        from axquant.capture import load_capture_activations

        load_capture_activations(
            activation_binding_path.parent,
            model=plan.source_model.model_id,
            revision=plan.source_model.revision,
        )
    else:
        sentinel = load_model(activation_binding_path, ActivationCaptureSentinel)
        from axquant.identity import semantic_plan_sha256

        if sentinel.plan_sha256 != semantic_plan_sha256(plan):
            raise ArtifactError("activation-capture sentinel binds another semantic plan")
    recomputed = candidate_key_from_artifacts(
        source_manifest=source_manifest,
        certification_policy_sha256=file_sha256(_path(root, request.certification_policy)),
        calibration_sha256=file_sha256(_path(root, request.calibration_manifest)),
        activation_capture_sha256=file_sha256(activation_binding_path),
        sensitivity_sha256=file_sha256(sensitivity_path),
        plan=plan,
        artifact_manifest=manifest,
    )
    m1 = checks_by_gate["M1"].issues
    if candidate != recomputed:
        m1.append("candidate key cannot be recomputed from the frozen semantic inputs")
    if campaign.candidate != candidate:
        m1.append("campaign and audit identify different candidate keys")
    if candidate.source != campaign.source:
        m1.append("candidate and campaign identify different source checkpoint keys")
    if manifest.target_class != campaign.target_class:
        m1.append("artifact target class differs from the frozen campaign")
    for key, path in {
        "candidate_key": _path(root, request.candidate_key),
        "source_checkpoint_manifest": _path(root, request.source_checkpoint_manifest),
        "certification_policy": _path(root, request.certification_policy),
        "calibration_manifest": _path(root, request.calibration_manifest),
        "activation_capture_or_sentinel": activation_binding_path,
        "sensitivity_report": sensitivity_path,
        "quantization_plan": plan_path,
        "artifact_manifest": manifest_path,
    }.items():
        _bind_evidence(checks_by_gate["M1"], key, path)

    m2 = checks_by_gate["M2"].issues
    m2.extend(
        _formal_completion_issues(
            completion_path=formal_path,
            completion=formal,
            campaign=campaign,
            candidate=candidate,
        )
    )
    _bind_evidence(checks_by_gate["M2"], "formal_holdout_completion", formal_path)
    for profile, bound in formal.result_file_by_profile.items():
        _bind_evidence(
            checks_by_gate["M2"],
            f"formal_{profile.replace('-', '_')}",
            (formal_path.parent / bound.path).resolve(),
        )

    m3 = checks_by_gate["M3"].issues
    m3.extend(
        _archive_issues(
            proof_path=archive_path,
            proof=archive,
            campaign=campaign,
            candidate=candidate,
            completion_path=formal_path,
            completion=formal,
        )
    )
    _bind_evidence(checks_by_gate["M3"], "flagship_archive_proof", archive_path)
    _bind_evidence(
        checks_by_gate["M3"],
        "flagship_archive_index",
        (archive_path.parent / archive.archive_index.path).resolve(),
    )

    m4 = checks_by_gate["M4"].issues
    if campaign.candidate != recomputed:
        m4.append("post-freeze candidate inputs differ from the campaign")
    _bind_evidence(checks_by_gate["M4"], "frozen_candidate_key", _path(root, request.candidate_key))

    m6 = checks_by_gate["M6"].issues
    formal_digests = {
        dataset.content_sha256
        for dataset in campaign.datasets
        if dataset.role
        in {
            CampaignDatasetRole.FORMAL_AGENT_CODING,
            CampaignDatasetRole.FORMAL_GENERAL,
        }
    }
    development_digests = {
        dataset.content_sha256
        for dataset in campaign.datasets
        if dataset.role
        not in {
            CampaignDatasetRole.FORMAL_AGENT_CODING,
            CampaignDatasetRole.FORMAL_GENERAL,
        }
    }
    if formal_digests & development_digests:
        m6.append("formal holdout content overlaps campaign development or calibration roles")
    for dataset in campaign.datasets:
        _bind_evidence(
            checks_by_gate["M6"],
            f"overlap_{dataset.role.value.replace('-', '_')}",
            (campaign_path.parent / dataset.overlap_report.path).resolve(),
        )

    hardware_path = Path(legacy_request.hardware_registry).expanduser()
    if not hardware_path.is_absolute():
        hardware_path = (legacy_request_path.parent / hardware_path).resolve()
    hardware = load_model(hardware_path, HardwareProfileRegistry)
    m7 = checks_by_gate["M7"].issues
    m7.extend(
        _host_issues(
            campaign_path=campaign_path,
            campaign=campaign,
            candidate=candidate,
            hardware_registry_path=hardware_path,
            hardware_registry=hardware,
            authorization_path=hardware_authorization_path,
            authorization=hardware_authorization,
        )
    )
    _bind_evidence(checks_by_gate["M7"], "hardware_registry_mbp_m5", hardware_path)
    _bind_evidence(
        checks_by_gate["M7"],
        "hardware_authorization",
        hardware_authorization_path,
    )
    _bind_evidence(
        checks_by_gate["M7"],
        "hardware_authorization_attestation",
        (hardware_authorization_path.parent / hardware_authorization.attestation.path).resolve(),
    )
    m7.extend(
        _formal_mtp_admissibility_issues(
            legacy_request_path=legacy_request_path,
            legacy_request=legacy_request,
            campaign=campaign,
            hardware_registry=hardware,
        )
    )
    _bind_evidence(checks_by_gate["M7"], "reproduction_review", reproduction_path)
    _bind_evidence(
        checks_by_gate["M7"],
        "reproduction_verification",
        (reproduction_path.parent / reproduction.reproduction_verification.path).resolve(),
    )
    m7.extend(
        _reproduction_issues(
            review_path=reproduction_path,
            review=reproduction,
            campaign=campaign,
            candidate=candidate,
        )
    )

    review_issues = _review_issues(
        review_path=review_path,
        review=review,
        campaign=campaign,
        candidate=candidate,
        legacy_audit_sha256=legacy_audit_sha,
    )
    m8 = checks_by_gate["M8"].issues
    m8.extend(review_issues)
    _bind_evidence(checks_by_gate["M8"], "independent_review", review_path)
    _bind_evidence(
        checks_by_gate["M8"],
        "independent_review_attestation",
        (review_path.parent / review.attestation.path).resolve(),
    )

    authorization_issues = [
        f"{check.gate_id}: {issue}" for check in ordered_checks[:8] for issue in check.issues
    ]
    legacy_m8 = next(check for check in legacy_audit.checks if check.gate_id == "M8")
    authorization_issues.extend(f"M8: {issue}" for issue in [*legacy_m8.issues, *review_issues])
    final_claim_issues = _final_claim_issues(
        request=request,
        root=root,
        campaign=campaign,
        candidate=candidate,
        legacy_audit_sha256=legacy_audit_sha,
        review=review,
    )
    m8.extend(final_claim_issues)
    if request.authorization_audit is not None:
        assert request.lifecycle_registry is not None
        assert request.public_claim is not None
        assert request.model_card is not None
        assert request.final_publication_review is not None
        for key, path in {
            "authorization_audit": _path(root, request.authorization_audit),
            "certified_lifecycle_registry": _path(root, request.lifecycle_registry),
            "public_claim_manifest": _path(root, request.public_claim),
            "certified_model_card": _path(root, request.model_card),
            "final_publication_review": _path(
                root,
                request.final_publication_review,
            ),
        }.items():
            _bind_evidence(checks_by_gate["M8"], key, path)
        publication_review_path = _path(root, request.final_publication_review)
        publication_review = load_model(
            publication_review_path,
            FinalPublicationReviewRecord,
        )
        _bind_evidence(
            checks_by_gate["M8"],
            "final_publication_review_attestation",
            (publication_review_path.parent / publication_review.attestation.path).resolve(),
        )

    checks = [
        ReleaseAuditCheck(
            gate_id=check.gate_id,
            name=check.name,
            passed=not check.issues,
            evidence_sha256=check.evidence_sha256,
            issues=check.issues,
        )
        for check in ordered_checks
    ]
    blockers = [f"{check.gate_id}: {issue}" for check in checks for issue in check.issues]
    return FlagshipReleaseAudit(
        request_sha256=stable_sha256(request),
        legacy_audit_sha256=legacy_audit_sha,
        campaign_sha256=stable_sha256(campaign),
        candidate=candidate,
        candidate_model=legacy_audit.candidate_model,
        source_model=legacy_audit.source_model,
        toolkit_version=legacy_audit.toolkit_version,
        wheel_sha256=legacy_audit.wheel_sha256,
        checks=checks,
        authorization_ready=not authorization_issues,
        authorization_issues=authorization_issues,
        release_ready=not blockers,
        blockers=blockers,
    )
