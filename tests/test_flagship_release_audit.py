from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from test_release_audit import _inputs

from axquant.campaign import campaign_bound_files, preflight_campaign
from axquant.certification.dispatch import build_certification_audit
from axquant.certification.flagship import build_flagship_release_audit
from axquant.claims import render_public_claim_request
from axquant.errors import ArtifactError, PublishingError
from axquant.identity import (
    candidate_key_from_artifacts,
    checkpoint_key_from_source_manifest,
    semantic_plan_sha256,
)
from axquant.lifecycle import transition_lifecycle
from axquant.naming import certified_mixed_precision_name
from axquant.publisher import _rerun_release_audit
from axquant.release_audit import build_release_audit
from axquant.schema import (
    ActivationCaptureSentinel,
    ArtifactLifecycleRegistry,
    ArtifactLifecycleState,
    ArtifactManifest,
    BoundFile,
    BoundMetricClaim,
    CampaignBaseline,
    CampaignDataset,
    CampaignDatasetManifest,
    CampaignDatasetRole,
    CampaignOverlapReport,
    CampaignRoles,
    CampaignState,
    CandidateInputBindings,
    EvidenceArchiveIndex,
    EvidenceArchiveRecord,
    FinalPublicationReviewRecord,
    FlagshipArchiveProof,
    FlagshipCampaign,
    FlagshipFrontierEntry,
    FlagshipFrontierIndex,
    FlagshipReleaseAudit,
    FlagshipReleaseAuditRequest,
    FormalHoldoutCompletion,
    FormalHostContract,
    FormalHostEvidenceBinding,
    FormalHostEvidenceKind,
    FormalHostEvidenceResult,
    FormalHostScopeEvidence,
    FrontierGate,
    FrontierGateResult,
    FrontierGateStatus,
    HardwareAuthorizationRecord,
    HardwareProfileRegistry,
    IndependentReviewRecord,
    LifecycleReason,
    PublicClaimRenderRequest,
    QuantizationPlan,
    ReleaseAuditRequest,
    ReproductionReviewRecord,
    SourceCheckpointFile,
    SourceCheckpointManifest,
    utc_now,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

_SOURCE_ID = "Qwen/Qwen3.6-27B"
_SOURCE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def _bound(root: Path, path: Path) -> BoundFile:
    return BoundFile(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _write_fixture(root: Path, relative: str, content: str = "{}\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _source_manifest(root: Path) -> tuple[Path, SourceCheckpointManifest]:
    source_dir = root / "source"
    source_dir.mkdir()
    config = _write_fixture(root, "source/config.json")
    tokenizer = _write_fixture(root, "source/tokenizer.json")
    index = _write_fixture(
        root,
        "source/model.safetensors.index.json",
        '{"metadata": {}, "weight_map": {"x": "model-00001-of-00001.safetensors"}}\n',
    )
    weight = _write_fixture(root, "source/model-00001-of-00001.safetensors", "weights")
    files = [
        SourceCheckpointFile(
            path=path.name,
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in (config, tokenizer, index, weight)
    ]
    manifest = SourceCheckpointManifest(
        source_model=load_model(root / "artifact/axquant_plan.json", QuantizationPlan).source_model,
        config_sha256=file_sha256(config),
        tokenizer_sha256=file_sha256(tokenizer),
        files=files,
    )
    path = root / "source-checkpoint-manifest.json"
    write_data(path, manifest)
    return path, manifest


def _flagship_fixture(
    root: Path, *, target_class: str = "4bit"
) -> tuple[Path, FlagshipReleaseAuditRequest]:
    probe = root / "probe.safetensors"
    save_file(
        {"model.layers.0.mlp.down_proj.weight": np.zeros((1,), dtype=np.float32)},
        probe,
    )
    measured_main_bpw = probe.stat().st_size * 8 / 1088
    display_name = certified_mixed_precision_name(
        _SOURCE_ID,
        measured_main_bpw,
        mtp=True,
    )
    candidate_repository = f"AutomatosX/{display_name}"
    legacy_request_path = _inputs(
        root,
        source_model_id=_SOURCE_ID,
        source_revision=_SOURCE_REVISION,
        candidate_model_id=candidate_repository,
        target_class_override=target_class,
    )
    legacy_request = load_model(legacy_request_path, ReleaseAuditRequest)
    legacy_audit = build_release_audit(legacy_request_path)
    assert legacy_audit.release_ready
    artifact = Path(legacy_request.artifact_directory)
    plan = load_model(artifact / "axquant_plan.json", QuantizationPlan)
    manifest = load_model(artifact / "axquant_manifest.json", ArtifactManifest)

    source_manifest_path, source_manifest = _source_manifest(root)
    policy = _write_fixture(root, "flagship-policy.json", '{"policy": "qwen36-mtp-v2"}\n')
    sentinel = root / "activation-capture-sentinel.json"
    write_data(
        sentinel,
        ActivationCaptureSentinel(plan_sha256=semantic_plan_sha256(plan)),
    )
    calibration = artifact / "calibration_manifest.json"
    sensitivity = Path(legacy_request.sensitivity_report)
    candidate = candidate_key_from_artifacts(
        source_manifest=source_manifest,
        certification_policy_sha256=file_sha256(policy),
        calibration_sha256=file_sha256(calibration),
        activation_capture_sha256=file_sha256(sentinel),
        sensitivity_sha256=file_sha256(sensitivity),
        plan=plan,
        artifact_manifest=manifest,
    )
    candidate_path = root / "candidate-key.json"
    write_data(candidate_path, candidate)
    frontier_evidence_path = _write_fixture(root, "frontier/all-gates.json")
    frontier_evidence = _bound(root, frontier_evidence_path)
    frontier_path = root / "frontier/index.json"
    write_data(
        frontier_path,
        FlagshipFrontierIndex(
            source=checkpoint_key_from_source_manifest(source_manifest),
            policy_sha256=file_sha256(policy),
            search_budget=1,
            search_used=1,
            entries=[
                FlagshipFrontierEntry(
                    candidate_id="candidate-001",
                    candidate=candidate,
                    gates=[
                        FrontierGateResult(
                            gate=gate,
                            status=FrontierGateStatus.PASSED,
                            evidence=frontier_evidence,
                        )
                        for gate in FrontierGate
                    ],
                    measured_main_bpw=measured_main_bpw,
                    measured_total_bpw=manifest.measured_total_bpw,
                    eligible_for_formal=True,
                )
            ],
            feasible_candidate_sha256=[stable_sha256(candidate)],
        ),
    )

    hardware_path = Path(legacy_request.hardware_registry)
    hardware = load_model(hardware_path, HardwareProfileRegistry)
    hardware_entry = hardware.entries[0]
    host_contract = FormalHostContract(
        hardware_id=f"mbp-m5/{hardware_entry.hardware.chip}",
        os_version=hardware_entry.hardware.os_version,
        power_mode=hardware_entry.hardware.power_mode,
        storage_contract="fixture durable evidence root with verified backup",
        thermal_protocol="cold start, fixed warmups, measured trials, cooldown",
        operator="runtime-owner",
    )
    host_scope_path = root / "formal-host-scope.json"
    host_evidence: list[FormalHostEvidenceBinding] = []
    for kind in FormalHostEvidenceKind:
        evidence_path = root / f"hardware/evidence/{kind.value}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_end = utc_now() - timedelta(minutes=1)
        write_data(
            evidence_path,
            FormalHostEvidenceResult(
                evidence_name=kind.value,
                kind=kind,
                subject_sha256=stable_sha256(host_contract),
                command=["fixture-host-check", kind.value],
                observations={"fixture": True},
                software_versions={"fixture": "1.0"},
                started_at=evidence_end - timedelta(seconds=1),
                completed_at=evidence_end,
            ),
        )
        host_evidence.append(
            FormalHostEvidenceBinding(
                name=kind.value,
                kind=kind,
                file=_bound(root, evidence_path),
            )
        )
    write_data(
        host_scope_path,
        FormalHostScopeEvidence(
            contract=host_contract,
            free_disk_bytes=1_000_000_000,
            evidence=host_evidence,
        ),
    )

    durable = root
    runtime_builds = {
        name: _bound(root, _write_fixture(root, f"runtime/{name}.json"))
        for name in ("ax-engine", "mlx", "mlx-lm")
    }
    dataset_digests = {
        role: f"{index_value + 1:x}" * 64 for index_value, role in enumerate(CampaignDatasetRole)
    }
    datasets: list[CampaignDataset] = []
    for index_value, role in enumerate(CampaignDatasetRole):
        attestation_file = _write_fixture(
            root,
            f"datasets/{index_value}/manifest-attestation.json",
        )
        manifest_file = root / f"datasets/{index_value}/manifest.json"
        write_data(
            manifest_file,
            CampaignDatasetManifest(
                dataset_id=f"dataset-{index_value}",
                role=role,
                content_sha256=dataset_digests[role],
                record_count=1,
                provenance=["clean-room fixture"],
                composition={"fixture": 1},
                scorer_versions={"fixture-scorer": "1.0"},
                raw_output_retention_policy="retain all raw outputs for campaign lifetime",
                sealed_by=(
                    "evaluation-custodian"
                    if role
                    in {
                        CampaignDatasetRole.FORMAL_AGENT_CODING,
                        CampaignDatasetRole.FORMAL_GENERAL,
                    }
                    else "certification-owner"
                ),
            ),
        )
        overlap = root / f"datasets/{index_value}/overlap.json"
        write_data(
            overlap,
            CampaignOverlapReport(
                dataset_sha256=dataset_digests[role],
                compared_dataset_sha256=sorted(
                    digest
                    for compared_role, digest in dataset_digests.items()
                    if compared_role is not role
                ),
                dataset_record_count=1,
                compared_record_count_by_sha256={
                    digest: 1
                    for compared_role, digest in dataset_digests.items()
                    if compared_role is not role
                },
                comparison_pair_count=len(CampaignDatasetRole) - 1,
                exact_match_count=0,
                near_duplicate_count=0,
                near_duplicate_threshold=0.9,
                passed=True,
            ),
        )
        datasets.append(
            CampaignDataset(
                dataset_id=f"dataset-{index_value}",
                role=role,
                content_sha256=dataset_digests[role],
                manifest=_bound(root, manifest_file),
                manifest_attestation=_bound(root, attestation_file),
                overlap_report=_bound(root, overlap),
                overlap_passed=True,
                sealed=True,
            )
        )
    baselines: list[CampaignBaseline] = []
    for kind in ("bf16", "uniform-4bit", "uniform-6bit"):
        checkpoint_files = [
            _bound(
                root,
                _write_fixture(
                    root,
                    f"baselines/{kind}/model.safetensors",
                    f"{kind}-weights",
                ),
            )
        ]
        baselines.append(
            CampaignBaseline(
                kind=kind,  # type: ignore[arg-type]
                source=checkpoint_key_from_source_manifest(source_manifest),
                artifact_manifest=_bound(
                    root,
                    _write_fixture(root, f"baselines/{kind}/axquant_manifest.json"),
                ),
                checkpoint_files=checkpoint_files,
                checkpoint_members_sha256=stable_sha256(
                    [
                        item.model_dump(mode="json")
                        for item in sorted(checkpoint_files, key=lambda item: item.path)
                    ]
                ),
                runtime_versions={
                    "ax-engine": "1.0",
                    "mlx": "1.0",
                    "mlx-lm": "1.0",
                },
                available=True,
            )
        )
    backup = _write_fixture(root, "storage/restore-readback.json")
    lifecycle_evidence = _bound(
        root,
        _write_fixture(root, "lifecycle/freeze-evidence.json"),
    )
    frozen_lifecycle = ArtifactLifecycleRegistry(registry_id="flagship", events=[])
    for lifecycle_state in (
        ArtifactLifecycleState.DEVELOPMENT,
        ArtifactLifecycleState.CANDIDATE,
        ArtifactLifecycleState.FROZEN,
    ):
        frozen_lifecycle = transition_lifecycle(
            registry=frozen_lifecycle,
            candidate=candidate,
            new_state=lifecycle_state,
            actor="certification-owner",
            reviewer="independent-reviewer",
            reason=LifecycleReason.PROVENANCE_ERROR,
            narrative=f"fixture transition to {lifecycle_state.value}",
            authorizing_evidence=lifecycle_evidence,
        )
    frozen_lifecycle_path = root / "lifecycle/frozen-registry.json"
    write_data(frozen_lifecycle_path, frozen_lifecycle)
    campaign = FlagshipCampaign(
        campaign_id="qwen36-27b-mtp-001",
        state=CampaignState.FROZEN,
        source=checkpoint_key_from_source_manifest(source_manifest),
        target_class=manifest.target_class,
        policy_file=_bound(root, policy),
        toolkit_wheel=_bound(root, Path(legacy_request.toolkit_wheel)),
        runtime_builds=runtime_builds,
        formal_host=host_contract,
        hardware_scope=_bound(root, host_scope_path),
        hardware_scope_evidence=host_evidence,
        datasets=datasets,
        baselines=baselines,
        candidate=candidate,
        candidate_inputs=CandidateInputBindings(
            source_checkpoint_manifest=_bound(root, source_manifest_path),
            calibration_manifest=_bound(root, calibration),
            activation_capture_or_sentinel=_bound(root, sentinel),
            sensitivity_report=_bound(root, sensitivity),
            plan=_bound(root, artifact / "axquant_plan.json"),
            artifact_manifest=_bound(root, artifact / "axquant_manifest.json"),
            candidate_frontier=_bound(root, frontier_path),
            frontier_evidence=[frontier_evidence],
        ),
        lifecycle_registry=_bound(root, frozen_lifecycle_path),
        durable_evidence_root=str(durable.resolve()),
        backup_verification=_bound(root, backup),
        required_free_disk_bytes=500_000_000,
        expected_stage_outputs={
            "formal-agent-coding": 10_000,
            "formal-general": 10_000,
            "hardware": 10_000,
        },
        roles=CampaignRoles(
            product_owner="product-owner",
            certification_owner="certification-owner",
            model_engineer="model-engineer",
            runtime_owner="runtime-owner",
            evaluation_custodian="evaluation-custodian",
            independent_reviewer="independent-reviewer",
            release_manager="release-manager",
        ),
        frozen_at=utc_now(),
        created_by="certification-owner",
    )
    campaign_path = root / "flagship-campaign.json"
    write_data(campaign_path, campaign)
    preflight_path = root / "flagship-campaign-preflight.json"
    preflight = preflight_campaign(
        campaign_path=campaign_path,
        output_path=preflight_path,
        observed_host_id="mbp-m5",
    )
    assert preflight.passed

    formal_results: dict[str, BoundFile] = {}
    for profile in ("agent-coding", "general"):
        formal_results[profile] = _bound(
            root,
            _write_fixture(root, f"formal/{profile}/result.json"),
        )
    formal_raw_path = _write_fixture(
        root,
        "formal/raw/task.json",
        '{"raw": true}\n',
    )
    formal_raw_index_path = root / "formal/raw-evidence-index.json"
    write_data(
        formal_raw_index_path,
        EvidenceArchiveIndex(
            records=[
                EvidenceArchiveRecord(
                    logical_name="formal-raw-task",
                    path="raw/task.json",
                    sha256=file_sha256(formal_raw_path),
                    size_bytes=formal_raw_path.stat().st_size,
                    durable_uri=f"file://{formal_raw_path}",
                )
            ],
            complete=True,
        ),
    )
    custodian_attestation_path = _write_fixture(
        root,
        "formal/custodian-attestation.sigstore.json",
        '{"verified": true}\n',
    )
    now = utc_now()
    formal_path = root / "formal-holdout-completion.json"
    write_data(
        formal_path,
        FormalHoldoutCompletion(
            campaign_sha256=stable_sha256(campaign),
            candidate_sha256=stable_sha256(candidate),
            started_at=now,
            completed_at=now + timedelta(minutes=1),
            dataset_sha256_by_profile={
                "agent-coding": next(
                    item.content_sha256
                    for item in datasets
                    if item.role is CampaignDatasetRole.FORMAL_AGENT_CODING
                ),
                "general": next(
                    item.content_sha256
                    for item in datasets
                    if item.role is CampaignDatasetRole.FORMAL_GENERAL
                ),
            },
            result_file_by_profile=formal_results,
            raw_evidence_index=_bound(root, formal_raw_index_path),
            evaluation_custodian="evaluation-custodian",
            custodian_attestation=_bound(root, custodian_attestation_path),
            verdict="pass",
            gate_issues=[],
        ),
    )

    verification_path = Path(legacy_request.reproduction_verification)
    reproduction_review_path = root / "reproduction-review.json"
    write_data(
        reproduction_review_path,
        ReproductionReviewRecord(
            candidate_sha256=stable_sha256(candidate),
            reproduction_host_id="clean-host",
            reproduction_verification=_bound(root, verification_path),
            reviewer="independent-reviewer",
        ),
    )
    hardware_attestation = _write_fixture(
        root,
        "hardware/mbp-m5-attestation.sigstore.json",
        '{"verified": true}\n',
    )
    hardware_authorization_path = root / "hardware-authorization.json"
    write_data(
        hardware_authorization_path,
        HardwareAuthorizationRecord(
            campaign_sha256=stable_sha256(campaign),
            candidate_sha256=stable_sha256(candidate),
            hardware_id=host_contract.hardware_id,
            hardware_registry=_bound(root, hardware_path),
            operator=host_contract.operator,
            attestation=_bound(root, hardware_attestation),
        ),
    )
    review_path = root / "independent-review.json"
    review_attestation = _write_fixture(
        root,
        "review/attestation.sigstore.json",
        '{"verified": true}\n',
    )
    write_data(
        review_path,
        IndependentReviewRecord(
            campaign_sha256=stable_sha256(campaign),
            candidate_sha256=stable_sha256(candidate),
            legacy_audit_sha256=stable_sha256(legacy_audit),
            reviewer="independent-reviewer",
            checks_reviewed=[
                "freeze",
                "candidate-selection",
                "raw-to-summary",
                "hardware-pareto",
                "reproduction",
                "public-claims",
            ],
            verdict="pass",
            issues=[],
            attestation=_bound(root, review_attestation),
        ),
    )

    archive_sources = [
        ("campaign", campaign_path),
        ("campaign-preflight", preflight_path),
        ("candidate-key", candidate_path),
        ("source-checkpoint-manifest", source_manifest_path),
        ("certification-policy", policy),
        ("calibration-manifest", calibration),
        ("activation-capture-or-sentinel", sentinel),
        ("formal-holdout-completion", formal_path),
        ("formal-raw-evidence-index", formal_raw_index_path),
        ("formal-custodian-attestation", custodian_attestation_path),
        ("formal-raw-task", formal_raw_path),
        ("formal-agent-coding", root / formal_results["agent-coding"].path),
        ("formal-general", root / formal_results["general"].path),
        ("hardware-mbp-m5", hardware_path),
        ("hardware-authorization", hardware_authorization_path),
        ("hardware-authorization-attestation", hardware_attestation),
        ("reproduction", verification_path),
        ("reproduction-review", reproduction_review_path),
        ("independent-review", review_path),
        ("independent-review-attestation", review_attestation),
    ]
    archived_bound_keys = {
        (file_sha256(path), path.stat().st_size) for _name, path in archive_sources
    }
    for index_value, bound in enumerate(campaign_bound_files(campaign)):
        key = (bound.sha256, bound.size_bytes)
        if key not in archived_bound_keys:
            path = root / bound.path
            archive_sources.append((f"campaign-bound-{index_value:03d}", path))
            archived_bound_keys.add(key)
    archive_records: list[tuple[str, Path]] = []
    for name, source_path in archive_sources:
        archived_path = durable / "records" / f"{name}.json"
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        archived_path.write_bytes(source_path.read_bytes())
        archive_records.append((name, archived_path))
    archive_index_path = durable / "evidence-archive-index.json"
    write_data(
        archive_index_path,
        EvidenceArchiveIndex(
            records=[
                EvidenceArchiveRecord(
                    logical_name=name,
                    path=path.resolve().relative_to(durable.resolve()).as_posix(),
                    sha256=file_sha256(path),
                    size_bytes=path.stat().st_size,
                    durable_uri=f"file://{path.resolve()}",
                )
                for name, path in archive_records
            ],
            complete=True,
        ),
    )
    archive_proof_path = durable / "archive-proof.json"
    write_data(
        archive_proof_path,
        FlagshipArchiveProof(
            campaign_sha256=stable_sha256(campaign),
            candidate_sha256=stable_sha256(candidate),
            durable_evidence_root=str(durable.resolve()),
            archive_index=_bound(durable, archive_index_path),
        ),
    )

    request = FlagshipReleaseAuditRequest(
        legacy_release_audit_request=legacy_request_path.name,
        campaign=campaign_path.name,
        campaign_preflight=preflight_path.name,
        candidate_key=candidate_path.name,
        source_checkpoint_manifest=source_manifest_path.name,
        certification_policy=policy.name,
        calibration_manifest=calibration.relative_to(root).as_posix(),
        activation_capture_or_sentinel=sentinel.name,
        formal_holdout_completion=formal_path.name,
        archive_proof=archive_proof_path.relative_to(root).as_posix(),
        independent_review=review_path.name,
        reproduction_review=reproduction_review_path.name,
        hardware_authorization=hardware_authorization_path.name,
    )
    request_path = root / "flagship-release-audit-request.json"
    write_data(request_path, request)
    return request_path, request


def test_flagship_audit_has_separate_authorization_and_publication_stages(
    tmp_path: Path,
) -> None:
    request_path, request = _flagship_fixture(tmp_path)
    authorization = build_flagship_release_audit(request_path)

    assert authorization.authorization_ready
    assert not authorization.release_ready
    assert authorization.checks[-1].gate_id == "M8"
    assert "publication may not" in authorization.checks[-1].issues[-1]

    authorization_path = tmp_path / "flagship-authorization-audit.json"
    write_data(authorization_path, authorization)
    candidate = authorization.candidate
    repository = authorization.candidate_model.model_id
    campaign = load_model(tmp_path / request.campaign, FlagshipCampaign)
    registry = load_model(
        tmp_path / campaign.lifecycle_registry.path,
        ArtifactLifecycleRegistry,
    )
    evidence = _bound(tmp_path, authorization_path)
    registry = transition_lifecycle(
        registry=registry,
        candidate=candidate,
        new_state=ArtifactLifecycleState.CERTIFIED,
        actor="release-manager",
        reviewer="independent-reviewer",
        reason=LifecycleReason.CERTIFICATION_PASSED,
        narrative="fixture transition to certified",
        authorizing_evidence=evidence,
        public_repository=repository,
    )
    lifecycle_path = tmp_path / "lifecycle-registry.json"
    write_data(lifecycle_path, registry)

    metric_evidence = _bound(tmp_path, tmp_path / "formal/agent-coding/result.json")
    quality_claims = [
        BoundMetricClaim(
            evidence=metric_evidence,
            profile=profile,  # type: ignore[arg-type]
            metric_key="quality.aggregate_retention",
            unit="ratio",
            value=value,
            comparison="higher-is-better",
        )
        for profile, value in (("agent-coding", 0.99), ("general", 0.98))
    ]
    performance_claims = [
        BoundMetricClaim(
            evidence=metric_evidence,
            profile="hardware",
            metric_key="hardware.effective_speedup",
            unit="x",
            value=1.25,
            numerator=1.25,
            denominator=1,
            comparison="ratio",
        )
    ]
    claim_request_path = tmp_path / "public-claim-render-request.json"
    write_data(
        claim_request_path,
        PublicClaimRenderRequest(
            authorization_audit=authorization_path.name,
            lifecycle_registry=lifecycle_path.name,
            artifact_manifest="artifact/axquant_manifest.json",
            public_owner=repository.split("/", 1)[0],
            quality_claims=quality_claims,
            performance_claims=performance_claims,
            limitations=["Certification applies only to the exact recorded evidence scope."],
            reviewer="independent-reviewer",
        ),
    )
    claim_path = tmp_path / "public-claim.json"
    card_path = tmp_path / "README.certified.md"
    claim = render_public_claim_request(
        request_path=claim_request_path,
        claim_path=claim_path,
        model_card_path=card_path,
    )
    assert claim.public_repository == repository
    publication_review_attestation = _write_fixture(
        tmp_path,
        "durable/final-review/attestation.sigstore.json",
        '{"verified": true}\n',
    )
    publication_review_path = tmp_path / "durable/final-publication-review.json"
    write_data(
        publication_review_path,
        FinalPublicationReviewRecord(
            campaign_sha256=stable_sha256(campaign),
            candidate_sha256=stable_sha256(candidate),
            authorization_audit_sha256=file_sha256(authorization_path),
            public_claim_sha256=file_sha256(claim_path),
            model_card_sha256=file_sha256(card_path),
            reviewer="independent-reviewer",
            verdict="pass",
            issues=[],
            attestation=_bound(tmp_path / "durable", publication_review_attestation),
        ),
    )
    final_request = request.model_copy(
        update={
            "authorization_audit": authorization_path.name,
            "lifecycle_registry": lifecycle_path.name,
            "public_claim": claim_path.name,
            "model_card": card_path.name,
            "final_publication_review": publication_review_path.relative_to(tmp_path).as_posix(),
        }
    )
    final_request_path = tmp_path / "flagship-release-audit-request.final.json"
    write_data(final_request_path, final_request)

    final = build_certification_audit(final_request_path)

    assert isinstance(final, FlagshipReleaseAudit)
    assert final.authorization_ready
    assert final.release_ready
    assert [check.gate_id for check in final.checks] == [f"M{index}" for index in range(9)]

    publication_review = load_model(
        publication_review_path,
        FinalPublicationReviewRecord,
    )
    write_data(
        publication_review_path,
        publication_review.model_copy(update={"model_card_sha256": "0" * 64}),
    )
    rejected = build_certification_audit(final_request_path)
    assert not rejected.release_ready
    assert any(
        "publication review binds another model card" in issue
        for issue in rejected.checks[8].issues
    )


def test_flagship_audit_rejects_host_scope_drift(tmp_path: Path) -> None:
    request_path, _request = _flagship_fixture(tmp_path)
    scope_path = tmp_path / "formal-host-scope.json"
    scope = load_model(scope_path, FormalHostScopeEvidence)
    scope.contract.os_version = "different macOS"
    write_data(scope_path, scope)

    audit = build_flagship_release_audit(request_path)

    assert not audit.authorization_ready
    assert any("bound file checksum changed" in issue for issue in audit.checks[0].issues)


def test_flagship_audit_rejects_failed_formal_verdict(tmp_path: Path) -> None:
    request_path, request = _flagship_fixture(tmp_path)
    formal_path = tmp_path / request.formal_holdout_completion
    formal = load_model(formal_path, FormalHoldoutCompletion)
    write_data(
        formal_path,
        formal.model_copy(
            update={
                "verdict": "fail",
                "gate_issues": ["fixture formal retention threshold failed"],
            }
        ),
    )

    audit = build_flagship_release_audit(request_path)

    assert not audit.authorization_ready
    assert any(
        "formal holdout completion did not pass" in issue for issue in audit.checks[2].issues
    )


def test_flagship_audit_rejects_changed_formal_raw_evidence(tmp_path: Path) -> None:
    request_path, _request = _flagship_fixture(tmp_path)
    (tmp_path / "formal/raw/task.json").write_text('{"tampered": true}\n', encoding="utf-8")

    audit = build_flagship_release_audit(request_path)

    assert not audit.authorization_ready
    assert any("raw/task.json" in issue for issue in audit.checks[2].issues)


def test_flagship_audit_requires_every_campaign_binding_in_durable_archive(
    tmp_path: Path,
) -> None:
    request_path, request = _flagship_fixture(tmp_path)
    proof_path = tmp_path / request.archive_proof
    proof = load_model(proof_path, FlagshipArchiveProof)
    index_path = proof_path.parent / proof.archive_index.path
    index = load_model(index_path, EvidenceArchiveIndex)
    removable = next(
        record for record in index.records if record.logical_name.startswith("campaign-bound-")
    )
    write_data(
        index_path,
        index.model_copy(
            update={"records": [record for record in index.records if record != removable]}
        ),
    )
    write_data(
        proof_path,
        proof.model_copy(update={"archive_index": _bound(proof_path.parent, index_path)}),
    )

    audit = build_flagship_release_audit(request_path)

    assert not audit.authorization_ready
    assert any("campaign bound file is absent" in issue for issue in audit.checks[3].issues)


def test_flagship_audit_cannot_be_rerun_through_legacy_request(tmp_path: Path) -> None:
    request_path, request = _flagship_fixture(tmp_path)
    audit = build_flagship_release_audit(request_path)

    with pytest.raises(PublishingError, match="cannot be downgraded"):
        _rerun_release_audit(
            audit=audit,
            request_path=tmp_path / request.legacy_release_audit_request,
        )


def test_flagship_audit_accepts_6bit_target_class(tmp_path: Path) -> None:
    request_path, _ = _flagship_fixture(tmp_path, target_class="6bit")

    audit = build_flagship_release_audit(request_path)

    m0 = next(check for check in audit.checks if check.gate_id == "M0")
    assert not any("target class" in issue for issue in m0.issues)
    assert audit.authorization_ready


def test_flagship_audit_rejects_unsupported_target_class(tmp_path: Path) -> None:
    request_path, _ = _flagship_fixture(tmp_path, target_class="5bit")

    audit = build_flagship_release_audit(request_path)

    m0 = next(check for check in audit.checks if check.gate_id == "M0")
    assert any("target class" in issue for issue in m0.issues)
    assert not audit.authorization_ready


def test_flagship_audit_fails_closed_on_a_damaged_mtp_bundle(
    tmp_path: Path,
) -> None:
    request_path, _ = _flagship_fixture(tmp_path)
    corrupted = next(tmp_path.rglob("agent-coding-axquant-mtp-on.json"))
    corrupted.write_text("{not valid json", encoding="utf-8")

    # Evidence integrity is a precondition of auditing at all: a damaged
    # bundle must abort the audit with a named checksum error (fail closed),
    # not produce gate verdicts against a tampered evidence set.
    with pytest.raises(ArtifactError, match="checksum does not match"):
        build_flagship_release_audit(request_path)
