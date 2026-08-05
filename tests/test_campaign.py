from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from test_release_audit import _inputs

from axquant.campaign import (
    build_flagship_frontier,
    close_campaign_no_go,
    complete_formal_campaign,
    formal_host_scope_evidence_issues,
    freeze_campaign,
    preflight_campaign,
    record_campaign_publication,
    start_formal_campaign,
)
from axquant.claims import build_public_claim
from axquant.errors import ArtifactError, ValidationGateError
from axquant.identity import (
    candidate_key_from_artifacts,
    checkpoint_key_from_source_manifest,
    semantic_plan_sha256,
)
from axquant.lifecycle import transition_lifecycle
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
    FlagshipCampaign,
    FlagshipFrontierEntry,
    FlagshipFrontierIndex,
    FlagshipFrontierRequest,
    FlagshipNoGoRecord,
    FlagshipPublicationVerification,
    FlagshipReleaseAudit,
    FormalHoldoutCompletion,
    FormalHostContract,
    FormalHostEvidenceBinding,
    FormalHostEvidenceKind,
    FormalHostEvidenceResult,
    FormalHostScopeEvidence,
    FrontierGate,
    FrontierGateResult,
    FrontierGateStatus,
    LifecycleReason,
    PostPublicationRuntimeVerification,
    PublicClaimManifest,
    QuantizationPlan,
    ReleaseAuditCheck,
    ReleaseAuditRequest,
    SourceCheckpointFile,
    SourceCheckpointManifest,
    utc_now,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

_SOURCE_ID = "Qwen/Qwen3.6-27B"
_SOURCE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def _bound(root: Path, name: str, content: str = "evidence") -> BoundFile:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return BoundFile(path=name, sha256=file_sha256(path), size_bytes=path.stat().st_size)


def _existing_bound(root: Path, path: Path) -> BoundFile:
    return BoundFile(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _source_manifest(root: Path, plan: QuantizationPlan) -> tuple[Path, SourceCheckpointManifest]:
    source_dir = root / "source"
    source_dir.mkdir()
    paths = [
        source_dir / "config.json",
        source_dir / "tokenizer.json",
        source_dir / "model.safetensors.index.json",
        source_dir / "model-00001-of-00001.safetensors",
    ]
    for path, content in zip(
        paths,
        (
            "{}\n",
            "{}\n",
            '{"metadata": {}, "weight_map": {"x": "model-00001-of-00001.safetensors"}}\n',
            "weights",
        ),
        strict=True,
    ):
        path.write_text(content, encoding="utf-8")
    manifest = SourceCheckpointManifest(
        source_model=plan.source_model,
        config_sha256=file_sha256(paths[0]),
        tokenizer_sha256=file_sha256(paths[1]),
        files=[
            SourceCheckpointFile(
                path=path.name,
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
            )
            for path in paths
        ],
    )
    manifest_path = root / "source-checkpoint-manifest.json"
    write_data(manifest_path, manifest)
    return manifest_path, manifest


def _campaign(root: Path, durable: Path) -> FlagshipCampaign:
    legacy_request_path = _inputs(
        root,
        source_model_id=_SOURCE_ID,
        source_revision=_SOURCE_REVISION,
        candidate_model_id="AutomatosX/AXQuant-Qwen3.6-27B-fixture",
        target_class_override="4bit",
    )
    legacy_request = load_model(legacy_request_path, ReleaseAuditRequest)
    artifact_dir = Path(legacy_request.artifact_directory)
    plan_path = artifact_dir / "axquant_plan.json"
    artifact_manifest_path = artifact_dir / "axquant_manifest.json"
    plan = load_model(plan_path, QuantizationPlan)
    artifact_manifest = load_model(artifact_manifest_path, ArtifactManifest)
    source_manifest_path, source_manifest = _source_manifest(root, plan)
    source = checkpoint_key_from_source_manifest(source_manifest)
    policy = _bound(root, "policy.json", '{"policy": "fixture"}\n')
    calibration_path = artifact_dir / "calibration_manifest.json"
    sensitivity_path = Path(legacy_request.sensitivity_report)
    sentinel_path = root / "activation-capture-sentinel.json"
    write_data(
        sentinel_path,
        ActivationCaptureSentinel(plan_sha256=semantic_plan_sha256(plan)),
    )
    candidate = candidate_key_from_artifacts(
        source_manifest=source_manifest,
        certification_policy_sha256=policy.sha256,
        calibration_sha256=file_sha256(calibration_path),
        activation_capture_sha256=file_sha256(sentinel_path),
        sensitivity_sha256=file_sha256(sensitivity_path),
        plan=plan,
        artifact_manifest=artifact_manifest,
    )
    frontier_evidence = _bound(root, "frontier/all-gates.json")
    frontier_path = root / "frontier/index.json"
    write_data(
        frontier_path,
        FlagshipFrontierIndex(
            source=source,
            policy_sha256=policy.sha256,
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
                    measured_main_bpw=5.2,
                    measured_total_bpw=5.3,
                    eligible_for_formal=True,
                )
            ],
            feasible_candidate_sha256=[stable_sha256(candidate)],
        ),
    )
    dataset_digests = {
        role: f"{index + 1:x}" * 64 for index, role in enumerate(CampaignDatasetRole)
    }
    datasets: list[CampaignDataset] = []
    for index, role in enumerate(CampaignDatasetRole):
        attestation = _bound(root, f"datasets/{index}/manifest-attestation.json")
        dataset_manifest_path = root / f"datasets/{index}/manifest.json"
        write_data(
            dataset_manifest_path,
            CampaignDatasetManifest(
                dataset_id=f"dataset-{index}",
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
        overlap_path = root / f"datasets/{index}/overlap.json"
        overlap_path.parent.mkdir(parents=True, exist_ok=True)
        write_data(
            overlap_path,
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
                dataset_id=f"dataset-{index}",
                role=role,
                content_sha256=dataset_digests[role],
                manifest=_existing_bound(root, dataset_manifest_path),
                manifest_attestation=attestation,
                overlap_report=_existing_bound(root, overlap_path),
                overlap_passed=True,
                sealed=True,
            )
        )
    baselines: list[CampaignBaseline] = []
    for kind in ("bf16", "uniform-4bit", "uniform-6bit"):
        checkpoint_files = [
            _bound(
                root,
                f"baselines/{kind}/model.safetensors",
                f"{kind}-weights",
            )
        ]
        baselines.append(
            CampaignBaseline(
                kind=kind,  # type: ignore[arg-type]
                source=source,
                artifact_manifest=_bound(
                    root,
                    f"baselines/{kind}/manifest.json",
                ),
                checkpoint_files=checkpoint_files,
                checkpoint_members_sha256=stable_sha256(
                    [
                        item.model_dump(mode="json")
                        for item in sorted(checkpoint_files, key=lambda item: item.path)
                    ]
                ),
                runtime_versions={
                    "mlx": "1.0",
                    "mlx-lm": "1.0",
                    "ax-engine": "1.0",
                },
                available=True,
            )
        )
    host_contract = FormalHostContract(
        hardware_id="mbp-m5/apple-m5/fixture",
        os_version="macOS fixture",
        power_mode="high-power",
        storage_contract="durable fixture",
        thermal_protocol="cold-start then fixed warmups",
        operator="runtime-owner",
    )
    host_scope_path = root / "hardware/mbp-m5.json"
    host_scope_path.parent.mkdir(parents=True, exist_ok=True)
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
                file=_existing_bound(root, evidence_path),
            )
        )
    write_data(
        host_scope_path,
        FormalHostScopeEvidence(
            contract=host_contract,
            free_disk_bytes=2_000_000_000,
            evidence=host_evidence,
        ),
    )
    lifecycle_evidence = _bound(root, "lifecycle/freeze-evidence.json")
    lifecycle = ArtifactLifecycleRegistry(registry_id="flagship", events=[])
    for state in (
        ArtifactLifecycleState.DEVELOPMENT,
        ArtifactLifecycleState.CANDIDATE,
        ArtifactLifecycleState.FROZEN,
    ):
        lifecycle = transition_lifecycle(
            registry=lifecycle,
            candidate=candidate,
            new_state=state,
            actor="certification-owner",
            reviewer="independent-reviewer",
            reason=LifecycleReason.PROVENANCE_ERROR,
            narrative=f"fixture transition to {state.value}",
            authorizing_evidence=lifecycle_evidence,
        )
    lifecycle_path = root / "lifecycle/registry.json"
    write_data(lifecycle_path, lifecycle)
    return FlagshipCampaign(
        campaign_id="qwen36-flagship-001",
        source=source,
        target_class="4bit",
        policy_file=policy,
        toolkit_wheel=_bound(root, "dist/axquant.whl"),
        runtime_builds={
            "ax-engine": _bound(root, "runtime/ax-engine.json"),
            "mlx": _bound(root, "runtime/mlx.json"),
            "mlx-lm": _bound(root, "runtime/mlx-lm.json"),
        },
        formal_host=host_contract,
        hardware_scope=_bound(
            root,
            "hardware/mbp-m5.json",
            host_scope_path.read_text(encoding="utf-8"),
        ),
        hardware_scope_evidence=host_evidence,
        datasets=datasets,
        baselines=baselines,
        candidate=candidate,
        candidate_inputs=CandidateInputBindings(
            source_checkpoint_manifest=_existing_bound(root, source_manifest_path),
            calibration_manifest=_existing_bound(root, calibration_path),
            activation_capture_or_sentinel=_existing_bound(root, sentinel_path),
            sensitivity_report=_existing_bound(root, sensitivity_path),
            plan=_existing_bound(root, plan_path),
            artifact_manifest=_existing_bound(root, artifact_manifest_path),
            candidate_frontier=_existing_bound(root, frontier_path),
            frontier_evidence=[frontier_evidence],
        ),
        lifecycle_registry=_bound(
            root,
            "lifecycle/registry.json",
            lifecycle_path.read_text(encoding="utf-8"),
        ),
        durable_evidence_root=str(durable),
        backup_verification=_bound(root, "storage/restore-drill.json"),
        required_free_disk_bytes=1_000_000_000,
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
        created_by="certification-owner",
    )


def test_freeze_and_preflight_are_fail_closed(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    request = tmp_path / "campaign-request.json"
    frozen_path = tmp_path / "campaign.json"
    write_data(request, campaign)

    frozen = freeze_campaign(request_path=request, output_path=frozen_path)
    result = preflight_campaign(
        campaign_path=frozen_path,
        observed_host_id="mbp-m5",
    )

    assert frozen.state.value == "frozen"
    assert result.passed
    assert not result.issues

    (tmp_path / campaign.policy_file.path).write_text("tampered", encoding="utf-8")
    changed = preflight_campaign(
        campaign_path=frozen_path,
        observed_host_id="mbp-m5",
    )
    assert not changed.passed
    assert any("policy.json" in issue for issue in changed.issues)


def test_frontier_builder_derives_feasible_summary_and_rechecks_evidence(
    tmp_path: Path,
) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    assert campaign.candidate_inputs is not None
    existing = load_model(
        tmp_path / campaign.candidate_inputs.candidate_frontier.path,
        FlagshipFrontierIndex,
    )
    request_path = tmp_path / "frontier-request.json"
    output_path = tmp_path / "frontier.json"
    write_data(
        request_path,
        FlagshipFrontierRequest(
            source=existing.source,
            policy_sha256=existing.policy_sha256,
            search_budget=existing.search_budget,
            search_used=existing.search_used,
            entries=existing.entries,
        ),
    )

    built = build_flagship_frontier(
        request_path=request_path,
        output_path=output_path,
    )

    assert built.feasible_candidate_sha256 == [stable_sha256(existing.entries[0].candidate)]

    evidence_path = tmp_path / campaign.candidate_inputs.frontier_evidence[0].path
    evidence_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValidationGateError, match="checksum changed"):
        build_flagship_frontier(
            request_path=request_path,
            output_path=output_path,
        )


def test_freeze_rejects_consumed_formal_holdout(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    formal = next(
        dataset
        for dataset in campaign.datasets
        if dataset.role is CampaignDatasetRole.FORMAL_GENERAL
    )
    formal.consumed = True
    request = tmp_path / "campaign-request.json"
    write_data(request, campaign)

    with pytest.raises(ValidationGateError, match="already consumed"):
        freeze_campaign(request_path=request, output_path=tmp_path / "campaign.json")


def test_freeze_rejects_formal_dataset_not_sealed_by_custodian(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    formal = next(
        dataset
        for dataset in campaign.datasets
        if dataset.role is CampaignDatasetRole.FORMAL_GENERAL
    )
    manifest_path = tmp_path / formal.manifest.path
    manifest = load_model(manifest_path, CampaignDatasetManifest)
    write_data(
        manifest_path,
        manifest.model_copy(update={"sealed_by": "model-engineer"}),
    )
    formal.manifest = _existing_bound(tmp_path, manifest_path)
    request = tmp_path / "campaign-request.json"
    write_data(request, campaign)

    with pytest.raises(ValidationGateError, match="evaluation custodian"):
        freeze_campaign(request_path=request, output_path=tmp_path / "campaign.json")


def test_preflight_rejects_any_host_other_than_mbp_m5(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    request = tmp_path / "campaign-request.json"
    frozen_path = tmp_path / "campaign.json"
    write_data(request, campaign)
    freeze_campaign(request_path=request, output_path=frozen_path)

    result = preflight_campaign(
        campaign_path=frozen_path,
        observed_host_id="macstudio-m2u",
    )

    assert not result.passed
    assert result.issues == ["campaign preflight must run on mbp-m5; observed macstudio-m2u"]


def test_campaign_roles_reject_placeholder_identity() -> None:
    with pytest.raises(ValueError, match="named accountable identities"):
        CampaignRoles(
            product_owner="product-owner",
            certification_owner="unassigned",
            model_engineer="model-engineer",
            runtime_owner="runtime-owner",
            evaluation_custodian="evaluation-custodian",
            independent_reviewer="independent-reviewer",
            release_manager="release-manager",
        )


def test_preflight_rejects_stale_host_scope(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    scope_path = tmp_path / campaign.hardware_scope.path
    scope = load_model(scope_path, FormalHostScopeEvidence)
    write_data(
        scope_path,
        scope.model_copy(update={"observed_at": utc_now() - timedelta(hours=2)}),
    )
    campaign.hardware_scope = _existing_bound(tmp_path, scope_path)
    request = tmp_path / "campaign-request.json"
    frozen_path = tmp_path / "campaign.json"
    write_data(request, campaign)
    freeze_campaign(request_path=request, output_path=frozen_path)

    result = preflight_campaign(
        campaign_path=frozen_path,
        observed_host_id="mbp-m5",
    )

    assert not result.passed
    assert "formal host scope evidence is stale" in result.issues


def test_formal_host_evidence_rejects_another_host_contract(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, tmp_path)
    scope_path = tmp_path / campaign.hardware_scope.path
    scope = load_model(scope_path, FormalHostScopeEvidence)
    evidence_path = tmp_path / scope.evidence[0].file.path
    evidence = load_model(evidence_path, FormalHostEvidenceResult)
    write_data(
        evidence_path,
        evidence.model_copy(update={"subject_sha256": "f" * 64}),
    )

    issues = formal_host_scope_evidence_issues(tmp_path, scope)

    assert any("binds another host contract" in issue for issue in issues)


def test_freeze_rejects_disposable_evidence_root(tmp_path: Path) -> None:
    disposable = tmp_path / ".internal" / "tmp" / "campaign"
    disposable.mkdir(parents=True)
    campaign = _campaign(tmp_path, disposable)
    request = tmp_path / "campaign-request.json"
    write_data(request, campaign)

    with pytest.raises(ValidationGateError, match=r"\.internal/tmp"):
        freeze_campaign(request_path=request, output_path=tmp_path / "campaign.json")


def test_freeze_rejects_campaign_root_outside_durable_root(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    durable = tmp_path / "durable"
    durable.mkdir()
    campaign = _campaign(campaign_root, durable)
    request = campaign_root / "campaign-request.json"
    write_data(request, campaign)

    with pytest.raises(ValidationGateError, match="inside the durable evidence root"):
        freeze_campaign(request_path=request, output_path=campaign_root / "campaign.json")


def test_freeze_rejects_symbolic_link_durable_root(tmp_path: Path) -> None:
    durable = tmp_path / "durable"
    durable.mkdir()
    durable_link = tmp_path / "durable-link"
    durable_link.symlink_to(durable, target_is_directory=True)
    campaign = _campaign(durable, durable_link)
    request = durable / "campaign-request.json"
    write_data(request, campaign)

    with pytest.raises(ValidationGateError, match="symbolic link"):
        freeze_campaign(request_path=request, output_path=durable / "campaign.json")


def test_formal_cycle_consumes_holdout_on_pass_or_failure(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    request = tmp_path / "campaign-request.json"
    frozen_path = tmp_path / "campaign.json"
    write_data(request, campaign)
    frozen = freeze_campaign(request_path=request, output_path=frozen_path)
    preflight = preflight_campaign(
        campaign_path=frozen_path,
        observed_host_id="mbp-m5",
    )
    running = start_formal_campaign(campaign=frozen, preflight=preflight)
    now = utc_now()
    result_files = {
        profile: _bound(tmp_path, f"formal/{profile}.json")
        for profile in ("agent-coding", "general")
    }
    raw_record_path = tmp_path / "formal/raw/task.json"
    raw_record_path.parent.mkdir(parents=True, exist_ok=True)
    raw_record_path.write_text('{"raw": true}\n', encoding="utf-8")
    raw_index_path = tmp_path / "formal/raw-evidence-index.json"
    write_data(
        raw_index_path,
        EvidenceArchiveIndex(
            records=[
                EvidenceArchiveRecord(
                    logical_name="formal-raw-task",
                    path="raw/task.json",
                    sha256=file_sha256(raw_record_path),
                    size_bytes=raw_record_path.stat().st_size,
                    durable_uri=f"file://{raw_record_path}",
                )
            ],
            complete=True,
        ),
    )
    custodian_attestation = _bound(tmp_path, "formal/custodian-attestation.json")
    completion = FormalHoldoutCompletion(
        campaign_sha256=stable_sha256(frozen),
        candidate_sha256=stable_sha256(frozen.candidate),  # type: ignore[arg-type]
        started_at=now,
        completed_at=now + timedelta(seconds=1),
        dataset_sha256_by_profile={
            "agent-coding": next(
                item.content_sha256
                for item in frozen.datasets
                if item.role is CampaignDatasetRole.FORMAL_AGENT_CODING
            ),
            "general": next(
                item.content_sha256
                for item in frozen.datasets
                if item.role is CampaignDatasetRole.FORMAL_GENERAL
            ),
        },
        result_file_by_profile=result_files,
        raw_evidence_index=_existing_bound(tmp_path, raw_index_path),
        evaluation_custodian=frozen.roles.evaluation_custodian,
        custodian_attestation=custodian_attestation,
        verdict="fail",
        gate_issues=["fixture formal quality gate failed"],
    )
    completion_path = tmp_path / "formal-completion.json"
    write_data(completion_path, completion)

    completed = complete_formal_campaign(
        campaign=running,
        completion_path=completion_path,
    )

    assert completed.state.value == "formal_failed"
    assert completed.formal_cycles_consumed == 1
    assert all(
        item.consumed
        for item in completed.datasets
        if item.role
        in {
            CampaignDatasetRole.FORMAL_AGENT_CODING,
            CampaignDatasetRole.FORMAL_GENERAL,
        }
    )


def test_formal_completion_must_be_inside_durable_root(tmp_path: Path) -> None:
    durable = tmp_path / "durable"
    durable.mkdir()
    campaign = _campaign(durable, durable)
    outside_completion = tmp_path / "formal-completion.json"
    outside_completion.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="inside the durable evidence root"):
        complete_formal_campaign(
            campaign=campaign,
            completion_path=outside_completion,
        )


def test_no_go_requires_exhausted_frontier_and_preserves_formal_holdout(
    tmp_path: Path,
) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    assert campaign.candidate is not None
    failed_evidence = _bound(tmp_path, "no-go/integrity.json")
    failed_frontier_path = tmp_path / "no-go/frontier.json"
    write_data(
        failed_frontier_path,
        FlagshipFrontierIndex(
            source=campaign.source,
            policy_sha256=campaign.policy_file.sha256,
            search_budget=1,
            search_used=1,
            entries=[
                FlagshipFrontierEntry(
                    candidate_id="failed-candidate",
                    candidate=campaign.candidate,
                    gates=[
                        FrontierGateResult(
                            gate=gate,
                            status=(
                                FrontierGateStatus.FAILED
                                if index == 0
                                else FrontierGateStatus.NOT_RUN
                            ),
                            evidence=failed_evidence if index == 0 else None,
                            issues=[
                                "integrity check failed"
                                if index == 0
                                else "skipped after integrity failure"
                            ],
                        )
                        for index, gate in enumerate(FrontierGate)
                    ],
                    eligible_for_formal=False,
                )
            ],
            feasible_candidate_sha256=[],
        ),
    )
    attestation = _bound(tmp_path, "no-go/attestation.json")
    campaign_path = tmp_path / "campaign.json"
    write_data(campaign_path, campaign)
    no_go_path = tmp_path / "no-go/record.json"
    write_data(
        no_go_path,
        FlagshipNoGoRecord(
            campaign_sha256=stable_sha256(campaign),
            frontier=_existing_bound(tmp_path, failed_frontier_path),
            binding_constraints=["candidate failed deterministic integrity"],
            reviewer=campaign.roles.independent_reviewer,
            attestation=attestation,
        ),
    )

    closed = close_campaign_no_go(
        campaign_path=campaign_path,
        no_go_record_path=no_go_path,
        output_path=tmp_path / "campaign.closed.json",
    )

    assert closed.state is CampaignState.CLOSED_NO_GO
    assert closed.no_go_record is not None
    assert all(
        not dataset.consumed
        for dataset in closed.datasets
        if dataset.role
        in {
            CampaignDatasetRole.FORMAL_AGENT_CODING,
            CampaignDatasetRole.FORMAL_GENERAL,
        }
    )


def test_no_go_rejects_frontier_with_eligible_candidate(tmp_path: Path) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    assert campaign.candidate_inputs is not None
    campaign_path = tmp_path / "campaign.json"
    write_data(campaign_path, campaign)
    attestation = _bound(tmp_path, "no-go/attestation.json")
    no_go_path = tmp_path / "no-go/record.json"
    write_data(
        no_go_path,
        FlagshipNoGoRecord(
            campaign_sha256=stable_sha256(campaign),
            frontier=campaign.candidate_inputs.candidate_frontier,
            binding_constraints=["fixture constraint"],
            reviewer=campaign.roles.independent_reviewer,
            attestation=attestation,
        ),
    )

    with pytest.raises(ValidationGateError, match="formally eligible candidate"):
        close_campaign_no_go(
            campaign_path=campaign_path,
            no_go_record_path=no_go_path,
            output_path=tmp_path / "campaign.closed.json",
        )


def test_publication_record_requires_downloaded_audit_claim_runtime_and_bytes(
    tmp_path: Path,
) -> None:
    durable = tmp_path
    campaign = _campaign(tmp_path, durable)
    assert campaign.candidate is not None
    now = utc_now()
    formal_roles = {
        CampaignDatasetRole.FORMAL_AGENT_CODING,
        CampaignDatasetRole.FORMAL_GENERAL,
    }
    release_ready = FlagshipCampaign.model_validate(
        campaign.model_copy(
            update={
                "state": CampaignState.RELEASE_READY,
                "frozen_at": now,
                "formal_holdout_consumed_at": now,
                "formal_cycles_consumed": 1,
                "datasets": [
                    item.model_copy(update={"consumed": True})
                    if item.role in formal_roles
                    else item
                    for item in campaign.datasets
                ],
            }
        ).model_dump(mode="python")
    )
    campaign_path = tmp_path / "campaign.release-ready.json"
    write_data(campaign_path, release_ready)

    download = tmp_path / "download"
    download.mkdir()
    authorization = _bound(tmp_path, "download/authorization-audit.json")
    registry = load_model(
        tmp_path / campaign.lifecycle_registry.path,
        ArtifactLifecycleRegistry,
    )
    repository = "owner/AX-Qwen3.6-27B-MLX-AXQ-MP-5p20bpw-MTP"
    registry = transition_lifecycle(
        registry=registry,
        candidate=campaign.candidate,
        new_state=ArtifactLifecycleState.CERTIFIED,
        actor=campaign.roles.release_manager,
        reviewer=campaign.roles.independent_reviewer,
        reason=LifecycleReason.CERTIFICATION_PASSED,
        narrative="fixture final authorization passed",
        authorizing_evidence=authorization,
        public_repository=repository,
    )
    lifecycle_path = download / "lifecycle-registry.json"
    write_data(lifecycle_path, registry)

    metric_evidence = _bound(tmp_path, "download/metric.json")
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
            value=1.2,
            numerator=1.2,
            denominator=1,
            comparison="ratio",
        )
    ]
    claim = build_public_claim(
        candidate=campaign.candidate,
        lifecycle=registry,
        audit_sha256=authorization.sha256,
        public_owner="owner",
        base_model=campaign.source.model.model_id,
        target_class="4bit",
        measured_main_bpw=5.2,
        measured_total_bpw=5.3,
        weight_bytes=1_000,
        runtime_versions={"ax-engine": "1.0", "mlx": "1.0", "mlx-lm": "1.0"},
        quality_claims=quality_claims,
        performance_claims=performance_claims,
        limitations=["Exact fixture scope only."],
        evidence_index=[metric_evidence],
    )
    claim_path = download / "public-claim.json"
    write_data(claim_path, claim)
    assert load_model(claim_path, PublicClaimManifest).public_repository == repository

    checks = [
        ReleaseAuditCheck(
            gate_id=f"M{index}",  # type: ignore[arg-type]
            name=f"fixture M{index}",
            passed=True,
            evidence_sha256=({"authorization_audit": authorization.sha256} if index == 8 else {}),
            issues=[],
        )
        for index in range(9)
    ]
    candidate_model = campaign.source.model.model_copy(
        update={
            "model_id": repository,
            "revision": "f" * 40,
            "local_path": str(download),
        }
    )
    audit = FlagshipReleaseAudit(
        request_sha256="1" * 64,
        legacy_audit_sha256="2" * 64,
        campaign_sha256=stable_sha256(campaign),
        candidate=campaign.candidate,
        candidate_model=candidate_model,
        source_model=campaign.source.model,
        toolkit_version="1.0.0",
        wheel_sha256="3" * 64,
        checks=checks,
        authorization_ready=True,
        authorization_issues=[],
        release_ready=True,
        blockers=[],
    )
    audit_path = download / "release-audit.json"
    write_data(audit_path, audit)
    card_path = download / "README.md"
    card_path.write_text("# downloaded certified model\n", encoding="utf-8")
    checkpoint_path = download / "model.safetensors"
    checkpoint_path.write_text("downloaded checkpoint", encoding="utf-8")

    revision = "a" * 40
    runtime_path = download / "post-publication-runtime.json"
    write_data(
        runtime_path,
        PostPublicationRuntimeVerification(
            candidate_sha256=stable_sha256(campaign.candidate),
            public_repository=repository,
            public_revision=revision,
            runtime_versions={"ax-engine": "1.0", "mlx": "1.0", "mlx-lm": "1.0"},
            verifier=campaign.roles.release_manager,
        ),
    )
    inventory_paths = {
        "release-audit": audit_path,
        "public-claim": claim_path,
        "lifecycle-registry": lifecycle_path,
        "model-card": card_path,
        "checkpoint-model.safetensors": checkpoint_path,
    }
    inventory_path = download / "download-inventory.json"
    write_data(
        inventory_path,
        EvidenceArchiveIndex(
            records=[
                EvidenceArchiveRecord(
                    logical_name=name,
                    path=path.name,
                    sha256=file_sha256(path),
                    size_bytes=path.stat().st_size,
                    durable_uri=(
                        f"https://huggingface.co/{repository}/resolve/{revision}/{path.name}"
                    ),
                )
                for name, path in inventory_paths.items()
            ],
            complete=True,
        ),
    )
    attestation = _bound(tmp_path, "download/publication-attestation.json")
    verification_path = tmp_path / "publication-verification.json"
    write_data(
        verification_path,
        FlagshipPublicationVerification(
            campaign_sha256=stable_sha256(release_ready),
            candidate_sha256=stable_sha256(campaign.candidate),
            public_repository=repository,
            public_revision=revision,
            hub_url=f"https://huggingface.co/{repository}/commit/{revision}",
            release_audit=_existing_bound(tmp_path, audit_path),
            public_claim=_existing_bound(tmp_path, claim_path),
            lifecycle_registry=_existing_bound(tmp_path, lifecycle_path),
            download_inventory=_existing_bound(tmp_path, inventory_path),
            runtime_verification=_existing_bound(tmp_path, runtime_path),
            verifier=campaign.roles.release_manager,
            attestation=attestation,
        ),
    )

    published = record_campaign_publication(
        campaign_path=campaign_path,
        verification_path=verification_path,
        output_path=tmp_path / "campaign.published.json",
    )

    assert published.state is CampaignState.PUBLISHED
    assert published.publication_verification is not None

    checkpoint_path.write_text("tampered after download", encoding="utf-8")
    with pytest.raises(ValidationGateError, match=r"model\.safetensors"):
        record_campaign_publication(
            campaign_path=campaign_path,
            verification_path=verification_path,
            output_path=tmp_path / "campaign.published-tampered.json",
        )
