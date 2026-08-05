from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from datetime import timedelta
from pathlib import Path

from axquant.errors import ArtifactError, ValidationGateError
from axquant.identity import candidate_key_from_artifacts, semantic_plan_sha256
from axquant.schema import (
    ActivationCaptureSentinel,
    ArtifactLifecycleRegistry,
    ArtifactManifest,
    BoundFile,
    CampaignDatasetManifest,
    CampaignDatasetRole,
    CampaignOverlapReport,
    CampaignPreflight,
    CampaignState,
    EvidenceArchiveIndex,
    FlagshipCampaign,
    FlagshipFrontierIndex,
    FlagshipFrontierRequest,
    FlagshipNoGoRecord,
    FlagshipPublicationVerification,
    FlagshipReleaseAudit,
    FormalHoldoutCompletion,
    FormalHostEvidenceResult,
    FormalHostScopeEvidence,
    PostPublicationRuntimeVerification,
    PublicClaimManifest,
    QuantizationPlan,
    QuantMethod,
    SourceCheckpointManifest,
    utc_now,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

FLAGSHIP_SOURCE_MODEL_ID = "Qwen/Qwen3.6-27B"
FLAGSHIP_SOURCE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
_REQUIRED_DATASET_ROLES = set(CampaignDatasetRole)
_REQUIRED_BASELINES = {"bf16", "uniform-4bit", "uniform-6bit"}
_FORMAL_SCOPE_MAX_AGE = timedelta(hours=1)


def _resolved(root: Path, relative_path: str) -> Path:
    normalized_root = root.resolve()
    candidate = normalized_root / relative_path
    current = normalized_root
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise ArtifactError(f"bound file path contains a symlink: {relative_path}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(normalized_root)
    except ValueError as exc:
        raise ArtifactError(f"bound file escapes campaign root: {relative_path}") from exc
    return resolved


def _durable_campaign_path(
    campaign: FlagshipCampaign,
    value: str | Path,
    *,
    label: str,
    must_exist: bool = False,
) -> Path:
    durable_issues = _durable_root_issues(campaign.durable_evidence_root)
    if durable_issues:
        raise ValidationGateError("; ".join(durable_issues))
    durable_root = Path(campaign.durable_evidence_root).expanduser().resolve()
    raw = Path(value).expanduser()
    lexical = Path(os.path.abspath(raw))
    try:
        relative = lexical.relative_to(durable_root)
    except ValueError as exc:
        raise ArtifactError(f"{label} must be inside the durable evidence root") from exc
    current = durable_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ArtifactError(f"{label} path cannot contain a symbolic link")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(durable_root)
    except ValueError as exc:
        raise ArtifactError(f"{label} escapes the durable evidence root") from exc
    if must_exist and not resolved.is_file():
        raise ArtifactError(f"{label} must be an existing file")
    return resolved


def campaign_bound_files(campaign: FlagshipCampaign) -> list[BoundFile]:
    files = [
        campaign.policy_file,
        campaign.toolkit_wheel,
        *campaign.runtime_builds.values(),
        campaign.hardware_scope,
        *(binding.file for binding in campaign.hardware_scope_evidence),
        campaign.lifecycle_registry,
        campaign.backup_verification,
    ]
    for dataset in campaign.datasets:
        files.extend(
            (
                dataset.manifest,
                dataset.manifest_attestation,
                dataset.overlap_report,
            )
        )
    for baseline in campaign.baselines:
        files.extend((baseline.artifact_manifest, *baseline.checkpoint_files))
    if campaign.candidate_inputs is not None:
        files.extend(
            (
                campaign.candidate_inputs.source_checkpoint_manifest,
                campaign.candidate_inputs.calibration_manifest,
                campaign.candidate_inputs.activation_capture_or_sentinel,
                campaign.candidate_inputs.sensitivity_report,
                campaign.candidate_inputs.plan,
                campaign.candidate_inputs.artifact_manifest,
                campaign.candidate_inputs.candidate_frontier,
                *campaign.candidate_inputs.frontier_evidence,
            )
        )
    if campaign.no_go_record is not None:
        files.append(campaign.no_go_record)
    if campaign.publication_verification is not None:
        files.append(campaign.publication_verification)
    unique: dict[tuple[str, str, int], BoundFile] = {}
    for bound in files:
        unique[(bound.path, bound.sha256, bound.size_bytes)] = bound
    return sorted(unique.values(), key=lambda item: item.path)


def _file_issues(root: Path, bound: BoundFile) -> list[str]:
    try:
        path = _resolved(root, bound.path)
    except ArtifactError as exc:
        return [str(exc)]
    if not path.is_file():
        return [f"bound file is missing: {bound.path}"]
    issues: list[str] = []
    if path.stat().st_size != bound.size_bytes:
        issues.append(f"bound file size changed: {bound.path}")
    if file_sha256(path) != bound.sha256:
        issues.append(f"bound file checksum changed: {bound.path}")
    return issues


def formal_completion_evidence_issues(
    completion_path: Path,
    completion: FormalHoldoutCompletion,
) -> list[str]:
    root = completion_path.parent
    issues = [
        issue
        for bound in (
            *completion.result_file_by_profile.values(),
            completion.raw_evidence_index,
            completion.custodian_attestation,
        )
        for issue in _file_issues(root, bound)
    ]
    raw_index_path = _resolved(root, completion.raw_evidence_index.path)
    if not raw_index_path.is_file():
        return issues
    try:
        raw_index = load_model(raw_index_path, EvidenceArchiveIndex)
    except (ArtifactError, OSError, ValueError) as exc:
        issues.append(f"formal raw evidence index is invalid: {exc}")
        return issues
    if not raw_index.complete:
        issues.append("formal raw evidence index is incomplete")
    for record in raw_index.records:
        issues.extend(
            _file_issues(
                raw_index_path.parent,
                BoundFile(
                    path=record.path,
                    sha256=record.sha256,
                    size_bytes=record.size_bytes,
                ),
            )
        )
    return issues


def build_flagship_frontier(
    *,
    request_path: str | Path,
    output_path: str | Path,
) -> FlagshipFrontierIndex:
    request_source = Path(request_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if request_source.parent != output.parent:
        raise ArtifactError("frontier output must share the request evidence root")
    request = load_model(request_source, FlagshipFrontierRequest)
    issues = [
        issue
        for entry in request.entries
        for result in entry.gates
        if result.evidence is not None
        for issue in _file_issues(request_source.parent, result.evidence)
    ]
    if issues:
        raise ValidationGateError("; ".join(issues))
    frontier = FlagshipFrontierIndex(
        source=request.source,
        policy_sha256=request.policy_sha256,
        search_budget=request.search_budget,
        search_used=request.search_used,
        formal_holdout_accessed=request.formal_holdout_accessed,
        entries=request.entries,
        feasible_candidate_sha256=sorted(
            stable_sha256(entry.candidate) for entry in request.entries if entry.eligible_for_formal
        ),
    )
    write_data(output, frontier)
    return frontier


def _durable_root_issues(value: str) -> list[str]:
    root = Path(value).expanduser()
    if not root.is_absolute():
        return ["durable evidence root must be an absolute path"]
    if root.is_symlink():
        return ["durable evidence root cannot be a symbolic link"]
    resolved = root.resolve()
    normalized = resolved.as_posix()
    if "/.internal/tmp" in normalized or normalized.endswith("/.internal"):
        return ["durable evidence root cannot be .internal/tmp or .internal"]
    if not resolved.is_dir():
        return ["durable evidence root does not exist"]
    if not os.access(resolved, os.W_OK):
        return ["durable evidence root is not writable"]
    return []


def _live_formal_host_issues(campaign: FlagshipCampaign) -> list[str]:
    issues: list[str] = []
    if platform.system() != "Darwin":
        issues.append("formal preflight requires macOS")
    if platform.machine() != "arm64":
        issues.append("formal preflight requires Apple Silicon arm64")
    try:
        version = subprocess.run(
            ["sw_vers", "-productVersion"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        build = subprocess.run(
            ["sw_vers", "-buildVersion"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        issues.append(f"cannot observe formal host macOS version: {exc}")
    else:
        observed_os = f"macOS {version} ({build})"
        if observed_os != campaign.formal_host.os_version:
            issues.append(
                f"formal host OS differs: frozen={campaign.formal_host.os_version!r}, "
                f"observed={observed_os!r}"
            )
    try:
        free_disk = shutil.disk_usage(Path(campaign.durable_evidence_root).resolve()).free
    except OSError as exc:
        issues.append(f"cannot observe durable-root free disk: {exc}")
    else:
        if free_disk < campaign.required_free_disk_bytes:
            issues.append("live durable-root free disk is below the frozen campaign requirement")
    return issues


def formal_host_scope_evidence_issues(
    root: Path,
    scope: FormalHostScopeEvidence,
) -> list[str]:
    issues: list[str] = []
    expected_subject_sha256 = stable_sha256(scope.contract)
    for binding in scope.evidence:
        issues.extend(_file_issues(root, binding.file))
        evidence_path = _resolved(root, binding.file.path)
        if not evidence_path.is_file():
            continue
        try:
            evidence = load_model(evidence_path, FormalHostEvidenceResult)
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"formal host evidence is invalid for {binding.name}: {exc}")
            continue
        if (
            evidence.evidence_name != binding.name
            or evidence.kind is not binding.kind
            or evidence.host_id != scope.contract.host_id
        ):
            issues.append(f"formal host evidence binding differs for {binding.name}")
        if evidence.subject_sha256 != expected_subject_sha256:
            issues.append(f"formal host evidence binds another host contract: {binding.name}")
        if evidence.completed_at > scope.observed_at:
            issues.append(f"formal host evidence postdates its scope capture: {binding.name}")
        elif scope.observed_at - evidence.completed_at > _FORMAL_SCOPE_MAX_AGE:
            issues.append(f"formal host evidence is stale: {binding.name}")
    return issues


def _freeze_issues(campaign: FlagshipCampaign, root: Path) -> list[str]:
    issues: list[str] = []
    durable_root = Path(campaign.durable_evidence_root).expanduser().resolve()
    try:
        root.resolve().relative_to(durable_root)
    except ValueError:
        issues.append("campaign evidence root must be inside the durable evidence root")
    source = campaign.source.model
    if source.model_id != FLAGSHIP_SOURCE_MODEL_ID or source.revision != FLAGSHIP_SOURCE_REVISION:
        issues.append("campaign source is not the accepted Qwen 3.6 27B immutable revision")
    if campaign.candidate is None:
        issues.append("campaign freeze requires exactly one candidate")
    elif campaign.candidate.source != campaign.source:
        issues.append("campaign candidate source key differs from the campaign source")
    if campaign.candidate_inputs is None:
        issues.append("campaign freeze requires candidate input bindings")
    elif campaign.candidate is not None:
        bindings = campaign.candidate_inputs
        try:
            source_manifest = load_model(
                _resolved(root, bindings.source_checkpoint_manifest.path),
                SourceCheckpointManifest,
            )
            plan = load_model(
                _resolved(root, bindings.plan.path),
                QuantizationPlan,
            )
            artifact = load_model(
                _resolved(root, bindings.artifact_manifest.path),
                ArtifactManifest,
            )
            frontier = load_model(
                _resolved(root, bindings.candidate_frontier.path),
                FlagshipFrontierIndex,
            )
            if plan.target_class != campaign.target_class:
                issues.append("campaign plan target class differs from frozen target class")
            if artifact.target_class != campaign.target_class:
                issues.append("campaign artifact target class differs from frozen target class")
            if (
                frontier.source != campaign.source
                or frontier.policy_sha256 != campaign.policy_file.sha256
            ):
                issues.append("campaign candidate frontier differs from frozen source or policy")
            candidate_digest = stable_sha256(campaign.candidate)
            if candidate_digest not in frontier.feasible_candidate_sha256:
                issues.append("campaign candidate is not formally eligible in its frontier")
            frontier_bindings = {
                (result.evidence.path, result.evidence.sha256, result.evidence.size_bytes)
                for entry in frontier.entries
                for result in entry.gates
                if result.evidence is not None
            }
            declared_frontier_bindings = {
                (bound.path, bound.sha256, bound.size_bytes) for bound in bindings.frontier_evidence
            }
            if frontier_bindings != declared_frontier_bindings:
                issues.append("campaign frontier evidence bindings are incomplete or extraneous")
            activation_path = _resolved(
                root,
                bindings.activation_capture_or_sentinel.path,
            )
            uses_capture = any(
                assignment.method in {QuantMethod.AWQ, QuantMethod.GPTQ}
                for assignment in plan.assignments
            )
            if uses_capture:
                from axquant.capture import load_capture_activations

                load_capture_activations(
                    activation_path.parent,
                    model=plan.source_model.model_id,
                    revision=plan.source_model.revision,
                )
            else:
                sentinel = load_model(activation_path, ActivationCaptureSentinel)
                if sentinel.plan_sha256 != semantic_plan_sha256(plan):
                    issues.append("campaign activation sentinel binds another semantic plan")
            recomputed_candidate = candidate_key_from_artifacts(
                source_manifest=source_manifest,
                certification_policy_sha256=campaign.policy_file.sha256,
                calibration_sha256=bindings.calibration_manifest.sha256,
                activation_capture_sha256=bindings.activation_capture_or_sentinel.sha256,
                sensitivity_sha256=bindings.sensitivity_report.sha256,
                plan=plan,
                artifact_manifest=artifact,
            )
            if recomputed_candidate != campaign.candidate:
                issues.append("campaign candidate key cannot be recomputed from bound inputs")
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"campaign candidate inputs are invalid: {exc}")
    roles = {dataset.role for dataset in campaign.datasets}
    dataset_ids = [dataset.dataset_id for dataset in campaign.datasets]
    dataset_digests = [dataset.content_sha256 for dataset in campaign.datasets]
    if len(dataset_ids) != len(set(dataset_ids)):
        issues.append("campaign dataset IDs must be unique")
    if len(dataset_digests) != len(set(dataset_digests)):
        issues.append("campaign dataset content cannot be reused across roles")
    if roles != _REQUIRED_DATASET_ROLES:
        missing = sorted(role.value for role in _REQUIRED_DATASET_ROLES - roles)
        extra = sorted(role.value for role in roles - _REQUIRED_DATASET_ROLES)
        issues.append(f"campaign dataset roles differ: missing={missing}, extra={extra}")
    expected_comparisons = {dataset.content_sha256 for dataset in campaign.datasets}
    for dataset in campaign.datasets:
        if not dataset.sealed:
            issues.append(f"campaign dataset is not sealed: {dataset.dataset_id}")
        if not dataset.overlap_passed:
            issues.append(f"campaign overlap report did not pass: {dataset.dataset_id}")
        manifest: CampaignDatasetManifest | None = None
        try:
            manifest_path = _resolved(root, dataset.manifest.path)
            manifest = load_model(manifest_path, CampaignDatasetManifest)
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"campaign dataset manifest is invalid for {dataset.dataset_id}: {exc}")
        else:
            if (
                manifest.dataset_id != dataset.dataset_id
                or manifest.role is not dataset.role
                or manifest.content_sha256 != dataset.content_sha256
            ):
                issues.append(f"campaign dataset manifest bindings differ for {dataset.dataset_id}")
            if (
                dataset.role
                in {
                    CampaignDatasetRole.FORMAL_AGENT_CODING,
                    CampaignDatasetRole.FORMAL_GENERAL,
                }
                and manifest.sealed_by != campaign.roles.evaluation_custodian
            ):
                issues.append(
                    "formal dataset was not sealed by the evaluation custodian: "
                    f"{dataset.dataset_id}"
                )
        try:
            overlap_path = _resolved(root, dataset.overlap_report.path)
            overlap = load_model(overlap_path, CampaignOverlapReport)
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"campaign overlap report is invalid for {dataset.dataset_id}: {exc}")
        else:
            if (
                overlap.dataset_sha256 != dataset.content_sha256
                or set(overlap.compared_dataset_sha256)
                != expected_comparisons - {dataset.content_sha256}
                or not overlap.passed
                or dataset.overlap_passed != overlap.passed
            ):
                issues.append(f"campaign overlap report bindings differ for {dataset.dataset_id}")
            if manifest is not None and overlap.dataset_record_count != manifest.record_count:
                issues.append(
                    f"campaign overlap record count differs from manifest for {dataset.dataset_id}"
                )
        if (
            dataset.role
            in {
                CampaignDatasetRole.FORMAL_AGENT_CODING,
                CampaignDatasetRole.FORMAL_GENERAL,
            }
            and dataset.consumed
        ):
            issues.append(f"formal holdout was already consumed: {dataset.dataset_id}")
    baseline_kinds = {baseline.kind for baseline in campaign.baselines}
    if len(campaign.baselines) != len(baseline_kinds) or baseline_kinds != _REQUIRED_BASELINES:
        issues.append("campaign requires exactly one BF16, uniform-4bit, and uniform-6bit baseline")
    for baseline in campaign.baselines:
        if baseline.source != campaign.source:
            issues.append(f"{baseline.kind} baseline source key differs from campaign source")
        if not baseline.available:
            issues.append(f"{baseline.kind} baseline is unavailable")
    baseline_member_digests = {
        baseline.checkpoint_members_sha256 for baseline in campaign.baselines
    }
    if len(baseline_member_digests) != len(campaign.baselines):
        issues.append("campaign baselines do not identify three byte-distinct checkpoints")
    if (
        campaign.candidate is not None
        and campaign.candidate.checkpoint_members_sha256 in baseline_member_digests
    ):
        issues.append("campaign candidate reuses a baseline checkpoint-member digest")
    if campaign.formal_host.operator != campaign.roles.runtime_owner:
        issues.append("formal host operator differs from the frozen runtime owner")
    if campaign.created_by != campaign.roles.certification_owner:
        issues.append("campaign creator differs from the frozen certification owner")
    issues.extend(_durable_root_issues(campaign.durable_evidence_root))
    try:
        lifecycle_path = _resolved(root, campaign.lifecycle_registry.path)
        from axquant.lifecycle import candidate_lifecycle_state
        from axquant.schema import ArtifactLifecycleRegistry, ArtifactLifecycleState

        lifecycle = load_model(lifecycle_path, ArtifactLifecycleRegistry)
        if (
            campaign.candidate is None
            or candidate_lifecycle_state(lifecycle, campaign.candidate)
            is not ArtifactLifecycleState.FROZEN
        ):
            issues.append("campaign candidate lifecycle must be frozen before campaign freeze")
    except (ArtifactError, OSError, ValueError) as exc:
        issues.append(f"campaign lifecycle registry is invalid: {exc}")
    for bound in campaign_bound_files(campaign):
        issues.extend(_file_issues(root, bound))
    return issues


def freeze_campaign(
    *,
    request_path: str | Path,
    output_path: str | Path,
) -> FlagshipCampaign:
    request_source = Path(request_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if request_source.parent != output.parent:
        raise ArtifactError("campaign freeze output must share the request evidence directory")
    campaign = load_model(request_source, FlagshipCampaign)
    if campaign.state is not CampaignState.DRAFT:
        raise ValidationGateError("campaign-freeze accepts only a draft campaign")
    if campaign.frozen_at is not None:
        raise ValidationGateError("draft campaign unexpectedly carries a freeze timestamp")
    issues = _freeze_issues(campaign, request_source.parent)
    if issues:
        raise ValidationGateError("; ".join(issues))
    from axquant.schema import utc_now

    frozen = campaign.model_copy(
        update={
            "state": CampaignState.FROZEN,
            "frozen_at": utc_now(),
        }
    )
    write_data(output, frozen)
    return frozen


def preflight_campaign(
    *,
    campaign_path: str | Path,
    output_path: str | Path | None = None,
    observed_host_id: str | None = None,
) -> CampaignPreflight:
    source = Path(campaign_path).expanduser().resolve()
    campaign = load_model(source, FlagshipCampaign)
    if output_path is not None:
        _durable_campaign_path(campaign, output_path, label="campaign preflight output")
    issues: list[str] = []
    if campaign.state is not CampaignState.FROZEN:
        issues.append("campaign preflight requires frozen state")
    observed = observed_host_id or socket.gethostname().split(".", 1)[0]
    if observed != campaign.formal_host.host_id:
        issues.append(
            f"campaign preflight must run on {campaign.formal_host.host_id}; observed {observed}"
        )
    issues.extend(_freeze_issues(campaign, source.parent))
    try:
        scope_path = _resolved(source.parent, campaign.hardware_scope.path)
        scope = load_model(scope_path, FormalHostScopeEvidence)
        if scope.contract != campaign.formal_host:
            issues.append("formal host scope differs from the frozen host contract")
        if scope.evidence != campaign.hardware_scope_evidence:
            issues.append("formal host evidence differs from the frozen campaign bindings")
        issues.extend(formal_host_scope_evidence_issues(source.parent, scope))
        if scope.free_disk_bytes < campaign.required_free_disk_bytes:
            issues.append("formal host free disk is below the frozen campaign requirement")
        now = utc_now()
        if scope.observed_at > now + timedelta(minutes=5):
            issues.append("formal host scope timestamp is implausibly in the future")
        elif now - scope.observed_at > _FORMAL_SCOPE_MAX_AGE:
            issues.append("formal host scope evidence is stale")
    except (ArtifactError, OSError, ValueError) as exc:
        issues.append(f"formal host scope is invalid: {exc}")
    if observed_host_id is None:
        issues.extend(_live_formal_host_issues(campaign))
    if campaign.formal_holdout_consumed_at is not None:
        issues.append("campaign formal holdout is already consumed")
    result = CampaignPreflight(
        campaign_id=campaign.campaign_id,
        campaign_sha256=stable_sha256(campaign),
        host_id=campaign.formal_host.host_id,
        passed=not issues,
        issues=issues,
        verified_files=campaign_bound_files(campaign),
    )
    if output_path is not None:
        write_data(output_path, result)
    return result


def start_formal_campaign(
    *,
    campaign: FlagshipCampaign,
    preflight: CampaignPreflight,
    output_path: str | Path | None = None,
) -> FlagshipCampaign:
    if campaign.state is not CampaignState.FROZEN:
        raise ValidationGateError("formal cycle can start only from frozen campaign state")
    if not preflight.passed or preflight.campaign_sha256 != stable_sha256(campaign):
        raise ValidationGateError("formal cycle requires a passing preflight for this campaign")
    if campaign.formal_cycles_consumed >= campaign.formal_cycle_limit:
        raise ValidationGateError("campaign formal-cycle budget is exhausted")
    if output_path is not None:
        _durable_campaign_path(campaign, output_path, label="formal campaign output")
    running = campaign.model_copy(update={"state": CampaignState.FORMAL_RUNNING})
    if output_path is not None:
        write_data(output_path, running)
    return running


def complete_formal_campaign(
    *,
    campaign: FlagshipCampaign,
    completion_path: str | Path,
    output_path: str | Path | None = None,
) -> FlagshipCampaign:
    completion_source = _durable_campaign_path(
        campaign,
        completion_path,
        label="formal completion",
        must_exist=True,
    )
    if output_path is not None:
        _durable_campaign_path(campaign, output_path, label="formal completion output")
    completion = load_model(completion_source, FormalHoldoutCompletion)
    if campaign.state is not CampaignState.FORMAL_RUNNING:
        raise ValidationGateError("formal completion requires formal_running campaign state")
    if completion.campaign_sha256 != stable_sha256(
        campaign.model_copy(update={"state": CampaignState.FROZEN})
    ):
        raise ValidationGateError("formal completion binds another frozen campaign")
    if campaign.candidate is None or completion.candidate_sha256 != stable_sha256(
        campaign.candidate
    ):
        raise ValidationGateError("formal completion binds another candidate")
    expected_datasets = {
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
    if completion.dataset_sha256_by_profile != expected_datasets:
        raise ValidationGateError("formal completion datasets differ from frozen holdouts")
    if completion.evaluation_custodian != campaign.roles.evaluation_custodian:
        raise ValidationGateError("formal completion signer is not the evaluation custodian")
    result_issues = formal_completion_evidence_issues(completion_source, completion)
    if result_issues:
        raise ValidationGateError("; ".join(result_issues))
    formal_roles = {
        CampaignDatasetRole.FORMAL_AGENT_CODING,
        CampaignDatasetRole.FORMAL_GENERAL,
    }
    consumed_datasets = [
        dataset.model_copy(update={"consumed": True}) if dataset.role in formal_roles else dataset
        for dataset in campaign.datasets
    ]
    completed = campaign.model_copy(
        update={
            "state": (
                CampaignState.RELEASE_READY
                if completion.verdict == "pass"
                else CampaignState.FORMAL_FAILED
            ),
            "datasets": consumed_datasets,
            "formal_holdout_consumed_at": completion.completed_at,
            "formal_cycles_consumed": campaign.formal_cycles_consumed + 1,
        }
    )
    completed = FlagshipCampaign.model_validate(completed.model_dump(mode="python"))
    if output_path is not None:
        write_data(output_path, completed)
    return completed


def close_campaign_no_go(
    *,
    campaign_path: str | Path,
    no_go_record_path: str | Path,
    output_path: str | Path,
) -> FlagshipCampaign:
    campaign_source = Path(campaign_path).expanduser().resolve()
    record_source = Path(no_go_record_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if campaign_source.parent != output.parent:
        raise ArtifactError("no-go output must share the campaign evidence root")
    try:
        record_relative = record_source.relative_to(output.parent).as_posix()
    except ValueError as exc:
        raise ArtifactError("no-go record must be inside the campaign evidence root") from exc
    if record_source.is_symlink() or not record_source.is_file():
        raise ArtifactError("no-go record must be an existing non-symlink file")
    campaign = load_model(campaign_source, FlagshipCampaign)
    _durable_campaign_path(
        campaign,
        campaign_source,
        label="no-go campaign",
        must_exist=True,
    )
    _durable_campaign_path(
        campaign,
        record_source,
        label="no-go record",
        must_exist=True,
    )
    _durable_campaign_path(campaign, output, label="no-go output")
    record = load_model(record_source, FlagshipNoGoRecord)
    if campaign.state not in {CampaignState.DRAFT, CampaignState.FROZEN}:
        raise ValidationGateError("no-go closure is only valid before a formal cycle starts")
    if campaign.formal_holdout_consumed_at is not None:
        raise ValidationGateError("no-go closure cannot discard a consumed formal cycle")
    if any(
        dataset.consumed
        for dataset in campaign.datasets
        if dataset.role
        in {
            CampaignDatasetRole.FORMAL_AGENT_CODING,
            CampaignDatasetRole.FORMAL_GENERAL,
        }
    ):
        raise ValidationGateError("no-go closure requires untouched formal holdouts")
    issues: list[str] = []
    if record.campaign_sha256 != stable_sha256(campaign):
        issues.append("no-go record binds another campaign")
    if record.reviewer != campaign.roles.independent_reviewer:
        issues.append("no-go record signer differs from the frozen independent reviewer")
    issues.extend(_file_issues(output.parent, record.frontier))
    issues.extend(_file_issues(output.parent, record.attestation))
    frontier_path = _resolved(output.parent, record.frontier.path)
    if frontier_path.is_file():
        frontier = load_model(frontier_path, FlagshipFrontierIndex)
        if (
            frontier.source != campaign.source
            or frontier.policy_sha256 != campaign.policy_file.sha256
        ):
            issues.append("no-go frontier differs from campaign source or policy")
        if frontier.feasible_candidate_sha256:
            issues.append("no-go frontier still contains a formally eligible candidate")
        if frontier.search_used != frontier.search_budget:
            issues.append("no-go frontier did not exhaust its declared search budget")
    if issues:
        raise ValidationGateError("; ".join(issues))
    bound_record = BoundFile(
        path=record_relative,
        sha256=file_sha256(record_source),
        size_bytes=record_source.stat().st_size,
    )
    closed = campaign.model_copy(
        update={
            "state": CampaignState.CLOSED_NO_GO,
            "no_go_record": bound_record,
        }
    )
    closed = FlagshipCampaign.model_validate(closed.model_dump(mode="python"))
    write_data(output, closed)
    return closed


def record_campaign_publication(
    *,
    campaign_path: str | Path,
    verification_path: str | Path,
    output_path: str | Path,
) -> FlagshipCampaign:
    campaign_source = Path(campaign_path).expanduser().resolve()
    verification_source = Path(verification_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if campaign_source.parent != output.parent:
        raise ArtifactError("publication output must share the campaign evidence root")
    try:
        verification_relative = verification_source.relative_to(output.parent).as_posix()
    except ValueError as exc:
        raise ArtifactError(
            "publication verification must be inside the campaign evidence root"
        ) from exc
    if verification_source.is_symlink() or not verification_source.is_file():
        raise ArtifactError("publication verification must be an existing non-symlink file")

    campaign = load_model(campaign_source, FlagshipCampaign)
    _durable_campaign_path(
        campaign,
        campaign_source,
        label="publication campaign",
        must_exist=True,
    )
    _durable_campaign_path(
        campaign,
        verification_source,
        label="publication verification",
        must_exist=True,
    )
    _durable_campaign_path(campaign, output, label="publication output")
    verification = load_model(
        verification_source,
        FlagshipPublicationVerification,
    )
    if campaign.state is not CampaignState.RELEASE_READY:
        raise ValidationGateError(
            "publication verification requires a release_ready formal campaign"
        )
    if campaign.candidate is None:
        raise ValidationGateError("publication verification requires a frozen candidate")
    issues: list[str] = []
    candidate_sha = stable_sha256(campaign.candidate)
    if (
        verification.campaign_sha256 != stable_sha256(campaign)
        or verification.candidate_sha256 != candidate_sha
    ):
        issues.append("publication verification binds another campaign or candidate")
    if verification.verifier != campaign.roles.release_manager:
        issues.append("publication verifier differs from the frozen release manager")
    expected_hub_url = (
        f"https://huggingface.co/{verification.public_repository}/commit/"
        f"{verification.public_revision}"
    )
    if verification.hub_url != expected_hub_url:
        issues.append("publication verification Hub URL differs from repository/revision")
    bound_inputs = (
        verification.release_audit,
        verification.public_claim,
        verification.lifecycle_registry,
        verification.download_inventory,
        verification.runtime_verification,
        verification.attestation,
    )
    for bound in bound_inputs:
        issues.extend(_file_issues(verification_source.parent, bound))

    audit_path = _resolved(verification_source.parent, verification.release_audit.path)
    claim_path = _resolved(verification_source.parent, verification.public_claim.path)
    lifecycle_path = _resolved(
        verification_source.parent,
        verification.lifecycle_registry.path,
    )
    inventory_path = _resolved(
        verification_source.parent,
        verification.download_inventory.path,
    )
    runtime_path = _resolved(
        verification_source.parent,
        verification.runtime_verification.path,
    )
    if all(
        path.is_file()
        for path in (audit_path, claim_path, lifecycle_path, inventory_path, runtime_path)
    ):
        audit = load_model(audit_path, FlagshipReleaseAudit)
        claim = load_model(claim_path, PublicClaimManifest)
        lifecycle = load_model(lifecycle_path, ArtifactLifecycleRegistry)
        inventory = load_model(inventory_path, EvidenceArchiveIndex)
        runtime = load_model(runtime_path, PostPublicationRuntimeVerification)
        if not audit.release_ready or audit.candidate != campaign.candidate:
            issues.append("downloaded flagship release audit is not ready for this candidate")
        if claim.candidate != campaign.candidate:
            issues.append("downloaded public claim binds another candidate")
        m8 = next((check for check in audit.checks if check.gate_id == "M8"), None)
        if m8 is None or m8.evidence_sha256.get("authorization_audit") != claim.audit_sha256:
            issues.append("downloaded public claim does not match final audit authorization")
        from axquant.lifecycle import require_active_certification

        try:
            active_event = require_active_certification(lifecycle, campaign.candidate)
        except ValidationGateError as exc:
            issues.append(str(exc))
        else:
            if (
                active_event.public_repository != verification.public_repository
                or claim.public_repository != verification.public_repository
            ):
                issues.append("publication repository differs across lifecycle and claim")
            if claim.lifecycle_event_sha256 != stable_sha256(active_event):
                issues.append("downloaded claim does not bind the active lifecycle event")
        if (
            runtime.candidate_sha256 != candidate_sha
            or runtime.public_repository != verification.public_repository
            or runtime.public_revision != verification.public_revision
            or runtime.verifier != verification.verifier
        ):
            issues.append("post-publication runtime verification binds another release")
        if not inventory.complete:
            issues.append("downloaded publication inventory is incomplete")
        inventory_names = {record.logical_name for record in inventory.records}
        required_names = {
            "lifecycle-registry",
            "model-card",
            "public-claim",
            "release-audit",
        }
        if missing := sorted(required_names - inventory_names):
            issues.append(f"downloaded publication inventory omits required records: {missing}")
        if not any(name.startswith("checkpoint-") for name in inventory_names):
            issues.append("downloaded publication inventory contains no checkpoint files")
        inventory_bindings = {
            (record.logical_name, record.sha256, record.size_bytes) for record in inventory.records
        }
        expected_named_bindings = {
            (
                "release-audit",
                verification.release_audit.sha256,
                verification.release_audit.size_bytes,
            ),
            (
                "public-claim",
                verification.public_claim.sha256,
                verification.public_claim.size_bytes,
            ),
            (
                "lifecycle-registry",
                verification.lifecycle_registry.sha256,
                verification.lifecycle_registry.size_bytes,
            ),
        }
        if not expected_named_bindings.issubset(inventory_bindings):
            issues.append("downloaded inventory does not bind audit, claim, and lifecycle bytes")
        for record in inventory.records:
            issues.extend(
                _file_issues(
                    inventory_path.parent,
                    BoundFile(
                        path=record.path,
                        sha256=record.sha256,
                        size_bytes=record.size_bytes,
                    ),
                )
            )
    if issues:
        raise ValidationGateError("; ".join(issues))

    bound_verification = BoundFile(
        path=verification_relative,
        sha256=file_sha256(verification_source),
        size_bytes=verification_source.stat().st_size,
    )
    published = campaign.model_copy(
        update={
            "state": CampaignState.PUBLISHED,
            "publication_verification": bound_verification,
        }
    )
    published = FlagshipCampaign.model_validate(published.model_dump(mode="python"))
    write_data(output, published)
    return published
