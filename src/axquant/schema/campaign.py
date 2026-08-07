from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.flagship import CandidateKey, CheckpointKey
from axquant.serde import stable_sha256

_SHA256 = r"^[0-9a-f]{64}$"
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not value
        or value != normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("bound file paths must be safe relative paths")
    return value


class BoundFile(StrictModel):
    path: str
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class CampaignState(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"
    FORMAL_RUNNING = "formal_running"
    FORMAL_FAILED = "formal_failed"
    RELEASE_READY = "release_ready"
    PUBLISHED = "published"
    CLOSED_NO_GO = "closed_no_go"


class CampaignDatasetRole(StrEnum):
    CALIBRATION = "calibration"
    DEVELOPMENT_AGENT_CODING = "development-agent-coding"
    DEVELOPMENT_GENERAL = "development-general"
    FORMAL_AGENT_CODING = "formal-agent-coding"
    FORMAL_GENERAL = "formal-general"
    REPRODUCTION_PARITY = "reproduction-parity"


class CampaignDataset(StrictModel):
    dataset_id: str = Field(pattern=_IDENTIFIER)
    role: CampaignDatasetRole
    content_sha256: str = Field(pattern=_SHA256)
    manifest: BoundFile
    manifest_attestation: BoundFile
    overlap_report: BoundFile
    overlap_passed: bool
    sealed: bool
    consumed: bool = False


class CampaignDatasetManifest(StrictModel):
    schema_version: Literal["axquant.campaign-dataset-manifest.v1"] = (
        "axquant.campaign-dataset-manifest.v1"
    )
    dataset_id: str = Field(pattern=_IDENTIFIER)
    role: CampaignDatasetRole
    content_sha256: str = Field(pattern=_SHA256)
    record_count: int = Field(gt=0)
    provenance: list[str] = Field(min_length=1)
    composition: dict[str, int]
    scorer_versions: dict[str, str]
    raw_output_retention_policy: str = Field(min_length=1)
    sealed_by: str = Field(min_length=1)
    sealed_at: datetime = Field(default_factory=utc_now)

    @field_validator("composition")
    @classmethod
    def positive_composition(cls, value: dict[str, int]) -> dict[str, int]:
        if not value or any(not key.strip() or count <= 0 for key, count in value.items()):
            raise ValueError("dataset composition requires positive named counts")
        return value

    @field_validator("scorer_versions")
    @classmethod
    def complete_scorer_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(
            not key.strip() or not version.strip() for key, version in value.items()
        ):
            raise ValueError("dataset scorer versions must be complete")
        return value

    @model_validator(mode="after")
    def composition_matches_records(self) -> CampaignDatasetManifest:
        if sum(self.composition.values()) != self.record_count:
            raise ValueError("dataset composition count differs from record count")
        if any(not item.strip() for item in self.provenance):
            raise ValueError("dataset provenance entries must be non-empty")
        return self


class CampaignOverlapMatch(StrictModel):
    dataset_record_sha256: str = Field(pattern=_SHA256)
    compared_dataset_sha256: str = Field(pattern=_SHA256)
    compared_record_sha256: str = Field(pattern=_SHA256)
    similarity: float = Field(ge=0, le=1)
    exact: bool

    @model_validator(mode="after")
    def exact_similarity_is_one(self) -> CampaignOverlapMatch:
        if self.exact and self.similarity != 1:
            raise ValueError("exact overlap match must have similarity exactly one")
        return self


class CampaignOverlapReport(StrictModel):
    schema_version: Literal["axquant.campaign-overlap-report.v1"] = (
        "axquant.campaign-overlap-report.v1"
    )
    normalization_algorithm: Literal["axquant-token-5gram-v2"] = "axquant-token-5gram-v2"
    dataset_sha256: str = Field(pattern=_SHA256)
    compared_dataset_sha256: list[str]
    dataset_record_count: int = Field(gt=0)
    compared_record_count_by_sha256: dict[str, int]
    comparison_pair_count: int = Field(gt=0)
    exact_match_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    matches: list[CampaignOverlapMatch] = Field(default_factory=list)
    passed: bool

    @model_validator(mode="after")
    def status_and_comparisons_are_consistent(self) -> CampaignOverlapReport:
        if len(self.compared_dataset_sha256) != len(set(self.compared_dataset_sha256)):
            raise ValueError("overlap report comparison digests must be unique")
        if set(self.compared_dataset_sha256) != set(self.compared_record_count_by_sha256) or any(
            count <= 0 for count in self.compared_record_count_by_sha256.values()
        ):
            raise ValueError("overlap report comparison record counts are incomplete")
        expected_pairs = self.dataset_record_count * sum(
            self.compared_record_count_by_sha256.values()
        )
        if self.comparison_pair_count != expected_pairs:
            raise ValueError("overlap report comparison-pair count is inconsistent")
        exact_count = sum(match.exact for match in self.matches)
        near_count = sum(not match.exact for match in self.matches)
        if self.exact_match_count != exact_count or self.near_duplicate_count != near_count:
            raise ValueError("overlap report match counts differ from match records")
        expected = self.exact_match_count == 0 and self.near_duplicate_count == 0
        if self.passed != expected:
            raise ValueError("overlap report pass state is inconsistent with match counts")
        return self


class CampaignBaseline(StrictModel):
    kind: Literal["bf16", "uniform-4bit", "uniform-6bit"]
    source: CheckpointKey
    artifact_manifest: BoundFile
    checkpoint_files: list[BoundFile] = Field(min_length=1)
    checkpoint_members_sha256: str = Field(pattern=_SHA256)
    runtime_versions: dict[str, str]
    available: bool

    @field_validator("runtime_versions")
    @classmethod
    def nonempty_runtime_versions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(
            not key.strip() or not version.strip() for key, version in value.items()
        ):
            raise ValueError("baseline runtime versions must be non-empty")
        return value

    @model_validator(mode="after")
    def checkpoint_members_are_bound(self) -> CampaignBaseline:
        paths = [record.path for record in self.checkpoint_files]
        if len(paths) != len(set(paths)):
            raise ValueError("baseline checkpoint file paths must be unique")
        expected = stable_sha256(
            [
                record.model_dump(mode="json")
                for record in sorted(self.checkpoint_files, key=lambda item: item.path)
            ]
        )
        if self.checkpoint_members_sha256 != expected:
            raise ValueError("baseline checkpoint-member digest is inconsistent")
        return self


class CandidateInputBindings(StrictModel):
    source_checkpoint_manifest: BoundFile
    calibration_manifest: BoundFile
    activation_capture_or_sentinel: BoundFile
    sensitivity_report: BoundFile
    plan: BoundFile
    artifact_manifest: BoundFile
    candidate_frontier: BoundFile
    frontier_evidence: list[BoundFile] = Field(min_length=1)


class FrontierGate(StrEnum):
    INTEGRITY = "integrity"
    SIZE = "size"
    RUNTIME = "runtime"
    DEVELOPMENT_AGENT_CODING = "development-agent-coding"
    DEVELOPMENT_GENERAL = "development-general"
    DEVELOPMENT_MTP = "development-mtp"
    COMPLETE_MEASUREMENT = "complete-measurement"


class FrontierGateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not-run"


class FrontierGateResult(StrictModel):
    gate: FrontierGate
    status: FrontierGateStatus
    evidence: BoundFile | None = None
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_is_evidenced(self) -> FrontierGateResult:
        if self.status is FrontierGateStatus.PASSED and (self.evidence is None or self.issues):
            raise ValueError("passing frontier gate requires evidence and no issues")
        if self.status is FrontierGateStatus.FAILED and (self.evidence is None or not self.issues):
            raise ValueError("failed frontier gate requires evidence and issues")
        if self.status is FrontierGateStatus.NOT_RUN and (
            self.evidence is not None or not self.issues
        ):
            raise ValueError("not-run frontier gate requires a reason and no evidence")
        return self


_FRONTIER_GATE_ORDER = tuple(FrontierGate)


class FlagshipFrontierEntry(StrictModel):
    candidate_id: str = Field(pattern=_IDENTIFIER)
    candidate: CandidateKey
    gates: list[FrontierGateResult]
    measured_main_bpw: float | None = Field(default=None, gt=0, le=16)
    measured_total_bpw: float | None = Field(default=None, gt=0, le=16)
    eligible_for_formal: bool

    @model_validator(mode="after")
    def gates_are_complete_and_cheapest_first(self) -> FlagshipFrontierEntry:
        gates = [result.gate for result in self.gates]
        if tuple(gates) != _FRONTIER_GATE_ORDER:
            raise ValueError("frontier gates must appear once in cheapest-failure-first order")
        terminal_seen = False
        for result in self.gates:
            if terminal_seen and result.status is not FrontierGateStatus.NOT_RUN:
                raise ValueError("frontier cannot run later gates after an earlier non-pass")
            if result.status is not FrontierGateStatus.PASSED:
                terminal_seen = True
        expected_eligible = all(result.status is FrontierGateStatus.PASSED for result in self.gates)
        if self.eligible_for_formal != expected_eligible:
            raise ValueError("frontier formal eligibility differs from gate results")
        if self.eligible_for_formal and (
            self.measured_main_bpw is None or self.measured_total_bpw is None
        ):
            raise ValueError("formally eligible frontier entry requires measured BPW")
        return self


class FlagshipFrontierIndex(StrictModel):
    schema_version: Literal["axquant.flagship-frontier.v1"] = "axquant.flagship-frontier.v1"
    source: CheckpointKey
    policy_sha256: str = Field(pattern=_SHA256)
    search_budget: int = Field(gt=0)
    search_used: int = Field(gt=0)
    formal_holdout_accessed: Literal[False] = False
    entries: list[FlagshipFrontierEntry] = Field(min_length=1)
    feasible_candidate_sha256: list[str]
    complete: Literal[True] = True
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def frontier_is_complete(self) -> FlagshipFrontierIndex:
        if self.search_used > self.search_budget:
            raise ValueError("frontier search use exceeds the frozen budget")
        if self.search_used != len(self.entries):
            raise ValueError("frontier search use must equal the retained candidate count")
        candidate_ids = [entry.candidate_id for entry in self.entries]
        candidate_digests = [stable_sha256(entry.candidate) for entry in self.entries]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("frontier candidate IDs must be unique")
        if len(candidate_digests) != len(set(candidate_digests)):
            raise ValueError("frontier candidate keys must be unique")
        expected = sorted(
            stable_sha256(entry.candidate) for entry in self.entries if entry.eligible_for_formal
        )
        if self.feasible_candidate_sha256 != expected:
            raise ValueError("frontier feasible-candidate summary is inconsistent")
        if any(
            entry.candidate.source != self.source
            or entry.candidate.certification_policy_sha256 != self.policy_sha256
            for entry in self.entries
        ):
            raise ValueError("frontier candidates differ from its source or policy")
        return self


class FlagshipFrontierRequest(StrictModel):
    schema_version: Literal["axquant.flagship-frontier-request.v1"] = (
        "axquant.flagship-frontier-request.v1"
    )
    source: CheckpointKey
    policy_sha256: str = Field(pattern=_SHA256)
    search_budget: int = Field(gt=0)
    search_used: int = Field(gt=0)
    formal_holdout_accessed: Literal[False] = False
    entries: list[FlagshipFrontierEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def search_use_is_bounded(self) -> FlagshipFrontierRequest:
        if self.search_used > self.search_budget:
            raise ValueError("frontier request search use exceeds its budget")
        if self.search_used != len(self.entries):
            raise ValueError("frontier request search use must equal the retained candidate count")
        return self


class FlagshipNoGoRecord(StrictModel):
    schema_version: Literal["axquant.flagship-no-go.v1"] = "axquant.flagship-no-go.v1"
    campaign_sha256: str = Field(pattern=_SHA256)
    frontier: BoundFile
    search_budget_exhausted: Literal[True] = True
    formal_holdout_unconsumed: Literal[True] = True
    binding_constraints: list[str] = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    attestation: BoundFile
    closed_at: datetime = Field(default_factory=utc_now)


class PostPublicationRuntimeVerification(StrictModel):
    schema_version: Literal["axquant.post-publication-runtime-verification.v1"] = (
        "axquant.post-publication-runtime-verification.v1"
    )
    candidate_sha256: str = Field(pattern=_SHA256)
    public_repository: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    public_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    downloaded_checksums_verified: Literal[True] = True
    ax_engine_passed: Literal[True] = True
    mlx_lm_passed: Literal[True] = True
    zero_fallback_verified: Literal[True] = True
    runtime_versions: dict[str, str]
    verifier: str = Field(min_length=1)
    verified_at: datetime = Field(default_factory=utc_now)

    @field_validator("runtime_versions")
    @classmethod
    def complete_runtime_versions(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"ax-engine", "mlx", "mlx-lm"}
        if set(value) != required or any(not version.strip() for version in value.values()):
            raise ValueError(
                "post-publication verification requires exact AX Engine, MLX, and MLX-LM versions"
            )
        return value


class FlagshipPublicationVerification(StrictModel):
    schema_version: Literal["axquant.flagship-publication-verification.v1"] = (
        "axquant.flagship-publication-verification.v1"
    )
    campaign_sha256: str = Field(pattern=_SHA256)
    candidate_sha256: str = Field(pattern=_SHA256)
    public_repository: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    public_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hub_url: str = Field(pattern=r"^https://huggingface\.co/[A-Za-z0-9._/-]+$")
    release_audit: BoundFile
    public_claim: BoundFile
    lifecycle_registry: BoundFile
    download_inventory: BoundFile
    runtime_verification: BoundFile
    verifier: str = Field(min_length=1)
    attestation: BoundFile
    verified_at: datetime = Field(default_factory=utc_now)


class CampaignRoles(StrictModel):
    product_owner: str = Field(min_length=1)
    certification_owner: str = Field(min_length=1)
    model_engineer: str = Field(min_length=1)
    runtime_owner: str = Field(min_length=1)
    evaluation_custodian: str = Field(min_length=1)
    independent_reviewer: str = Field(min_length=1)
    release_manager: str = Field(min_length=1)

    @field_validator(
        "product_owner",
        "certification_owner",
        "model_engineer",
        "runtime_owner",
        "evaluation_custodian",
        "independent_reviewer",
        "release_manager",
    )
    @classmethod
    def named_accountable_role(cls, value: str) -> str:
        normalized = value.strip()
        disallowed = {
            "anonymous",
            "none",
            "pending",
            "repository owner",
            "tbd",
            "unknown",
            "unassigned",
        }
        if normalized.casefold() in disallowed:
            raise ValueError("campaign roles require named accountable identities")
        return normalized

    @model_validator(mode="after")
    def reviewer_is_independent(self) -> CampaignRoles:
        operators = {
            self.product_owner,
            self.certification_owner,
            self.model_engineer,
            self.runtime_owner,
            self.evaluation_custodian,
            self.release_manager,
        }
        if self.independent_reviewer in operators:
            raise ValueError(
                "independent reviewer must differ from candidate and release operators"
            )
        return self


class FormalHostContract(StrictModel):
    host_id: Literal["df-macbookpro-m5"] = "df-macbookpro-m5"
    hardware_id: str = Field(min_length=1)
    os_version: str = Field(min_length=1)
    power_mode: str = Field(min_length=1)
    storage_contract: str = Field(min_length=1)
    thermal_protocol: str = Field(min_length=1)
    operator: str = Field(min_length=1)


class FormalHostEvidenceKind(StrEnum):
    AX_ENGINE_DOCTOR = "ax-engine-doctor"
    METAL = "metal"
    ZERO_FALLBACK = "zero-fallback"
    STORAGE = "storage"
    POWER = "power"
    THERMAL = "thermal"


class FormalHostEvidenceBinding(StrictModel):
    name: str = Field(pattern=_IDENTIFIER)
    kind: FormalHostEvidenceKind
    file: BoundFile


class FormalHostEvidenceResult(StrictModel):
    schema_version: Literal["axquant.formal-host-evidence.v1"] = "axquant.formal-host-evidence.v1"
    evidence_name: str = Field(pattern=_IDENTIFIER)
    kind: FormalHostEvidenceKind
    host_id: Literal["df-macbookpro-m5"] = "df-macbookpro-m5"
    subject_sha256: str = Field(pattern=_SHA256)
    passed: Literal[True] = True
    command: list[str] = Field(min_length=1)
    observations: dict[str, str | int | float | bool]
    software_versions: dict[str, str]
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def evidence_is_complete(self) -> FormalHostEvidenceResult:
        if self.completed_at <= self.started_at:
            raise ValueError("formal host evidence must end after it starts")
        if not self.observations or any(not key.strip() for key in self.observations):
            raise ValueError("formal host evidence requires named observations")
        if not self.software_versions or any(
            not key.strip() or not version.strip()
            for key, version in self.software_versions.items()
        ):
            raise ValueError("formal host evidence requires complete software versions")
        if any(not argument for argument in self.command):
            raise ValueError("formal host evidence command arguments must be non-empty")
        return self


class FormalHostScopeEvidence(StrictModel):
    schema_version: Literal["axquant.formal-host-scope.v1"] = "axquant.formal-host-scope.v1"
    contract: FormalHostContract
    doctor_passed: Literal[True] = True
    metal_available: Literal[True] = True
    zero_fallback_controls_passed: Literal[True] = True
    free_disk_bytes: int = Field(gt=0)
    evidence: list[FormalHostEvidenceBinding]
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def evidence_is_complete(self) -> FormalHostScopeEvidence:
        names = [binding.name for binding in self.evidence]
        if len(names) != len(set(names)):
            raise ValueError("formal host evidence names must be unique")
        required = set(FormalHostEvidenceKind)
        observed = {binding.kind for binding in self.evidence}
        if observed != required:
            missing = sorted(kind.value for kind in required - observed)
            extra = sorted(kind.value for kind in observed - required)
            raise ValueError(f"formal host evidence kinds differ: missing={missing}, extra={extra}")
        return self


class FlagshipCampaign(StrictModel):
    schema_version: Literal["axquant.flagship-campaign.v1"] = "axquant.flagship-campaign.v1"
    campaign_id: str = Field(pattern=_IDENTIFIER)
    state: CampaignState = CampaignState.DRAFT
    certification_track: Literal["qwen36-mtp-v2"] = "qwen36-mtp-v2"
    source: CheckpointKey
    target_class: str = Field(min_length=1)
    policy_file: BoundFile
    toolkit_wheel: BoundFile
    runtime_builds: dict[str, BoundFile]
    formal_host: FormalHostContract
    hardware_scope: BoundFile
    hardware_scope_evidence: list[FormalHostEvidenceBinding]
    datasets: list[CampaignDataset]
    baselines: list[CampaignBaseline]
    candidate: CandidateKey | None = None
    candidate_inputs: CandidateInputBindings | None = None
    lifecycle_registry: BoundFile
    durable_evidence_root: str = Field(min_length=1)
    backup_verification: BoundFile
    required_free_disk_bytes: int = Field(gt=0)
    expected_stage_outputs: dict[str, int]
    roles: CampaignRoles
    formal_cycle_limit: int = Field(default=1, ge=1)
    formal_cycles_consumed: int = Field(default=0, ge=0)
    frozen_at: datetime | None = None
    formal_holdout_consumed_at: datetime | None = None
    no_go_record: BoundFile | None = None
    publication_verification: BoundFile | None = None
    created_by: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @field_validator("runtime_builds")
    @classmethod
    def required_runtime_builds(cls, value: dict[str, BoundFile]) -> dict[str, BoundFile]:
        required = {"ax-engine", "mlx", "mlx-lm"}
        if set(value) != required:
            raise ValueError(f"runtime builds must contain exactly {sorted(required)}")
        return value

    @field_validator("expected_stage_outputs")
    @classmethod
    def declared_stage_outputs(cls, value: dict[str, int]) -> dict[str, int]:
        if not value or any(not name.strip() or size <= 0 for name, size in value.items()):
            raise ValueError("expected campaign stage outputs require positive byte estimates")
        return value

    @field_validator("hardware_scope_evidence")
    @classmethod
    def declared_host_evidence(
        cls,
        value: list[FormalHostEvidenceBinding],
    ) -> list[FormalHostEvidenceBinding]:
        names = [binding.name for binding in value]
        if len(names) != len(set(names)):
            raise ValueError("campaign formal host evidence names must be unique")
        if {binding.kind for binding in value} != set(FormalHostEvidenceKind):
            raise ValueError("campaign formal host evidence must cover every required kind")
        return value

    @model_validator(mode="after")
    def state_invariants(self) -> FlagshipCampaign:
        roles = [dataset.role for dataset in self.datasets]
        if len(roles) != len(set(roles)):
            raise ValueError("campaign dataset roles must be unique")
        if self.formal_cycles_consumed > self.formal_cycle_limit:
            raise ValueError("campaign formal-cycle budget is exceeded")
        frozen_or_later = self.state in {
            CampaignState.FROZEN,
            CampaignState.FORMAL_RUNNING,
            CampaignState.FORMAL_FAILED,
            CampaignState.RELEASE_READY,
            CampaignState.PUBLISHED,
        }
        if frozen_or_later and (self.candidate is None or self.frozen_at is None):
            raise ValueError("frozen and later campaign states require candidate and freeze time")
        if frozen_or_later and self.candidate_inputs is None:
            raise ValueError("frozen and later campaign states require candidate input bindings")
        if self.state is CampaignState.DRAFT and self.frozen_at is not None:
            raise ValueError("draft campaign cannot carry a freeze time")
        if self.state is CampaignState.CLOSED_NO_GO and self.formal_holdout_consumed_at is not None:
            raise ValueError("closed-no-go campaign cannot claim formal holdout consumption")
        if (self.state is CampaignState.CLOSED_NO_GO) != (self.no_go_record is not None):
            raise ValueError("closed-no-go state requires exactly one bound no-go record")
        if (self.state is CampaignState.PUBLISHED) != (self.publication_verification is not None):
            raise ValueError("published state requires exactly one publication verification")
        if self.state is CampaignState.PUBLISHED and self.formal_holdout_consumed_at is None:
            raise ValueError("published campaign requires consumed formal holdout evidence")
        if self.state in {CampaignState.FORMAL_FAILED, CampaignState.RELEASE_READY} and (
            self.formal_holdout_consumed_at is None
        ):
            raise ValueError("formal terminal campaign states require consumed holdout evidence")
        if self.formal_holdout_consumed_at is not None and not all(
            dataset.consumed
            for dataset in self.datasets
            if dataset.role
            in {
                CampaignDatasetRole.FORMAL_AGENT_CODING,
                CampaignDatasetRole.FORMAL_GENERAL,
            }
        ):
            raise ValueError("formal holdout consumption timestamp requires all formal datasets")
        return self


class CampaignPreflight(StrictModel):
    schema_version: Literal["axquant.flagship-campaign-preflight.v1"] = (
        "axquant.flagship-campaign-preflight.v1"
    )
    campaign_id: str
    campaign_sha256: str = Field(pattern=_SHA256)
    checked_at: datetime = Field(default_factory=utc_now)
    host_id: Literal["df-macbookpro-m5"] = "df-macbookpro-m5"
    passed: bool
    issues: list[str]
    verified_files: list[BoundFile]

    @model_validator(mode="after")
    def pass_matches_issues(self) -> CampaignPreflight:
        if self.passed == bool(self.issues):
            raise ValueError("campaign preflight pass state must equal absence of issues")
        return self
