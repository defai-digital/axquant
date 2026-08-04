from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from axquant.certification.policy import direct_policy_sha256
from axquant.errors import ArtifactError, PublishingError
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    ArtifactManifest,
    CertifiedCheckpointEntry,
    CertifiedCheckpointRegistry,
    Qwen3NextReleaseAudit,
)
from axquant.serde import file_sha256, load_model, write_data

DIRECT_CERTIFICATION_ALLOWED_CLAIMS: Final[tuple[str, ...]] = (
    "AXQuant exact-checkpoint certified non-MTP direct-decode artifact",
    "Measured metrics limited to the audit hardware scope",
)


def load_checkpoint_registry(path: str | Path) -> CertifiedCheckpointRegistry:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return CertifiedCheckpointRegistry()
    registry = load_model(source, CertifiedCheckpointRegistry)
    policy_sha256 = direct_policy_sha256()
    issues: list[str] = []
    for entry in registry.entries:
        if entry.policy_sha256 != policy_sha256:
            issues.append(f"{entry.entry_id} uses another certification policy")
        if any(claim not in DIRECT_CERTIFICATION_ALLOWED_CLAIMS for claim in entry.allowed_claims):
            issues.append(f"{entry.entry_id} exceeds the direct certification claim scope")
        if not is_immutable_revision(entry.candidate_model.revision):
            issues.append(f"{entry.entry_id} candidate revision is not immutable")
        if entry.artifact_manifest_sha256 != entry.certification_scope.artifact_manifest_sha256:
            issues.append(f"{entry.entry_id} artifact manifest differs from its exact scope")
    if issues:
        raise ArtifactError("certification registry trust validation failed: " + "; ".join(issues))
    return registry


def append_certified_checkpoint(
    *,
    registry_path: str | Path,
    audit_path: str | Path,
    artifact_directory: str | Path,
    candidate_id: str,
    measured_bpw: float,
    allowed_claims: list[str],
    supersedes_entry_id: str | None = None,
) -> CertifiedCheckpointRegistry:
    audit_source = Path(audit_path).expanduser().resolve()
    audit = load_model(audit_source, Qwen3NextReleaseAudit)
    if not audit.release_ready:
        raise PublishingError("only a release-ready N0-N8 audit may enter the registry")
    if audit.policy_sha256 != direct_policy_sha256():
        raise PublishingError("registry audit does not use the wheel-owned direct policy")
    if not is_immutable_revision(audit.candidate_model.revision):
        raise PublishingError("registry candidate identity requires an immutable revision")
    if not allowed_claims or any(
        claim not in DIRECT_CERTIFICATION_ALLOWED_CLAIMS for claim in allowed_claims
    ):
        raise PublishingError("registry claims exceed the direct certification claim scope")
    artifact = Path(artifact_directory).expanduser().resolve()
    manifest_path = artifact / "axquant_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactError(f"artifact manifest does not exist: {manifest_path}")
    manifest = load_model(manifest_path, ArtifactManifest)
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != audit.certification_scope.artifact_manifest_sha256:
        raise PublishingError("audit and registry artifact manifest differ")
    # The audit binds the manifest digest, but the artifact tree can drift
    # between audit creation and registry append. Recheck every manifest-bound
    # file and the Safetensors membership before certifying the checkpoint.
    from axquant.release_audit import _artifact_issues

    artifact_issues = _artifact_issues(artifact, manifest)
    if artifact_issues:
        raise PublishingError(
            "registry artifact integrity check failed: " + "; ".join(artifact_issues)
        )
    if abs(manifest.measured_total_bpw - measured_bpw) > 1e-9:
        raise PublishingError("registry measured BPW differs from the artifact")

    registry_source = Path(registry_path).expanduser().resolve()
    registry = load_checkpoint_registry(registry_source)
    audit_sha256 = file_sha256(audit_source)
    entry_id = f"{candidate_id}-{audit.certification_scope.target_class.value}-{audit_sha256[:12]}"
    for existing in registry.entries:
        if existing.entry_id == entry_id:
            if existing.release_audit_sha256 != audit_sha256:
                raise PublishingError("registry entry ID collides with another audit")
            expected = {
                "certification_scope": audit.certification_scope,
                "candidate_model": audit.candidate_model,
                "candidate_id": candidate_id,
                "policy_sha256": audit.policy_sha256,
                "artifact_manifest_sha256": manifest_sha256,
                "measured_bpw": measured_bpw,
                "allowed_claims": allowed_claims,
                "hardware_scope_ids": audit.certification_scope.hardware_scope_ids,
                "supersedes_entry_id": supersedes_entry_id,
            }
            if any(getattr(existing, field) != value for field, value in expected.items()):
                raise PublishingError("existing registry entry differs from the requested append")
            return registry

    entry = CertifiedCheckpointEntry(
        entry_id=entry_id,
        certification_scope=audit.certification_scope,
        candidate_model=audit.candidate_model,
        candidate_id=candidate_id,
        policy_sha256=audit.policy_sha256,
        artifact_manifest_sha256=manifest_sha256,
        release_audit_sha256=audit_sha256,
        measured_bpw=measured_bpw,
        allowed_claims=allowed_claims,
        hardware_scope_ids=audit.certification_scope.hardware_scope_ids,
        certified_at=datetime.now(UTC),
        supersedes_entry_id=supersedes_entry_id,
    )
    updated = CertifiedCheckpointRegistry(entries=[*registry.entries, entry])
    write_data(registry_source, updated)
    return updated
