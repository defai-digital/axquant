"""Artifact evidence-binding sidecar writer and verifier (AXQ-045 MH3).

Release tooling writes ``axquant_evidence_binding.json`` next to
``axquant_manifest.json`` to bind an artifact to the certificate files that
authorize release claims. ``release_audit`` and ``publisher`` reject artifact
directories without a valid sidecar. Conversion never writes this file:
convert output remains development evidence by design.
"""

from __future__ import annotations

from pathlib import Path

from axquant.errors import ArtifactError
from axquant.schema import (
    ArtifactEvidenceBinding,
    ArtifactManifest,
    EvidenceKind,
    utc_now,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

ARTIFACT_MANIFEST_FILENAME = "axquant_manifest.json"
ARTIFACT_EVIDENCE_BINDING_FILENAME = "axquant_evidence_binding.json"


def load_artifact_evidence_binding(directory: str | Path) -> ArtifactEvidenceBinding | None:
    path = Path(directory) / ARTIFACT_EVIDENCE_BINDING_FILENAME
    if not path.is_file():
        return None
    return load_model(path, ArtifactEvidenceBinding)


def _certificate_sha256(path: str | Path | None, label: str) -> str | None:
    if path is None:
        return None
    source = Path(path).expanduser()
    if not source.is_file():
        raise ArtifactError(f"{label} certificate file does not exist: {source}")
    return file_sha256(source)


def write_artifact_evidence_binding(
    *,
    artifact_directory: str | Path,
    evidence_kind: EvidenceKind,
    tier1_certificate_path: str | Path | None = None,
    tier2_certificate_path: str | Path | None = None,
) -> ArtifactEvidenceBinding:
    """Bind an artifact manifest to at least one certificate file.

    The manifest digest is the semantic ``stable_sha256`` of the loaded
    ``ArtifactManifest``; certificate digests are byte SHA-256 of the
    certificate files. At least one certificate binding is required for
    release claims.
    """

    if tier1_certificate_path is None and tier2_certificate_path is None:
        raise ArtifactError(
            "at least one certificate path is required "
            "(tier1_certificate_path and/or tier2_certificate_path)"
        )
    directory = Path(artifact_directory).expanduser().resolve()
    manifest_path = directory / ARTIFACT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactError(f"artifact manifest does not exist: {manifest_path}")
    manifest = load_model(manifest_path, ArtifactManifest)
    binding = ArtifactEvidenceBinding(
        artifact_manifest_sha256=stable_sha256(manifest),
        tier1_certificate_sha256=_certificate_sha256(tier1_certificate_path, "tier1"),
        tier2_certificate_sha256=_certificate_sha256(tier2_certificate_path, "tier2"),
        evidence_kind=evidence_kind,
    )
    write_data(directory / ARTIFACT_EVIDENCE_BINDING_FILENAME, binding)
    return binding


def artifact_evidence_binding_issues(
    directory: str | Path,
    manifest: ArtifactManifest,
) -> list[str]:
    """Return release-blocking issues for the artifact evidence binding."""

    root = Path(directory)
    path = root / ARTIFACT_EVIDENCE_BINDING_FILENAME
    if not path.is_file():
        return [f"artifact evidence binding sidecar is missing: {path.name}"]
    try:
        binding = load_model(path, ArtifactEvidenceBinding)
    except (ArtifactError, OSError, ValueError) as exc:
        return [f"artifact evidence binding sidecar is invalid: {exc}"]
    issues: list[str] = []
    if stable_sha256(manifest) != binding.artifact_manifest_sha256:
        issues.append(
            f"artifact evidence binding manifest digest does not match {ARTIFACT_MANIFEST_FILENAME}"
        )
    if binding.tier1_certificate_sha256 is None and binding.tier2_certificate_sha256 is None:
        issues.append("artifact evidence binding carries no certificate binding")
    return issues


def refresh_artifact_evidence_binding(
    *,
    artifact_directory: str | Path,
    manifest: ArtifactManifest,
) -> ArtifactEvidenceBinding | None:
    """Rebind the manifest digest after publish-prepare mutates the manifest.

    Certificate bindings and evidence kind are preserved; only the manifest
    digest and timestamp move. Returns None when no sidecar exists.
    """

    directory = Path(artifact_directory)
    path = directory / ARTIFACT_EVIDENCE_BINDING_FILENAME
    if not path.is_file():
        return None
    binding = load_model(path, ArtifactEvidenceBinding)
    refreshed = binding.model_copy(
        update={
            "artifact_manifest_sha256": stable_sha256(manifest),
            "created_at": utc_now(),
        }
    )
    write_data(path, refreshed)
    return refreshed
