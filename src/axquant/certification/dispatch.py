from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from axquant.errors import ArtifactError
from axquant.schema import (
    Qwen3NextReleaseAudit,
    Qwen3NextReleaseAuditRequest,
    ReleaseAudit,
    ReleaseAuditRequest,
)
from axquant.serde import load_model, read_data

CertificationRequest: TypeAlias = ReleaseAuditRequest | Qwen3NextReleaseAuditRequest
CertificationAudit: TypeAlias = ReleaseAudit | Qwen3NextReleaseAudit

_MTP_REQUEST_VERSION = "axquant.release-audit-request.v4"
_DIRECT_REQUEST_VERSION = "axquant.qwen3-next-release-audit-request.v1"
_MTP_AUDIT_VERSION = "axquant.release-audit.v4"
_DIRECT_AUDIT_VERSION = "axquant.qwen3-next-release-audit.v1"


def _schema_version(path: str | Path) -> str:
    payload = read_data(path)
    if not isinstance(payload, dict):
        raise ArtifactError(f"certification artifact must contain an object: {path}")
    version = payload.get("schema_version")
    if not isinstance(version, str) or not version:
        raise ArtifactError(f"certification artifact has no schema_version: {path}")
    return version


def load_certification_request(path: str | Path) -> CertificationRequest:
    version = _schema_version(path)
    if version == _MTP_REQUEST_VERSION:
        return load_model(path, ReleaseAuditRequest)
    if version == _DIRECT_REQUEST_VERSION:
        return load_model(path, Qwen3NextReleaseAuditRequest)
    raise ArtifactError(f"unsupported release-audit request schema: {version}")


def load_certification_audit(path: str | Path) -> CertificationAudit:
    version = _schema_version(path)
    if version == _MTP_AUDIT_VERSION:
        return load_model(path, ReleaseAudit)
    if version == _DIRECT_AUDIT_VERSION:
        return load_model(path, Qwen3NextReleaseAudit)
    raise ArtifactError(f"unsupported release-audit schema: {version}")


def build_certification_audit(path: str | Path) -> CertificationAudit:
    request = load_certification_request(path)
    if isinstance(request, ReleaseAuditRequest):
        from axquant.release_audit import build_release_audit

        return build_release_audit(path)

    from axquant.certification.qwen3_next_direct import build_qwen3_next_release_audit

    return build_qwen3_next_release_audit(path)
