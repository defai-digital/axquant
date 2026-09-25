"""Artifact evidence-binding sidecar model (AXQ-045 MH3).

The sidecar binds an artifact manifest to the certificate files that
authorize release claims for that exact artifact state. It is written by
release tooling (release-audit / publish-prepare), never during convert:
conversion output remains development evidence by design.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.enums import EvidenceKind


class ArtifactEvidenceBinding(StrictModel):
    schema_version: Literal["axquant.artifact-evidence-binding.v1"] = (
        "axquant.artifact-evidence-binding.v1"
    )
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tier1_certificate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tier2_certificate_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evidence_kind: EvidenceKind
    created_at: datetime = Field(default_factory=utc_now)
