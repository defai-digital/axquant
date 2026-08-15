from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now

CERTIFICATION_VERIFICATION_SCHEMA_VERSION: Literal["axquant.certification-verification.v1"] = (
    "axquant.certification-verification.v1"
)


class CertificationVerificationCheck(StrictModel):
    """One independently reportable offline certificate check."""

    check_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    passed: bool
    message: str = Field(min_length=1)


class CertificationVerificationReport(StrictModel):
    """Machine-readable result emitted by ``axquant verify-cert``."""

    schema_version: Literal["axquant.certification-verification.v1"] = (
        CERTIFICATION_VERIFICATION_SCHEMA_VERSION
    )
    certificate_path: str = Field(min_length=1)
    certificate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    certificate_schema_version: str = Field(min_length=1)
    certificate_status: str | None = None
    artifact_path: str | None = None
    hub_repo_id: str | None = None
    product_class: str | None = None
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recomputed_main_bpw: float | None = Field(default=None, gt=0.0)
    passed: bool
    checks: list[CertificationVerificationCheck] = Field(min_length=1)
    issues: list[str] = Field(default_factory=list)
    verified_files: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verdict_matches_checks(self) -> CertificationVerificationReport:
        expected = all(check.passed for check in self.checks) and not self.issues
        if self.passed != expected:
            raise ValueError("verification verdict must match its checks and issues")
        check_ids = [check.check_id for check in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("verification check IDs must be unique")
        return self
