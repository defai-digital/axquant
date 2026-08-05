from __future__ import annotations

from pathlib import Path

from axquant.errors import ValidationGateError
from axquant.lifecycle import require_active_certification
from axquant.naming import certified_mixed_precision_name
from axquant.schema import (
    ArtifactLifecycleRegistry,
    ArtifactManifest,
    BoundFile,
    BoundMetricClaim,
    CandidateKey,
    PublicClaimManifest,
    PublicClaimRenderRequest,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data, write_text

_ALLOWED_METRIC_PREFIXES = (
    "artifact.",
    "quality.",
    "perplexity_",
    "task.",
    "mtp.",
    "hardware.",
    "integrity.",
)


def _validate_metric_claims(claims: list[BoundMetricClaim]) -> None:
    seen: set[tuple[str, str]] = set()
    for claim in claims:
        if not claim.metric_key.startswith(_ALLOWED_METRIC_PREFIXES):
            raise ValidationGateError(f"unsupported public metric claim: {claim.metric_key}")
        key = (claim.profile, claim.metric_key)
        if key in seen:
            raise ValidationGateError(
                f"duplicate public metric claim: {claim.profile}/{claim.metric_key}"
            )
        seen.add(key)


def _validate_claim_partitions(
    quality_claims: list[BoundMetricClaim],
    performance_claims: list[BoundMetricClaim],
) -> None:
    quality_profiles = {claim.profile for claim in quality_claims}
    if quality_profiles != {"agent-coding", "general"}:
        raise ValidationGateError(
            "public quality claims must cover agent-coding and general profiles exactly"
        )
    if any(claim.profile != "hardware" for claim in performance_claims):
        raise ValidationGateError("public performance claims must use the hardware profile")


def build_public_claim(
    *,
    candidate: CandidateKey,
    lifecycle: ArtifactLifecycleRegistry,
    audit_sha256: str,
    public_owner: str,
    base_model: str,
    target_class: str,
    measured_main_bpw: float,
    measured_total_bpw: float,
    weight_bytes: int,
    runtime_versions: dict[str, str],
    quality_claims: list[BoundMetricClaim],
    performance_claims: list[BoundMetricClaim],
    limitations: list[str],
    evidence_index: list[BoundFile],
    mtp: bool = True,
) -> PublicClaimManifest:
    event = require_active_certification(lifecycle, candidate)
    _validate_metric_claims([*quality_claims, *performance_claims])
    _validate_claim_partitions(quality_claims, performance_claims)
    display_name = certified_mixed_precision_name(
        base_model,
        measured_main_bpw,
        mtp=mtp,
    )
    repository = f"{public_owner}/{display_name}"
    if event.public_repository != repository:
        raise ValidationGateError(
            "certified lifecycle repository differs from measured-BPW generated identity"
        )
    return PublicClaimManifest(
        candidate=candidate,
        lifecycle_event_sha256=stable_sha256(event),
        audit_sha256=audit_sha256,
        public_repository=repository,
        display_name=display_name,
        target_class=target_class,
        measured_main_bpw=measured_main_bpw,
        measured_total_bpw=measured_total_bpw,
        weight_bytes=weight_bytes,
        hardware_scope_ids=["mbp-m5"],
        runtime_versions=runtime_versions,
        quality_claims=quality_claims,
        performance_claims=performance_claims,
        limitations=limitations,
        evidence_index=evidence_index,
    )


def render_certified_model_card(
    *,
    claim: PublicClaimManifest,
    source_model_id: str,
    source_revision: str,
    reviewer: str,
) -> str:
    expected = certified_mixed_precision_name(
        source_model_id,
        claim.measured_main_bpw,
        mtp=claim.display_name.endswith("-MTP"),
    )
    if claim.display_name != expected or not claim.public_repository.endswith(f"/{expected}"):
        raise ValidationGateError("public claim repository does not match measured-BPW naming")

    def rows(claims: list[BoundMetricClaim]) -> str:
        return "\n".join(
            f"| `{item.profile}` | `{item.metric_key}` | {item.value:g} {item.unit} | "
            f"`{item.evidence.path}` |"
            for item in claims
        )

    limitations = "\n".join(f"- {item}" for item in claim.limitations)
    runtime = "\n".join(
        f"- `{name}`: `{version}`" for name, version in sorted(claim.runtime_versions.items())
    )
    return f"""---
library_name: mlx
base_model: {source_model_id}
tags:
  - mlx
  - quantized
  - mixed-precision
  - mtp
---

# {claim.display_name}

This is an AXQuant-certified **mixed-precision** MTP checkpoint. Its measured main-weight
precision is **{claim.measured_main_bpw:.4f} BPW** and its measured total precision is
**{claim.measured_total_bpw:.4f} BPW**. The `{claim.target_class}` value is a planning class,
not a fixed storage claim.

## Identity and scope

- Repository: `{claim.public_repository}`
- Exact source: `{source_model_id}@{source_revision}`
- Candidate key: `{stable_sha256(claim.candidate)}`
- Release audit: `{claim.audit_sha256}`
- Lifecycle event: `{claim.lifecycle_event_sha256}`
- Authorizing performance host: `mbp-m5`
- Independent evidence reviewer: {reviewer}

Calibration data, development evaluation data, formal holdouts, and reproduction data were
separated and digest-bound by the flagship campaign. The internal clean-host reproduction
verified the same semantic candidate independently of absolute filesystem paths.

## Quality evidence

| Profile | Metric | Result | Evidence |
| --- | --- | ---: | --- |
{rows(claim.quality_claims)}

## Runtime and performance evidence

| Profile | Metric | Result | Evidence |
| --- | --- | ---: | --- |
{rows(claim.performance_claims)}

Runtime versions:

{runtime}

MLX-LM compatibility is limited to the exact compatibility evidence named in the release package.
The `mbp-m5` results authorize only the recorded hardware and software scope.

## Limitations

{limitations}

## Verification

Run `axquant release-audit --request flagship-release-audit-request.json` from the package root,
then verify all files against `public-claim.json` and the packaged evidence indexes. A later
`superseded` or `revoked` lifecycle event terminates this certified status without rewriting the
historical audit.
"""


def write_public_claim(
    *,
    claim: PublicClaimManifest,
    claim_path: str | Path,
    model_card_path: str | Path,
    source_model_id: str,
    source_revision: str,
    reviewer: str,
) -> None:
    write_data(claim_path, claim)
    write_text(
        model_card_path,
        render_certified_model_card(
            claim=claim,
            source_model_id=source_model_id,
            source_revision=source_revision,
            reviewer=reviewer,
        ),
    )


def render_public_claim_request(
    *,
    request_path: str | Path,
    claim_path: str | Path,
    model_card_path: str | Path,
) -> PublicClaimManifest:
    from axquant.schema import FlagshipReleaseAudit

    source = Path(request_path).expanduser().resolve()
    request = load_model(source, PublicClaimRenderRequest)
    root = source.parent

    def request_file(relative: str) -> Path:
        unresolved = root / relative
        resolved = unresolved.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValidationGateError(
                f"public claim input escapes request root: {relative}"
            ) from exc
        if unresolved.is_symlink() or not resolved.is_file():
            raise ValidationGateError(f"public claim input is missing or unsafe: {relative}")
        return resolved

    audit_path = request_file(request.authorization_audit)
    lifecycle_path = request_file(request.lifecycle_registry)
    artifact_path = request_file(request.artifact_manifest)
    audit = load_model(audit_path, FlagshipReleaseAudit)
    if not audit.authorization_ready:
        raise ValidationGateError("public claims require a flagship authorization-ready audit")
    lifecycle = load_model(lifecycle_path, ArtifactLifecycleRegistry)
    artifact = load_model(artifact_path, ArtifactManifest)
    lifecycle_event = require_active_certification(lifecycle, audit.candidate)
    if artifact.source_model.model_id != audit.source_model.model_id or (
        artifact.source_model.revision != audit.source_model.revision
    ):
        raise ValidationGateError("public claim artifact source differs from authorization audit")
    if request.reviewer != lifecycle_event.reviewer:
        raise ValidationGateError("public claim reviewer differs from certified lifecycle review")
    evidence_index: dict[tuple[str, str, int], BoundFile] = {}
    for metric in [*request.quality_claims, *request.performance_claims]:
        evidence = metric.evidence
        unresolved_evidence = root / evidence.path
        evidence_path = unresolved_evidence.resolve()
        try:
            evidence_path.relative_to(root)
        except ValueError as exc:
            raise ValidationGateError(
                f"public metric evidence escapes request root: {evidence.path}"
            ) from exc
        if (
            unresolved_evidence.is_symlink()
            or not evidence_path.is_file()
            or evidence_path.stat().st_size != evidence.size_bytes
            or file_sha256(evidence_path) != evidence.sha256
        ):
            raise ValidationGateError(
                f"public metric evidence is missing or changed: {evidence.path}"
            )
        evidence_index[(evidence.path, evidence.sha256, evidence.size_bytes)] = evidence
    runtime_versions = {
        key.replace("_", "-"): value
        for key, value in artifact.software_versions.model_dump(mode="python").items()
        if value is not None
    }
    claim = build_public_claim(
        candidate=audit.candidate,
        lifecycle=lifecycle,
        audit_sha256=file_sha256(audit_path),
        public_owner=request.public_owner,
        base_model=audit.source_model.model_id,
        target_class=artifact.target_class,
        measured_main_bpw=artifact.measured_main_bpw,
        measured_total_bpw=artifact.measured_total_bpw,
        weight_bytes=artifact.weight_file_size_bytes,
        runtime_versions=runtime_versions,
        quality_claims=request.quality_claims,
        performance_claims=request.performance_claims,
        limitations=request.limitations,
        evidence_index=sorted(evidence_index.values(), key=lambda item: item.path),
        mtp=artifact.mtp_present,
    )
    if audit.candidate_model.model_id != claim.public_repository:
        raise ValidationGateError(
            "authorization candidate repository differs from generated measured-BPW name"
        )
    write_public_claim(
        claim=claim,
        claim_path=claim_path,
        model_card_path=model_card_path,
        source_model_id=audit.source_model.model_id,
        source_revision=audit.source_model.revision or "",
        reviewer=request.reviewer,
    )
    return claim
