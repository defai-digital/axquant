from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from axquant.artifact_paths import artifact_member_path, artifact_tree_files
from axquant.errors import ArtifactError
from axquant.identity import same_model_identity
from axquant.model_card import _AXQ_NAME
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    MTP_SCHEMA_VERSION,
    ArtifactManifest,
    CertificationVerificationCheck,
    CertificationVerificationReport,
    PublicCheckpointCertification,
    PublicMtpAccelerationCertification,
    QuantizationPlan,
)
from axquant.schema.public_certification import CHECKPOINT_SCHEMA_VERSION
from axquant.serde import file_sha256, read_data, stable_sha256

_CERTIFIED_MTP_STATUSES = frozenset({"certified", "certified-scoped", "certified-see-tier2-record"})
_PLANNING_BPW_CLASS = re.compile(r"^[0-9]+p[0-9]+bpw$")
_CERTIFICATE_MAIN_BPW_KEYS = (
    "measured_main_bpw",
    "candidate_measured_main_bpw",
)
_CERTIFICATE_TOTAL_BPW_KEYS = (
    "measured_total_bpw",
    "candidate_measured_bpw",
)
_CERTIFICATE_WEIGHT_BYTE_KEYS = (
    "candidate_weight_bytes",
    "candidate_bytes",
    "weight_bytes",
)


def _mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _resolve_certificate(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ArtifactError(f"certificate must not be a symlink: {source}")
    resolved = source.resolve()
    if not resolved.is_file():
        raise ArtifactError(f"certificate does not exist: {resolved}")
    return resolved


def _resolve_artifact(path: str | Path | None, certificate: Path) -> Path | None:
    if path is None:
        inferred = certificate.parent
        if (inferred / "axquant_manifest.json").is_file() and (
            inferred / "axquant_plan.json"
        ).is_file():
            return inferred
        return None
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ArtifactError(f"artifact directory must not be a symlink: {source}")
    resolved = source.resolve()
    if not resolved.is_dir():
        raise ArtifactError(f"artifact directory does not exist: {resolved}")
    try:
        artifact_tree_files(resolved)
    except ValueError as exc:
        raise ArtifactError(f"artifact directory is unsafe: {exc}") from exc
    return resolved


def _class_sku_agrees(product_class: str, repository_leaf: str) -> bool:
    """Accept exact SKU matches and legacy planning-class labels on 4/6-bit repos.

    Historical Qwen 3.6 4-bit certificates record ``product_class=5p6bpw`` while
    the Hub leaf stays ``...-MLX-AXQ-4bit[-MTP]``. New Spec 1.0 issuance still
    uses 4bit/6bit SKUs; verification must not invalidate the published catalog.
    """

    match = _AXQ_NAME.fullmatch(repository_leaf)
    if match is None:
        return False
    repo_class = match.group("product_class")
    if product_class == repo_class:
        return True
    return (
        repo_class in {"4bit", "6bit"} and _PLANNING_BPW_CLASS.fullmatch(product_class) is not None
    )


def _edition_binding_is_consistent(edition: str | None, tag: str | None) -> bool:
    if edition is None and tag is None:
        return True
    if edition is None or tag is None:
        return False
    normalized_edition = edition if edition.startswith("v") else f"v{edition}"
    return normalized_edition == tag and tag.startswith("v") and tag[1:].isdigit()


def _tier2_pointer_issues(
    certificate_path: Path,
    certificate: PublicCheckpointCertification,
) -> list[str]:
    block = certificate.mtp_acceleration
    if block.status not in _CERTIFIED_MTP_STATUSES:
        return []
    if block.tier2_certificate is None:
        return ["Tier 1 certificate asserts MTP acceleration without a Tier 2 certificate pointer"]
    try:
        pointer = artifact_member_path(certificate_path.parent, block.tier2_certificate)
    except ValueError as exc:
        return [f"Tier 2 certificate pointer is unsafe: {exc}"]
    if not pointer.is_file():
        return [f"Tier 2 certificate pointer is missing: {block.tier2_certificate}"]
    try:
        payload = read_data(pointer)
    except ArtifactError as exc:
        return [f"Tier 2 certificate cannot be read: {exc}"]
    mapping = _mapping(payload)
    if mapping is None or mapping.get("schema_version") != MTP_SCHEMA_VERSION:
        return ["Tier 2 certificate pointer does not name a supported Tier 2 schema"]
    try:
        tier2 = PublicMtpAccelerationCertification.model_validate(mapping)
    except ValidationError as exc:
        return [f"Tier 2 certificate is invalid: {exc}"]
    issues: list[str] = []
    if tier2.status != "certified":
        issues.append("Tier 1 MTP acceleration claim points to a non-certified Tier 2 record")
    if tier2.artifact.hub_repo_id != certificate.artifact.hub_repo_id:
        issues.append("Tier 2 certificate repository differs from the Tier 1 certificate")
    if tier2.artifact.hub_commit != certificate.artifact.hub_commit:
        issues.append("Tier 2 certificate commit differs from the Tier 1 certificate")
    return issues


def _manifest_file_issues(
    artifact: Path,
    manifest: ArtifactManifest,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    verified: list[str] = []
    for record in manifest.files:
        try:
            path = artifact_member_path(artifact, record.path)
        except ValueError as exc:
            issues.append(f"manifest file path is unsafe ({record.path}): {exc}")
            continue
        if not path.is_file():
            issues.append(f"manifest-bound file is missing: {record.path}")
            continue
        if path.stat().st_size != record.size_bytes:
            issues.append(f"manifest-bound file size changed: {record.path}")
            continue
        if file_sha256(path) != record.sha256:
            issues.append(f"manifest-bound file checksum changed: {record.path}")
            continue
        verified.append(record.path)
    return issues, verified


def _certificate_weight_file_issues(
    artifact: Path,
    certificate: PublicCheckpointCertification,
) -> tuple[list[str], list[str]]:
    if certificate.weight_files is None:
        return [], []
    issues: list[str] = []
    verified: list[str] = []
    for relative_name, expected in sorted(certificate.weight_files.items()):
        if not isinstance(expected, str) or len(expected) != 64:
            issues.append(f"certificate weight hash is invalid: {relative_name}")
            continue
        try:
            path = artifact_member_path(artifact, relative_name)
        except ValueError as exc:
            issues.append(f"certificate weight path is unsafe ({relative_name}): {exc}")
            continue
        if not path.is_file():
            issues.append(f"certificate-bound weight file is missing: {relative_name}")
            continue
        if file_sha256(path) != expected:
            issues.append(f"certificate-bound weight checksum changed: {relative_name}")
            continue
        verified.append(relative_name)
    return issues, verified


def verify_certificate(
    *,
    certificate_path: str | Path,
    artifact_dir: str | Path | None = None,
) -> CertificationVerificationReport:
    """Verify a public checkpoint certificate and optional local artifact bundle offline."""

    source = _resolve_certificate(certificate_path)
    artifact = _resolve_artifact(artifact_dir, source)
    certificate_digest = file_sha256(source)
    payload = read_data(source)
    raw = _mapping(payload)
    raw_schema = raw.get("schema_version") if raw is not None else None
    schema_version = raw_schema if isinstance(raw_schema, str) else "unknown"
    raw_status = raw.get("status") if raw is not None else None
    status = raw_status if isinstance(raw_status, str) else None
    checks: list[CertificationVerificationCheck] = []
    issues: list[str] = []

    def add(check_id: str, passed: bool, success: str, failure: str) -> None:
        message = success if passed else failure
        checks.append(
            CertificationVerificationCheck(
                check_id=check_id,
                passed=passed,
                message=message,
            )
        )
        if not passed:
            issues.append(failure)

    known_schema = schema_version == CHECKPOINT_SCHEMA_VERSION
    add(
        "certificate.schema",
        known_schema,
        f"supported checkpoint certificate schema: {CHECKPOINT_SCHEMA_VERSION}",
        f"unsupported certificate schema: {schema_version}",
    )

    certificate: PublicCheckpointCertification | None = None
    if known_schema and raw is not None:
        try:
            certificate = PublicCheckpointCertification.model_validate(raw)
        except ValidationError as exc:
            add(
                "certificate.envelope",
                False,
                "certificate envelope is valid",
                f"certificate envelope is invalid: {exc}",
            )
        else:
            add(
                "certificate.envelope",
                True,
                "certificate envelope is valid",
                "certificate envelope is invalid",
            )

    hub_repo_id: str | None = None
    product_class: str | None = None
    if certificate is not None:
        hub_repo_id = certificate.artifact.hub_repo_id
        product_class = certificate.artifact.product_class
        leaf = hub_repo_id.rsplit("/", 1)[-1]
        repo_class_ok = _class_sku_agrees(product_class, leaf)
        add(
            "certificate.class-repository",
            repo_class_ok,
            "certificate product class agrees with the class-SKU repository",
            "certificate product class does not agree with a class-SKU repository leaf",
        )
        add(
            "certificate.immutable-commit",
            is_immutable_revision(certificate.artifact.hub_commit),
            "certificate binds a full immutable Hub commit",
            "certificate Hub commit is not a full immutable commit SHA",
        )
        edition_ok = _edition_binding_is_consistent(
            certificate.artifact.artifact_edition,
            certificate.artifact.hub_tag,
        )
        add(
            "certificate.edition-tag",
            edition_ok,
            "certificate edition and immutable tag binding is consistent",
            "certificate artifact edition and Hub tag binding is incomplete or inconsistent",
        )
        tier2_issues = _tier2_pointer_issues(source, certificate)
        add(
            "certificate.tier-boundary",
            not tier2_issues,
            "Tier 1 does not make an unbound Tier 2 acceleration claim",
            "; ".join(tier2_issues),
        )
        evidence_kind = certificate.plan.get("evidence_kind")
        # v1 public records may document architecture_prior planning. That is
        # historical catalog fact, not a new promotion. optimize/deployment
        # still refuse to label prior evidence as measured or certified.
        add(
            "certificate.evidence-kind",
            True,
            (
                "legacy v1 certificate records architecture-prior planning evidence"
                if evidence_kind == "architecture_prior"
                else "certificate evidence kind is internally consistent for v1"
            ),
            "certificate evidence kind is inconsistent",
        )

    manifest_digest: str | None = None
    plan_digest: str | None = None
    recomputed_main_bpw: float | None = None
    verified_files: list[str] = []
    if artifact is None:
        add(
            "artifact.local-bundle",
            True,
            "artifact bundle was not supplied; certificate-local checks only",
            "artifact bundle is unavailable",
        )
    elif certificate is not None:
        manifest_path = artifact / "axquant_manifest.json"
        plan_path = artifact / "axquant_plan.json"
        required_files_ok = manifest_path.is_file() and plan_path.is_file()
        add(
            "artifact.required-files",
            required_files_ok,
            "artifact contains axquant_manifest.json and axquant_plan.json",
            "artifact must contain axquant_manifest.json and axquant_plan.json",
        )
        manifest_raw: dict[str, Any] | None = None
        plan_raw: dict[str, Any] | None = None
        manifest: ArtifactManifest | None = None
        plan: QuantizationPlan | None = None
        if manifest_path.is_file():
            manifest_digest = file_sha256(manifest_path)
            loaded_manifest = read_data(manifest_path)
            manifest_raw = _mapping(loaded_manifest)
            try:
                if manifest_raw is None:
                    raise ValueError("manifest root must be an object")
                manifest = ArtifactManifest.model_validate(manifest_raw)
            except (ValidationError, ValueError) as exc:
                add(
                    "artifact.manifest-envelope",
                    False,
                    "artifact manifest is valid",
                    f"artifact manifest is invalid: {exc}",
                )
            else:
                add(
                    "artifact.manifest-envelope",
                    True,
                    "artifact manifest is valid",
                    "artifact manifest is invalid",
                )
        if plan_path.is_file():
            loaded_plan = read_data(plan_path)
            plan_raw = _mapping(loaded_plan)
            try:
                if plan_raw is None:
                    raise ValueError("plan root must be an object")
                plan = QuantizationPlan.model_validate(plan_raw)
            except (ValidationError, ValueError) as exc:
                add(
                    "artifact.plan-envelope",
                    False,
                    "quantization plan is valid",
                    f"quantization plan is invalid: {exc}",
                )
            else:
                plan_digest = stable_sha256(plan)
                add(
                    "artifact.plan-envelope",
                    True,
                    "quantization plan is valid",
                    "quantization plan is invalid",
                )

        class_values: list[tuple[str, Any]] = [
            ("certificate artifact", certificate.artifact.product_class),
        ]
        if manifest_raw is not None:
            class_values.append(("manifest", manifest_raw.get("target_class")))
        if plan_raw is not None:
            class_values.append(("plan", plan_raw.get("target_class")))
        cert_plan_class = certificate.plan.get("target_class")
        if cert_plan_class is not None:
            class_values.append(("certificate plan", cert_plan_class))
        class_agreement = all(
            isinstance(value, str) and value == certificate.artifact.product_class
            for _label, value in class_values
        )
        class_detail = ", ".join(f"{label}={value!r}" for label, value in class_values)
        add(
            "artifact.class-agreement",
            class_agreement,
            f"class-SKU agreement holds ({class_detail})",
            f"class-SKU agreement failed ({class_detail})",
        )

        if plan_raw is not None:
            packaged_evidence = plan_raw.get("evidence_kind")
            certificate_evidence = certificate.plan.get("evidence_kind")
            evidence_agreement = packaged_evidence == certificate_evidence
            add(
                "artifact.evidence-kind",
                evidence_agreement,
                "certificate and packaged plan evidence kinds agree",
                "certificate and packaged plan evidence kinds differ",
            )

        expected_manifest_digest = certificate.artifact.candidate_manifest_sha256
        digest_ok = expected_manifest_digest is None or manifest_digest == expected_manifest_digest
        add(
            "artifact.manifest-digest",
            digest_ok,
            (
                "certificate does not bind a manifest digest (legacy optional field)"
                if expected_manifest_digest is None
                else "artifact manifest digest matches the certificate"
            ),
            "artifact manifest digest does not match the certificate",
        )

        if manifest_raw is not None:
            raw_bytes = manifest_raw.get("main_weight_file_size_bytes")
            raw_parameters = manifest_raw.get("main_logical_parameters")
            raw_bpw = manifest_raw.get("measured_main_bpw")
            accounting_values_ok = (
                type(raw_bytes) is int
                and raw_bytes > 0
                and type(raw_parameters) is int
                and raw_parameters > 0
                and isinstance(raw_bpw, (int, float))
                and not isinstance(raw_bpw, bool)
            )
            if accounting_values_ok:
                assert isinstance(raw_bytes, int)
                assert isinstance(raw_parameters, int)
                assert isinstance(raw_bpw, (int, float))
                recomputed_main_bpw = 8.0 * raw_bytes / raw_parameters
                exact_bpw = recomputed_main_bpw == raw_bpw
            else:
                exact_bpw = False
            add(
                "artifact.main-bpw",
                exact_bpw,
                "manifest measured main BPW exactly matches recomputed byte accounting",
                "manifest measured main BPW does not exactly match recomputed byte accounting",
            )
            certificate_bpw_values = [
                (key, certificate.size[key], recomputed_main_bpw)
                for key in _CERTIFICATE_MAIN_BPW_KEYS
                if key in certificate.size
            ]
            raw_total_bpw = manifest_raw.get("measured_total_bpw")
            certificate_bpw_values.extend(
                (key, certificate.size[key], raw_total_bpw)
                for key in _CERTIFICATE_TOTAL_BPW_KEYS
                if key in certificate.size
            )
            certificate_bpw_ok = all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                and value == expected
                for _key, value, expected in certificate_bpw_values
            )
            if not certificate_bpw_values:
                certificate_bpw_ok = True
            add(
                "artifact.certificate-main-bpw",
                certificate_bpw_ok,
                (
                    "certificate has no optional measured-BPW size field"
                    if not certificate_bpw_values
                    else "certificate measured BPW matches artifact accounting"
                ),
                "certificate measured BPW does not match artifact accounting",
            )

        if manifest is not None and plan is not None:
            plan_binding_ok = manifest.plan_sha256 == stable_sha256(plan)
            add(
                "artifact.plan-binding",
                plan_binding_ok,
                "manifest semantic plan digest matches axquant_plan.json",
                "manifest semantic plan digest does not match axquant_plan.json",
            )
            identity_ok = same_model_identity(manifest.source_model, plan.source_model)
            add(
                "artifact.model-identity",
                identity_ok,
                "manifest and plan source model identities agree",
                "manifest and plan source model identities differ",
            )
            file_issues, manifest_verified = _manifest_file_issues(artifact, manifest)
            verified_files.extend(manifest_verified)
            add(
                "artifact.manifest-files",
                not file_issues,
                "all manifest-bound artifact files match size and SHA-256",
                "; ".join(file_issues),
            )
            certificate_file_issues, certificate_verified = _certificate_weight_file_issues(
                artifact,
                certificate,
            )
            verified_files.extend(certificate_verified)
            add(
                "artifact.certificate-weight-files",
                not certificate_file_issues,
                "all certificate-bound weight files match SHA-256",
                "; ".join(certificate_file_issues),
            )
            recorded_weight_bytes = [
                certificate.size[key]
                for key in _CERTIFICATE_WEIGHT_BYTE_KEYS
                if key in certificate.size
            ]
            weight_bytes_ok = all(
                type(value) is int and value == manifest.weight_file_size_bytes
                for value in recorded_weight_bytes
            )
            add(
                "artifact.certificate-weight-bytes",
                weight_bytes_ok,
                (
                    "certificate has no optional candidate weight-byte field"
                    if not recorded_weight_bytes
                    else "certificate candidate weight bytes match the manifest"
                ),
                "certificate candidate weight bytes do not match the manifest",
            )

    return CertificationVerificationReport(
        certificate_path=str(source),
        certificate_sha256=certificate_digest,
        certificate_schema_version=schema_version,
        certificate_status=status,
        artifact_path=str(artifact) if artifact is not None else None,
        hub_repo_id=hub_repo_id,
        product_class=product_class,
        manifest_sha256=manifest_digest,
        plan_sha256=plan_digest,
        recomputed_main_bpw=recomputed_main_bpw,
        passed=not issues,
        checks=checks,
        issues=issues,
        verified_files=sorted(set(verified_files)),
    )
