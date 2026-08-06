from __future__ import annotations

import re
import shutil
from pathlib import Path

import structlog
from huggingface_hub import HfApi
from pydantic import ValidationError

from axquant.artifact_paths import artifact_tree_files
from axquant.certification.common import bound_file
from axquant.certification.dispatch import (
    CertificationAudit,
    build_certification_audit,
    load_certification_audit,
    load_certification_request,
)
from axquant.certification.packaging import prepare_direct_publication
from axquant.certification.registry import (
    DIRECT_CERTIFICATION_ALLOWED_CLAIMS,
    append_certified_checkpoint,
)
from axquant.errors import AxquantError, PublishingError
from axquant.lifecycle import require_active_certification
from axquant.release_audit import build_release_audit
from axquant.reporting import prepare_publication
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    ArtifactLifecycleRegistry,
    ArtifactManifest,
    DirectQualityEvaluation,
    DirectReleaseValidationIndex,
    FlagshipReleaseAudit,
    FlagshipReleaseAuditRequest,
    PublicClaimManifest,
    Qwen3NextReleaseAudit,
    Qwen3NextReleaseAuditCheck,
    Qwen3NextReleaseAuditRequest,
    ReleaseAudit,
    ReleaseAuditCheck,
    ReleaseAuditRequest,
    ReleaseValidationIndex,
)
from axquant.serde import file_sha256, load_model, read_data, stable_sha256

_LOG = structlog.get_logger()
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_TEXT_PUBLICATION_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
_MAX_PUBLIC_TEXT_SCAN_BYTES = 10_000_000
_LARGE_MODEL_TEXT_FILES = {"tokenizer.json"}
_SENSITIVE_CONTENT_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "macOS home path": re.compile(r"(?<![A-Za-z0-9_])/Users/[^/\s\"']+"),
    "macOS volume path": re.compile(r"(?<![A-Za-z0-9_])/Volumes/[^/\s\"']+"),
    "Linux home path": re.compile(r"(?<![A-Za-z0-9_])/home/[^/\s\"']+"),
    "macOS private temporary path": re.compile(r"(?<![A-Za-z0-9_])/private/var/"),
    "POSIX temporary path": re.compile(r"(?<![A-Za-z0-9_])/(?:private/)?(?:var/)?tmp/"),
    "Windows home path": re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s\"']+"),
    "private IPv4 address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
    "credential assignment": re.compile(
        r"(?i)(?:\"(?:token|password|secret|api[_-]?key)\"|"
        r"\b(?:token|password|secret|api[_-]?key))\s*[:=]\s*"
        r"(?!\"?(?:null|none|redacted|<redacted>)\b)[\"']?[A-Za-z0-9_./+-]{8,}"
    ),
}


def _publication_files(directory: Path) -> list[Path]:
    try:
        return artifact_tree_files(directory)
    except ValueError as exc:
        raise PublishingError(str(exc)) from exc


def publication_privacy_issues(directory: Path) -> list[str]:
    issues: list[str] = []
    for path in _publication_files(directory):
        relative = path.relative_to(directory)
        lowered_parts = [part.casefold() for part in relative.parts]
        lowered_name = relative.name.casefold()
        sensitive_stem = Path(lowered_name).stem
        if (
            lowered_name == ".env"
            or lowered_name.startswith(".env.")
            or sensitive_stem in {"credentials", "secrets"}
            or lowered_name in {"id_rsa", "id_ed25519", "private_key", "private-key"}
        ):
            issues.append(f"sensitive publication filename: {relative.as_posix()}")
        if "formal" in lowered_parts and "raw" in lowered_parts:
            issues.append(f"formal raw evidence must not be published: {relative.as_posix()}")
        if path.suffix.casefold() not in _TEXT_PUBLICATION_SUFFIXES:
            continue
        if (
            path.stat().st_size > _MAX_PUBLIC_TEXT_SCAN_BYTES
            and lowered_name not in _LARGE_MODEL_TEXT_FILES
        ):
            issues.append(
                f"text publication file exceeds privacy-scan limit: {relative.as_posix()}"
            )
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"text publication file is not valid UTF-8: {relative.as_posix()}")
            continue
        for label, pattern in _SENSITIVE_CONTENT_PATTERNS.items():
            if pattern.search(content):
                issues.append(f"{label} found in publication file: {relative.as_posix()}")
    return sorted(set(issues))


def require_publication_privacy(directory: Path) -> None:
    issues = publication_privacy_issues(directory)
    if issues:
        raise PublishingError("publication privacy scan failed: " + "; ".join(issues))


def _copy_exact_publication_file(source: Path, target: Path, *, label: str) -> Path:
    """Copy a packaged evidence file without overwriting or nesting into collisions."""

    if not source.is_file():
        raise PublishingError(f"{label} source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise PublishingError(f"{label} target is not a regular file: {target}")
        if file_sha256(target) != file_sha256(source):
            raise PublishingError(f"packaged {label} differs from the authorizing source")
        return target
    if source != target.resolve():
        shutil.copy2(source, target)
    return target


def _load_release_validation(
    *,
    validation_index_path: str | Path,
    repo_id: str,
) -> None:
    """Validate the exact release index and its direct-track dependencies."""

    source = Path(validation_index_path).expanduser().resolve()
    validation_payload = read_data(source)
    if (
        isinstance(validation_payload, dict)
        and validation_payload.get("schema_version") == "axquant.direct-release-validation-index.v1"
    ):
        direct_validation = load_model(source, DirectReleaseValidationIndex)
        if not direct_validation.release_ready:
            raise PublishingError("publication requires a release-ready validation index")
        for entry in direct_validation.entries:
            bound_file(
                source.parent,
                entry.evaluation_manifest_file,
                entry.evaluation_manifest_sha256,
                f"{entry.profile.value} evaluation manifest",
            )
            bound_file(
                source.parent,
                entry.reference_evaluation_file,
                entry.reference_evaluation_sha256,
                f"{entry.profile.value} reference evaluation",
            )
            candidate_path = bound_file(
                source.parent,
                entry.candidate_evaluation_file,
                entry.candidate_evaluation_sha256,
                f"{entry.profile.value} candidate evaluation",
            )
            candidate = load_model(candidate_path, DirectQualityEvaluation)
            if candidate.model.model_id != repo_id:
                raise PublishingError("release validation candidate does not match the repository")
        bound_file(
            source.parent,
            direct_validation.general_calibration_overlap_report_file,
            direct_validation.general_calibration_overlap_report_sha256,
            "general calibration overlap report",
        )
        return

    release_validation = load_model(source, ReleaseValidationIndex)
    if not release_validation.release_ready:
        raise PublishingError("publication requires a release-ready validation index")
    validation_models = {entry.candidate_model.model_id for entry in release_validation.entries}
    if validation_models != {repo_id}:
        raise PublishingError("release validation candidate does not match the repository")


def _require_release_validation(
    *,
    validation_index_path: str | Path,
    repo_id: str,
) -> None:
    try:
        _load_release_validation(
            validation_index_path=validation_index_path,
            repo_id=repo_id,
        )
    except PublishingError:
        raise
    except (AxquantError, ValidationError, OSError, ValueError) as exc:
        raise PublishingError(f"release validation is invalid: {exc}") from exc


def _require_direct_request_inputs(
    *,
    request_path: str | Path,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
) -> Qwen3NextReleaseAuditRequest:
    source = Path(request_path).expanduser().resolve()
    request = load_certification_request(source)
    if not isinstance(request, Qwen3NextReleaseAuditRequest):
        raise PublishingError("direct publication requires a direct certification request")
    expected = {
        "validation index": (source.parent / request.release_validation_index).resolve(),
        "hardware registry": (source.parent / request.hardware_registry).resolve(),
        "Pareto report": (source.parent / request.pareto_report).resolve(),
    }
    supplied = {
        "validation index": Path(validation_index_path).expanduser().resolve(),
        "hardware registry": Path(hardware_registry_path).expanduser().resolve(),
        "Pareto report": Path(pareto_report_path).expanduser().resolve(),
    }
    mismatched = [label for label in expected if expected[label] != supplied[label]]
    if mismatched:
        raise PublishingError(
            f"direct publication arguments do not match its request: {sorted(mismatched)}"
        )
    return request


def _require_release_audit(
    *,
    audit_path: str | Path | None,
    model_dir: Path,
    repo_id: str,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
) -> CertificationAudit:
    if audit_path is None:
        raise PublishingError("executed publication requires --release-audit")
    source = Path(audit_path).expanduser().resolve()
    if not source.is_file():
        raise PublishingError(f"release audit does not exist: {source}")
    audit = load_certification_audit(source)
    if isinstance(audit, FlagshipReleaseAudit):
        if not audit.release_ready:
            raise PublishingError(
                "executed flagship publication requires a release-ready qwen36-mtp-v2 audit"
            )
        if audit.candidate_model.model_id != repo_id or not is_immutable_revision(
            audit.candidate_model.revision
        ):
            raise PublishingError("flagship audit candidate identity does not match repository")
        candidate_path = audit.candidate_model.local_path
        if candidate_path is None or Path(candidate_path).expanduser().resolve() != model_dir:
            raise PublishingError(
                "flagship audit candidate path does not match the publication artifact"
            )
        return audit
    if isinstance(audit, Qwen3NextReleaseAudit):
        if not audit.release_ready:
            raise PublishingError("executed publication requires a release-ready N0-N8 audit")
        if audit.candidate_model.model_id != repo_id or not is_immutable_revision(
            audit.candidate_model.revision
        ):
            raise PublishingError("release audit candidate identity does not match the repository")
        candidate_path = audit.candidate_model.local_path
        if candidate_path is None or Path(candidate_path).expanduser().resolve() != model_dir:
            raise PublishingError(
                "release audit candidate path does not match the publication artifact"
            )
        direct_checks: dict[str, Qwen3NextReleaseAuditCheck] = {
            check.gate_id.value: check for check in audit.checks
        }
        expected_bindings = {
            ("N1", "artifact_manifest"): model_dir / "axquant_manifest.json",
            ("N1", "plan"): model_dir / "axquant_plan.json",
            ("N4", "release_validation_index"): Path(validation_index_path).expanduser().resolve(),
            ("N7", "hardware_registry"): Path(hardware_registry_path).expanduser().resolve(),
            ("N7", "pareto_report"): Path(pareto_report_path).expanduser().resolve(),
        }
        for (gate_id, evidence_name), evidence_path in expected_bindings.items():
            expected_sha256 = direct_checks[gate_id].evidence_sha256.get(evidence_name)
            if (
                not evidence_path.is_file()
                or expected_sha256 is None
                or file_sha256(evidence_path) != expected_sha256
            ):
                raise PublishingError(
                    f"release audit {gate_id} evidence is stale or mismatched: {evidence_name}"
                )
        return audit

    if not audit.release_ready:
        raise PublishingError("executed publication requires a release-ready M0-M8 audit")
    if audit.toolkit_version != "1.0.0":
        raise PublishingError("executed publication requires an exact AXQuant 1.0.0 audit")
    if audit.candidate_model.model_id != repo_id or not is_immutable_revision(
        audit.candidate_model.revision
    ):
        raise PublishingError("release audit candidate identity does not match the repository")
    candidate_path = audit.candidate_model.local_path
    if candidate_path is None or Path(candidate_path).expanduser().resolve() != model_dir:
        raise PublishingError(
            "release audit candidate path does not match the publication artifact"
        )

    mtp_checks: dict[str, ReleaseAuditCheck] = {check.gate_id: check for check in audit.checks}
    expected_bindings = {
        ("M1", "artifact_manifest"): model_dir / "axquant_manifest.json",
        ("M1", "plan"): model_dir / "axquant_plan.json",
        ("M2", "release_validation_index"): Path(validation_index_path).expanduser().resolve(),
        ("M7", "hardware_registry"): Path(hardware_registry_path).expanduser().resolve(),
        ("M7", "pareto_report"): Path(pareto_report_path).expanduser().resolve(),
    }
    for (gate_id, evidence_name), evidence_path in expected_bindings.items():
        expected_sha256 = mtp_checks[gate_id].evidence_sha256.get(evidence_name)
        if (
            not evidence_path.is_file()
            or expected_sha256 is None
            or file_sha256(evidence_path) != expected_sha256
        ):
            raise PublishingError(
                f"release audit {gate_id} evidence is stale or mismatched: {evidence_name}"
            )
    return audit


def _package_release_audit(audit_path: str | Path, model_dir: Path) -> Path:
    source = Path(audit_path).expanduser().resolve()
    audit = load_certification_audit(source)
    _publication_files(model_dir)
    target = (
        model_dir / "certification" / "audit.json"
        if isinstance(audit, Qwen3NextReleaseAudit)
        else (
            model_dir / "certification" / "flagship-release-audit.json"
            if isinstance(audit, FlagshipReleaseAudit)
            else model_dir / "release_audit.json"
        )
    )
    return _copy_exact_publication_file(source, target, label="release audit")


def _rerun_release_audit(
    *,
    audit: CertificationAudit,
    request_path: str | Path,
) -> None:
    source = Path(request_path).expanduser().resolve()
    if not source.is_file():
        raise PublishingError(f"release audit request does not exist: {source}")
    if isinstance(audit, Qwen3NextReleaseAudit):
        request = load_certification_request(source)
        if not isinstance(request, Qwen3NextReleaseAuditRequest):
            raise PublishingError("release audit and request use different certification tracks")
    elif isinstance(audit, FlagshipReleaseAudit):
        request = load_certification_request(source)
        if not isinstance(request, FlagshipReleaseAuditRequest):
            raise PublishingError("flagship audit cannot be downgraded to an older request")
    rerun = (
        build_release_audit(source)
        if isinstance(audit, ReleaseAudit)
        else build_certification_audit(source)
    )
    if not rerun.release_ready:
        track = (
            "qwen36-mtp-v2 M0-M8"
            if isinstance(audit, FlagshipReleaseAudit)
            else ("M0-M8" if isinstance(audit, ReleaseAudit) else "N0-N8")
        )
        raise PublishingError(f"fresh {track} release audit did not pass")
    audit_payload = audit.model_dump(mode="json", exclude={"created_at"})
    rerun_payload = rerun.model_dump(mode="json", exclude={"created_at"})
    if rerun_payload != audit_payload:
        raise PublishingError("authorizing release audit does not match a fresh audit rerun")


def _require_flagship_request_inputs(
    *,
    request_path: str | Path,
    audit: FlagshipReleaseAudit,
    model_dir: Path,
    repo_id: str,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
) -> FlagshipReleaseAuditRequest:
    source = Path(request_path).expanduser().resolve()
    request = load_certification_request(source)
    if not isinstance(request, FlagshipReleaseAuditRequest):
        raise PublishingError("flagship publication requires a qwen36-mtp-v2 request")
    if (
        request.public_claim is None
        or request.lifecycle_registry is None
        or request.model_card is None
    ):
        raise PublishingError(
            "flagship publication requires final lifecycle, claim, and model card"
        )
    if audit.request_sha256 != stable_sha256(request):
        raise PublishingError("flagship audit does not bind the supplied final request")
    legacy_request_path = (source.parent / request.legacy_release_audit_request).resolve()
    legacy_request = load_model(legacy_request_path, ReleaseAuditRequest)

    def legacy_path(value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (legacy_request_path.parent / path).resolve()

    expected = {
        "validation index": legacy_path(legacy_request.release_validation_index),
        "hardware registry": legacy_path(legacy_request.hardware_registry),
        "Pareto report": legacy_path(legacy_request.pareto_report),
    }
    supplied = {
        "validation index": Path(validation_index_path).expanduser().resolve(),
        "hardware registry": Path(hardware_registry_path).expanduser().resolve(),
        "Pareto report": Path(pareto_report_path).expanduser().resolve(),
    }
    mismatched = [label for label in expected if expected[label] != supplied[label]]
    if mismatched:
        raise PublishingError(
            f"flagship publication arguments do not match its base request: {mismatched}"
        )
    claim_path = (source.parent / request.public_claim).resolve()
    lifecycle_path = (source.parent / request.lifecycle_registry).resolve()
    model_card_path = (source.parent / request.model_card).resolve()
    expected_package_paths = {
        "public claim": (claim_path, model_dir / "public-claim.json"),
        "lifecycle registry": (
            lifecycle_path,
            model_dir / "certification" / "lifecycle-registry.json",
        ),
        "model card": (model_card_path, model_dir / "README.md"),
    }
    for label, (actual, expected_path) in expected_package_paths.items():
        if actual != expected_path.resolve() or not actual.is_file():
            raise PublishingError(f"flagship {label} is not at its canonical package path")
    claim = load_model(claim_path, PublicClaimManifest)
    lifecycle = load_model(lifecycle_path, ArtifactLifecycleRegistry)
    event = require_active_certification(lifecycle, audit.candidate)
    if claim.candidate != audit.candidate:
        raise PublishingError("flagship public claim binds another candidate")
    if claim.lifecycle_event_sha256 != stable_sha256(event):
        raise PublishingError("flagship claim does not bind active certified lifecycle event")
    if claim.public_repository != repo_id or event.public_repository != repo_id:
        raise PublishingError("flagship measured-BPW repository identity differs from --repo")
    require_publication_privacy(model_dir)
    return request


def prepare_flagship_publication(
    *,
    model_dir: str | Path,
    repo_id: str,
    request_path: str | Path,
    audit_path: str | Path,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
) -> list[Path]:
    directory = Path(model_dir).expanduser().resolve()
    audit = _require_release_audit(
        audit_path=audit_path,
        model_dir=directory,
        repo_id=repo_id,
        validation_index_path=validation_index_path,
        hardware_registry_path=hardware_registry_path,
        pareto_report_path=pareto_report_path,
    )
    if not isinstance(audit, FlagshipReleaseAudit):
        raise PublishingError("qwen36-mtp-v2 package requires a flagship release audit")
    _require_flagship_request_inputs(
        request_path=request_path,
        audit=audit,
        model_dir=directory,
        repo_id=repo_id,
        validation_index_path=validation_index_path,
        hardware_registry_path=hardware_registry_path,
        pareto_report_path=pareto_report_path,
    )
    _rerun_release_audit(audit=audit, request_path=request_path)
    return _publication_files(directory)


def publish_model(
    *,
    model_dir: str | Path,
    repo_id: str,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
    release_audit_path: str | Path | None = None,
    release_audit_request_path: str | Path | None = None,
    certification_registry_path: str | Path | None = None,
    execute: bool = False,
    private: bool = False,
) -> list[str]:
    if type(execute) is not bool or type(private) is not bool:
        raise PublishingError("publication execute/private controls must be booleans")
    supplied_directory = Path(model_dir).expanduser()
    if supplied_directory.is_symlink():
        raise PublishingError("artifact root is a symlink")
    directory = supplied_directory.resolve()
    if not directory.is_dir():
        raise PublishingError(f"model directory does not exist: {directory}")
    _publication_files(directory)
    if not _REPO_ID.fullmatch(repo_id):
        raise PublishingError("Hub repository must use the owner/name form")
    request_schema: str | None = None
    if release_audit_request_path is not None:
        payload = read_data(release_audit_request_path)
        if isinstance(payload, dict):
            raw_schema = payload.get("schema_version")
            request_schema = raw_schema if isinstance(raw_schema, str) else None
    flagship_request = request_schema == "axquant.flagship-release-audit-request.v1"
    direct_request = request_schema == "axquant.qwen3-next-release-audit-request.v1"
    declares_flagship = (directory / "public-claim.json").exists() or (
        directory / "certification" / "lifecycle-registry.json"
    ).exists()
    if declares_flagship and not flagship_request:
        raise PublishingError(
            "flagship package cannot be published through an older certification request"
        )
    audit: CertificationAudit | None = None
    if execute or flagship_request:
        audit = _require_release_audit(
            audit_path=release_audit_path,
            model_dir=directory,
            repo_id=repo_id,
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto_report_path,
        )
        if release_audit_request_path is None:
            raise PublishingError("executed publication requires --release-audit-request")
        if isinstance(audit, Qwen3NextReleaseAudit) and certification_registry_path is None:
            raise PublishingError("executed non-MTP publication requires --certification-registry")
        # Executed publication reruns the audit after preparation so it binds
        # the exact artifact state that is uploaded.
    if flagship_request:
        if release_audit_request_path is None or not isinstance(audit, FlagshipReleaseAudit):
            raise PublishingError("flagship publication requires matching request and audit")
        _require_flagship_request_inputs(
            request_path=release_audit_request_path,
            audit=audit,
            model_dir=directory,
            repo_id=repo_id,
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto_report_path,
        )
    if flagship_request:
        if audit is None or release_audit_request_path is None:
            raise PublishingError("flagship publication requires matching request and audit")
        _rerun_release_audit(audit=audit, request_path=release_audit_request_path)
    elif direct_request:
        if release_audit_request_path is None:
            raise PublishingError("direct publication requires --release-audit-request")
        _require_direct_request_inputs(
            request_path=release_audit_request_path,
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto_report_path,
        )
    _require_release_validation(
        validation_index_path=validation_index_path,
        repo_id=repo_id,
    )
    if direct_request:
        if release_audit_request_path is None:
            raise PublishingError("direct publication requires --release-audit-request")
        prepare_direct_publication(
            model_dir=directory,
            repo_id=repo_id,
            request_path=release_audit_request_path,
        )
    elif not flagship_request:
        # Flagship packages are pre-assembled and validated above; the legacy
        # preparer would overwrite the certified model card (README.md).
        prepare_publication(
            model_dir=directory,
            repo_id=repo_id,
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto_report_path,
        )
    if execute:
        if audit is None:
            raise PublishingError("executed publication requires a release audit")
        if release_audit_request_path is None:
            raise PublishingError("executed publication requires --release-audit-request")
        _rerun_release_audit(audit=audit, request_path=release_audit_request_path)
        _require_release_audit(
            audit_path=release_audit_path,
            model_dir=directory,
            repo_id=repo_id,
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto_report_path,
        )
        if release_audit_path is None:
            raise PublishingError("executed publication requires --release-audit")
        if isinstance(audit, Qwen3NextReleaseAudit):
            if certification_registry_path is None:
                raise PublishingError(
                    "executed non-MTP publication requires --certification-registry"
                )
            manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
            registry = append_certified_checkpoint(
                registry_path=certification_registry_path,
                audit_path=release_audit_path,
                artifact_directory=directory,
                candidate_id=repo_id.split("/", maxsplit=1)[1],
                measured_bpw=manifest.measured_total_bpw,
                allowed_claims=list(DIRECT_CERTIFICATION_ALLOWED_CLAIMS),
            )
            registry_source = Path(certification_registry_path).expanduser().resolve()
            registry_target = directory / "certification" / "certified_checkpoint_registry.json"
            _copy_exact_publication_file(
                registry_source,
                registry_target,
                label="certification registry",
            )
            if not registry.entries:
                raise PublishingError("certification registry append produced no entry")
        _package_release_audit(release_audit_path, directory)
    _require_release_validation(
        validation_index_path=validation_index_path,
        repo_id=repo_id,
    )
    if flagship_request:
        # Audit packaging can add public text artifacts after the request-level
        # scan. Re-scan the exact final tree immediately before previewing or
        # uploading it.
        require_publication_privacy(directory)
    files = [path.relative_to(directory).as_posix() for path in _publication_files(directory)]
    if not execute:
        _LOG.info("publication_preview", repo=repo_id, files=len(files), private=private)
        return files
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=directory,
            commit_message="Publish AXQuant model artifact",
        )
    except Exception as exc:
        raise PublishingError(f"Hub publication failed: {exc}") from exc
    _LOG.info("publication_completed", repo=repo_id, files=len(files), private=private)
    return files
