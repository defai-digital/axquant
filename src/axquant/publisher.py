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
from axquant.release_audit import build_release_audit
from axquant.reporting import prepare_publication
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    ArtifactManifest,
    DirectQualityEvaluation,
    DirectReleaseValidationIndex,
    Qwen3NextReleaseAudit,
    Qwen3NextReleaseAuditCheck,
    Qwen3NextReleaseAuditRequest,
    ReleaseAudit,
    ReleaseAuditCheck,
    ReleaseValidationIndex,
)
from axquant.serde import file_sha256, load_model, read_data

_LOG = structlog.get_logger()
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _publication_files(directory: Path) -> list[Path]:
    try:
        return artifact_tree_files(directory)
    except ValueError as exc:
        raise PublishingError(str(exc)) from exc


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
        else model_dir / "release_audit.json"
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
    rerun = (
        build_release_audit(source)
        if isinstance(audit, ReleaseAudit)
        else build_certification_audit(source)
    )
    if not rerun.release_ready:
        track = "M0-M8" if isinstance(audit, ReleaseAudit) else "N0-N8"
        raise PublishingError(f"fresh {track} release audit did not pass")
    audit_payload = audit.model_dump(mode="json", exclude={"created_at"})
    rerun_payload = rerun.model_dump(mode="json", exclude={"created_at"})
    if rerun_payload != audit_payload:
        raise PublishingError("authorizing release audit does not match a fresh audit rerun")


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
    audit: CertificationAudit | None = None
    if execute:
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
        # The audit rerun happens once, after preparation, so it binds the
        # exact artifact state that is uploaded.
    direct_request = False
    if release_audit_request_path is not None:
        payload = read_data(release_audit_request_path)
        direct_request = (
            isinstance(payload, dict)
            and payload.get("schema_version") == "axquant.qwen3-next-release-audit-request.v1"
        )
    if direct_request:
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
    else:
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
