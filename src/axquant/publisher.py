from __future__ import annotations

import re
import shutil
from pathlib import Path

import structlog
from huggingface_hub import HfApi

from axquant.errors import PublishingError
from axquant.release_audit import build_release_audit
from axquant.reporting import prepare_publication
from axquant.schema import ReleaseAudit, ReleaseAuditCheck, ReleaseValidationIndex
from axquant.serde import file_sha256, load_model

_LOG = structlog.get_logger()
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _require_release_audit(
    *,
    audit_path: str | Path | None,
    model_dir: Path,
    repo_id: str,
    validation_index_path: str | Path,
    hardware_registry_path: str | Path,
    pareto_report_path: str | Path,
) -> ReleaseAudit:
    if audit_path is None:
        raise PublishingError("executed publication requires --release-audit")
    source = Path(audit_path).expanduser().resolve()
    if not source.is_file():
        raise PublishingError(f"release audit does not exist: {source}")
    audit = load_model(source, ReleaseAudit)
    if not audit.release_ready:
        raise PublishingError("executed publication requires a release-ready M0-M8 audit")
    if audit.toolkit_version != "1.0.0":
        raise PublishingError("executed publication requires an exact AXQuant 1.0.0 audit")
    if audit.candidate_model.model_id != repo_id or not audit.candidate_model.revision:
        raise PublishingError("release audit candidate identity does not match the repository")
    candidate_path = audit.candidate_model.local_path
    if candidate_path is None or Path(candidate_path).expanduser().resolve() != model_dir:
        raise PublishingError(
            "release audit candidate path does not match the publication artifact"
        )

    checks: dict[str, ReleaseAuditCheck] = {check.gate_id: check for check in audit.checks}
    expected_bindings = {
        ("M1", "artifact_manifest"): model_dir / "axquant_manifest.json",
        ("M1", "plan"): model_dir / "axquant_plan.json",
        ("M2", "release_validation_index"): Path(validation_index_path).expanduser().resolve(),
        ("M7", "hardware_registry"): Path(hardware_registry_path).expanduser().resolve(),
        ("M7", "pareto_report"): Path(pareto_report_path).expanduser().resolve(),
    }
    for (gate_id, evidence_name), evidence_path in expected_bindings.items():
        expected_sha256 = checks[gate_id].evidence_sha256.get(evidence_name)
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
    target = model_dir / "release_audit.json"
    if target.is_file():
        if file_sha256(target) != file_sha256(source):
            raise PublishingError("packaged release_audit.json differs from the authorizing audit")
        return target
    if source != target.resolve():
        shutil.copy2(source, target)
    return target


def _rerun_release_audit(
    *,
    audit: ReleaseAudit,
    request_path: str | Path,
) -> None:
    source = Path(request_path).expanduser().resolve()
    if not source.is_file():
        raise PublishingError(f"release audit request does not exist: {source}")
    rerun = build_release_audit(source)
    if not rerun.release_ready:
        raise PublishingError("fresh M0-M8 release audit did not pass")
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
    execute: bool = False,
    private: bool = False,
) -> list[str]:
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise PublishingError(f"model directory does not exist: {directory}")
    if not _REPO_ID.fullmatch(repo_id):
        raise PublishingError("Hub repository must use the owner/name form")
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
        _rerun_release_audit(audit=audit, request_path=release_audit_request_path)
    prepare_publication(
        model_dir=directory,
        repo_id=repo_id,
        validation_index_path=validation_index_path,
        hardware_registry_path=hardware_registry_path,
        pareto_report_path=pareto_report_path,
    )
    if execute:
        _require_release_audit(
            audit_path=release_audit_path,
            model_dir=directory,
            repo_id=repo_id,
            validation_index_path=validation_index_path,
            hardware_registry_path=hardware_registry_path,
            pareto_report_path=pareto_report_path,
        )
        assert release_audit_path is not None
        _package_release_audit(release_audit_path, directory)
    validation_index = load_model(validation_index_path, ReleaseValidationIndex)
    if not validation_index.release_ready:
        raise PublishingError("publication requires a release-ready validation index")
    validation_models = {entry.candidate_model.model_id for entry in validation_index.entries}
    if validation_models != {repo_id}:
        raise PublishingError("release validation candidate does not match the repository")
    files = [
        path.relative_to(directory).as_posix()
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    ]
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
