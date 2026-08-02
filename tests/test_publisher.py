from __future__ import annotations

from pathlib import Path

import pytest

from axquant import publisher
from axquant.errors import PublishingError
from axquant.publisher import (
    _package_release_audit,
    _require_release_audit,
    _rerun_release_audit,
    publish_model,
)
from axquant.schema import (
    ModelIdentity,
    ProfileName,
    ReleaseAudit,
    ReleaseAuditCheck,
    ReleaseValidationEntry,
    ReleaseValidationIndex,
)
from axquant.serde import file_sha256, load_model, write_data


def _release_audit(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    paths = {
        "artifact_manifest": artifact / "axquant_manifest.json",
        "plan": artifact / "axquant_plan.json",
        "release_validation_index": tmp_path / "release-validation-index.json",
        "hardware_registry": tmp_path / "hardware-registry.json",
        "pareto_report": tmp_path / "pareto.json",
    }
    for name, path in paths.items():
        path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
    evidence = {
        "M1": {
            "artifact_manifest": file_sha256(paths["artifact_manifest"]),
            "plan": file_sha256(paths["plan"]),
        },
        "M2": {
            "release_validation_index": file_sha256(paths["release_validation_index"]),
        },
        "M7": {
            "hardware_registry": file_sha256(paths["hardware_registry"]),
            "pareto_report": file_sha256(paths["pareto_report"]),
        },
    }
    checks = [
        ReleaseAuditCheck(
            gate_id=gate_id,
            name=f"{gate_id} fixture",
            passed=True,
            evidence_sha256=evidence.get(gate_id, {}),
        )
        for gate_id in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
    ]
    audit = ReleaseAudit(
        request_sha256="request",
        candidate_model=ModelIdentity(
            model_id="AutomatosX/AXQuant-test",
            revision="candidate-revision",
            local_path=str(artifact.resolve()),
        ),
        source_model=ModelIdentity(
            model_id="Qwen/Qwen3.6-test",
            revision="source-revision",
        ),
        toolkit_version="1.0.0",
        wheel_sha256="wheel",
        checks=checks,
        release_ready=True,
    )
    audit_path = tmp_path / "release-audit.json"
    write_data(audit_path, audit)
    return audit_path, paths


def test_executed_publication_requires_release_audit(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    with pytest.raises(PublishingError, match="requires --release-audit"):
        publish_model(
            model_dir=artifact,
            repo_id="AutomatosX/AXQuant-test",
            validation_index_path=tmp_path / "validation.json",
            hardware_registry_path=tmp_path / "hardware.json",
            pareto_report_path=tmp_path / "pareto.json",
            execute=True,
        )


def test_release_audit_gate_accepts_exact_bound_inputs(tmp_path: Path) -> None:
    audit_path, paths = _release_audit(tmp_path)

    audit = _require_release_audit(
        audit_path=audit_path,
        model_dir=tmp_path / "artifact",
        repo_id="AutomatosX/AXQuant-test",
        validation_index_path=paths["release_validation_index"],
        hardware_registry_path=paths["hardware_registry"],
        pareto_report_path=paths["pareto_report"],
    )

    assert audit.release_ready


def test_release_audit_gate_rejects_stale_evidence(tmp_path: Path) -> None:
    audit_path, paths = _release_audit(tmp_path)
    paths["pareto_report"].write_text('{"fixture":"changed"}\n', encoding="utf-8")

    with pytest.raises(PublishingError, match="M7 evidence is stale or mismatched: pareto_report"):
        _require_release_audit(
            audit_path=audit_path,
            model_dir=tmp_path / "artifact",
            repo_id="AutomatosX/AXQuant-test",
            validation_index_path=paths["release_validation_index"],
            hardware_registry_path=paths["hardware_registry"],
            pareto_report_path=paths["pareto_report"],
        )


def test_executed_publication_requires_release_audit_request(tmp_path: Path) -> None:
    audit_path, paths = _release_audit(tmp_path)

    with pytest.raises(PublishingError, match="requires --release-audit-request"):
        publish_model(
            model_dir=tmp_path / "artifact",
            repo_id="AutomatosX/AXQuant-test",
            validation_index_path=paths["release_validation_index"],
            hardware_registry_path=paths["hardware_registry"],
            pareto_report_path=paths["pareto_report"],
            release_audit_path=audit_path,
            execute=True,
        )


def test_release_audit_rerun_must_match_authorizing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, _paths = _release_audit(tmp_path)
    audit = load_model(audit_path, ReleaseAudit)
    request_path = tmp_path / "release-audit-request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    changed = audit.model_copy(update={"wheel_sha256": "another-wheel"})
    monkeypatch.setattr(publisher, "build_release_audit", lambda _path: changed)

    with pytest.raises(PublishingError, match="does not match a fresh audit rerun"):
        _rerun_release_audit(audit=audit, request_path=request_path)


def test_executed_publication_rechecks_audit_after_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, paths = _release_audit(tmp_path)
    audit = load_model(audit_path, ReleaseAudit)
    request_path = tmp_path / "release-audit-request.json"
    request_path.write_text("{}\n", encoding="utf-8")

    def mutate_audited_manifest(**_kwargs: object) -> list[Path]:
        paths["artifact_manifest"].write_text('{"fixture":"changed"}\n', encoding="utf-8")
        return []

    monkeypatch.setattr(publisher, "build_release_audit", lambda _path: audit)
    monkeypatch.setattr(publisher, "prepare_publication", mutate_audited_manifest)

    with pytest.raises(
        PublishingError,
        match="M1 evidence is stale or mismatched: artifact_manifest",
    ):
        publish_model(
            model_dir=tmp_path / "artifact",
            repo_id="AutomatosX/AXQuant-test",
            validation_index_path=paths["release_validation_index"],
            hardware_registry_path=paths["hardware_registry"],
            pareto_report_path=paths["pareto_report"],
            release_audit_path=audit_path,
            release_audit_request_path=request_path,
            execute=True,
        )


def test_authorizing_release_audit_is_packaged_without_overwrite(tmp_path: Path) -> None:
    audit_path, _paths = _release_audit(tmp_path)
    artifact = tmp_path / "artifact"

    packaged = _package_release_audit(audit_path, artifact)

    assert packaged == artifact / "release_audit.json"
    assert file_sha256(packaged) == file_sha256(audit_path)

    packaged.write_text('{"fixture":"different"}\n', encoding="utf-8")
    with pytest.raises(PublishingError, match="differs from the authorizing audit"):
        _package_release_audit(audit_path, artifact)


def _release_ready_validation_index(repo_id: str) -> ReleaseValidationIndex:
    candidate = ModelIdentity(model_id=repo_id, revision="candidate-revision")
    reference = ModelIdentity(model_id="fixture/reference", revision="rev")
    return ReleaseValidationIndex(
        entries=[
            ReleaseValidationEntry(
                profile=profile,
                validation_file=f"{profile.value}-validation.json",
                validation_sha256=digest * 64,
                benchmark_index_file=f"{profile.value}-benchmark.json",
                benchmark_index_sha256=digest * 64,
                reference_model=reference,
                candidate_model=candidate,
                dataset_sha256=digest * 64,
                passed=True,
            )
            for profile, digest in ((ProfileName.AGENT_CODING, "a"), (ProfileName.GENERAL, "b"))
        ],
        release_ready=True,
        issues=[],
    )


def test_dry_run_publication_never_touches_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # execute defaults to False; the dry-run/preview path must never
    # construct HfApi, regardless of how far preparation gets.
    calls: list[str] = []
    monkeypatch.setattr(
        publisher,
        "HfApi",
        lambda: calls.append("HfApi") or pytest.fail("dry run must not construct HfApi"),
    )

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    repo_id = "AutomatosX/AXQuant-test"
    validation_index_path = tmp_path / "validation-index.json"
    write_data(validation_index_path, _release_ready_validation_index(repo_id))
    monkeypatch.setattr(publisher, "prepare_publication", lambda **_kwargs: [])

    files = publish_model(
        model_dir=artifact,
        repo_id=repo_id,
        validation_index_path=validation_index_path,
        hardware_registry_path=tmp_path / "hardware.json",
        pareto_report_path=tmp_path / "pareto.json",
        execute=False,
    )

    assert calls == []
    assert files == []


def test_executed_publication_uploads_only_after_every_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "AutomatosX/AXQuant-test"
    audit_path, paths = _release_audit(tmp_path)
    audit = load_model(audit_path, ReleaseAudit)

    # Replace the placeholder release-validation-index fixture with a real,
    # schema-valid ReleaseValidationIndex (publish_model loads and checks it
    # directly, unlike the other M-gate evidence files which stay opaque
    # bytes bound only by checksum), then re-bind the audit's M2 checksum to
    # match the new content.
    write_data(paths["release_validation_index"], _release_ready_validation_index(repo_id))
    rebound_checks = [
        check.model_copy(
            update={
                "evidence_sha256": {
                    **check.evidence_sha256,
                    "release_validation_index": file_sha256(paths["release_validation_index"]),
                }
            }
        )
        if check.gate_id == "M2"
        else check
        for check in audit.checks
    ]
    audit = audit.model_copy(update={"checks": rebound_checks})
    write_data(audit_path, audit)

    request_path = tmp_path / "release-audit-request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "build_release_audit", lambda _path: audit)
    monkeypatch.setattr(publisher, "prepare_publication", lambda **_kwargs: [])

    calls: dict[str, dict[str, object]] = {}

    class _FakeHfApi:
        def create_repo(self, **kwargs: object) -> None:
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs: object) -> None:
            calls["upload_folder"] = kwargs

    monkeypatch.setattr(publisher, "HfApi", _FakeHfApi)

    files = publish_model(
        model_dir=tmp_path / "artifact",
        repo_id=repo_id,
        validation_index_path=paths["release_validation_index"],
        hardware_registry_path=paths["hardware_registry"],
        pareto_report_path=paths["pareto_report"],
        release_audit_path=audit_path,
        release_audit_request_path=request_path,
        execute=True,
    )

    assert calls["create_repo"]["repo_id"] == repo_id
    assert calls["upload_folder"]["repo_id"] == repo_id
    assert calls["upload_folder"]["folder_path"] == tmp_path / "artifact"
    assert isinstance(files, list)
