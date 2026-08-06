from __future__ import annotations

from pathlib import Path

import pytest

from axquant import publisher
from axquant.errors import PublishingError
from axquant.publisher import (
    _copy_exact_publication_file,
    _package_release_audit,
    _require_release_audit,
    _require_release_validation,
    _rerun_release_audit,
    publish_model,
)
from axquant.schema import (
    DirectQualityEvaluation,
    DirectQualityTaskOutcome,
    DirectReleaseValidationIndex,
    DirectValidationEntry,
    ModelIdentity,
    ProfileName,
    QualityGenerationConfig,
    ReleaseAudit,
    ReleaseAuditCheck,
    ReleaseValidationEntry,
    ReleaseValidationIndex,
    SoftwareVersions,
)
from axquant.serde import file_sha256, load_model, write_data

_SOURCE_REVISION = "a" * 40
_REFERENCE_REVISION = "b" * 40
_CANDIDATE_REVISION = "c" * 40


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
            revision=_CANDIDATE_REVISION,
            local_path=str(artifact.resolve()),
        ),
        source_model=ModelIdentity(
            model_id="Qwen/Qwen3.6-test",
            revision=_SOURCE_REVISION,
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
    audit = _bind_release_ready_validation(audit_path, paths)
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
    with pytest.raises(PublishingError, match="differs from the authorizing source"):
        _package_release_audit(audit_path, artifact)


def test_packaged_publication_files_reject_directory_collisions(tmp_path: Path) -> None:
    audit_path, _paths = _release_audit(tmp_path)
    artifact = tmp_path / "artifact"
    audit_target = artifact / "release_audit.json"
    audit_target.mkdir()

    with pytest.raises(PublishingError, match="target is not a regular file"):
        _package_release_audit(audit_path, artifact)
    assert not (audit_target / audit_path.name).exists()

    registry_source = tmp_path / "registry.json"
    registry_source.write_text("{}\n", encoding="utf-8")
    registry_target = artifact / "certification" / "certified_checkpoint_registry.json"
    registry_target.mkdir(parents=True)
    with pytest.raises(PublishingError, match="target is not a regular file"):
        _copy_exact_publication_file(
            registry_source,
            registry_target,
            label="certification registry",
        )
    assert not (registry_target / registry_source.name).exists()


def _release_ready_validation_index(repo_id: str) -> ReleaseValidationIndex:
    candidate = ModelIdentity(model_id=repo_id, revision=_CANDIDATE_REVISION)
    reference = ModelIdentity(model_id="fixture/reference", revision=_REFERENCE_REVISION)
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


def _direct_release_ready_validation_index(
    tmp_path: Path,
    repo_id: str,
) -> tuple[Path, Path]:
    generation = QualityGenerationConfig(
        prompt_format="raw",
        max_sequence_length=128,
        max_generation_tokens=32,
    )
    software = SoftwareVersions(
        axquant="1.0.0",
        python="3.13",
        safetensors="0.6",
        pydantic="2",
    )
    entries: list[DirectValidationEntry] = []
    candidate_to_mutate: Path | None = None
    for profile in (ProfileName.AGENT_CODING, ProfileName.GENERAL):
        prefix = profile.value
        evaluation_manifest = tmp_path / f"{prefix}-evaluation-manifest.json"
        reference_path = tmp_path / f"{prefix}-reference.json"
        candidate_path = tmp_path / f"{prefix}-candidate.json"
        evaluation_manifest.write_text('{"fixture":"manifest"}\n', encoding="utf-8")
        common = {
            "profile": profile,
            "model_artifact_sha256": "a" * 64,
            "evaluation_manifest_sha256": file_sha256(evaluation_manifest),
            "dataset_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "generation": generation,
            "random_seed": 7,
            "evaluated_tokens": 1,
            "software_versions": software,
            "perplexity": 1.0,
            "outcomes": [
                DirectQualityTaskOutcome(
                    task_id=f"{prefix}-task",
                    score=1.0,
                    scored_tokens=1,
                    output_sha256="d" * 64,
                )
            ],
        }
        write_data(
            reference_path,
            DirectQualityEvaluation(
                model=ModelIdentity(
                    model_id="Qwen/reference",
                    revision=_REFERENCE_REVISION,
                ),
                **common,
            ),
        )
        write_data(
            candidate_path,
            DirectQualityEvaluation(
                model=ModelIdentity(model_id=repo_id, revision=_CANDIDATE_REVISION),
                **common,
            ),
        )
        entries.append(
            DirectValidationEntry(
                profile=profile,
                evaluation_manifest_file=evaluation_manifest.name,
                evaluation_manifest_sha256=file_sha256(evaluation_manifest),
                reference_evaluation_file=reference_path.name,
                reference_evaluation_sha256=file_sha256(reference_path),
                candidate_evaluation_file=candidate_path.name,
                candidate_evaluation_sha256=file_sha256(candidate_path),
                passed=True,
            )
        )
        candidate_to_mutate = candidate_path
    overlap = tmp_path / "general-overlap.json"
    overlap.write_text('{"fixture":"overlap"}\n', encoding="utf-8")
    index_path = tmp_path / "direct-validation-index.json"
    write_data(
        index_path,
        DirectReleaseValidationIndex(
            entries=entries,
            general_calibration_overlap_report_file=overlap.name,
            general_calibration_overlap_report_sha256=file_sha256(overlap),
            release_ready=True,
        ),
    )
    assert candidate_to_mutate is not None
    return index_path, candidate_to_mutate


def test_direct_publication_rejects_stale_validation_dependency(tmp_path: Path) -> None:
    repo_id = "AutomatosX/AXQuant-test"
    validation_path, candidate_path = _direct_release_ready_validation_index(tmp_path, repo_id)
    _require_release_validation(validation_index_path=validation_path, repo_id=repo_id)

    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PublishingError, match="checksum does not match its index"):
        _require_release_validation(validation_index_path=validation_path, repo_id=repo_id)


def _bind_release_ready_validation(
    audit_path: Path,
    paths: dict[str, Path],
) -> ReleaseAudit:
    repo_id = "AutomatosX/AXQuant-test"
    write_data(paths["release_validation_index"], _release_ready_validation_index(repo_id))
    audit = load_model(audit_path, ReleaseAudit)
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
    return audit


def test_executed_publication_reruns_full_audit_after_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, paths = _release_audit(tmp_path)
    audit = _bind_release_ready_validation(audit_path, paths)
    request_path = tmp_path / "release-audit-request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    prepared = False

    def prepare(**_kwargs: object) -> list[Path]:
        nonlocal prepared
        prepared = True
        return []

    def rebuild(_path: Path) -> ReleaseAudit:
        if prepared:
            return audit.model_copy(update={"wheel_sha256": "post-prepare-drift"})
        return audit

    monkeypatch.setattr(publisher, "prepare_publication", prepare)
    monkeypatch.setattr(publisher, "build_release_audit", rebuild)
    monkeypatch.setattr(
        publisher,
        "HfApi",
        lambda: pytest.fail("post-preparation audit drift must prevent Hub access"),
    )

    with pytest.raises(PublishingError, match="does not match a fresh audit rerun"):
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


def test_publication_rejects_symlinks_before_preparation_or_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (artifact / "leak.bin").symlink_to(outside)
    monkeypatch.setattr(
        publisher,
        "prepare_publication",
        lambda **_kwargs: pytest.fail("unsafe artifact must fail before preparation"),
    )

    with pytest.raises(PublishingError, match="artifact tree contains symlinks"):
        publish_model(
            model_dir=artifact,
            repo_id="AutomatosX/AXQuant-test",
            validation_index_path=tmp_path / "validation.json",
            hardware_registry_path=tmp_path / "hardware.json",
            pareto_report_path=tmp_path / "pareto.json",
        )


@pytest.mark.parametrize(
    ("execute", "private"),
    [
        ("false", False),
        (False, 1),
    ],
)
def test_publication_rejects_non_boolean_execution_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execute: object,
    private: object,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(
        publisher,
        "prepare_publication",
        lambda **_kwargs: pytest.fail("invalid controls must fail before preparation"),
    )

    with pytest.raises(PublishingError, match="controls must be booleans"):
        publish_model(
            model_dir=artifact,
            repo_id="AutomatosX/AXQuant-test",
            validation_index_path=tmp_path / "validation.json",
            hardware_registry_path=tmp_path / "hardware.json",
            pareto_report_path=tmp_path / "pareto.json",
            execute=execute,  # type: ignore[arg-type]
            private=private,  # type: ignore[arg-type]
        )


def test_executed_publication_uploads_only_after_every_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "AutomatosX/AXQuant-test"
    audit_path, paths = _release_audit(tmp_path)
    audit = _bind_release_ready_validation(audit_path, paths)

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


def test_flagship_package_rejects_legacy_request_downgrade(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "public-claim.json").write_text("{}\n", encoding="utf-8")
    legacy_request = tmp_path / "legacy-request.json"
    write_data(
        legacy_request,
        {
            "schema_version": "axquant.release-audit-request.v4",
        },
    )

    with pytest.raises(PublishingError, match="cannot be published through an older"):
        publish_model(
            model_dir=artifact,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-MP-5p30bpw-MTP",
            validation_index_path=tmp_path / "validation.json",
            hardware_registry_path=tmp_path / "hardware.json",
            pareto_report_path=tmp_path / "pareto.json",
            release_audit_request_path=legacy_request,
            execute=False,
        )


def test_flagship_preview_publication_preserves_certified_model_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FlagshipAudit:
        pass

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    readme = artifact / "README.md"
    readme.write_text(
        "# Certified flagship model card\n\nBound by release-audit M8.\n",
        encoding="utf-8",
    )
    certified_card = readme.read_bytes()
    (artifact / "public-claim.json").write_text("{}\n", encoding="utf-8")
    request = tmp_path / "flagship-request.json"
    write_data(
        request,
        {
            "schema_version": "axquant.flagship-release-audit-request.v1",
        },
    )

    monkeypatch.setattr(publisher, "FlagshipReleaseAudit", _FlagshipAudit)
    monkeypatch.setattr(
        publisher,
        "_require_release_audit",
        lambda **_kwargs: _FlagshipAudit(),
    )
    monkeypatch.setattr(
        publisher,
        "_require_flagship_request_inputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(publisher, "_rerun_release_audit", lambda **_kwargs: None)
    monkeypatch.setattr(publisher, "_require_release_validation", lambda **_kwargs: None)
    monkeypatch.setattr(
        publisher,
        "prepare_publication",
        lambda **_kwargs: pytest.fail("flagship publication must skip legacy preparation"),
    )

    files = publish_model(
        model_dir=artifact,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-MP-5p30bpw-MTP",
        validation_index_path=tmp_path / "validation.json",
        hardware_registry_path=tmp_path / "hardware.json",
        pareto_report_path=tmp_path / "pareto.json",
        release_audit_request_path=request,
        execute=False,
    )

    assert "README.md" in files
    assert readme.read_bytes() == certified_card
