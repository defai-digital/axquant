from __future__ import annotations

from pathlib import Path

import pytest

from axquant.artifact_evidence_binding import (
    ARTIFACT_EVIDENCE_BINDING_FILENAME,
    artifact_evidence_binding_issues,
    load_artifact_evidence_binding,
    refresh_artifact_evidence_binding,
    write_artifact_evidence_binding,
)
from axquant.errors import ArtifactError
from axquant.schema import (
    ArtifactEvidenceBinding,
    ArtifactManifest,
    AxEngineOptimizationMetadata,
    EvidenceKind,
    ModelIdentity,
    MtpPolicy,
    MtpRuntimeMetadata,
    OptimizationScope,
    PrecisionShare,
    ProfileName,
    RuntimeMetadata,
    RuntimeName,
    RuntimeProfile,
    RuntimeSupportLevel,
    SoftwareVersions,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

_SOURCE_REVISION = "a" * 40


def _manifest(target_class: str = "mixed-4.8bpw") -> ArtifactManifest:
    return ArtifactManifest(
        axquant_version="1.9.0",
        source_model=ModelIdentity(
            model_id="Qwen/Qwen3.6-test",
            revision=_SOURCE_REVISION,
        ),
        plan_sha256="b" * 64,
        profile=ProfileName.AGENT_CODING,
        target_class=target_class,
        effective_bpw=4.8,
        logical_parameters=8,
        main_logical_parameters=8,
        weight_file_size_bytes=8,
        main_weight_file_size_bytes=8,
        mtp_weight_file_size_bytes=0,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=8.0,
        measured_main_bpw=8.0,
        weight_distribution={"4bit": PrecisionShare(parameters=8, fraction=1.0)},
        mtp_distribution={},
        mtp_present=False,
        mtp_policy=MtpPolicy(mode="disabled", candidate_bits=(16,), min_bits=16),
        runtime=RuntimeMetadata(
            primary_runtime=RuntimeProfile(
                name=RuntimeName.AX_ENGINE,
                compatibility_level="A",
                support_level=RuntimeSupportLevel.OPTIMIZED,
                standard_inference=True,
                mtp_support="none",
            ),
            compatible_runtimes=[],
            optimization_scope=OptimizationScope.TEXT_PATH,
            mtp=MtpRuntimeMetadata(detected=False),
            ax_engine=AxEngineOptimizationMetadata(),
        ),
        software_versions=SoftwareVersions(
            axquant="1.9.0",
            python="3.13",
            safetensors="0.6",
            pydantic="2",
        ),
        files=[],
    )


def _certificate(path: Path) -> Path:
    path.write_text('{"fixture": "certificate"}\n', encoding="utf-8")
    return path


def test_sidecar_round_trip_binds_manifest_and_certificates(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(artifact / "axquant_manifest.json", _manifest())
    tier1 = _certificate(tmp_path / "tier1-cert.json")
    tier2 = _certificate(tmp_path / "tier2-cert.json")

    binding = write_artifact_evidence_binding(
        artifact_directory=artifact,
        evidence_kind=EvidenceKind.MEASURED,
        tier1_certificate_path=tier1,
        tier2_certificate_path=tier2,
    )

    assert binding.schema_version == "axquant.artifact-evidence-binding.v1"
    assert binding.artifact_manifest_sha256 == stable_sha256(
        load_model(artifact / "axquant_manifest.json", ArtifactManifest)
    )
    assert binding.tier1_certificate_sha256 == file_sha256(tier1)
    assert binding.tier2_certificate_sha256 == file_sha256(tier2)
    assert binding.evidence_kind is EvidenceKind.MEASURED

    loaded = load_artifact_evidence_binding(artifact)
    assert loaded == binding
    reloaded = load_model(
        artifact / ARTIFACT_EVIDENCE_BINDING_FILENAME,
        ArtifactEvidenceBinding,
    )
    assert reloaded == binding
    assert artifact_evidence_binding_issues(artifact, _manifest()) == []


def test_binding_issues_reject_a_missing_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(artifact / "axquant_manifest.json", _manifest())

    assert artifact_evidence_binding_issues(artifact, _manifest()) == [
        f"artifact evidence binding sidecar is missing: {ARTIFACT_EVIDENCE_BINDING_FILENAME}"
    ]
    assert load_artifact_evidence_binding(artifact) is None


def test_binding_issues_reject_a_manifest_digest_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(artifact / "axquant_manifest.json", _manifest())
    write_artifact_evidence_binding(
        artifact_directory=artifact,
        evidence_kind=EvidenceKind.MEASURED,
        tier1_certificate_path=_certificate(tmp_path / "tier1-cert.json"),
    )

    changed = _manifest(target_class="mixed-5.0bpw")
    write_data(artifact / "axquant_manifest.json", changed)
    issues = artifact_evidence_binding_issues(artifact, changed)

    assert issues == [
        "artifact evidence binding manifest digest does not match axquant_manifest.json"
    ]


def test_binding_issues_reject_a_sidecar_without_certificates(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = _manifest()
    write_data(artifact / "axquant_manifest.json", manifest)
    write_data(
        artifact / ARTIFACT_EVIDENCE_BINDING_FILENAME,
        ArtifactEvidenceBinding(
            artifact_manifest_sha256=stable_sha256(manifest),
            evidence_kind=EvidenceKind.MEASURED,
        ),
    )

    assert artifact_evidence_binding_issues(artifact, manifest) == [
        "artifact evidence binding carries no certificate binding"
    ]


def test_binding_issues_reject_an_invalid_sidecar(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = _manifest()
    write_data(artifact / "axquant_manifest.json", manifest)
    (artifact / ARTIFACT_EVIDENCE_BINDING_FILENAME).write_text("{not json\n", encoding="utf-8")

    issues = artifact_evidence_binding_issues(artifact, manifest)

    assert len(issues) == 1
    assert issues[0].startswith("artifact evidence binding sidecar is invalid:")


def test_writer_requires_at_least_one_certificate(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(artifact / "axquant_manifest.json", _manifest())

    with pytest.raises(ArtifactError, match="at least one certificate path is required"):
        write_artifact_evidence_binding(
            artifact_directory=artifact,
            evidence_kind=EvidenceKind.MEASURED,
        )


def test_writer_rejects_a_missing_certificate_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(artifact / "axquant_manifest.json", _manifest())

    with pytest.raises(ArtifactError, match="tier2 certificate file does not exist"):
        write_artifact_evidence_binding(
            artifact_directory=artifact,
            evidence_kind=EvidenceKind.MEASURED,
            tier2_certificate_path=tmp_path / "missing-cert.json",
        )


def test_refresh_rebinds_the_manifest_digest_and_preserves_certificates(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(artifact / "axquant_manifest.json", _manifest())
    tier1 = _certificate(tmp_path / "tier1-cert.json")
    original = write_artifact_evidence_binding(
        artifact_directory=artifact,
        evidence_kind=EvidenceKind.MEASURED,
        tier1_certificate_path=tier1,
    )

    changed = _manifest(target_class="mixed-5.0bpw")
    write_data(artifact / "axquant_manifest.json", changed)
    refreshed = refresh_artifact_evidence_binding(
        artifact_directory=artifact,
        manifest=changed,
    )

    assert refreshed is not None
    assert refreshed.artifact_manifest_sha256 == stable_sha256(changed)
    assert refreshed.tier1_certificate_sha256 == original.tier1_certificate_sha256
    assert refreshed.tier2_certificate_sha256 is None
    assert refreshed.evidence_kind is EvidenceKind.MEASURED
    assert refreshed.created_at >= original.created_at
    assert artifact_evidence_binding_issues(artifact, changed) == []


def test_refresh_without_a_sidecar_is_a_no_op(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = _manifest()
    write_data(artifact / "axquant_manifest.json", manifest)

    assert refresh_artifact_evidence_binding(artifact_directory=artifact, manifest=manifest) is None
    assert not (artifact / ARTIFACT_EVIDENCE_BINDING_FILENAME).exists()
