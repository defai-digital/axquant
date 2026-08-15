from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from axquant.analyzer import architecture_prior_report
from axquant.certification.verify import verify_certificate
from axquant.cli import main
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    CalibrationEvidence,
    EvidenceKind,
    PlanRequest,
    ProfileName,
    PublicCertArtifact,
    PublicCheckpointCertification,
    PublicIndexMeta,
    PublicMtpAccelerationBlock,
)
from axquant.serde import file_sha256, read_data, stable_sha256, write_data


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "tiny", "architectures": ["TinyForCausalLM"]}),
        encoding="utf-8",
    )
    save_file(
        {"model.layers.0.mlp.down_proj.weight": np.zeros((64, 64), dtype=np.float32)},
        source / "model.safetensors",
    )
    inventory = inspect_model(
        source,
        model_id="owner/Tiny-4B",
        revision="a" * 40,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.GENERAL)
    report = report.model_copy(
        update={
            "evidence_kind": EvidenceKind.MEASURED,
            "calibration": CalibrationEvidence(
                dataset_id="fixture/calibration",
                dataset_sha256="b" * 64,
                samples=8,
                domains=["general"],
                sequence_length=128,
                backend="fixture",
                reference="fixture-cache",
            ),
        }
    )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            target_bpw=4.8,
            allow_unmeasured=True,
        ),
    )

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny",
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": np.zeros((64, 8), dtype=np.uint32),
            "model.layers.0.mlp.down_proj.scales": np.ones((64, 1), dtype=np.float32),
            "model.layers.0.mlp.down_proj.biases": np.zeros((64, 1), dtype=np.float32),
        },
        artifact / "model.safetensors",
    )
    write_data(artifact / "axquant_plan.json", plan)
    runtime = build_runtime_metadata(plan, artifact)
    weight_path = artifact / "model.safetensors"
    weight_bytes = weight_path.stat().st_size
    logical_parameters = sum(allocation.parameters for allocation in plan.assignments)
    measured_bpw = 8.0 * weight_bytes / logical_parameters
    manifest = ArtifactManifest(
        axquant_version="1.8.0",
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        calibration=plan.calibration,
        profile=plan.profile,
        target_class=plan.target_class,
        effective_bpw=plan.effective_bpw,
        logical_parameters=logical_parameters,
        main_logical_parameters=logical_parameters,
        weight_file_size_bytes=weight_bytes,
        main_weight_file_size_bytes=weight_bytes,
        mtp_weight_file_size_bytes=0,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=measured_bpw,
        measured_main_bpw=measured_bpw,
        weight_distribution=plan.weight_distribution,
        mtp_distribution=plan.mtp_distribution,
        mtp_present=False,
        mtp_policy=plan.mtp,
        runtime=runtime,
        software_versions=plan.software_versions,
        files=[
            ArtifactFile(
                path=weight_path.name,
                size_bytes=weight_bytes,
                sha256=file_sha256(weight_path),
            )
        ],
    )
    manifest_path = artifact / "axquant_manifest.json"
    write_data(manifest_path, manifest)

    certificate = PublicCheckpointCertification(
        status="certified",
        host_id="df-macbookpro-m5",
        certified_at="2026-08-14T12:00:00+00:00",
        artifact=PublicCertArtifact(
            hub_repo_id="owner/AX-Tiny-4B-MLX-AXQ-4bit",
            hub_commit="c" * 40,
            product_class="4bit",
            candidate_manifest_sha256=file_sha256(manifest_path),
        ),
        plan={"evidence_kind": "measured", "target_class": "4bit"},
        size={
            "candidate_weight_bytes": weight_bytes,
            "measured_main_bpw": measured_bpw,
        },
        quality={"general": {"retention": 1.0}},
        thresholds={"minimum_quality_retention": 0.98},
        mtp_acceleration=PublicMtpAccelerationBlock(status="not-applicable"),
        toolchain={"axquant": "1.8.0"},
        public_index=PublicIndexMeta(
            display_name="Tiny 4B AXQ 4-bit",
            sort_order=1,
            edition_label="fixture",
        ),
    )
    certificate_path = tmp_path / "checkpoint-tier1.json"
    write_data(certificate_path, certificate)
    return certificate_path, artifact


def test_consistent_certificate_bundle_passes(tmp_path: Path) -> None:
    certificate, artifact = _bundle(tmp_path)

    report = verify_certificate(certificate_path=certificate, artifact_dir=artifact)

    assert report.passed is True
    assert report.issues == []
    assert report.recomputed_main_bpw is not None


def test_manifest_digest_tamper_fails(tmp_path: Path) -> None:
    certificate, artifact = _bundle(tmp_path)
    payload = read_data(certificate)
    payload["artifact"]["candidate_manifest_sha256"] = "d" * 64
    write_data(certificate, payload)

    report = verify_certificate(certificate_path=certificate, artifact_dir=artifact)

    assert report.passed is False
    assert any("manifest digest" in issue for issue in report.issues)


def test_class_and_repository_mismatch_fails(tmp_path: Path) -> None:
    certificate, artifact = _bundle(tmp_path)
    payload = read_data(certificate)
    payload["artifact"]["product_class"] = "6bit"
    write_data(certificate, payload)

    report = verify_certificate(certificate_path=certificate, artifact_dir=artifact)

    assert report.passed is False
    assert any("class-SKU" in issue for issue in report.issues)


def test_exact_bpw_rewrite_fails_and_cli_writes_report(tmp_path: Path) -> None:
    certificate, artifact = _bundle(tmp_path)
    manifest_path = artifact / "axquant_manifest.json"
    payload = read_data(manifest_path)
    payload["measured_main_bpw"] += 1e-10
    write_data(manifest_path, payload)
    output = tmp_path / "verification.json"

    exit_code = main(
        [
            "verify-cert",
            "--certificate",
            str(certificate),
            "--artifact",
            str(artifact),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert output.is_file()
    report = read_data(output)
    assert report["schema_version"] == "axquant.certification-verification.v1"
    assert report["passed"] is False
    assert any("measured main BPW" in issue for issue in report["issues"])


def test_published_v1_catalog_certificate_passes_locally() -> None:
    """Historical catalog records must remain verifiable without a local pack."""

    certificate = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "certifications"
        / "qwen36-27b-axq4-tier1.json"
    )
    report = verify_certificate(certificate_path=certificate)

    assert report.passed is True
    assert report.product_class == "5p6bpw"
    assert report.hub_repo_id == "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP"
