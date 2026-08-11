from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.model_card import (
    prepare_development_model_card,
    render_development_model_card,
    resolve_public_certification_claim,
)
from axquant.planner import plan_quantization
from axquant.public_cert_index import public_row_for_repo
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    CheckpointCertificationClaim,
    PlanRequest,
    ProfileName,
    ProtectedTensorSidecarManifest,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantizerExecutionRecord,
)
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data


def _prepare_card(**kwargs: object) -> list[Path]:
    """Fixture helper: skip the live public cert index unless a test opts in."""

    kwargs.setdefault("use_public_certification", False)
    return prepare_development_model_card(**kwargs)  # type: ignore[arg-type]


def _file(path: Path, *, relative_to: Path) -> ArtifactFile:
    return ArtifactFile(
        path=path.relative_to(relative_to).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )


def _development_artifact(
    qwen36_model_dir: Path,
    tmp_path: Path,
    *,
    include_native_manifest: bool = True,
) -> Path:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
    )
    report = architecture_prior_report(inventory, profile=ProfileName.GENERAL)
    plan = plan_quantization(
        report,
        PlanRequest(profile=ProfileName.GENERAL, target_bpw=14.0, allow_unmeasured=True),
    )
    directory = tmp_path / "AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP"
    directory.mkdir()
    config = json.loads((qwen36_model_dir / "config.json").read_text(encoding="utf-8"))
    config["text_config"]["max_position_embeddings"] = 262_144
    write_data(directory / "config.json", config)
    (directory / "README.md").write_text("# old card\n", encoding="utf-8")
    (directory / "LICENSE").write_text("Apache License 2.0 fixture\n", encoding="utf-8")
    if include_native_manifest:
        (directory / "model-manifest.json").write_text("{}\n", encoding="utf-8")
    save_file(
        {"model.layers.0.mlp.down_proj.weight": np.zeros((1,), dtype=np.float32)},
        directory / "model.safetensors",
    )
    save_file(
        {"mtp.fc.weight": np.zeros((1,), dtype=np.float32)},
        directory / "mtp.safetensors",
    )
    save_file(
        {"visual.patch_embed.weight": np.zeros((1,), dtype=np.float32)},
        directory / "vision.safetensors",
    )
    write_data(directory / "axquant_plan.json", plan)
    runtime = build_runtime_metadata(plan, directory)
    write_data(directory / "axquant_runtime.json", runtime)
    allocation = plan.assignments[0]
    execution = QuantizerExecutionManifest(
        plan_sha256=stable_sha256(plan),
        records=[
            QuantizerExecutionRecord(
                method=allocation.method,
                module_path=allocation.module_path,
                bits=allocation.bits,
                group_size=allocation.group_size,
                success=True,
            )
        ],
    )
    write_data(directory / "axquant_quantizer_execution.json", execution)
    source_record = ArtifactFile(path="source.safetensors", size_bytes=1, sha256="a" * 64)
    mtp_sidecar = ProtectedTensorSidecarManifest(
        source_model=plan.source_model,
        role="mtp",
        tensor_count=2,
        parameters=64,
        dtypes=("BF16",),
        tensor_names_sha256="b" * 64,
        source_files=[source_record],
        output=_file(directory / "mtp.safetensors", relative_to=directory),
    )
    vision_sidecar = ProtectedTensorSidecarManifest(
        source_model=plan.source_model,
        role="vision",
        tensor_count=3,
        parameters=96,
        dtypes=("BF16",),
        tensor_names_sha256="c" * 64,
        source_files=[source_record],
        output=_file(directory / "vision.safetensors", relative_to=directory),
    )
    write_data(directory / "axquant_mtp_sidecar_manifest.json", mtp_sidecar)
    write_data(directory / "axquant_vision_sidecar_manifest.json", vision_sidecar)
    files = [
        _file(path, relative_to=directory) for path in sorted(directory.iterdir()) if path.is_file()
    ]
    manifest = ArtifactManifest(
        axquant_version="1.0.0",
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        profile=plan.profile,
        target_class="4bit",
        effective_bpw=6.0,
        logical_parameters=16,
        main_logical_parameters=13,
        weight_file_size_bytes=16,
        main_weight_file_size_bytes=13,
        mtp_weight_file_size_bytes=3,
        protected_weight_file_size_bytes=6,
        measured_total_bpw=8.0,
        measured_main_bpw=8.0,
        weight_distribution=plan.weight_distribution,
        mtp_distribution=plan.mtp_distribution,
        mtp_present=True,
        mtp_policy=plan.mtp,
        runtime=runtime,
        software_versions=plan.software_versions,
        files=files,
    )
    write_data(directory / "axquant_manifest.json", manifest)
    return directory


def test_development_model_card_is_detailed_sanitized_and_bound(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    original_local_path = str(qwen36_model_dir.resolve())

    changed = _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
    )

    assert directory / "README.md" in changed
    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert len(readme.split()) > 700
    assert "base_model: Qwen/Qwen3.6-27B" in readme
    assert "storage-budget product class" in readme
    assert "no MTP speedup claim" in readme
    assert "Vision-language quality" in readme
    assert "262,144 tokens" in readme
    assert "Not certified" in readme
    assert original_local_path not in readme

    manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
    plan = load_model(directory / "axquant_plan.json", QuantizationPlan)
    execution = load_model(
        directory / "axquant_quantizer_execution.json",
        QuantizerExecutionManifest,
    )
    assert manifest.source_model.local_path is None
    assert plan.source_model.local_path is None
    assert manifest.plan_sha256 == stable_sha256(plan) == execution.plan_sha256
    records = {record.path: record for record in manifest.files}
    assert {"README.md", "LICENSE", "model-manifest.json"}.issubset(records)
    assert records["README.md"].sha256 == file_sha256(directory / "README.md")
    assert records["README.md"].size_bytes == (directory / "README.md").stat().st_size
    for name in (
        "axquant_manifest.json",
        "axquant_plan.json",
        "axquant_mtp_sidecar_manifest.json",
        "axquant_vision_sidecar_manifest.json",
    ):
        assert original_local_path not in json.dumps(read_data(directory / name))


def _claim(directory: Path, **overrides: object) -> CheckpointCertificationClaim:
    fields: dict[str, object] = {
        "hub_repo_id": "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        "hub_commit": "cdd13bf81cf21818a01cf59a31fc116ef84326bc",
        "candidate_manifest_sha256": file_sha256(directory / "axquant_manifest.json"),
        "host_id": "df-macbookpro-m5",
        "certified_at": "2026-08-08T21:20:00+00:00",
        "mtp_acceleration_status": "not-certified",
    }
    fields.update(overrides)
    return CheckpointCertificationClaim(**fields)  # type: ignore[arg-type]


def _certified_card(directory: Path, claim: CheckpointCertificationClaim) -> str:
    return render_development_model_card(
        directory=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        product_class="6bit",
        manifest=load_model(directory / "axquant_manifest.json", ArtifactManifest),
        plan=load_model(directory / "axquant_plan.json", QuantizationPlan),
        execution=load_model(
            directory / "axquant_quantizer_execution.json",
            QuantizerExecutionManifest,
        ),
        mtp_sidecar=None,
        vision_sidecar=None,
        certification=claim,
    )


def test_model_card_states_tier_1_without_implying_an_acceleration_claim(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    """A Tier 1 certificate replaces the development banner and nothing more.

    Tier 1 binds size, quality, and conversion integrity. The card must say so
    and must keep saying MTP acceleration is uncertified, because that is what
    the certificate itself records for this pack.
    """
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
    )

    card = _certified_card(directory, _claim(directory))

    assert "Checkpoint Tier 1 certified" in card
    assert "df-macbookpro-m5" in card
    assert "2026-08-08" in card
    assert "cdd13bf81cf2" in card
    assert "Development evidence — not a certified AXQuant release" not in card
    # Tier 1 must never read as a speed claim.
    assert "not certified**; no MTP speedup claim" in card
    assert "M0-M8 release" in card


def test_model_card_renders_a_scoped_acceleration_status_from_the_certificate(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
    )

    card = _certified_card(
        directory,
        _claim(
            directory,
            mtp_acceleration_status="certified-scoped",
            mtp_acceleration_note="decode-heavy authorizing profiles only",
        ),
    )

    assert "authorizing profiles only" in card
    assert "decode-heavy authorizing profiles only" in card
    assert "no speedup claim" in card


def test_model_card_refuses_a_certificate_that_does_not_bind_the_artifact(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    """An unbound certificate is an error, never a silent development card.

    Re-deriving the manifest digest from disk is what makes the rendered Tier 1
    statement evidence instead of a caller assertion, so both a foreign
    repository and a stale digest must fail closed.
    """
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
    )

    with pytest.raises(ArtifactError, match="different repository"):
        _certified_card(
            directory,
            _claim(directory, hub_repo_id="AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP"),
        )

    with pytest.raises(ArtifactError, match="does not bind this artifact"):
        _certified_card(directory, _claim(directory, candidate_manifest_sha256="0" * 64))


def test_checkpoint_certification_claim_requires_an_immutable_commit() -> None:
    with pytest.raises(ValidationError):
        CheckpointCertificationClaim(
            hub_repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
            hub_commit="main",
            candidate_manifest_sha256="0" * 64,
            host_id="df-macbookpro-m5",
            certified_at="2026-08-08T21:20:00+00:00",  # type: ignore[arg-type]
        )


def test_development_model_card_rejects_product_class_mismatch(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)

    with pytest.raises(ArtifactError, match="product class"):
        _prepare_card(
            artifact_dir=directory,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
            product_class="4bit",
        )


def test_development_model_card_supports_versioned_fleet_names(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP",
        product_class="6bit",
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "| Artifact edition | `v2` |" in readme
    assert "AX-Qwen3.6-27B-MLX-AXQ-4bit-v2-MTP" in readme
    assert "AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP" in readme


def test_development_model_card_marks_v2_at_stable_repository_name(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        product_class="6bit",
        artifact_edition=2,
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "| Artifact edition | `v2` |" in readme
    assert "**Stable-name v2.**" in readme
    assert "`legacy-pre-v2`" in readme
    assert "AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP" in readme
    assert "AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP" in readme
    assert "AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP" not in readme


def test_development_model_card_rejects_conflicting_artifact_edition(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)

    with pytest.raises(ArtifactError, match="edition does not match"):
        _prepare_card(
            artifact_dir=directory,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP",
            artifact_edition=3,
        )


def test_embedding_model_card_links_4bit_and_8bit_v2_siblings(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-4bit-v2",
        product_class="4bit",
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-4bit-v2" in readme
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-8bit-v2" in readme
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-6bit-v2" not in readme


def test_embedding_model_card_links_stable_4bit_and_8bit_siblings(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-4bit",
        product_class="4bit",
        artifact_edition=2,
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-4bit" in readme
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-8bit" in readme
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-6bit" not in readme
    assert "AX-Qwen3-Embedding-8B-MLX-AXQ-4bit-v2" not in readme


def test_floor_collapsed_model_card_omits_4bit_sibling_link(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP",
        product_class="6bit",
        artifact_edition=2,
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "Why there is no AXQ-4bit pack" in readme
    assert "no distinct 4bit pack" in readme
    assert "6.97 BPW" in readme or "~6.97 BPW" in readme
    assert "AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP" in readme
    assert "AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP" not in readme
    assert "does **not** publish a separate" in readme


def test_development_model_card_does_not_claim_ax_engine_without_native_manifest(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(
        qwen36_model_dir,
        tmp_path,
        include_native_manifest=False,
    )
    _prepare_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP",
        product_class="6bit",
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "## AX Engine status" in readme
    assert "does **not** include a validated native `model-manifest.json`" in readme
    assert "Not established; no validated native manifest is included" in readme
    assert "ax-engine serve" not in readme


def test_multimodal_model_cards_use_executable_runtime_commands(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
    plan = load_model(directory / "axquant_plan.json", QuantizationPlan)
    execution = load_model(
        directory / "axquant_quantizer_execution.json",
        QuantizerExecutionManifest,
    )

    asr_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={
                    "adapter_id": "qwen3-asr-v1",
                    "product_family": "qwen3-asr",
                    "config_model_type": "qwen3_asr",
                    "vision_present": False,
                    "audio_present": True,
                }
            )
        }
    )
    asr_card = render_development_model_card(
        directory=directory,
        repo_id="AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit",
        product_class="6bit",
        manifest=manifest.model_copy(update={"mtp_present": False}),
        plan=asr_plan,
        execution=execution,
        mtp_sidecar=None,
        vision_sidecar=None,
        artifact_edition=2,
    )
    assert "library_name: mlx-audio" in asr_card
    assert "pipeline_tag: automatic-speech-recognition" in asr_card
    assert "--output-path ./transcript" in asr_card
    assert "--format txt" in asr_card
    assert "If an OptiQ repository is published separately" in asr_card

    vlm_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={
                    "adapter_id": "qwen3-vl-v1",
                    "product_family": "qwen3-vl",
                    "config_model_type": "qwen3_vl",
                    "vision_present": True,
                    "audio_present": False,
                }
            )
        }
    )
    vlm_card = render_development_model_card(
        directory=directory,
        repo_id="AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit",
        product_class="6bit",
        manifest=manifest.model_copy(update={"mtp_present": False}),
        plan=vlm_plan,
        execution=execution,
        mtp_sidecar=None,
        vision_sidecar=None,
        artifact_edition=2,
    )
    assert "library_name: mlx" in vlm_card
    assert "pipeline_tag: image-text-to-text" in vlm_card
    assert "--temperature 0.0" in vlm_card
    assert "--temp 0.0" not in vlm_card


def test_development_model_card_rejects_stale_execution_before_mutating(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    plan_path = directory / "axquant_plan.json"
    original_plan = load_model(plan_path, QuantizationPlan)
    execution_path = directory / "axquant_quantizer_execution.json"
    execution = load_model(execution_path, QuantizerExecutionManifest)
    execution.plan_sha256 = "f" * 64
    write_data(execution_path, execution)

    with pytest.raises(ArtifactError, match="execution does not bind"):
        _prepare_card(
            artifact_dir=directory,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        )

    assert load_model(plan_path, QuantizationPlan) == original_plan


def test_development_model_card_rejects_symlinked_artifact_root(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    linked = tmp_path / "linked-artifact"
    linked.symlink_to(directory, target_is_directory=True)

    with pytest.raises(ArtifactError, match="must not be a symlink"):
        _prepare_card(
            artifact_dir=linked,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        )


def _write_public_cert(
    cert_dir: Path,
    *,
    repo_id: str,
    hub_commit: str,
    manifest_sha256: str,
    status: str = "certified",
) -> Path:
    cert_dir.mkdir(parents=True, exist_ok=True)
    stem = "fixture-axq6-tier1"
    path = cert_dir / f"{stem}.json"
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": status,
        "certification_tier": "checkpoint",
        "certified_at": "2026-08-08T21:20:00+00:00",
        "host_id": "df-macbookpro-m5",
        "artifact": {
            "hub_repo_id": repo_id,
            "hub_commit": hub_commit,
            "product_class": "6bit",
            "candidate_manifest_sha256": manifest_sha256,
            "source_model_id": "Qwen/Qwen3.6-27B",
            "source_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        },
        "plan": {"evidence_kind": "architecture_prior"},
        "size": {"pass": True},
        "quality": {"general": {"retention": 1.0}},
        "thresholds": {"minimum_quality_retention": 0.98},
        "mtp_acceleration": {"status": "not-certified", "reason": "fixture"},
        "toolchain": {"axquant": "1.6.1"},
        "public_index": {
            "display_name": "Fixture AXQ 6-bit",
            "sort_order": 1,
            "edition_label": "main@`cdd13bf8`",
            "listed": True,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (cert_dir / f"{stem}.md").write_text("# fixture cert\n", encoding="utf-8")
    return path


def test_prepare_binds_public_certificate_when_manifest_matches(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    # Refresh public file digests the way prepare does before cert bind.
    repo_id = "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP"
    hub_commit = "cdd13bf81cf21818a01cf59a31fc116ef84326bc"
    # Prepare once without public certs to materialize sanitized manifests, then
    # register a certificate against the resulting digest and re-prepare.
    _prepare_card(artifact_dir=directory, repo_id=repo_id)
    digest = file_sha256(directory / "axquant_manifest.json")
    cert_dir = tmp_path / "certs"
    _write_public_cert(
        cert_dir,
        repo_id=repo_id,
        hub_commit=hub_commit,
        manifest_sha256=digest,
    )
    claim = resolve_public_certification_claim(repo_id, certifications_dir=cert_dir)
    assert claim is not None
    assert claim.candidate_manifest_sha256 == digest

    prepare_development_model_card(
        artifact_dir=directory,
        repo_id=repo_id,
        use_public_certification=True,
        certifications_dir=cert_dir,
    )
    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "Checkpoint Tier 1 certified" in readme
    assert "df-macbookpro-m5" in readme
    assert "Development evidence — not a certified AXQuant release" not in readme


def test_prepare_fails_closed_when_public_certificate_does_not_bind(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    repo_id = "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP"
    cert_dir = tmp_path / "certs"
    _write_public_cert(
        cert_dir,
        repo_id=repo_id,
        hub_commit="cdd13bf81cf21818a01cf59a31fc116ef84326bc",
        manifest_sha256="0" * 64,
    )
    with pytest.raises(ArtifactError, match="does not bind this artifact"):
        prepare_development_model_card(
            artifact_dir=directory,
            repo_id=repo_id,
            use_public_certification=True,
            certifications_dir=cert_dir,
        )


def test_prepare_stays_development_without_public_certificate(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    cert_dir = tmp_path / "empty-certs"
    cert_dir.mkdir()
    prepare_development_model_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        use_public_certification=True,
        certifications_dir=cert_dir,
    )
    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "Development evidence — not a certified AXQuant release" in readme
    assert (
        public_row_for_repo(
            "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
            cert_dir=cert_dir,
            listed_only=False,
        )
        is None
    )
