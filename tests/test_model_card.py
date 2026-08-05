from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.model_card import prepare_development_model_card
from axquant.planner import plan_quantization
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    PlanRequest,
    ProfileName,
    ProtectedTensorSidecarManifest,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantizerExecutionRecord,
)
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data


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

    changed = prepare_development_model_card(
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


def test_development_model_card_rejects_product_class_mismatch(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)

    with pytest.raises(ArtifactError, match="product class"):
        prepare_development_model_card(
            artifact_dir=directory,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
            product_class="4bit",
        )


def test_development_model_card_supports_versioned_fleet_names(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    prepare_development_model_card(
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
    prepare_development_model_card(
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
        prepare_development_model_card(
            artifact_dir=directory,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP",
            artifact_edition=3,
        )


def test_embedding_model_card_links_4bit_and_8bit_v2_siblings(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(qwen36_model_dir, tmp_path)
    prepare_development_model_card(
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
    prepare_development_model_card(
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


def test_development_model_card_does_not_claim_ax_engine_without_native_manifest(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    directory = _development_artifact(
        qwen36_model_dir,
        tmp_path,
        include_native_manifest=False,
    )
    prepare_development_model_card(
        artifact_dir=directory,
        repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP",
        product_class="6bit",
    )

    readme = (directory / "README.md").read_text(encoding="utf-8")
    assert "## AX Engine status" in readme
    assert "does **not** include a validated native `model-manifest.json`" in readme
    assert "Not established; no validated native manifest is included" in readme
    assert "ax-engine serve" not in readme


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
        prepare_development_model_card(
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
        prepare_development_model_card(
            artifact_dir=linked,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        )
