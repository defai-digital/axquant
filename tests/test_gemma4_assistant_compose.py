"""Tests for ST2 Gemma4 AXQ + assistant composite composition and formal profile."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.benchmark import (
    GEMMA4_ASSISTANT_EXACT_MTP_PROFILE_ENV,
    QWEN36_EXACT_MTP_PROFILE_ENV,
)
from axquant.errors import ArtifactError
from axquant.gemma4_assistant_compose import (
    ASSISTANT_CONTRACT_NAME,
    COMPOSITE_MANIFEST_NAME,
    Gemma4AssistantComposeRequest,
    bind_gemma4_assistant_runtime_metadata,
    compose_gemma4_assistant_mtp,
    load_composite_manifest,
    refresh_gemma4_assistant_composite_manifest,
    validate_gemma4_assistant_composite,
    validate_known_gemma4_assistant_pair,
)
from axquant.gemma4_vlm import (
    GEMMA4_MLX_VLM_VISION_LAYOUT,
    normalize_gemma4_vision_tensor_names,
    validate_gemma4_mlx_vlm_vision_layout,
)
from axquant.schema import (
    AxEngineOptimizationMetadata,
    MtpRuntimeMetadata,
    OptimizationScope,
    RuntimeMetadata,
    RuntimeName,
    RuntimeProfile,
    RuntimeSupportLevel,
)
from axquant.schema.artifacts import ALLOWED_BENCHMARK_RUNTIME_ENV_KEYS
from axquant.serde import file_sha256, load_model, write_data


def _write_minimal_target(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}),
        encoding="utf-8",
    )
    main_name = "language_model.model.embed_tokens.weight"
    vision_names = (
        "embed_vision.embedding_projection.weight",
        "vision_tower.encoder.layers.0.input_layernorm.weight",
    )
    save_file(
        {main_name: np.zeros((2, 2), dtype=np.float32)},
        root / "model.safetensors",
        metadata={"format": "mlx"},
    )
    save_file(
        {name: np.zeros((2, 2), dtype=np.float32) for name in vision_names},
        root / "vision.safetensors",
        metadata={
            "format": "mlx",
            "axquant_role": "protected-vision",
            "axquant_layout": GEMMA4_MLX_VLM_VISION_LAYOUT,
        },
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": 12, "total_size": 48},
                "weight_map": {
                    main_name: "model.safetensors",
                    **{name: "vision.safetensors" for name in vision_names},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")


def _write_minimal_assistant(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "gemma4_assistant"}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"assistant-weight-bytes-v1")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")


def test_known_pair_validation() -> None:
    validate_known_gemma4_assistant_pair("gemma-4-26b-a4b-it", "gemma-4-26b-a4b-it-assistant")
    validate_known_gemma4_assistant_pair("google/gemma-4-31b-it", "google/gemma-4-31b-it-assistant")
    with pytest.raises(ArtifactError, match="known Gemma4"):
        validate_known_gemma4_assistant_pair("gemma-4-unknown-it", "gemma-4-unknown-it-assistant")
    with pytest.raises(ArtifactError, match="must be"):
        validate_known_gemma4_assistant_pair("gemma-4-26b-a4b-it", "gemma-4-31b-it-assistant")


def test_refresh_composite_manifest_rebinds_public_metadata(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    _write_minimal_assistant(assistant)
    (target / "README.md").write_text("# Before\n", encoding="utf-8")
    compose_gemma4_assistant_mtp(
        Gemma4AssistantComposeRequest(
            target_dir=target,
            assistant_dir=assistant,
            output_dir=output,
            target_model_id="gemma-4-31b-it",
            assistant_model_id="gemma-4-31b-it-assistant",
            prefer_hardlink=False,
        )
    )
    (output / "README.md").write_text("# Public card\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="digest mismatch"):
        validate_gemma4_assistant_composite(output)

    refreshed = refresh_gemma4_assistant_composite_manifest(output)

    assert refreshed["base_weight_digests"]["README.md"] == file_sha256(output / "README.md")


def test_bind_assistant_runtime_metadata_declares_external_drafter(tmp_path: Path) -> None:
    runtime = RuntimeMetadata(
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
    )
    write_data(tmp_path / "axquant_runtime.json", runtime)

    bound = bind_gemma4_assistant_runtime_metadata(tmp_path, max_depth=2)

    assert bound is not None
    assert bound.mtp.detected
    assert bound.mtp.sidecar_file == "assistant"
    assert bound.mtp.draft_tokens == 2
    assert bound.mtp.verification_mode == "external-assistant"
    assert bound.primary_runtime.mtp_support == "native"
    assert [item.name for item in bound.compatible_runtimes] == [RuntimeName.MLX_VLM]
    assert load_model(tmp_path / "axquant_runtime.json", RuntimeMetadata) == bound


def test_gemma4_mlx_vlm_layout_rejects_source_prefix(tmp_path: Path) -> None:
    target = tmp_path / "target"
    _write_minimal_target(target)
    assert len(validate_gemma4_mlx_vlm_vision_layout(target)) == 2

    save_file(
        {"model.embed_vision.embedding_projection.weight": np.zeros((2, 2), dtype=np.float32)},
        target / "vision.safetensors",
        metadata={
            "format": "mlx",
            "axquant_role": "protected-vision",
            "axquant_layout": GEMMA4_MLX_VLM_VISION_LAYOUT,
        },
    )
    with pytest.raises(ArtifactError, match="source-prefixed"):
        validate_gemma4_mlx_vlm_vision_layout(target)


def test_gemma4_mlx_vlm_layout_rejects_index_drift(tmp_path: Path) -> None:
    target = tmp_path / "target"
    _write_minimal_target(target)
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"].pop("embed_vision.embedding_projection.weight")
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ArtifactError, match="do not exactly match"):
        validate_gemma4_mlx_vlm_vision_layout(target)


def test_gemma4_vision_name_normalization_rejects_collisions() -> None:
    with pytest.raises(ArtifactError, match="collision"):
        normalize_gemma4_vision_tensor_names(
            (
                "model.vision_tower.encoder.weight",
                "vision_tower.encoder.weight",
            )
        )


def test_gemma4_unified_layout_accepts_vision_and_audio_modules(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    names = (
        "vision_embedder.patch_dense.weight",
        "embed_vision.embedding_projection.weight",
        "embed_audio.embedding_projection.weight",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "vision_config": {"hidden_size": 8},
                "audio_config": {"hidden_size": 8},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {name: np.zeros((2, 2), dtype=np.float32) for name in names},
        target / "vision.safetensors",
        metadata={
            "format": "mlx",
            "axquant_role": "protected-vision",
            "axquant_layout": GEMMA4_MLX_VLM_VISION_LAYOUT,
        },
    )
    save_file(
        {"language_model.model.embed_tokens.weight": np.zeros((2, 2), dtype=np.float32)},
        target / "model.safetensors",
        metadata={"format": "mlx"},
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": 16, "total_size": 64},
                "weight_map": {
                    "language_model.model.embed_tokens.weight": "model.safetensors",
                    **{name: "vision.safetensors" for name in names},
                },
            }
        ),
        encoding="utf-8",
    )
    (target / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")

    assert validate_gemma4_mlx_vlm_vision_layout(target) == tuple(sorted(names))
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_assistant(assistant)
    result = compose_gemma4_assistant_mtp(
        Gemma4AssistantComposeRequest(
            target_dir=target,
            assistant_dir=assistant,
            output_dir=output,
            target_model_id="gemma-4-12b-it",
            assistant_model_id="gemma-4-12b-it-assistant",
            base_pack_id="AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit",
            base_tier1_certificate="docs/certifications/gemma4-12b-axq6-tier1.md",
            assistant_source_id="google/gemma-4-12b-it-assistant",
            max_depth=1,
            prefer_hardlink=False,
            axquant_version="test",
        )
    )
    assert result.output_dir == output.resolve()
    assert validate_gemma4_assistant_composite(output)["target_model_id"] == "gemma-4-12b-it"


def test_gemma4_unified_layout_requires_audio_module_when_configured(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    names = (
        "vision_embedder.patch_dense.weight",
        "embed_vision.embedding_projection.weight",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_unified",
                "vision_config": {"hidden_size": 8},
                "audio_config": {"hidden_size": 8},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {name: np.zeros((2, 2), dtype=np.float32) for name in names},
        target / "vision.safetensors",
        metadata={
            "format": "mlx",
            "axquant_role": "protected-vision",
            "axquant_layout": GEMMA4_MLX_VLM_VISION_LAYOUT,
        },
    )

    with pytest.raises(ArtifactError, match="missing MLX-VLM multimodal modules"):
        validate_gemma4_mlx_vlm_vision_layout(target)


def test_compose_preserves_base_digests(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    _write_minimal_assistant(assistant)
    # Divergent assistant tokenizer must be overwritten by target sync.
    (assistant / "tokenizer.json").write_text('{"version":"assistant-only"}', encoding="utf-8")
    base_digest = file_sha256(target / "model.safetensors")

    result = compose_gemma4_assistant_mtp(
        Gemma4AssistantComposeRequest(
            target_dir=target,
            assistant_dir=assistant,
            output_dir=output,
            target_model_id="gemma-4-26b-a4b-it",
            assistant_model_id="gemma-4-26b-a4b-it-assistant",
            base_pack_id="AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit",
            base_tier1_certificate="docs/certifications/gemma4-26b-a4b-axq6-tier1.md",
            assistant_source_id="google/gemma-4-26b-a4b-it-assistant",
            max_depth=1,
            prefer_hardlink=False,
            axquant_version="test",
        )
    )

    assert result.output_dir == output.resolve()
    assert (output / ASSISTANT_CONTRACT_NAME).is_file()
    assert (output / COMPOSITE_MANIFEST_NAME).is_file()
    assert (output / "assistant" / "model.safetensors").is_file()
    assert file_sha256(output / "model.safetensors") == base_digest
    assert result.base_weight_digests["model.safetensors"] == base_digest
    assert (output / "assistant" / "tokenizer.json").read_text(encoding="utf-8") == (
        output / "tokenizer.json"
    ).read_text(encoding="utf-8")

    contract = json.loads((output / ASSISTANT_CONTRACT_NAME).read_text(encoding="utf-8"))
    assert contract["schema_version"] == "ax.gemma4_assistant_mtp.v1"
    assert contract["backend"] == "gemma4_assistant"
    assert contract["target_model_id"] == "gemma-4-26b-a4b-it"
    assert contract["assistant_model_id"] == "gemma-4-26b-a4b-it-assistant"
    assert contract["assistant_path"] == "assistant"

    manifest = load_composite_manifest(output / COMPOSITE_MANIFEST_NAME)
    assert manifest["schema_version"] == "axquant.composite-pack-manifest.v1"
    assert manifest["contract_sha256"] == result.contract_sha256
    assert manifest["base_pack_id"] == "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit"
    assert validate_gemma4_assistant_composite(output)["contract_sha256"] == (
        result.contract_sha256
    )


def test_composite_validation_rejects_assistant_digest_drift(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    _write_minimal_assistant(assistant)
    compose_gemma4_assistant_mtp(
        Gemma4AssistantComposeRequest(
            target_dir=target,
            assistant_dir=assistant,
            output_dir=output,
            target_model_id="gemma-4-26b-a4b-it",
            assistant_model_id="gemma-4-26b-a4b-it-assistant",
            prefer_hardlink=False,
        )
    )
    (output / "assistant" / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ArtifactError, match="digest mismatch"):
        validate_gemma4_assistant_composite(output)


def test_composite_validation_rejects_unbound_assistant_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    _write_minimal_assistant(assistant)
    compose_gemma4_assistant_mtp(
        Gemma4AssistantComposeRequest(
            target_dir=target,
            assistant_dir=assistant,
            output_dir=output,
            target_model_id="gemma-4-26b-a4b-it",
            assistant_model_id="gemma-4-26b-a4b-it-assistant",
            prefer_hardlink=False,
        )
    )
    (output / "assistant" / "unbound.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="file set"):
        validate_gemma4_assistant_composite(output)


def test_compose_rejects_non_empty_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    _write_minimal_assistant(assistant)
    output.mkdir()
    (output / "stale.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(ArtifactError, match="empty or new"):
        compose_gemma4_assistant_mtp(
            Gemma4AssistantComposeRequest(
                target_dir=target,
                assistant_dir=assistant,
                output_dir=output,
                target_model_id="gemma-4-26b-a4b-it",
                assistant_model_id="gemma-4-26b-a4b-it-assistant",
            )
        )


def test_compose_rejects_wrong_assistant_model_type(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    assistant.mkdir()
    (assistant / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    (assistant / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ArtifactError, match="gemma4_assistant"):
        compose_gemma4_assistant_mtp(
            Gemma4AssistantComposeRequest(
                target_dir=target,
                assistant_dir=assistant,
                output_dir=output,
                target_model_id="gemma-4-26b-a4b-it",
                assistant_model_id="gemma-4-26b-a4b-it-assistant",
            )
        )


def test_gemma4_formal_profile_keys_are_allowlisted() -> None:
    for key in GEMMA4_ASSISTANT_EXACT_MTP_PROFILE_ENV:
        assert key in ALLOWED_BENCHMARK_RUNTIME_ENV_KEYS
    assert GEMMA4_ASSISTANT_EXACT_MTP_PROFILE_ENV["AX_MLX_GEMMA4_ASSISTANT_MTP"] == "1"
    assert GEMMA4_ASSISTANT_EXACT_MTP_PROFILE_ENV["AX_MLX_MTP_MIN_REMAINING_TOKENS"] == "0"
    # Profiles must not share a single required exclusive flag set with Qwen.
    assert "AX_MLX_QWEN_LINEAR_MTP_EXACT" not in GEMMA4_ASSISTANT_EXACT_MTP_PROFILE_ENV
    assert "AX_MLX_GEMMA4_ASSISTANT_MTP" not in QWEN36_EXACT_MTP_PROFILE_ENV
