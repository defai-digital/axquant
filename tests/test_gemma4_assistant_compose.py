"""Tests for ST2 Gemma4 AXQ + assistant composite composition and formal profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.benchmark import (
    GEMMA4_ASSISTANT_EXACT_MTP_PROFILE_ENV,
    QWEN36_EXACT_MTP_PROFILE_ENV,
)
from axquant.errors import ArtifactError
from axquant.gemma4_assistant_compose import (
    ASSISTANT_CONTRACT_NAME,
    COMPOSITE_MANIFEST_NAME,
    Gemma4AssistantComposeRequest,
    compose_gemma4_assistant_mtp,
    load_composite_manifest,
    validate_known_gemma4_assistant_pair,
)
from axquant.schema.artifacts import ALLOWED_BENCHMARK_RUNTIME_ENV_KEYS
from axquant.serde import file_sha256


def _write_minimal_target(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"target-weight-bytes-v1")
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
    validate_known_gemma4_assistant_pair(
        "google/gemma-4-31b-it", "google/gemma-4-31b-it-assistant"
    )
    with pytest.raises(ArtifactError, match="known Gemma4"):
        validate_known_gemma4_assistant_pair("gemma-4-unknown-it", "gemma-4-unknown-it-assistant")
    with pytest.raises(ArtifactError, match="must be"):
        validate_known_gemma4_assistant_pair("gemma-4-26b-a4b-it", "gemma-4-31b-it-assistant")


def test_compose_preserves_base_digests(tmp_path: Path) -> None:
    target = tmp_path / "target"
    assistant = tmp_path / "assistant"
    output = tmp_path / "composite"
    _write_minimal_target(target)
    _write_minimal_assistant(assistant)
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
    (assistant / "config.json").write_text(
        json.dumps({"model_type": "gemma4"}), encoding="utf-8"
    )
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
