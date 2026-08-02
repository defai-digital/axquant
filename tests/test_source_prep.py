"""Convert-time source preparation (Gemma-4 unified → gemma4 text path)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.errors import ArtifactError
from axquant.source_prep import (
    needs_conversion_prep,
    needs_gemma4_unified_prep,
    prepare_conversion_source,
    prepare_gemma4_unified_source,
)


def _write_gemma4_unified_fixture(root: Path) -> Path:
    model_dir = root / "gemma4-unified"
    model_dir.mkdir()
    config = {
        "model_type": "gemma4_unified",
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "enable_moe_block": False,
            "num_experts": None,
        },
        "vision_config": {"hidden_size": 8},
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    # Language + multimodal tensors (f32 so numpy/safetensors work without MLX).
    tensors = {
        "model.language_model.embed_tokens.weight": np.zeros((32, 8), dtype=np.float32),
        "model.language_model.layers.0.mlp.down_proj.weight": np.zeros((8, 16), dtype=np.float32),
        "model.language_model.norm.weight": np.zeros((8,), dtype=np.float32),
        "model.vision_embedder.patch_dense.weight": np.zeros((8, 8), dtype=np.float32),
        "model.embed_vision.embedding_projection.weight": np.zeros((8, 8), dtype=np.float32),
        "model.embed_audio.embedding_projection.weight": np.zeros((8, 8), dtype=np.float32),
    }
    save_file(tensors, model_dir / "model.safetensors")
    return model_dir


def test_needs_gemma4_unified_prep() -> None:
    assert needs_gemma4_unified_prep({"model_type": "gemma4_unified"})
    assert not needs_gemma4_unified_prep({"model_type": "gemma4"})


def test_prepare_gemma4_unified_filters_multimodal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlx = pytest.importorskip("mlx.core")
    source = _write_gemma4_unified_fixture(tmp_path)
    assert needs_conversion_prep(source)

    # Re-save fixture weights via MLX so prepare can load them.
    weights = {
        "model.language_model.embed_tokens.weight": mlx.zeros((32, 8)),
        "model.language_model.layers.0.mlp.down_proj.weight": mlx.zeros((8, 16)),
        "model.language_model.norm.weight": mlx.zeros((8,)),
        "model.vision_embedder.patch_dense.weight": mlx.zeros((8, 8)),
        "model.embed_vision.embedding_projection.weight": mlx.zeros((8, 8)),
        "model.embed_audio.embedding_projection.weight": mlx.zeros((8, 8)),
    }
    mlx.save_safetensors(str(source / "model.safetensors"), weights)

    prepared = prepare_gemma4_unified_source(source, work_dir=tmp_path / "work")
    cfg = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "gemma4"
    loaded = mlx.load(str(prepared / "model.safetensors"))
    assert "model.language_model.layers.0.mlp.down_proj.weight" in loaded
    assert "model.vision_embedder.patch_dense.weight" not in loaded
    assert "model.embed_audio.embedding_projection.weight" not in loaded
    # Original source still has multimodal tensors for sidecar extraction.
    original = mlx.load(str(source / "model.safetensors"))
    assert "model.vision_embedder.patch_dense.weight" in original


def test_prepare_conversion_source_noop_for_qwen(tmp_path: Path) -> None:
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 2}}),
        encoding="utf-8",
    )
    assert prepare_conversion_source(model_dir, work_dir=tmp_path / "work") is None


def test_prepare_rejects_wrong_type(tmp_path: Path) -> None:
    model_dir = tmp_path / "plain"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="gemma4_unified"):
        prepare_gemma4_unified_source(model_dir, work_dir=tmp_path / "work")


def test_needs_tekken_tokenizer_prep(tmp_path: Path) -> None:
    from axquant.source_prep import needs_tekken_tokenizer_prep

    model_dir = tmp_path / "devstral"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}), encoding="utf-8")
    (model_dir / "tekken.json").write_text("{}", encoding="utf-8")
    assert needs_tekken_tokenizer_prep(model_dir)
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert not needs_tekken_tokenizer_prep(model_dir)


def test_resolve_tekken_tokenizer_repo_for_devstral(tmp_path: Path) -> None:
    from axquant.source_prep import _resolve_tekken_tokenizer_repo

    model_dir = tmp_path / "Devstral-Small-2505-bf16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}), encoding="utf-8")
    repo = _resolve_tekken_tokenizer_repo(
        model_dir,
        model_id="mistralai/Devstral-Small-2505",
        config={"model_type": "mistral"},
    )
    assert repo == "mlx-community/Devstral-Small-2505-bf16"
