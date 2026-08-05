"""Convert-time source preparation (Gemma-4 unified → gemma4 text path)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from axquant import source_prep
from axquant.errors import ArtifactError
from axquant.source_prep import (
    needs_conversion_prep,
    needs_gemma4_unified_prep,
    prepare_conversion_source,
    prepare_gemma4_unified_source,
)


class _NumpyMlx:
    @staticmethod
    def load(path: str) -> dict[str, np.ndarray]:
        return load_file(path)

    @staticmethod
    def save_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
        save_file({name: np.asarray(value) for name, value in tensors.items()}, path)


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


def test_filter_sharded_rejects_path_traversal_in_weight_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A checkpoint's own index.json is semi-trusted (it can come from any Hub
    # repo). A `weight_map` entry pointing outside the checkpoint directory
    # must be rejected the same way `inspector.py`'s indexed-shard scan
    # already rejects it, not silently followed with `mx.load`.
    # Force the MLX import path to explode so a regression that reorders
    # validation after `_mlx_core()` fails the same way Ubuntu CI does.
    def _mlx_must_not_run() -> object:
        raise AssertionError("shard path validation must run before MLX import")

    monkeypatch.setattr(source_prep, "_mlx_core", _mlx_must_not_run)
    source = _write_gemma4_unified_fixture(tmp_path)
    (source / "model.safetensors").unlink()
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.embed_tokens.weight": (
                        "../../../../etc/evil.safetensors"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="unsafe shard path"):
        prepare_gemma4_unified_source(source, work_dir=tmp_path / "work")


def test_filter_sharded_rejects_unindexed_tensor_in_referenced_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_gemma4_unified_fixture(tmp_path)
    (source / "model.safetensors").unlink()
    shard = source / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.language_model.norm.weight": np.zeros((8,), dtype=np.float32),
            "unindexed.injected.weight": np.ones((8, 8), dtype=np.float32),
        },
        shard,
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.norm.weight": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(source_prep, "_mlx_core", lambda: _NumpyMlx)

    with pytest.raises(ArtifactError, match="unindexed"):
        prepare_gemma4_unified_source(source, work_dir=tmp_path / "work")


def test_filter_sharded_rejects_non_string_shard_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mlx_must_not_run() -> object:
        raise AssertionError("shard type validation must run before MLX import")

    monkeypatch.setattr(source_prep, "_mlx_core", _mlx_must_not_run)
    source = _write_gemma4_unified_fixture(tmp_path)
    (source / "model.safetensors").unlink()
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.norm.weight": 7,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="non-string shard reference"):
        prepare_gemma4_unified_source(source, work_dir=tmp_path / "work")


def test_prepare_rejects_output_that_overlaps_source_without_deleting_it(
    tmp_path: Path,
) -> None:
    source = _write_gemma4_unified_fixture(tmp_path)
    overlapping = tmp_path / "gemma4-text-path"
    source.rename(overlapping)

    with pytest.raises(ArtifactError, match="must not overlap"):
        prepare_gemma4_unified_source(overlapping, work_dir=tmp_path)

    assert (overlapping / "config.json").is_file()
    assert (overlapping / "model.safetensors").is_file()


def test_filter_single_shard_verifies_backend_output_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_gemma4_unified_fixture(tmp_path)

    class _DroppingMlx(_NumpyMlx):
        @staticmethod
        def save_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
            first_name = next(iter(tensors))
            save_file({first_name: np.asarray(tensors[first_name])}, path)

    monkeypatch.setattr(source_prep, "_mlx_core", lambda: _DroppingMlx)
    with pytest.raises(ArtifactError, match="output coverage mismatch"):
        prepare_gemma4_unified_source(source, work_dir=tmp_path / "work")


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
    from axquant.source_prep import (
        _resolve_tekken_tokenizer_pack,
        _resolve_tekken_tokenizer_repo,
    )

    model_dir = tmp_path / "Devstral-Small-2505-bf16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}), encoding="utf-8")
    repo = _resolve_tekken_tokenizer_repo(
        model_dir,
        model_id="mistralai/Devstral-Small-2505",
        config={"model_type": "mistral"},
    )
    assert repo == "mlx-community/Devstral-Small-2505-bf16"
    pack_repo, revision = _resolve_tekken_tokenizer_pack(
        model_dir,
        model_id="mistralai/Devstral-Small-2505",
        config={"model_type": "mistral"},
    )
    assert pack_repo == repo
    assert len(revision) == 40
    assert revision.isalnum()


def test_tekken_prep_does_not_mutate_existing_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    model_dir = tmp_path / "Devstral-Small-2505-bf16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mistral"}),
        encoding="utf-8",
    )
    (model_dir / "tekken.json").write_text("{}", encoding="utf-8")
    source_provenance = model_dir / "axquant_tekken_tokenizer_provenance.json"
    source_provenance.write_text("source sentinel", encoding="utf-8")
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        (downloads / filename).write_text(f"downloaded {filename}", encoding="utf-8")

    def _download(*, repo_id: str, filename: str, revision: str) -> str:
        assert repo_id == "mlx-community/Devstral-Small-2505-bf16"
        assert len(revision) == 40
        return str(downloads / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    prepared = source_prep.prepare_tekken_tokenizer_source(
        model_dir,
        work_dir=tmp_path / "work",
        model_id="mistralai/Devstral-Small-2505",
    )

    assert source_provenance.read_text(encoding="utf-8") == "source sentinel"
    prepared_provenance = json.loads(
        (prepared / "axquant_tekken_tokenizer_provenance.json").read_text(encoding="utf-8")
    )
    assert set(prepared_provenance["fetched_sha256"]) == {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
