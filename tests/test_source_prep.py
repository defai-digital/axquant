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


@pytest.mark.parametrize("quantized,mlx_layout", [(True, True), (False, True), (True, False)])
def test_quantized_qwen_mtp_text_view(tmp_path: Path, quantized: bool, mlx_layout: bool) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = {"model_type": "qwen3_5_moe"}
    if quantized:
        config["quantization"] = {"bits": 6, "group_size": 64}  # type: ignore[assignment]
    (source / "config.json").write_text(json.dumps(config))
    conv = "language_model.model.layers.0.linear_attn.conv1d.weight"
    norm = "language_model.model.layers.0.input_layernorm.weight"
    mtp = "language_model.mtp.norm.weight"
    save_file(
        {
            conv: np.ones((4, 4, 1) if mlx_layout else (4, 1, 4), dtype=np.float32),
            norm: np.ones(4, dtype=np.float32),
        },
        source / "model-00001.safetensors",
    )
    save_file({mtp: np.ones((4, 4), dtype=np.float32)}, source / "model-mtp.safetensors")
    index = {
        "weight_map": {
            conv: "model-00001.safetensors",
            norm: "model-00001.safetensors",
            mtp: "model-mtp.safetensors",
        }
    }
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    prepared = prepare_conversion_source(source, work_dir=tmp_path / "work")
    if quantized and mlx_layout:
        assert needs_conversion_prep(source)
        assert prepared is not None
        assert not (prepared / "model-mtp.safetensors").exists()
        assert (prepared / "model-00001.safetensors").read_bytes() == before[
            "model-00001.safetensors"
        ]
        assert (
            mtp
            not in json.loads((prepared / "model.safetensors.index.json").read_text())["weight_map"]
        )
    else:
        assert prepared is None
        view = source_prep._quantized_qwen_mtp_view(source, config)
        assert view is None or not view[1]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def _write_quantized_qwen_mtp_fixture(root: Path, conv_shape: tuple[int, ...]) -> dict[str, str]:
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "quantization": {"bits": 6, "group_size": 64}})
    )
    conv = "language_model.model.layers.0.linear_attn.conv1d.weight"
    norm = "language_model.model.norm.weight"
    save_file(
        {conv: np.ones(conv_shape, dtype=np.float32), norm: np.ones(4, dtype=np.float32)},
        root / "model-00001.safetensors",
    )
    mtp = "language_model.mtp.weight"
    save_file({mtp: np.ones((4, 4), dtype=np.float32)}, root / "model-mtp.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    conv: "model-00001.safetensors",
                    norm: "model-00001.safetensors",
                    mtp: "model-mtp.safetensors",
                }
            }
        )
    )
    return {"conv": conv, "norm": norm, "mtp": mtp}


def test_quantized_qwen_mtp_mixed_shard_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    keys = _write_quantized_qwen_mtp_fixture(source, (4, 4, 1))
    index_text = (source / "model.safetensors.index.json").read_text()
    index = json.loads(index_text)
    for key in keys.values():
        index["weight_map"][key] = "model-00001.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError, match="isolated MTP shards"):
        needs_conversion_prep(source)
    with pytest.raises(ArtifactError, match="isolated MTP shards"):
        prepare_conversion_source(source, work_dir=tmp_path / "work")


def test_quantized_qwen_mtp_hf_layout_mixed_shard_stays_unprepped(tmp_path: Path) -> None:
    """HF-layout sources keep MTP keys visible so MLX-LM applies its norm shift."""
    source = tmp_path / "source"
    keys = _write_quantized_qwen_mtp_fixture(source, (4, 1, 4))
    index_text = (source / "model.safetensors.index.json").read_text()
    index = json.loads(index_text)
    index["weight_map"][keys["mtp"]] = "model-00001.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    assert not needs_conversion_prep(source)
    assert prepare_conversion_source(source, work_dir=tmp_path / "work") is None


def test_quantized_qwen_mtp_own_conv_layout_does_not_block_isolation(tmp_path: Path) -> None:
    """Only trunk convolutions gate the layout check, not MTP convolutions."""
    source = tmp_path / "source"
    keys = _write_quantized_qwen_mtp_fixture(source, (4, 4, 1))
    mtp_conv = "language_model.mtp.layers.0.linear_attn.conv1d.weight"
    mtp_shard = source / "model-mtp.safetensors"
    save_file(
        {**load_file(mtp_shard), mtp_conv: np.ones((4, 1, 4), dtype=np.float32)},
        mtp_shard,
    )
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"][mtp_conv] = "model-mtp.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    assert needs_conversion_prep(source)
    prepared = prepare_conversion_source(source, work_dir=tmp_path / "work")
    assert prepared is not None
    weight_map = json.loads((prepared / "model.safetensors.index.json").read_text())["weight_map"]
    assert keys["mtp"] not in weight_map and mtp_conv not in weight_map


@pytest.mark.parametrize(
    "payload",
    [
        {"metadata": {}},
        ["not", "an", "object"],
        {"weight_map": []},
        {"weight_map": {"language_model.model.norm.weight": 7}},
        {"weight_map": {"language_model.model.norm.weight": ""}},
        '{"weight_map": ',
        '{"weight_map": {}, "weight_map": {}}',
    ],
)
def test_quantized_qwen_mtp_malformed_index_fails_closed(tmp_path: Path, payload: object) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "quantization": {"bits": 6}})
    )
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (source / "model.safetensors.index.json").write_text(text)
    with pytest.raises(ArtifactError):
        needs_conversion_prep(source)
    with pytest.raises(ArtifactError):
        prepare_conversion_source(source, work_dir=tmp_path / "work")


def test_quantized_qwen_mtp_reserved_shard_name_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    keys = _write_quantized_qwen_mtp_fixture(source, (4, 4, 1))
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"][keys["norm"]] = "model.safetensors.index.json"
    modified_text = json.dumps(index)
    (source / "model.safetensors.index.json").write_text(modified_text)
    with pytest.raises(ArtifactError, match="plain Safetensors file"):
        prepare_conversion_source(source, work_dir=tmp_path / "work")
    assert (source / "model.safetensors.index.json").read_text() == modified_text


@pytest.mark.parametrize("missing", ["model-00001-of-00002.safetensors", "sub\\shard.safetensors"])
def test_quantized_qwen_mtp_missing_conv_shard_fails_closed(tmp_path: Path, missing: str) -> None:
    source = tmp_path / "source"
    keys = _write_quantized_qwen_mtp_fixture(source, (4, 4, 1))
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"][keys["conv"]] = missing
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError, match="missing"):
        needs_conversion_prep(source)


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
    assert "vision_config" not in cfg
    assert "audio_config" not in cfg
    loaded = mlx.load(str(prepared / "model.safetensors"))
    assert "model.language_model.layers.0.mlp.down_proj.weight" in loaded
    assert "model.vision_embedder.patch_dense.weight" not in loaded
    assert "model.embed_audio.embedding_projection.weight" not in loaded
    # Original source still has multimodal tensors for sidecar extraction.
    original = mlx.load(str(source / "model.safetensors"))
    assert "model.vision_embedder.patch_dense.weight" in original
    source_cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    assert "vision_config" in source_cfg


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


def test_ministral3_model_prefix_prep_rewrites_keys_and_forces_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nemotron-3-Embed style: bare layers.* keys + missing lm_head."""
    import numpy as np

    from axquant.source_prep import (
        needs_ministral3_model_prefix_prep,
        prepare_ministral3_model_prefix_source,
    )

    model_dir = tmp_path / "Nemotron-3-Embed-1B-BF16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ministral3",
                "num_hidden_layers": 2,
                "tie_word_embeddings": False,
                "architectures": ["Ministral3Model"],
            }
        ),
        encoding="utf-8",
    )

    weights = {
        "embed_tokens.weight": np.zeros((8, 4), dtype=np.float32),
        "layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "layers.0.mlp.gate_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "norm.weight": np.zeros((4,), dtype=np.float32),
    }

    class _FakeMlx:
        @staticmethod
        def load(path: str) -> dict[str, np.ndarray]:
            del path
            return dict(weights)

        @staticmethod
        def save_safetensors(path: str, payload: dict[str, np.ndarray]) -> None:
            # Persist names only for verification (values unused).
            Path(path).write_text("\n".join(sorted(payload)), encoding="utf-8")

    monkeypatch.setattr(source_prep, "_mlx_core", lambda: _FakeMlx)
    # Bypass real safetensors round-trip verification (fake writer).
    monkeypatch.setattr(
        source_prep,
        "_verify_saved_tensor_names",
        lambda mx, path, expected: None,
    )
    # Sample keys from our synthetic map without needing a real safetensors file.
    monkeypatch.setattr(
        source_prep,
        "_sample_safetensor_keys",
        lambda directory, limit=32: list(weights)[:limit],
    )
    (model_dir / "model.safetensors").write_bytes(b"fake")

    assert needs_ministral3_model_prefix_prep(model_dir, {"model_type": "ministral3"})
    prepared = prepare_ministral3_model_prefix_source(model_dir, work_dir=tmp_path / "work")
    cfg = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    assert cfg["tie_word_embeddings"] is True
    saved = (prepared / "model.safetensors").read_text(encoding="utf-8").splitlines()
    assert "model.embed_tokens.weight" in saved
    assert "model.layers.0.mlp.gate_proj.weight" in saved
    assert "embed_tokens.weight" not in saved


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
