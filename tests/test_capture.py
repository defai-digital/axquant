"""Tests for the calibration activation capture artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant.activation_cache import tokenize_calibration
from axquant.capture import (
    CAPTURE_ACTIVATIONS_DIR,
    CAPTURE_MANIFEST_NAME,
    capture_calibration_activations,
    load_capture_activations,
)
from axquant.errors import CaptureError
from axquant.schema import (
    ActivationCaptureEntry,
    ActivationCaptureManifest,
    ModelIdentity,
    ProfileName,
)
from axquant.serde import file_sha256, load_model, write_data

_HIDDEN = 32
_INTERMEDIATE = 64
_VOCAB = 64


def _entry(name: str, rows: int, features: int, file: str, sha256: str) -> ActivationCaptureEntry:
    return ActivationCaptureEntry(
        module_path=name,
        rows=rows,
        in_features=features,
        file=file,
        sha256=sha256,
    )


def _manifest(
    entries: tuple[ActivationCaptureEntry, ...], **overrides: object
) -> ActivationCaptureManifest:
    fields: dict[str, object] = {
        "model": "test-model",
        "revision": "rev1",
        "tokenized_cache_manifest_sha256": "a" * 64,
        "cache_key_sha256": "b" * 64,
        "calibration_dataset_id": "calibration-dataset",
        "max_rows": 16,
        "entries": entries,
    }
    fields.update(overrides)
    return ActivationCaptureManifest(**fields)  # type: ignore[arg-type]


def _write_capture_dir(root: Path) -> Path:
    capture = root / "capture"
    activations = capture / CAPTURE_ACTIVATIONS_DIR
    activations.mkdir(parents=True)
    rng = np.random.default_rng(0)
    specs = (
        ("model.layers.0.mlp.down_proj", 4, 8),
        ("model.layers.0.self_attn.q_proj", 6, 8),
    )
    entries: list[ActivationCaptureEntry] = []
    for index, (name, rows, features) in enumerate(specs):
        x_rows = rng.standard_normal((rows, features)).astype(np.float16)
        filename = f"{index:04d}-{name.replace('.', '-')}.npz"
        np.savez(activations / filename, x_rows=x_rows)
        entries.append(_entry(name, rows, features, filename, file_sha256(activations / filename)))
    write_data(capture / CAPTURE_MANIFEST_NAME, _manifest(tuple(entries)))
    return capture


class TestManifestSchema:
    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            _manifest((), unexpected_field="nope")

    def test_round_trip_via_serde(self, tmp_path: Path) -> None:
        manifest = _manifest(
            (
                _entry("model.layers.0.mlp.down_proj", 4, 8, "0000-a.npz", "c" * 64),
                _entry("model.layers.0.self_attn.q_proj", 6, 8, "0001-b.npz", "d" * 64),
            )
        )
        path = tmp_path / CAPTURE_MANIFEST_NAME
        write_data(path, manifest)
        loaded = load_model(path, ActivationCaptureManifest)
        assert loaded.model_dump(mode="json") == manifest.model_dump(mode="json")
        assert isinstance(loaded.entries, tuple)
        assert loaded.schema_version == "axquant.activation-capture.v1"
        assert loaded.entries[0].module_path == "model.layers.0.mlp.down_proj"


class TestLoadCaptureActivations:
    def test_happy_path_returns_fp16_mapping(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        loaded = load_capture_activations(capture, model="test-model", revision="rev1")
        assert set(loaded) == {
            "model.layers.0.mlp.down_proj",
            "model.layers.0.self_attn.q_proj",
        }
        assert loaded["model.layers.0.mlp.down_proj"].shape == (4, 8)
        assert loaded["model.layers.0.mlp.down_proj"].dtype == np.float16

    def test_model_mismatch_fails_closed(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        with pytest.raises(CaptureError, match="does not match"):
            load_capture_activations(capture, model="other-model")

    def test_revision_mismatch_fails_closed(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        with pytest.raises(CaptureError, match="revision"):
            load_capture_activations(capture, model="test-model", revision="rev2")

    def test_tampered_npz_fails_closed(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        manifest = load_model(capture / CAPTURE_MANIFEST_NAME, ActivationCaptureManifest)
        target = capture / CAPTURE_ACTIVATIONS_DIR / manifest.entries[0].file
        np.savez(target, x_rows=np.zeros((4, 8), dtype=np.float16))
        with pytest.raises(CaptureError, match="checksum mismatch"):
            load_capture_activations(capture, model="test-model")

    def test_missing_npz_fails_closed(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        manifest = load_model(capture / CAPTURE_MANIFEST_NAME, ActivationCaptureManifest)
        (capture / CAPTURE_ACTIVATIONS_DIR / manifest.entries[1].file).unlink()
        with pytest.raises(CaptureError, match="missing"):
            load_capture_activations(capture, model="test-model")


class _SmallTokenizer:
    """Deterministic offline tokenizer whose ids stay inside the tiny vocab."""

    pad_token_id = 0
    eos_token_id = 2
    special_tokens_map: ClassVar[dict[str, str]] = {
        "eos_token": "</s>",
        "pad_token": "<pad>",
    }

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "def": 1, "</s>": 2}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> list[int]:
        assert add_special_tokens
        assert truncation
        ids = [len(word) % (_VOCAB - 4) + 3 for word in text.split()]
        return [*ids, self.eos_token_id][:max_length]


def _write_tiny_llama(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": _HIDDEN,
                "intermediate_size": _INTERMEDIATE,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": _VOCAB,
                "max_position_embeddings": 128,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)

    def weight(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    save_file(
        {
            "model.embed_tokens.weight": weight(_VOCAB, _HIDDEN),
            "model.layers.0.self_attn.q_proj.weight": weight(_HIDDEN, _HIDDEN),
            "model.layers.0.self_attn.k_proj.weight": weight(16, _HIDDEN),
            "model.layers.0.self_attn.v_proj.weight": weight(16, _HIDDEN),
            "model.layers.0.self_attn.o_proj.weight": weight(_HIDDEN, _HIDDEN),
            "model.layers.0.mlp.gate_proj.weight": weight(_INTERMEDIATE, _HIDDEN),
            "model.layers.0.mlp.up_proj.weight": weight(_INTERMEDIATE, _HIDDEN),
            "model.layers.0.mlp.down_proj.weight": weight(_HIDDEN, _INTERMEDIATE),
            "model.layers.0.input_layernorm.weight": np.ones((_HIDDEN,), dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones((_HIDDEN,), dtype=np.float32),
            "model.norm.weight": np.ones((_HIDDEN,), dtype=np.float32),
        },
        str(model_dir / "model.safetensors"),
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "BPE",
                    "vocab": {f"tok{i}": i for i in range(_VOCAB)},
                    "merges": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 128}),
        encoding="utf-8",
    )


@pytest.fixture
def mlx_calibration_cache(tmp_path: Path) -> Path:
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"text": text})
            for text in (
                "def sort_list(items): return sorted(items)",
                "Fix the bug in this function",
                "Generate a JSON response",
            )
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    tokenize_calibration(
        model=ModelIdentity(model_id="tiny-llama", revision="rev0"),
        dataset_path=dataset,
        output_dir=cache_dir,
        profile=ProfileName.AGENT_CODING,
        sequence_length=32,
        random_seed=7,
        tokenizer=_SmallTokenizer(),
    )
    return cache_dir


class TestCaptureEndToEnd:
    def test_capture_and_load_round_trip(self, tmp_path: Path, mlx_calibration_cache: Path) -> None:
        pytest.importorskip("mlx.core")
        model_dir = tmp_path / "tiny-llama"
        _write_tiny_llama(model_dir)
        output_dir = tmp_path / "capture"

        manifest = capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=mlx_calibration_cache,
            output_dir=output_dir,
            max_rows=8,
        )

        assert manifest.model == "tiny-llama"
        assert manifest.revision == "rev0"
        assert manifest.cache_key_sha256
        assert manifest.tokenized_cache_manifest_sha256
        assert manifest.calibration_dataset_id
        assert manifest.entries
        by_path = {entry.module_path: entry for entry in manifest.entries}
        assert "model.layers.0.mlp.down_proj" in by_path
        assert "model.layers.0.self_attn.q_proj" in by_path
        assert not any("embed" in path or "lm_head" in path for path in by_path)
        for entry in manifest.entries:
            assert 0 < entry.rows <= 8
            assert (output_dir / CAPTURE_ACTIVATIONS_DIR / entry.file).is_file()
        assert by_path["model.layers.0.self_attn.q_proj"].in_features == _HIDDEN
        assert by_path["model.layers.0.mlp.down_proj"].in_features == _INTERMEDIATE

        loaded = load_capture_activations(output_dir, model="tiny-llama", revision="rev0")
        assert set(loaded) == set(by_path)
        for entry in manifest.entries:
            rows = loaded[entry.module_path]
            assert rows.shape == (entry.rows, entry.in_features)
            assert rows.dtype == np.float16

    def test_target_modules_filter_and_unresolved_target(
        self, tmp_path: Path, mlx_calibration_cache: Path
    ) -> None:
        pytest.importorskip("mlx.core")
        model_dir = tmp_path / "tiny-llama"
        _write_tiny_llama(model_dir)

        manifest = capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=mlx_calibration_cache,
            output_dir=tmp_path / "capture",
            target_modules=["model.layers.0.mlp.down_proj"],
            max_rows=4,
        )
        assert [entry.module_path for entry in manifest.entries] == ["model.layers.0.mlp.down_proj"]

        with pytest.raises(CaptureError, match="did not resolve"):
            capture_calibration_activations(
                model_dir=model_dir,
                cache_dir=mlx_calibration_cache,
                output_dir=tmp_path / "capture-missing",
                target_modules=["model.layers.9.mlp.down_proj"],
                max_rows=4,
            )
