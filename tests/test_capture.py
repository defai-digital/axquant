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
    CAPTURE_COMPLETION_SCHEMA,
    CAPTURE_MANIFEST_NAME,
    capture_calibration_activations,
    load_capture_activations,
)
from axquant.errors import CaptureError
from axquant.schema import (
    ActivationCaptureEntry,
    ActivationCaptureManifest,
    CaptureProgress,
    ModelIdentity,
    ProfileName,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

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


def _write_completion(capture: Path, manifest: ActivationCaptureManifest) -> None:
    (capture / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": CAPTURE_COMPLETION_SCHEMA,
                "complete": True,
                "cache_key_sha256": manifest.cache_key_sha256,
                "manifest_sha256": stable_sha256(manifest),
                "modules": len(manifest.entries),
                "rows": sum(entry.rows for entry in manifest.entries),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


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
    manifest = _manifest(tuple(entries))
    write_data(capture / CAPTURE_MANIFEST_NAME, manifest)
    _write_completion(capture, manifest)
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

    def test_rejects_duplicate_modules_and_rows_above_limit(self) -> None:
        entry = _entry("model.layers.0.mlp.down_proj", 4, 8, "a.npz", "c" * 64)
        with pytest.raises(ValidationError, match="module paths must be unique"):
            _manifest((entry, entry))
        with pytest.raises(ValidationError, match="exceed max_rows"):
            _manifest((_entry("model.layers.0.mlp.up_proj", 17, 8, "b.npz", "d" * 64),))


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
        assert loaded.manifest_sha256 == stable_sha256(loaded.manifest)
        assert loaded.source_dir == capture.resolve()
        assert loaded["model.layers.0.mlp.down_proj"].flags.writeable is False

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

    def test_missing_completion_marker_fails_closed(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        (capture / "completion.json").unlink()
        with pytest.raises(CaptureError, match="incomplete"):
            load_capture_activations(capture, model="test-model")

    def test_completion_marker_manifest_mismatch_fails_closed(self, tmp_path: Path) -> None:
        capture = _write_capture_dir(tmp_path)
        marker_path = capture / "completion.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["manifest_sha256"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        with pytest.raises(CaptureError, match="manifest checksum mismatch"):
            load_capture_activations(capture, model="test-model")

    def test_sharded_layout_round_trip(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture"
        activations = capture / CAPTURE_ACTIVATIONS_DIR
        activations.mkdir(parents=True)
        rng = np.random.default_rng(1)
        specs = (
            ("model.layers.0.mlp.down_proj", 4, 8),
            ("model.layers.0.self_attn.q_proj", 6, 8),
        )
        arrays = {
            f"rows::{name}": rng.standard_normal((rows, features)).astype(np.float16)
            for name, rows, features in specs
        }
        np.savez_compressed(activations / "shard-0000.npz", **arrays)
        shard_sha256 = file_sha256(activations / "shard-0000.npz")
        entries = tuple(
            ActivationCaptureEntry(
                module_path=name,
                rows=rows,
                in_features=features,
                file="shard-0000.npz",
                sha256=shard_sha256,
                array_key=f"rows::{name}",
            )
            for name, rows, features in specs
        )
        manifest = _manifest(entries)
        write_data(capture / CAPTURE_MANIFEST_NAME, manifest)
        _write_completion(capture, manifest)
        loaded = load_capture_activations(capture, model="test-model", revision="rev1")
        assert set(loaded) == {name for name, _, _ in specs}
        for name, _, _ in specs:
            np.testing.assert_array_equal(loaded[name], arrays[f"rows::{name}"])

    def test_sharded_array_key_mismatch_fails_closed(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture"
        activations = capture / CAPTURE_ACTIVATIONS_DIR
        activations.mkdir(parents=True)
        np.savez_compressed(
            activations / "shard-0000.npz",
            **{"rows::model.layers.0.mlp.down_proj": np.zeros((4, 8), dtype=np.float16)},
        )
        entries = (
            ActivationCaptureEntry(
                module_path="model.layers.0.mlp.down_proj",
                rows=4,
                in_features=8,
                file="shard-0000.npz",
                sha256=file_sha256(activations / "shard-0000.npz"),
                array_key="rows::model.layers.0.self_attn.q_proj",
            ),
        )
        manifest = _manifest(entries)
        write_data(capture / CAPTURE_MANIFEST_NAME, manifest)
        _write_completion(capture, manifest)
        with pytest.raises(CaptureError, match="array key mismatch"):
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


@pytest.fixture
def mlx_multi_segment_cache(tmp_path: Path) -> Path:
    """Cache with enough packed batches to span several replay segments."""
    dataset = tmp_path / "calibration-multi.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"text": text})
            for text in (
                "def sort_list(items): return sorted(items) and filter them all",
                "Fix the bug in this function and add regression coverage now",
                "Generate a JSON response with every field populated fully",
                "Refactor the parser into smaller composable helper pieces",
                "Write unit tests for the tokenizer edge cases today please",
                "Document the public API surface with accurate examples here",
                "Profile the hot loop and remove the extra allocations soon",
                "Validate all checksums before publishing the artifact bundle",
                "Handle empty inputs gracefully across every code path now",
                "Cache the compiled graph to avoid repeated tracing overhead",
                "Clamp the gradients before the optimizer step is applied",
                "Shuffle the calibration samples with a deterministic seed",
            )
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache-multi"
    tokenize_calibration(
        model=ModelIdentity(model_id="tiny-llama", revision="rev0"),
        dataset_path=dataset,
        output_dir=cache_dir,
        profile=ProfileName.AGENT_CODING,
        sequence_length=16,
        random_seed=7,
        tokenizer=_SmallTokenizer(),
    )
    return cache_dir


class TestCaptureResume:
    def test_resume_matches_single_shot(
        self,
        tmp_path: Path,
        mlx_multi_segment_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("mlx.core")
        model_dir = tmp_path / "tiny-llama"
        _write_tiny_llama(model_dir)
        output_dir = tmp_path / "capture"
        control_dir = tmp_path / "capture-control"

        import axquant.capture as capture_module

        real_replay = capture_module._replay_segment
        calls = {"count": 0}

        def boom(model: object, mlx: object, batches: list[object]) -> None:
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("simulated interruption")
            real_replay(model, mlx, batches)

        monkeypatch.setattr(capture_module, "_replay_segment", boom)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            capture_calibration_activations(
                model_dir=model_dir,
                cache_dir=mlx_multi_segment_cache,
                output_dir=output_dir,
                max_rows=8,
                segment_batches=1,
            )
        progress = load_model(output_dir / "capture_progress.json", CaptureProgress)
        assert progress.segments_completed == 2
        assert (output_dir / CAPTURE_ACTIVATIONS_DIR / ".partial").is_dir()
        assert not (output_dir / "completion.json").exists()
        with pytest.raises(CaptureError, match="incomplete"):
            load_capture_activations(output_dir, model="tiny-llama")

        monkeypatch.undo()
        resumed = capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=mlx_multi_segment_cache,
            output_dir=output_dir,
            max_rows=8,
            segment_batches=1,
        )
        assert not (output_dir / "capture_progress.json").exists()
        assert not (output_dir / CAPTURE_ACTIVATIONS_DIR / ".partial").exists()
        assert (output_dir / "completion.json").is_file()

        control = capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=mlx_multi_segment_cache,
            output_dir=control_dir,
            max_rows=8,
            segment_batches=1,
        )
        assert resumed.model_dump(mode="json", exclude={"created_at"}) == control.model_dump(
            mode="json", exclude={"created_at"}
        )
        resumed_rows = load_capture_activations(output_dir, model="tiny-llama", revision="rev0")
        control_rows = load_capture_activations(control_dir, model="tiny-llama", revision="rev0")
        assert set(resumed_rows) == set(control_rows)
        for name in resumed_rows:
            np.testing.assert_array_equal(resumed_rows[name], control_rows[name])

    def test_resume_binding_mismatch_fails_closed(
        self,
        tmp_path: Path,
        mlx_multi_segment_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("mlx.core")
        model_dir = tmp_path / "tiny-llama"
        _write_tiny_llama(model_dir)
        output_dir = tmp_path / "capture"

        import axquant.capture as capture_module

        real_replay = capture_module._replay_segment
        calls = {"count": 0}

        def boom(model: object, mlx: object, batches: list[object]) -> None:
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("simulated interruption")
            real_replay(model, mlx, batches)

        monkeypatch.setattr(capture_module, "_replay_segment", boom)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            capture_calibration_activations(
                model_dir=model_dir,
                cache_dir=mlx_multi_segment_cache,
                output_dir=output_dir,
                max_rows=8,
                segment_batches=1,
            )
        assert (output_dir / "capture_progress.json").is_file()
        monkeypatch.undo()
        with pytest.raises(CaptureError, match="fresh output directory"):
            capture_calibration_activations(
                model_dir=model_dir,
                cache_dir=mlx_multi_segment_cache,
                output_dir=output_dir,
                max_rows=16,
                segment_batches=1,
            )


class TestCaptureSharding:
    def test_sharded_capture_round_trip(self, tmp_path: Path, mlx_calibration_cache: Path) -> None:
        pytest.importorskip("mlx.core")
        model_dir = tmp_path / "tiny-llama"
        _write_tiny_llama(model_dir)
        sharded_dir = tmp_path / "capture-sharded"
        control_dir = tmp_path / "capture-control"

        sharded = capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=mlx_calibration_cache,
            output_dir=sharded_dir,
            max_rows=8,
            modules_per_shard=3,
        )
        control = capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=mlx_calibration_cache,
            output_dir=control_dir,
            max_rows=8,
        )

        assert len(sharded.entries) == len(control.entries) == 7
        for entry in sharded.entries:
            assert entry.array_key == f"rows::{entry.module_path}"
            assert entry.file.startswith("shard-")
        shard_files = {entry.file for entry in sharded.entries}
        assert shard_files == {"shard-0000.npz", "shard-0001.npz", "shard-0002.npz"}
        by_file: dict[str, set[str]] = {}
        for entry in sharded.entries:
            by_file.setdefault(entry.file, set()).add(entry.sha256)
        assert all(len(digests) == 1 for digests in by_file.values())
        assert len(by_file["shard-0000.npz"]) == 1

        import zipfile

        shard_path = sharded_dir / CAPTURE_ACTIVATIONS_DIR / "shard-0000.npz"
        with zipfile.ZipFile(shard_path) as archive:
            assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())

        sharded_rows = load_capture_activations(sharded_dir, model="tiny-llama", revision="rev0")
        control_rows = load_capture_activations(control_dir, model="tiny-llama", revision="rev0")
        assert set(sharded_rows) == set(control_rows)
        for name in sharded_rows:
            np.testing.assert_array_equal(sharded_rows[name], control_rows[name])

    def test_invalid_segment_and_shard_sizes(self, tmp_path: Path) -> None:
        with pytest.raises(CaptureError, match="segment_batches"):
            capture_calibration_activations(
                model_dir=tmp_path,
                cache_dir=tmp_path,
                output_dir=tmp_path / "out",
                segment_batches=0,
            )
        with pytest.raises(CaptureError, match="modules_per_shard"):
            capture_calibration_activations(
                model_dir=tmp_path,
                cache_dir=tmp_path,
                output_dir=tmp_path / "out",
                modules_per_shard=0,
            )


class TestActivationRowsF16:
    """Regression: BF16 checkpoint activations must convert to numpy."""

    def test_bfloat16_mlx_array_converts_to_fp16_rows(self) -> None:
        mx = pytest.importorskip("mlx.core")
        from axquant.capture import _activation_rows_f16

        x = mx.array([[1.5, -2.25], [3.0, 4.5]], dtype=mx.bfloat16)
        rows = _activation_rows_f16(mx, x)
        assert rows.dtype == np.float16
        assert rows.shape == (2, 2)
        np.testing.assert_allclose(rows, [[1.5, -2.25], [3.0, 4.5]], rtol=1e-3)

    def test_bfloat16_direct_numpy_conversion_still_fails(self) -> None:
        """Guards the regression premise: raw np.asarray on bf16 must stay broken."""
        mx = pytest.importorskip("mlx.core")

        x = mx.array([[1.0]], dtype=mx.bfloat16)
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            np.asarray(x.reshape(-1, 1), dtype=np.float16)
