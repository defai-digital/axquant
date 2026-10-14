"""Development storage packs must decode exactly and fail on file tampering."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp4", "mxfp8"])
def test_storage_roundtrip_and_digest_rejection(tmp_path, mode):
    from axquant.nvfp4 import dequantize_nvfp4, quantize_nvfp4_reference

    weight = np.random.default_rng(42).normal(size=(2, 29)).astype(np.float32)
    matrix = np.pad(weight, ((0, 0), (0, 3)))
    entry = {"shape": [2, 29], "padded_columns": 32, "dtype": "float32", "method": mode}
    if mode == "nvfp4":
        q = quantize_nvfp4_reference(matrix)
        packed, scales = q.packed, q.scale_bytes
        entry["global_scale"] = q.global_scale
        expected = dequantize_nvfp4(q)[:, :29]
    else:
        mx = pytest.importorskip("mlx.core")
        bits = 4 if mode == "mxfp4" else 8
        p, s = mx.quantize(mx.array(matrix), group_size=32, bits=bits, mode=mode)
        packed, scales = np.array(p), np.array(s)
        expected = np.array(
            mx.dequantize(p, s, group_size=32, bits=bits, mode=mode).astype(mx.float32)
        )[:, :29]
    entry["group_size"] = 16 if mode == "nvfp4" else 32
    entry["max_absolute_error"] = float(np.max(np.abs(expected - weight)))
    file = tmp_path / "model.safetensors"
    storage = {
        "w.axq_packed": packed,
        "w.axq_scales": scales,
        "decoder.anchors": np.array([np.inf], dtype=np.float32),
    }
    save_file(storage, str(file))
    manifest = {
        "schema_version": "axq-doclayout.development/1",
        "format": mode,
        "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()},
        "tensors": {
            "w": entry,
            "decoder.anchors": {
                "shape": [1],
                "dtype": "float32",
                "method": "preserved",
            },
        },
        "source_model": "AutomatosX/AXDocLayout-M-v1.0",
        "source_revision": "69ddc0da17a7fc35dfde29b06adbb83a75284ae1",
        "source_weight_sha256": "4d9ee36db616f3f4ef293588e4258b5f6f42d7a84097a5afb520779076045bf2",
        "quantized_parameters": 58,
        "total_parameters_and_buffers": 59,
        "max_absolute_weight_error": entry["max_absolute_error"],
    }
    (tmp_path / "axq-manifest.json").write_text(json.dumps(manifest))
    from axquant import doclayout_reference as loader

    np.testing.assert_array_equal(loader.read_state(tmp_path)["w"], expected)
    assert np.isposinf(loader.read_state(tmp_path)["decoder.anchors"]).all()
    # Hub snapshots point at sibling blobs, while arbitrary symlink escapes fail.
    cache = tmp_path / "models--AutomatosX--AXDocLayout-M-v1.0-AXQ-NVFP4"
    snapshot = cache / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    blobs = cache / "blobs"
    blobs.mkdir()
    blob = blobs / hashlib.sha256(file.read_bytes()).hexdigest()
    blob.write_bytes(file.read_bytes())
    (snapshot / file.name).symlink_to(blob)
    (snapshot / "axq-manifest.json").write_text(json.dumps(manifest))
    np.testing.assert_array_equal(loader.read_state(snapshot)["w"], expected)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(file.read_bytes())
    (snapshot / file.name).unlink()
    (snapshot / file.name).symlink_to(outside)
    with pytest.raises(ValueError, match="Invalid pack file"):
        loader.read_state(snapshot)
    manifest["quality_certified"] = True
    (tmp_path / "axq-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quality_certified"):
        loader.read_state(tmp_path)
    manifest["quality_certified"] = False
    (tmp_path / "axq-manifest.json").write_text(json.dumps(manifest))
    content = bytearray(file.read_bytes())
    content[-1] ^= 1
    file.write_bytes(content)
    with pytest.raises(ValueError, match="digest mismatch"):
        loader.read_state(tmp_path)
    storage["decoder.anchors"][0] = np.nan
    save_file(storage, str(file))
    manifest["files"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path / "axq-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Nonfinite decoded weight"):
        loader.read_state(tmp_path)


def test_detection_heads_and_buffers_are_protected():
    from axquant import doclayout as converter

    matrix = np.ones((4, 32), dtype=np.float32)
    assert converter.eligible("encoder.linear1.weight", matrix)
    assert converter.eligible("encoder.attn.in_proj_weight", matrix)
    assert not converter.eligible("decoder.enc_score_head.weight", matrix)
    assert not converter.eligible("decoder.dec_bbox_head.0.layers.0.weight", matrix)
    assert not converter.eligible("decoder.anchors", matrix)
    assert not converter.eligible("norm.weight", np.ones(32, dtype=np.float32))


def test_cli_requires_explicit_unmeasured_opt_in(tmp_path, capsys):
    from axquant.cli import main
    from axquant.doclayout import REVISION

    assert (
        main(
            [
                "quantize-doclayout",
                "--model",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
                "--revision",
                REVISION,
                "--q-mode",
                "nvfp4",
            ]
        )
        == 2
    )
    assert "requires --allow-unmeasured" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("revision", ["main", "v1.0", "0" * 40])
def test_unadmitted_revisions_fail_before_conversion(tmp_path, revision):
    from axquant.doclayout import convert_doclayout
    from axquant.errors import ArtifactError

    with pytest.raises(ArtifactError, match="immutable source revision"):
        convert_doclayout(
            tmp_path, tmp_path / "out", q_mode="nvfp4", revision=revision, allow_unmeasured=True
        )


def test_changed_source_metadata_is_rejected(tmp_path):
    from axquant.doclayout import verify_source
    from axquant.errors import ArtifactError

    for name in ["config.json", "taxonomy.json", "preprocessor_config.json", "LICENSE", "NOTICE"]:
        (tmp_path / name).write_text("tampered")
    with pytest.raises(ArtifactError, match="differs from admitted revision"):
        verify_source(tmp_path)


def test_missing_numpy_reports_the_install_extra(monkeypatch, tmp_path):
    import builtins

    from axquant.doclayout import REVISION, convert_doclayout
    from axquant.errors import BackendUnavailableError

    original = builtins.__import__

    def without_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("NumPy not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_numpy)
    with pytest.raises(BackendUnavailableError, match=r"install axquant\[doclayout\]"):
        convert_doclayout(
            tmp_path, tmp_path / "out", q_mode="nvfp4", revision=REVISION, allow_unmeasured=True
        )


def test_schema_rejects_invalid_layout_and_missing_scale():
    from pydantic import ValidationError

    from axquant.schema.doclayout import DocLayoutTensor

    with pytest.raises(ValidationError, match="Invalid quantized tensor layout"):
        DocLayoutTensor(
            shape=(2, 29),
            dtype="float32",
            method="mxfp4",
            group_size=16,
            padded_columns=32,
            max_absolute_error=0.1,
        )
    with pytest.raises(ValidationError, match="Only NVFP4"):
        DocLayoutTensor(
            shape=(2, 29),
            dtype="float32",
            method="nvfp4",
            group_size=16,
            padded_columns=32,
            max_absolute_error=0.1,
        )
