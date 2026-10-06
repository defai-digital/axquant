from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from axquant.cli import main
from axquant.errors import ArtifactError
from axquant.ngram_layout import _parse_safetensors_file, relayout_ngram_table
from axquant.runtime_compatibility import (
    inspect_runtime_export,
    require_runtime_compatibility_record,
    write_runtime_compatibility,
)
from axquant.runtime_export import export_runtime_pack
from axquant.serde import file_sha256


def _pack(directory: Path, *, mode: str = "affine", model_type: str = "qwen4_exp") -> Path:
    directory.mkdir()
    tensors = {}
    quantization = {"bits": 6, "group_size": 32}
    for shard in range(12):
        prefix = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{shard}"
        # Deliberately distinct rows reveal lexicographic ordering (0, 1, 10, 11, 2).
        tensors[f"{prefix}.weight"] = np.full((2, 16), shard, dtype=np.uint32)
        tensors[f"{prefix}.scales"] = np.full((2, 2), shard + 1, dtype=np.float32)
        tensors[f"{prefix}.biases"] = np.full((2, 2), -shard, dtype=np.float32)
        quantization[prefix] = {"bits": 8, "group_size": 32, "mode": "affine"}
    tensors["language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"] = np.zeros(
        4, dtype=np.float32
    )
    tensors["language_model.model.layers.0.self_attn.q_proj.weight"] = np.zeros(
        (4, 4), dtype=np.float32
    )
    quantization["language_model.model.layers.0.mlp.switch_mlp.down_proj"] = {
        "bits": 4,
        "group_size": 32,
        "mode": mode,
    }
    save_file(tensors, directory / "model.safetensors")
    mtp = {
        "mtp.pre_fc_norm_embedding.weight": np.zeros(4, dtype=np.float32),
        "mtp.pre_fc_norm_hidden.weight": np.zeros(16, dtype=np.float32),
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight": np.zeros(16, dtype=np.float32),
        "mtp.layers.0.mlp.experts.gate_up_proj": np.arange(48, dtype=np.float32).reshape(2, 6, 4),
        "mtp.layers.0.mlp.experts.down_proj": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
    }
    save_file(mtp, directory / "mtp.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
                "weight_map": dict.fromkeys(tensors, "model.safetensors"),
            }
        )
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "text_config": {
                    "split_ngram_parts": 12,
                    "hidden_size": 4,
                    "moe_intermediate_size": 3,
                    "num_experts": 2,
                    "mlx_lm_extra_tensors": list(mtp),
                },
                "quantization": quantization,
                "quantization_config": quantization,
                "mlx_lm_extra_tensors": list(mtp),
            }
        )
    )
    (directory / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen4-exp-mtp",
                "mtp_norm_layout": "raw_hf_delta",
                "trunk_norm_layout": "raw_hf_delta",
                "mtp_depth_max": 1,
            }
        )
    )
    (directory / "axquant_manifest.json").write_text('{"source": "original"}')
    (directory / "runtime_check.json").write_text('{"passed": true}')
    return directory


def test_numeric_concatenation_preserves_eight_bit_recipe_and_row_bytes(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    output = tmp_path / "variant"
    relayout_ngram_table(source, output)
    layout = _parse_safetensors_file(output / "ngram-table.safetensors")
    assert layout.metadata == {"format": "mlx", "ngram_bits": "8", "ngram_group_size": "32"}
    table = load_file(output / "ngram-table.safetensors")
    original = load_file(source / "model.safetensors")
    for component in ("weight", "scales", "biases"):
        expected = np.concatenate(
            [
                original[
                    f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{i}.{component}"
                ]
                for i in range(12)
            ]
        )
        np.testing.assert_array_equal(table[f"ngram.{component}"], expected)
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["ngram_sidecar"] is True
    assert not any("ngram_embedding" in key for key in config["quantization"])


def test_omlx_variant_exposes_native_mtp_to_index_loader_and_preserves_payloads(
    tmp_path: Path,
) -> None:
    source = _pack(tmp_path / "source", mode="mxfp4")
    original = {p.name: file_sha256(p) for p in source.iterdir()}
    output = tmp_path / "omlx"
    report = export_runtime_pack(source, output, target="omlx")
    assert report["targets"]["omlx"]["status"] == "static-compatible"
    assert report["targets"]["omlx"]["runtime_verified"] is False
    index = json.loads((output / "model.safetensors.index.json").read_text())
    mtp = load_file(output / "mtp.safetensors")
    assert all(index["weight_map"][name] == "mtp.safetensors" for name in mtp)
    assert file_sha256(output / "mtp.safetensors") == original["mtp.safetensors"]
    assert file_sha256(output / "model.safetensors") == original["model.safetensors"]
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert config["mlx_lm_extra_tensors"] == []
    assert config["text_config"]["mlx_lm_extra_tensors"] == []
    assert not (output / "axquant_manifest.json").exists()
    assert not (output / "runtime_check.json").exists()
    receipt = json.loads((output / "axquant_runtime_export.json").read_text())
    assert receipt["model_settings"] == {"qwen4_ple_ssd_offload": True}
    assert {p.name: file_sha256(p) for p in source.iterdir()} == original


@pytest.mark.parametrize("mode", ["affine", "mxfp4"])
def test_mtplx_variant_rebases_norms_and_splits_head_experts_without_requantization(
    tmp_path: Path,
    mode: str,
) -> None:
    source = _pack(tmp_path / "source", mode=mode)
    output = tmp_path / "mtplx"
    report = export_runtime_pack(source, output, target="mtplx")
    assert report["targets"]["mtplx"]["status"] == "static-compatible"
    trunk = load_file(output / "model.safetensors")
    assert not any("ngram_embedding" in name for name in trunk)
    np.testing.assert_array_equal(
        trunk["language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"], np.ones(4)
    )
    before, after = load_file(source / "mtp.safetensors"), load_file(output / "mtp.safetensors")
    combined = before["mtp.layers.0.mlp.experts.gate_up_proj"]
    np.testing.assert_array_equal(
        after["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"], combined[:, :3, :]
    )
    np.testing.assert_array_equal(
        after["mtp.layers.0.mlp.switch_mlp.up_proj.weight"], combined[:, 3:, :]
    )
    np.testing.assert_array_equal(
        after["mtp.layers.0.mlp.switch_mlp.down_proj.weight"],
        before["mtp.layers.0.mlp.experts.down_proj"],
    )
    np.testing.assert_array_equal(
        after["mtp.pre_fc_norm_embedding.weight"], before["mtp.pre_fc_norm_embedding.weight"]
    )
    np.testing.assert_array_equal(
        after["mtp.layers.0.attn_hyper_connection.hc_norm.weight"], np.ones(16)
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert not any(name.startswith("mtp.") for name in index["weight_map"])
    receipt = json.loads((output / "axquant_runtime_export.json").read_text())
    assert receipt["runtime_verified"] is False
    assert receipt["launch_environment"]["MTPLX_NGRAM_RESIDENT"] == "0"
    assert receipt["launch_environment"]["MTPLX_QWEN4_FIXED_M4_VERIFY"] == "0"
    assert receipt["modality_constraints"] == ["MTPLX image input requires MTP generation mode"]
    config = json.loads((output / "config.json").read_text())
    assert (
        config["quantization"]["language_model.model.layers.0.mlp.switch_mlp.down_proj"]["mode"]
        == mode
    )
    assert all(file_sha256(output / name) == digest for name, digest in receipt["files"].items())


@pytest.mark.parametrize("model_type", ["unregistered_future_family", "nemotron_h"])
def test_future_architectures_fail_closed_without_explicit_profile(
    tmp_path: Path, model_type: str
) -> None:
    source = _pack(tmp_path / "source", model_type=model_type)
    report = inspect_runtime_export(source)
    assert report["targets"]["omlx"]["status"] == "unsupported"
    for target in ("omlx", "mtplx"):
        with pytest.raises(ArtifactError, match="No audited"):
            export_runtime_pack(source, tmp_path / target, target=target, dry_run=True)
        assert not (tmp_path / target).exists()


def test_mtplx_mxfp8_is_rejected_before_writes(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source", mode="mxfp8")
    output = tmp_path / "mtplx"
    with pytest.raises(ArtifactError, match="supports affine/MXFP4 trunk only"):
        export_runtime_pack(source, output, target="mtplx")
    assert not output.exists()


@pytest.mark.parametrize(
    "fault", ["missing-component", "mixed-bits", "unsafe-path", "missing-shard"]
)
def test_relayout_rejects_incomplete_or_ambiguous_tables(tmp_path: Path, fault: str) -> None:
    source = _pack(tmp_path / "source")
    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    index, config = json.loads(index_path.read_text()), json.loads(config_path.read_text())
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    if fault == "missing-component":
        data = load_file(source / "model.safetensors")
        data.pop(prefix + ".scales")
        save_file(data, source / "model.safetensors")
        index["weight_map"].pop(prefix + ".scales")
    elif fault == "mixed-bits":
        config["quantization"][prefix]["bits"] = 4
    elif fault == "unsafe-path":
        index["weight_map"][prefix + ".weight"] = "../escape.safetensors"
    else:
        config["text_config"]["split_ngram_parts"] = 13
    index_path.write_text(json.dumps(index))
    config_path.write_text(json.dumps(config))
    with pytest.raises(ArtifactError):
        relayout_ngram_table(source, tmp_path / "variant", dry_run=True)
    assert not (tmp_path / "variant").exists()


def test_export_runtime_cli_dry_run_and_current_header_check(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    assert (
        main(
            [
                "export-runtime",
                "--directory",
                str(source),
                "--output",
                str(tmp_path / "out"),
                "--target",
                "omlx",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (tmp_path / "out").exists()
    assert (
        main(
            [
                "check-runtime-compatibility",
                "--directory",
                str(source),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    assert (
        json.loads((tmp_path / "report.json").read_text())["targets"]["omlx"]["status"]
        == "requires-export"
    )


def test_manifest_gate_rejects_absent_unbound_and_stale_reports(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    with pytest.raises(ArtifactError, match="no runtime compatibility record"):
        require_runtime_compatibility_record(source)
    write_runtime_compatibility(source)
    with pytest.raises(ArtifactError, match="not bound"):
        require_runtime_compatibility_record(source)
    manifest_path = source / "axquant_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "axquant_compatibility.json",
                        "sha256": file_sha256(source / "axquant_compatibility.json"),
                    }
                ]
            }
        )
    )
    require_runtime_compatibility_record(source)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["model_type"] = "new_future_architecture"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ArtifactError, match="stale"):
        require_runtime_compatibility_record(source)


def test_omlx_preserves_companion_pointer_objects(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["mlx_lm_extra_tensors"] = {"mtp_file": "mtp.safetensors"}
    config_path.write_text(json.dumps(config))
    output = tmp_path / "omlx"
    export_runtime_pack(source, output, target="omlx")
    assert json.loads((output / "config.json").read_text())["mlx_lm_extra_tensors"] == {
        "mtp_file": "mtp.safetensors"
    }


def test_dry_run_rejects_undeclared_norms_without_copying(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    runtime_path = source / "mtplx_runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime.pop("trunk_norm_layout")
    runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(ArtifactError, match="declared raw_hf_delta trunk"):
        export_runtime_pack(source, tmp_path / "out", target="mtplx", dry_run=True)
    assert not (tmp_path / "out").exists()


def test_explicit_legacy_norm_declaration_is_bound_without_mutating_source(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    runtime_path = source / "mtplx_runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime.pop("trunk_norm_layout")
    runtime_path.write_text(json.dumps(runtime))
    original = file_sha256(runtime_path)
    output = tmp_path / "legacy-variant"
    export_runtime_pack(source, output, target="mtplx", source_trunk_norm_layout="raw_hf_delta")
    receipt = json.loads((output / "axquant_runtime_export.json").read_text())
    assert receipt["source_trunk_norm_layout"] == "raw_hf_delta"
    assert receipt["source_binding"]["mtplx_runtime.json"] == original
    assert file_sha256(runtime_path) == original


def test_explicit_norm_declaration_cannot_override_existing_contract(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    with pytest.raises(ArtifactError, match="conflicts"):
        export_runtime_pack(
            source,
            tmp_path / "out",
            target="omlx",
            source_trunk_norm_layout="mlx_multiplier",
            dry_run=True,
        )


def test_existing_canonical_table_is_checked_from_actual_header(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    output = tmp_path / "mtplx"
    export_runtime_pack(source, output, target="mtplx")
    table_path = output / "ngram-table.safetensors"
    table = load_file(table_path)
    save_file(table, table_path, metadata={"ngram_bits": "4", "ngram_group_size": "32"})
    verdict = inspect_runtime_export(output)["targets"]["mtplx"]
    assert verdict["status"] == "unsupported"
    assert any("disagrees" in issue for issue in verdict["blockers"])
    table_path.unlink()
    assert inspect_runtime_export(output)["targets"]["mtplx"]["status"] == "unsupported"


def test_atomic_export_failure_leaves_no_variant(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    data = load_file(source / "mtp.safetensors")
    # An alternate expert orientation must never be guessed or transposed.
    data["mtp.layers.0.mlp.experts.gate_up_proj"] = np.zeros((2, 4, 6), dtype=np.float32)
    save_file(data, source / "mtp.safetensors")
    original = {p.name: file_sha256(p) for p in source.iterdir()}
    with pytest.raises(ArtifactError, match="native floating"):
        export_runtime_pack(source, tmp_path / "out", target="mtplx")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.export.*"))
    assert {p.name: file_sha256(p) for p in source.iterdir()} == original


@pytest.mark.integration
def test_public_mlx_dequantization_preserves_cross_shard_rows(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        source = _pack(tmp_path / "source")
        output = tmp_path / "table"
        relayout_ngram_table(source, output)
        table = mx.load(str(output / "ngram-table.safetensors"))
        original = mx.load(str(source / "model.safetensors"))
        for row in (0, 1, 2, 19, 20, 21, 23):
            prefix = (
                f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{row // 2}"
            )
            before = mx.dequantize(
                original[prefix + ".weight"][row % 2 : row % 2 + 1],
                original[prefix + ".scales"][row % 2 : row % 2 + 1],
                original[prefix + ".biases"][row % 2 : row % 2 + 1],
                group_size=32,
                bits=8,
            )
            after = mx.dequantize(
                table["ngram.weight"][row : row + 1],
                table["ngram.scales"][row : row + 1],
                table["ngram.biases"][row : row + 1],
                group_size=32,
                bits=8,
            )
            mx.eval(before, after)
            np.testing.assert_array_equal(np.array(before), np.array(after))
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("Metal device is unavailable in this sandbox")
        raise
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.integration
def test_norm_rebase_matches_public_mlx_float32_rounding(tmp_path: Path, dtype_name: str) -> None:
    from axquant.ngram_layout import _TensorPart
    from axquant.runtime_export import _shift_norm

    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        raw = mx.array([0.0, 0.1, -0.2, -1.0, 0.333, -0.001], dtype=getattr(mx, dtype_name))
        path = tmp_path / "norm.safetensors"
        mx.save_safetensors(str(path), {"norm.weight": raw})
        layout = _parse_safetensors_file(path)
        shifted = _shift_norm(_TensorPart(path, layout, layout.tensors["norm.weight"]))
        expected = (raw.astype(mx.float32) + 1.0).astype(raw.dtype)
        mx.eval(expected)
        bits = expected.view(mx.uint32 if dtype_name == "float32" else mx.uint16)
        np.testing.assert_array_equal(
            np.frombuffer(
                shifted.payload, dtype=np.uint32 if dtype_name == "float32" else np.uint16
            ),
            np.array(bits),
        )
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("Metal device is unavailable in this sandbox")
        raise
    finally:
        mx.set_default_device(previous)
