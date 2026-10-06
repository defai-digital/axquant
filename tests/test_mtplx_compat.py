"""MTPLX config contract: container mode, MTP pointer, and n-gram table shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.errors import ArtifactError
from axquant.mtplx_compat import apply_mtplx_load_contract, ngram_table_matches_mtplx


def _config() -> dict:
    return {
        "model_type": "qwen3_5",
        "quantization": {
            "bits": 8,
            "group_size": 32,
            "mode": "affine",
            "language_model.model.layers.0.mlp.down_proj": {
                "bits": 8,
                "group_size": 32,
                "mode": "mxfp8",
            },
        },
        "quantization_config": {
            "bits": 8,
            "group_size": 32,
            "mode": "affine",
            "language_model.model.layers.0.mlp.down_proj": {
                "bits": 8,
                "group_size": 32,
                "mode": "mxfp8",
            },
        },
        "text_config": {"model_type": "qwen3_5_text"},
    }


def _write_safetensors(path: Path, header: dict) -> None:
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)


def test_container_mode_follows_uniform_mxfp8_modules(tmp_path: Path) -> None:
    config = _config()
    _write_safetensors(
        tmp_path / "mtp.safetensors", {"mtp.fc.weight": {"dtype": "BF16", "shape": [1]}}
    )

    notes = apply_mtplx_load_contract(config, tmp_path)

    assert config["quantization"]["mode"] == "mxfp8"
    assert config["quantization_config"]["mode"] == "mxfp8"
    assert config["mlx_lm_extra_tensors"] == {"mtp_file": "mtp.safetensors"}
    assert "ngram_file" not in config["mlx_lm_extra_tensors"]
    assert config["mtplx_mtp_contract"]["mtp_position_mode"] == "local"
    assert "mtp_quant_bits" not in config["mtplx_mtp_contract"]
    assert "mtplx_mtp_quantization" not in config
    assert notes


def test_mixed_module_modes_leave_the_container_alone(tmp_path: Path) -> None:
    config = _config()
    config["quantization"]["language_model.lm_head"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    config["quantization_config"]["language_model.lm_head"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }

    apply_mtplx_load_contract(config, tmp_path)

    assert config["quantization"]["mode"] == "affine"
    assert "mlx_lm_extra_tensors" not in config
    assert "mtplx_mtp_contract" not in config


def test_quantized_mtp_sidecar_is_not_declared_bf16(tmp_path: Path) -> None:
    config = _config()
    _write_safetensors(
        tmp_path / "mtp.safetensors",
        {
            "mtp.fc.weight": {"dtype": "U32", "shape": [1]},
            "mtp.fc.scales": {"dtype": "U8", "shape": [1]},
        },
    )

    apply_mtplx_load_contract(config, tmp_path)

    assert "mtplx_mtp_contract" not in config
    assert config["mlx_lm_extra_tensors"]["mtp_file"] == "mtp.safetensors"


def test_sharded_ngram_table_is_rejected(tmp_path: Path) -> None:
    config = _config()
    config["model_type"] = "qwen4_exp"
    config["text_config"] = {"model_type": "qwen4_exp_text", "ngram_sidecar": False}
    _write_safetensors(
        tmp_path / "ngram-table.safetensors",
        {"language_model.model.layers.0.ple.ngram_embedding.shards.0.weight": {"dtype": "U32"}},
    )

    with pytest.raises(ArtifactError, match="not an MTPLX n-gram table"):
        apply_mtplx_load_contract(config, tmp_path)
    assert "ngram_file" not in config.get("mlx_lm_extra_tensors", {})


def test_youssofal_ngram_table_is_declared(tmp_path: Path) -> None:
    header = {
        "__metadata__": {
            "ngram_bits": "4",
            "ngram_group_size": "32",
            "rows": "8",
            "dim": "4",
        },
        "ngram.weight": {"dtype": "U32", "shape": [8, 1], "data_offsets": [0, 32]},
        "ngram.scales": {"dtype": "BF16", "shape": [8, 1], "data_offsets": [32, 48]},
        "ngram.biases": {"dtype": "BF16", "shape": [8, 1], "data_offsets": [48, 64]},
    }
    path = tmp_path / "ngram-table.safetensors"
    _write_safetensors(path, header)
    assert ngram_table_matches_mtplx(path) is True
    config = {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": "qwen4_exp_text"},
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
    }

    apply_mtplx_load_contract(config, tmp_path)

    assert config["mlx_lm_extra_tensors"]["ngram_file"] == "ngram-table.safetensors"
    assert config["text_config"]["ngram_sidecar"] is True
    assert "mtplx_mtp_contract" not in config


def test_missing_declared_ngram_file_fails(tmp_path: Path) -> None:
    config = {
        "model_type": "qwen4_exp",
        "mlx_lm_extra_tensors": {"ngram_file": "ngram-table.safetensors"},
    }

    with pytest.raises(ArtifactError, match="absent"):
        apply_mtplx_load_contract(config, tmp_path)
