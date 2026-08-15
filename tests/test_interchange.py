from __future__ import annotations

import json
from pathlib import Path

from axquant.interchange import check_affine_u32_pack


def test_existing_affine_u32_fixture_conforms(packed_model_dir: Path) -> None:
    assert check_affine_u32_pack(packed_model_dir) == []


def test_non_affine_quantization_is_rejected(packed_model_dir: Path) -> None:
    config_path = packed_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization"]["mode"] = "mxfp4"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    issues = check_affine_u32_pack(packed_model_dir)

    assert any("non-affine" in issue for issue in issues)


def test_missing_affine_bias_metadata_is_rejected(packed_model_dir: Path) -> None:
    from safetensors import safe_open
    from safetensors.numpy import save_file

    weight_path = packed_model_dir / "model.safetensors"
    with safe_open(weight_path, framework="numpy") as handle:
        tensors = {
            name: handle.get_tensor(name)
            for name in list(handle.keys())
            if not name.endswith(".biases")
        }
    save_file(tensors, weight_path)

    issues = check_affine_u32_pack(packed_model_dir)

    assert any("lacks metadata" in issue for issue in issues)
