"""Grafted Qwen3.5 MoE MTP extract/pack + compose onto Holo3-class packs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.errors import ArtifactError
from axquant.grafted_mtp import (
    QWEN35_MOE_PACKED_MTP_SHAPES,
    compose_grafted_mtp_onto_pack,
    prepare_grafted_qwen_moe_mtp,
)
from axquant.schema import ModelIdentity


def _tiny_shapes(*, experts: int = 2, hidden: int = 4, inter: int = 3) -> dict[str, tuple[int, ...]]:
    return {
        "mtp.fc.weight": (hidden, hidden * 2),
        "mtp.layers.0.input_layernorm.weight": (hidden,),
        "mtp.layers.0.mlp.experts.down_proj": (experts, hidden, inter),
        "mtp.layers.0.mlp.experts.gate_up_proj": (experts, inter * 2, hidden),
        "mtp.layers.0.mlp.gate.weight": (experts, hidden),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": (hidden, inter),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (inter, hidden),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": (inter, hidden),
        "mtp.layers.0.mlp.shared_expert_gate.weight": (1, hidden),
        "mtp.layers.0.post_attention_layernorm.weight": (hidden,),
        "mtp.layers.0.self_attn.k_norm.weight": (2,),
        "mtp.layers.0.self_attn.k_proj.weight": (4, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (hidden, hidden * 2),
        "mtp.layers.0.self_attn.q_norm.weight": (2,),
        "mtp.layers.0.self_attn.q_proj.weight": (8, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (4, hidden),
        "mtp.norm.weight": (hidden,),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
    }


def _write_unpacked_donor(
    src: Path,
    *,
    experts: int = 2,
    hidden: int = 4,
    inter: int = 3,
) -> None:
    src.mkdir(parents=True, exist_ok=True)
    weights: dict[str, np.ndarray] = {
        "mtp.fc.weight": np.zeros((hidden, hidden * 2), dtype=np.float32),
        "mtp.layers.0.input_layernorm.weight": np.ones((hidden,), dtype=np.float32),
        "mtp.layers.0.mlp.gate.weight": np.zeros((experts, hidden), dtype=np.float32),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": np.zeros(
            (hidden, inter), dtype=np.float32
        ),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": np.zeros(
            (inter, hidden), dtype=np.float32
        ),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": np.zeros(
            (inter, hidden), dtype=np.float32
        ),
        "mtp.layers.0.mlp.shared_expert_gate.weight": np.zeros((1, hidden), dtype=np.float32),
        "mtp.layers.0.post_attention_layernorm.weight": np.ones((hidden,), dtype=np.float32),
        "mtp.layers.0.self_attn.k_norm.weight": np.ones((2,), dtype=np.float32),
        "mtp.layers.0.self_attn.k_proj.weight": np.zeros((4, hidden), dtype=np.float32),
        "mtp.layers.0.self_attn.o_proj.weight": np.zeros((hidden, hidden * 2), dtype=np.float32),
        "mtp.layers.0.self_attn.q_norm.weight": np.ones((2,), dtype=np.float32),
        "mtp.layers.0.self_attn.q_proj.weight": np.zeros((8, hidden), dtype=np.float32),
        "mtp.layers.0.self_attn.v_proj.weight": np.zeros((4, hidden), dtype=np.float32),
        "mtp.norm.weight": np.ones((hidden,), dtype=np.float32),
        "mtp.pre_fc_norm_embedding.weight": np.ones((hidden,), dtype=np.float32),
        "mtp.pre_fc_norm_hidden.weight": np.ones((hidden,), dtype=np.float32),
        # Non-MTP noise must be ignored.
        "model.layers.0.mlp.gate.weight": np.zeros((experts, hidden), dtype=np.float32),
    }
    for expert in range(experts):
        weights[f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"] = np.full(
            (inter, hidden), float(expert), dtype=np.float32
        )
        weights[f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight"] = np.full(
            (inter, hidden), float(expert + 10), dtype=np.float32
        )
        weights[f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight"] = np.full(
            (hidden, inter), float(expert + 20), dtype=np.float32
        )
    save_file(weights, src / "model-00001-of-00001.safetensors")
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in weights
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _trunk() -> ModelIdentity:
    return ModelIdentity(
        model_id="Hcompany/Holo3-35B-A3B",
        revision="208d5ae3a03f99d561f32ab5e606f73397a390ea",
    )


def _donor() -> ModelIdentity:
    return ModelIdentity(
        model_id="Qwen/Qwen3.5-35B-A3B",
        revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    )


def test_production_packed_shape_table_is_19_tensors() -> None:
    assert len(QWEN35_MOE_PACKED_MTP_SHAPES) == 19
    assert QWEN35_MOE_PACKED_MTP_SHAPES["mtp.layers.0.mlp.experts.gate_up_proj"] == (
        256,
        1024,
        2048,
    )
    assert QWEN35_MOE_PACKED_MTP_SHAPES["mtp.layers.0.mlp.experts.down_proj"] == (
        256,
        2048,
        512,
    )


def test_prepare_grafted_qwen_moe_mtp_packs_unpacked_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlx.core")
    import axquant.grafted_mtp as grafted_mtp

    experts, hidden, inter = 2, 4, 3
    shapes = _tiny_shapes(experts=experts, hidden=hidden, inter=inter)
    monkeypatch.setattr(grafted_mtp, "QWEN35_MOE_PACKED_MTP_SHAPES", shapes)

    donor_dir = tmp_path / "donor"
    _write_unpacked_donor(donor_dir, experts=experts, hidden=hidden, inter=inter)
    out = tmp_path / "graft"
    bundle = prepare_grafted_qwen_moe_mtp(
        donor_dir,
        output_dir=out,
        trunk=_trunk(),
        donor=_donor(),
    )
    assert bundle.sidecar.is_file()
    assert bundle.manifest.role == "mtp"
    assert bundle.manifest.tensor_count == 19
    assert bundle.manifest.source_model.model_id == "Hcompany/Holo3-35B-A3B"
    graft = json.loads(bundle.graft_record.read_text(encoding="utf-8"))
    assert graft["graft_kind"] == "parent-qwen35-moe-mtp"
    assert graft["donor_model"]["revision"] == "59d61f3ce65a6d9863b86d2e96597125219dc754"
    assert "not co-trained" in graft["notes"][0]

    import mlx.core as mx

    packed = mx.load(str(bundle.sidecar))
    assert set(packed) == set(shapes)
    gate_up = packed["mtp.layers.0.mlp.experts.gate_up_proj"]
    down = packed["mtp.layers.0.mlp.experts.down_proj"]
    assert tuple(int(x) for x in gate_up.shape) == (experts, inter * 2, hidden)
    assert tuple(int(x) for x in down.shape) == (experts, hidden, inter)
    # gate expert0 all 0, up expert0 all 10 → first half of concat axis -2 is 0.
    assert float(np.array(gate_up[0, 0, 0])) == 0.0
    assert float(np.array(gate_up[0, inter, 0])) == 10.0
    assert float(np.array(down[1, 0, 0])) == 21.0


def test_prepare_grafted_rejects_missing_mtp(tmp_path: Path) -> None:
    donor_dir = tmp_path / "empty"
    donor_dir.mkdir()
    save_file(
        {"model.layers.0.weight": np.zeros((2, 2), dtype=np.float32)},
        donor_dir / "model.safetensors",
    )
    with pytest.raises(ArtifactError, match="no mtp"):
        prepare_grafted_qwen_moe_mtp(
            donor_dir,
            output_dir=tmp_path / "out",
            trunk=_trunk(),
            donor=_donor(),
        )


def test_prepare_grafted_rejects_incomplete_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlx.core")
    import axquant.grafted_mtp as grafted_mtp

    shapes = _tiny_shapes(experts=2)
    monkeypatch.setattr(grafted_mtp, "QWEN35_MOE_PACKED_MTP_SHAPES", shapes)
    donor_dir = tmp_path / "donor"
    _write_unpacked_donor(donor_dir, experts=2)
    # Drop one expert tensor from the index so packing sees a gap.
    index_path = donor_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["mtp.layers.0.mlp.experts.1.up_proj.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ArtifactError, match="expert"):
        prepare_grafted_qwen_moe_mtp(
            donor_dir,
            output_dir=tmp_path / "out",
            trunk=_trunk(),
            donor=_donor(),
        )


def test_compose_grafted_mtp_onto_pack_copies_sidecar_and_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlx.core")
    import axquant.grafted_mtp as grafted_mtp

    shapes = _tiny_shapes()
    monkeypatch.setattr(grafted_mtp, "QWEN35_MOE_PACKED_MTP_SHAPES", shapes)
    donor_dir = tmp_path / "donor"
    _write_unpacked_donor(donor_dir)
    graft_dir = tmp_path / "graft"
    prepare_grafted_qwen_moe_mtp(
        donor_dir,
        output_dir=graft_dir,
        trunk=_trunk(),
        donor=_donor(),
    )

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "model.safetensors").write_bytes(b"trunk-weights")
    (pack / "axquant_manifest.json").write_text(
        json.dumps({"mtp_present": False, "schema_version": "test"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (pack / "axquant_runtime.json").write_text(
        json.dumps({"mtp": {"detected": False}}, indent=2) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "pack-mtp"
    composed = compose_grafted_mtp_onto_pack(pack, graft_dir, output_dir=out)
    assert composed == out.resolve()
    assert (out / "mtp.safetensors").is_file()
    assert (out / "axquant_mtp_sidecar_manifest.json").is_file()
    assert (out / "axquant_mtp_graft.json").is_file()
    # Source pack unchanged.
    assert not (pack / "mtp.safetensors").exists()
    assert json.loads((pack / "axquant_manifest.json").read_text())["mtp_present"] is False
    # Composed pack flags MTP present; trunk file preserved.
    assert (out / "model.safetensors").read_bytes() == b"trunk-weights"
    manifest = json.loads((out / "axquant_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mtp_present"] is True
    assert manifest["mtp_weight_file_size_bytes"] > 0
    runtime = json.loads((out / "axquant_runtime.json").read_text(encoding="utf-8"))
    assert runtime["mtp"]["detected"] is True
    assert runtime["mtp"]["sidecar_file"] == "mtp.safetensors"
    assert runtime["mtp"]["optimized"] is False


def test_compose_rejects_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mlx.core")
    import axquant.grafted_mtp as grafted_mtp

    monkeypatch.setattr(grafted_mtp, "QWEN35_MOE_PACKED_MTP_SHAPES", _tiny_shapes())
    donor_dir = tmp_path / "donor"
    _write_unpacked_donor(donor_dir)
    graft_dir = tmp_path / "graft"
    prepare_grafted_qwen_moe_mtp(
        donor_dir,
        output_dir=graft_dir,
        trunk=_trunk(),
        donor=_donor(),
    )
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "model.safetensors").write_bytes(b"x")
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(ArtifactError, match="already exists"):
        compose_grafted_mtp_onto_pack(pack, graft_dir, output_dir=out)
