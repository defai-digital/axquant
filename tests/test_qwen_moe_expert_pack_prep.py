"""Unpacked Qwen3.5 MoE expert packing for MLX-LM convert (Ornith path)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.source_prep import (
    needs_qwen_moe_unpacked_expert_prep,
    prepare_qwen_moe_packed_experts_source,
)


def _write_unpacked_moe(src: Path, *, experts: int = 2, hidden: int = 4, inter: int = 3) -> None:
    src.mkdir(parents=True, exist_ok=True)
    weights: dict[str, np.ndarray] = {
        "lm_head.weight": np.zeros((8, hidden), dtype=np.float32),
        "model.language_model.layers.0.mlp.gate.weight": np.zeros((experts, hidden), dtype=np.float32),
    }
    for expert in range(experts):
        weights[f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj.weight"] = (
            np.full((inter, hidden), float(expert), dtype=np.float32)
        )
        weights[f"model.language_model.layers.0.mlp.experts.{expert}.up_proj.weight"] = (
            np.full((inter, hidden), float(expert + 10), dtype=np.float32)
        )
        weights[f"model.language_model.layers.0.mlp.experts.{expert}.down_proj.weight"] = (
            np.full((hidden, inter), float(expert + 20), dtype=np.float32)
        )
    save_file(weights, src / "model-00001-of-00001.safetensors")
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in weights},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (src / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "num_hidden_layers": 1,
                    "num_experts": experts,
                    "hidden_size": hidden,
                    "moe_intermediate_size": inter,
                },
            }
        ),
        encoding="utf-8",
    )


def test_needs_qwen_moe_unpacked_expert_prep_detects_ornith_layout(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_unpacked_moe(src)
    assert needs_qwen_moe_unpacked_expert_prep(src) is True


def test_prepare_qwen_moe_packed_experts_stacks_gate_up_down(tmp_path: Path) -> None:
    mlx = pytest.importorskip("mlx.core")
    src = tmp_path / "src"
    _write_unpacked_moe(src, experts=2, hidden=4, inter=3)
    prepared = prepare_qwen_moe_packed_experts_source(src, work_dir=tmp_path / "prep")
    index = json.loads((prepared / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in weight_map
    assert "model.language_model.layers.0.mlp.experts.down_proj" in weight_map
    assert "model.language_model.layers.0.mlp.gate.weight" in weight_map
    assert not any(".experts.0." in name for name in weight_map)
    gate_up = mlx.load(
        str(prepared / weight_map["model.language_model.layers.0.mlp.experts.gate_up_proj"])
    )["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    down = mlx.load(
        str(prepared / weight_map["model.language_model.layers.0.mlp.experts.down_proj"])
    )["model.language_model.layers.0.mlp.experts.down_proj"]
    assert tuple(gate_up.shape) == (2, 6, 4)
    assert tuple(down.shape) == (2, 4, 3)
