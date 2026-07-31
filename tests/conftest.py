from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def tiny_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyForCausalLM"],
                "model_type": "tiny",
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": np.zeros((16, 8), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 8), dtype=np.float32),
            "model.layers.0.mlp.down_proj.weight": np.zeros((8, 8), dtype=np.float32),
            "model.norm.weight": np.zeros((8,), dtype=np.float32),
            "lm_head.weight": np.zeros((16, 8), dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    save_file(
        {
            "mtp.projection.weight": np.zeros((8, 8), dtype=np.float32),
            "mtp.output_head.weight": np.zeros((16, 8), dtype=np.float32),
        },
        model_dir / "mtp.safetensors",
    )
    return model_dir


@pytest.fixture
def qwen36_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "Qwen3.6-27B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "language_model_only": False,
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "model_type": "qwen3_5_text",
                    "mtp_num_hidden_layers": 1,
                    "num_hidden_layers": 64,
                    "vocab_size": 248320,
                },
                "vision_config": {
                    "depth": 27,
                    "hidden_size": 1152,
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight": np.zeros(
                (8, 8),
                dtype=np.float32,
            ),
            "language_model.model.layers.0.linear_attn.conv1d.weight": np.zeros(
                (8, 4, 1),
                dtype=np.float32,
            ),
            "language_model.model.layers.0.mlp.down_proj.weight": np.zeros(
                (8, 8),
                dtype=np.float32,
            ),
            "language_model.lm_head.weight": np.zeros((16, 8), dtype=np.float32),
            "visual.patch_embed.proj.weight": np.zeros((8, 8), dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    save_file(
        {
            "mtp.fc.weight": np.zeros((8, 8), dtype=np.float32),
            "mtp.layers.0.self_attn.q_proj.weight": np.zeros((8, 8), dtype=np.float32),
        },
        model_dir / "mtp.safetensors",
    )
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen3_5",
                "mtp_depth_max": 2,
                "mtp_sidecar": "INT8 quantized projections, bf16 norms",
            }
        ),
        encoding="utf-8",
    )
    return model_dir


@pytest.fixture
def packed_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "packed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["TinyForCausalLM"],
                "model_type": "tiny",
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": np.zeros(
                (8, 2),
                dtype=np.uint32,
            ),
            "model.layers.0.mlp.down_proj.scales": np.zeros(
                (8, 1),
                dtype=np.float32,
            ),
            "model.layers.0.mlp.down_proj.biases": np.zeros(
                (8, 1),
                dtype=np.float32,
            ),
            "model.norm.weight": np.zeros((8,), dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    return model_dir
