from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.architectures import adapter_for
from axquant.cuda import plan_cuda_nvfp4
from axquant.schema import SupportTier, TensorRole
from axquant.serde import write_data


@pytest.mark.parametrize("size", ["4B", "8B"])
def test_qwen3_vl_cuda_allocation_preserves_multimodal_and_tied_heads(
    tmp_path: Path, size: str
) -> None:
    identity = f"Qwen/Qwen3-VL-{size}-Instruct"
    source = tmp_path / identity.split("/")[-1]
    source.mkdir()
    config = {
        "model_type": "qwen3_vl",
        "text_config": {"num_hidden_layers": 1, "tie_word_embeddings": size == "4B"},
        "tie_word_embeddings": size == "4B",
        "vision_config": {"depth": 1},
    }
    write_data(source / "config.json", config)
    write_data(source / "chat_template.json", {"chat_template": "test-template"})
    names = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "lm_head.weight",
    ]
    save_file(
        {name: np.ones((32, 32), dtype=np.float32) for name in names}, source / "model.safetensors"
    )
    adapter = adapter_for(identity, config)
    assert adapter is not None
    profile = adapter.profile(identity, config)
    assert profile.support_tier == SupportTier.CONVERTIBLE
    assert not profile.mtp_declared
    plan = plan_cuda_nvfp4(source, model_id=identity, allow_unmeasured=True)
    assert any(item.path == "chat_template.json" for item in plan.source_files)
    allocations = {item.tensor_name: item for item in plan.allocations}
    assert all(allocations[name].method == "nvfp4" for name in names[:4])
    assert all(allocations[name].method == "preserve" for name in names[4:])
    assert allocations[names[5]].role == TensorRole.VISION
