from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.cuda import _quantization_config, convert_cuda_nvfp4, plan_cuda_nvfp4
from axquant.serde import write_data


@pytest.mark.parametrize(
    ("members", "fused"),
    [
        (("in_proj_qkv", "in_proj_z"), "in_proj_qkvz"),
        (("in_proj_b", "in_proj_a"), "in_proj_ba"),
    ],
)
def test_gdn_fused_scales_match_across_shards(
    tmp_path: Path, members: tuple[str, str], fused: str
) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    source = tmp_path / "source"
    source.mkdir()
    write_data(
        source / "config.json",
        {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 1}},
    )
    prefix = "model.language_model.layers.0.linear_attn."
    names = [prefix + member + ".weight" for member in members]
    for index, name in enumerate(names):
        save_file(
            {name: np.full((8, 32), 2.0 + index * 18.0, dtype=np.float32)},
            source / f"model-{index}.safetensors",
        )
    save_file(
        {prefix + "out_proj.weight": np.ones((8, 32), dtype=np.float32)},
        source / "output.safetensors",
    )
    plan = plan_cuda_nvfp4(source, model_id="Qwen/Qwen3.5-9B", allow_unmeasured=True)
    assert {item.scale_group for item in plan.allocations if item.tensor_name in names} == {
        prefix + fused
    }
    output = tmp_path / "pack"
    convert_cuda_nvfp4(source, plan, output, device="cpu", allow_unmeasured=True)
    scales = [
        load_file(output / f"model-{index}.safetensors")[
            name.removesuffix(".weight") + ".weight_global_scale"
        ]
        for index, name in enumerate(names)
    ]
    assert torch.equal(scales[0], scales[1])
    assert scales[0].item() == pytest.approx(2688 / 20)
    protected = plan_cuda_nvfp4(
        source,
        model_id="Qwen/Qwen3.5-9B",
        keep_patterns=[names[0]],
        allow_unmeasured=True,
    )
    assert all(
        item.method == "preserve" and item.scale_group is None
        for item in protected.allocations
        if item.tensor_name in names
    )
    assert prefix + fused in _quantization_config(protected)["ignore"]
