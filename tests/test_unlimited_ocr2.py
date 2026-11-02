from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.architectures.registry import adapter_for
from axquant.cuda import convert_cuda_nvfp4, plan_cuda_nvfp4
from axquant.errors import ArtifactError, PlanningError
from axquant.schema import SupportTier
from axquant.serde import file_sha256, read_data, write_data


def config() -> dict:
    return {
        "model_type": "unlimited-ocr",
        "architectures": ["UnlimitedOCRForCausalLM"],
        "hidden_size": 1280,
        "num_hidden_layers": 12,
        "num_attention_heads": 10,
        "num_key_value_heads": 10,
        "n_routed_experts": 64,
        "num_experts_per_tok": 6,
        "vocab_size": 129280,
        "vision_config": {"model_type": "vision"},
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "dash_router": "entmax_topk_perlayer",
        },
    }


@pytest.fixture
def source(tmp_path: Path) -> Path:
    source = tmp_path / "renamed-source"
    source.mkdir()
    write_data(source / "config.json", config())
    save_file(
        {
            name: np.ones((8, 32), dtype=np.float32)
            for name in (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.dash_router.q_proj.weight",
                "model.layers.0.self_attn.indexer.k_proj.weight",
                "model.sam_model.blocks.0.attn.qkv.weight",
            )
        },
        source / "model.safetensors",
    )
    return source


def test_sparse_checkpoint_cannot_match_v1_even_after_rename() -> None:
    for reference in ("baidu/Unlimited-OCR-2", "/local/renamed-source", "baidu/Unlimited-OCR"):
        adapter = adapter_for(reference, config())
        assert adapter is not None
        assert adapter.adapter_id == "unlimited-ocr2-v1"


@pytest.mark.parametrize("change", [{"hidden_size": 2048}, {"num_hidden_layers": True}])
def test_drifted_signature_remains_inspection_only(source: Path, change: dict) -> None:
    write_data(source / "config.json", config() | change)
    adapter = adapter_for("baidu/Unlimited-OCR-2", config() | change)
    assert adapter is not None
    assert (
        adapter.profile("baidu/Unlimited-OCR-2", config() | change).support_tier
        is SupportTier.INSPECT_ONLY
    )
    with pytest.raises(PlanningError, match="registered convertible"):
        plan_cuda_nvfp4(source, model_id="baidu/Unlimited-OCR-2", allow_unmeasured=True)


def test_cuda_export_preserves_sparse_routing_and_config(source: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    plan = plan_cuda_nvfp4(source, model_id="baidu/Unlimited-OCR-2", allow_unmeasured=True)
    selected = [a.tensor_name for a in plan.allocations if a.method == "nvfp4"]
    assert selected == ["model.layers.0.self_attn.q_proj.weight"]
    output = source.parent / "pack"
    manifest = convert_cuda_nvfp4(source, plan, output, device="cpu", allow_unmeasured=True)
    before = load_file(str(source / "model.safetensors"))
    after = load_file(str(output / "model.safetensors"))
    for name in before:
        if name not in selected:
            assert before[name].dtype == after[name].dtype
            assert torch.equal(before[name], after[name])
    assert (
        read_data(output / "config.json")["sparse_attention_config"]
        == config()["sparse_attention_config"]
    )
    assert manifest.backend == "numpy-reference"
    assert manifest.runtime_verified is False
    assert manifest.quality_certified is False


def mx_plan(source: Path, mode: str):
    from axquant.inspector import inspect_model
    from axquant.manual import manual_quantization_plan
    from axquant.schema import ManualPlanRecipe

    method, bits = ("mxfp4", 4) if mode == "mxfp4" else ("affine", 8)
    recipe = ManualPlanRecipe.model_validate(
        {
            "default_bits": 16,
            "default_method": "bf16",
            "group_size": 32,
            "target_bpw": 16,
            "rules": [
                {
                    "rule_id": "text",
                    "roles": ["attention", "mlp", "expert"],
                    "bits": bits,
                    "method": method,
                    "group_size": 32,
                    "reason": "Source-layout development storage",
                }
            ],
            "hardware": {"supported_methods": ["affine", "mxfp4", "bf16"]},
        }
    )
    inventory = inspect_model(source, model_id="baidu/Unlimited-OCR-2", revision="a" * 40)
    return manual_quantization_plan(inventory, recipe)


@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
def test_source_mx_full_conversion_preserves_sparse_graph(
    source: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mx = pytest.importorskip("mlx.core")
    import axquant.converter as converter
    from axquant.source_binding import build_source_plan_binding

    # Contiguous individual experts must retain individual source names.
    weights = mx.load(str(source / "model.safetensors"))
    weights["model.layers.0.self_attn.q_proj.weight"] = mx.arange(256).reshape(8, 32) / 97 - 1
    for expert in (0, 1):
        weights[f"model.layers.1.mlp.experts.{expert}.up_proj.weight"] = mx.ones((8, 32))
    weights["model.layers.0.self_attn.dash_router.state"] = mx.array([1, 2], dtype=mx.int64)
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    write_data(source / "credentials.json", {"test_only": "never copy"})
    original = (source / "config.json").read_bytes()
    plan = mx_plan(source, mode)

    def graph_not_allowed(*args, **kwargs):
        raise AssertionError("source-layout storage must not invoke a model/graph backend")

    monkeypatch.setattr(converter, "prepare_conversion_source", graph_not_allowed)
    monkeypatch.setattr(converter, "_mlx_api", graph_not_allowed)
    output = source.parent / mode
    manifest = converter.convert_model(
        model=str(source),
        plan=plan,
        output=output,
        q_mode=mode,
        allow_unmeasured=True,
        ax_engine_manifest="skip",
        expert_stream="off",
        source_binding=build_source_plan_binding(plan, source),
    )
    assert not manifest.runtime.primary_runtime.standard_inference
    assert manifest.runtime.compatible_runtimes == []
    assert not (output / "credentials.json").exists()
    assert (source / "config.json").read_bytes() == original
    before, after = weights, mx.load(str(output / "model.safetensors"))
    for allocation in plan.assignments:
        name = allocation.tensor
        if allocation.bits == 16:
            assert before[name].dtype == after[name].dtype
            assert bool(mx.all(before[name] == after[name]).item())
        else:
            prefix = name.removesuffix(".weight")
            reconstructed = mx.dequantize(
                after[name],
                after[prefix + ".scales"],
                group_size=32,
                bits=allocation.bits,
                mode=mode,
            )
            assert reconstructed.shape == before[name].shape
            assert bool(mx.all(mx.isfinite(reconstructed)).item())
            error = mx.linalg.norm(reconstructed - before[name]) / mx.linalg.norm(before[name])
            assert error.item() < (0.15 if mode == "mxfp4" else 1 / 16)
            inputs = mx.ones((1, 32))
            product = mx.quantized_matmul(
                inputs,
                after[name],
                after[prefix + ".scales"],
                group_size=32,
                bits=allocation.bits,
                mode=mode,
            )
            assert bool(mx.allclose(product, inputs @ reconstructed.T, atol=1e-4).item())
    report = read_data(output / "axquant_compatibility.json")
    assert all(t["status"] == "unsupported" for t in report["targets"].values())
    for member in manifest.files:
        assert member.sha256 == file_sha256(output / member.path)


def test_source_mx_rejects_router_quantization(source: Path) -> None:
    from axquant.source_mx import preflight_source_mx

    plan = mx_plan(source, "mxfp8")
    assignments = [
        a.model_copy(update={"bits": 8, "group_size": 32, "method": plan.assignments[0].method})
        if a.role.value == "router"
        else a
        for a in plan.assignments
    ]
    forged = plan.model_copy(update={"assignments": assignments})
    with pytest.raises(PlanningError, match="unsupported or unprotected"):
        preflight_source_mx(source, forged)


def test_source_mx_rejects_partial_coverage(source: Path) -> None:
    from axquant.source_mx import preflight_source_mx

    plan = mx_plan(source, "mxfp8")
    with pytest.raises(PlanningError, match="every source tensor"):
        preflight_source_mx(source, plan.model_copy(update={"assignments": plan.assignments[:-1]}))


def test_source_mx_missing_scales_rolls_back(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mx = pytest.importorskip("mlx.core")
    from axquant.converter import convert_model
    from axquant.source_binding import build_source_plan_binding

    plan = mx_plan(source, "mxfp8")
    original_save = mx.save_safetensors

    def broken_save(path, weights, **kwargs):
        broken = {k: v for k, v in weights.items() if not k.endswith(".scales")}
        original_save(path, broken, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", broken_save)
    output = source.parent / "broken"
    with pytest.raises(ArtifactError):
        convert_model(
            model=str(source),
            plan=plan,
            output=output,
            q_mode="mxfp8",
            allow_unmeasured=True,
            ax_engine_manifest="skip",
            expert_stream="off",
            source_binding=build_source_plan_binding(plan, source),
        )
    assert not output.exists()


def test_source_mx_invalid_e8m0_rolls_back(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mx = pytest.importorskip("mlx.core")
    from axquant.converter import convert_model
    from axquant.source_binding import build_source_plan_binding

    plan = mx_plan(source, "mxfp8")
    original_quantize = mx.quantize

    def invalid_quantize(weight, **kwargs):
        packed, scales = original_quantize(weight, **kwargs)
        return packed, mx.full(scales.shape, 255, dtype=mx.uint8)

    monkeypatch.setattr(mx, "quantize", invalid_quantize)
    output = source.parent / "invalid-scale"
    with pytest.raises(ArtifactError, match="invalid data"):
        convert_model(
            model=str(source),
            plan=plan,
            output=output,
            q_mode="mxfp8",
            allow_unmeasured=True,
            ax_engine_manifest="skip",
            expert_stream="off",
            source_binding=build_source_plan_binding(plan, source),
        )
    assert not output.exists()
