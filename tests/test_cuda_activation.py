from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant.cli import main
from axquant.cuda import (
    _quantization_config,
    convert_cuda_nvfp4_w4a4,
    plan_cuda_nvfp4,
    plan_cuda_nvfp4_w4a4,
)
from axquant.errors import PlanningError
from axquant.schema.cuda import CudaFileDigest, CudaQuantizationPlan
from axquant.schema.cuda_activation import CudaActivationCalibration
from axquant.serde import read_data, stable_sha256, write_data


@pytest.fixture
def expert_source(tmp_path: Path) -> Path:
    source = tmp_path / "Qwen3-CUDA-Experts"
    source.mkdir()
    write_data(source / "config.json", {"model_type": "qwen3", "num_hidden_layers": 1})
    rng = np.random.default_rng(1)
    tensors = {
        f"model.layers.0.mlp.experts.{expert}.{projection}.weight": (
            rng.normal(size=(32, 32)).astype(np.float32) * (expert + 1)
        )
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    tensors["model.layers.0.self_attn.q_proj.weight"] = rng.normal(size=(32, 32)).astype(np.float32)
    tensors["model.norm.weight"] = np.ones(32, dtype=np.float32)
    save_file(tensors, source / "model.safetensors")
    return source


def calibration_for(plan: CudaQuantizationPlan) -> CudaActivationCalibration:
    return CudaActivationCalibration(
        weight_plan_sha256=stable_sha256(plan),
        input_files=[CudaFileDigest(path="page.png", sha256="a" * 64, size_bytes=16)],
        runtime="vllm",
        runtime_version="test",
        source_precision=plan.activation_dtype,
        statistics=[
            {
                "tensor_name": item.tensor_name,
                "input_columns": item.shape[1],
                "sample_count": 16,
                "absolute_maximum": float(index + 1),
            }
            for index, item in enumerate(plan.allocations)
            if item.method == "nvfp4"
        ],
    )


def test_keep_one_expert_preserves_complete_runtime_table(expert_source: Path) -> None:
    plan = plan_cuda_nvfp4(
        expert_source, keep_patterns=["*.experts.0.down_proj.weight"], allow_unmeasured=True
    )
    assert all(
        item.method == "preserve" for item in plan.allocations if ".experts." in item.tensor_name
    )
    assert any(item.method == "nvfp4" for item in plan.allocations)
    assert "model.layers.0.mlp.experts" in _quantization_config(plan)["ignore"]


@pytest.mark.parametrize("mutation", ["identity", "missing", "shape", "precision"])
def test_calibration_must_bind_source_and_complete_inputs(
    expert_source: Path, mutation: str
) -> None:
    plan = plan_cuda_nvfp4(expert_source, allow_unmeasured=True)
    capture = calibration_for(plan)
    payload = capture.model_dump(mode="json")
    if mutation == "identity":
        payload["weight_plan_sha256"] = "0" * 64
    elif mutation == "missing":
        payload["statistics"].pop()
    elif mutation == "shape":
        payload["statistics"][0]["input_columns"] += 16
    else:
        payload["source_precision"] = "float16"
    with pytest.raises(PlanningError):
        plan_cuda_nvfp4_w4a4(plan, CudaActivationCalibration.model_validate(payload))


@pytest.mark.parametrize("maximum", [float("nan"), float("inf"), -1.0])
def test_invalid_activation_maximum_fails(expert_source: Path, maximum: float) -> None:
    plan = plan_cuda_nvfp4(expert_source, allow_unmeasured=True)
    payload = calibration_for(plan).model_dump(mode="json")
    payload["statistics"][0]["absolute_maximum"] = maximum
    with pytest.raises(ValidationError):
        CudaActivationCalibration.model_validate(payload)


def test_w4a4_export_has_scales_and_complete_expert_fusion(expert_source: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    plan = plan_cuda_nvfp4(expert_source, allow_unmeasured=True)
    w4a4 = plan_cuda_nvfp4_w4a4(plan, calibration_for(plan))
    output = expert_source.parent / "w4a4"
    manifest = convert_cuda_nvfp4_w4a4(
        expert_source, w4a4, output, device="cpu", allow_unmeasured=True
    )
    assert manifest.activation_precision == 4 and not manifest.runtime_verified
    tensors = load_file(str(output / "model.safetensors"))
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            assert prefix + ".input_global_scale" in tensors
            assert tensors[prefix + ".input_global_scale"].dtype == torch.float32
    for suffix in ("weight_global_scale", "input_global_scale"):
        values = [
            tensors[f"model.layers.0.mlp.experts.{e}.{p}.{suffix}"].item()
            for e in range(2)
            for p in ("gate_proj", "up_proj")
        ]
        assert len(set(values)) == 1
    assert torch.equal(tensors["model.norm.weight"], torch.ones(32))
    config = read_data(output / "config.json")
    assert (
        config["quantization_config"]["config_groups"]["nvfp4"]["input_activations"]["num_bits"]
        == 4
    )
    assert (
        read_data(output / "axquant_cuda_plan.json")["schema_version"]
        == "axquant.cuda-w4a4-plan.v1"
    )


def test_w4a4_cli_requires_calibration_before_output(expert_source: Path) -> None:
    output = expert_source.parent / "plan.json"
    assert (
        main(
            [
                "plan-cuda",
                str(expert_source),
                "--activation-bits",
                "4",
                "--allow-unmeasured",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_ocr_smoke_rejects_repetitive_prefix_even_with_expected_text() -> None:
    from scripts.smoke_cuda_ocr import require_smoke_text

    expected = "AXQuant NVFP4\nInvoice 12345\nTotal USD 42.50"
    require_smoke_text(expected)
    with pytest.raises(RuntimeError, match="repetitive"):
        require_smoke_text("1.\n1.\n1.\n" + expected)


@pytest.mark.parametrize("prefix", ["model.", "model.language_model."])
def test_calibration_maps_runtime_language_aliases(prefix: str) -> None:
    from scripts.capture_cuda_ocr import runtime_source_name

    selected = {prefix + "layers.0.self_attn.q_proj.weight": {}}
    assert runtime_source_name("language_model.model.layers.0.self_attn.q_proj", selected) == (
        prefix + "layers.0.self_attn.q_proj"
    )


def test_native_kernel_check_rejects_fallback() -> None:
    from types import SimpleNamespace

    from scripts.smoke_cuda_ocr import inspect_native_fp4

    scheme_type = type("CompressedTensorsW4A4Fp4", (), {})
    scheme = scheme_type()
    scheme.use_a16 = False
    scheme.kernel = type("CutlassNvFp4LinearKernel", (), {})()
    layer = SimpleNamespace(scheme=scheme)
    model = SimpleNamespace(named_modules=lambda: [("layer", layer)])
    assert inspect_native_fp4(model)["linear_kernels"] == {"CutlassNvFp4LinearKernel": 1}
    layer.quant_method = SimpleNamespace(nvfp4_backend=SimpleNamespace(name="MARLIN"))
    with pytest.raises(RuntimeError, match="MoE check rejected"):
        inspect_native_fp4(model)
    del layer.quant_method
    scheme.use_a16 = True
    with pytest.raises(RuntimeError, match="Linear check rejected"):
        inspect_native_fp4(model)


def test_preserved_fused_attention_is_ignored_by_runtime(expert_source: Path) -> None:
    plan = plan_cuda_nvfp4(expert_source, keep_patterns=["*.self_attn.*"], allow_unmeasured=True)
    assert "model.layers.0.self_attn.qkv_proj" in _quantization_config(plan)["ignore"]


def test_unlimited_runtime_rejects_unaddressable_preserved_attention() -> None:
    from scripts.smoke_cuda_ocr import require_runtime_layout

    config = {
        "architectures": ["UnlimitedOCRForCausalLM"],
        "quantization_config": {"ignore": ["model.layers.0.self_attn.q_proj"]},
    }
    with pytest.raises(ValueError, match="lacks Linear prefixes"):
        require_runtime_layout(config)
    config["quantization_config"]["ignore"] = ["model.layers.0.mlp.gate_up_proj"]
    require_runtime_layout(config)


def test_kernel_coverage_requires_every_planned_runtime_unit(expert_source: Path) -> None:
    from scripts.smoke_cuda_ocr import require_kernel_coverage

    plan = plan_cuda_nvfp4(expert_source, allow_unmeasured=True)
    w4a4 = plan_cuda_nvfp4_w4a4(plan, calibration_for(plan))
    record = {
        "linear_kernels": {"CutlassNvFp4LinearKernel": 1},
        "moe_backends": {"VLLM_CUTLASS": 1},
    }
    require_kernel_coverage(w4a4.model_dump(mode="json"), record)
    record["moe_backends"] = {}
    with pytest.raises(RuntimeError, match="MoE coverage differs"):
        require_kernel_coverage(w4a4.model_dump(mode="json"), record)
