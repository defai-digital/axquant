from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant import cuda_fp8
from axquant.cli import main
from axquant.errors import ArtifactError, PlanningError
from axquant.fp8 import quantize_fp8_channel
from axquant.schema.cuda_fp8 import CudaFp8QuantizationPlan
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data


@pytest.fixture
def fp8_source(tmp_path: Path) -> Path:
    source = tmp_path / "Qwen3-FP8"
    source.mkdir()
    write_data(
        source / "config.json",
        {
            "model_type": "qwen3",
            "torch_dtype": "bfloat16",
            "num_hidden_layers": 1,
        },
    )
    values = np.random.default_rng(42).normal(size=(16, 32)).astype(np.float32)
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "vision_model.transformer.layers.0.mlp.fc1.weight",
    ]
    save_file({name: values for name in names}, source / "model.safetensors")
    save_file({"mtp.projection.weight": values}, source / "mtp.safetensors")
    write_data(source / "tokenizer_config.json", {"model_max_length": 2048})
    write_data(source / "credentials.json", {"private": True})
    return source


def test_fp8_protection_fused_keep_and_exact_binding(fp8_source: Path) -> None:
    plan = cuda_fp8.plan_cuda_fp8(fp8_source, allow_unmeasured=True)
    selected = {item.tensor_name for item in plan.allocations if item.method == "fp8"}
    assert len(selected) == 4
    assert plan.activation_precision == 8 and plan.activation_strategy == "dynamic-token"
    assert str(fp8_source.parent) not in plan.model_dump_json()
    assert "credentials.json" not in plan.model_dump_json()
    kept = cuda_fp8.plan_cuda_fp8(
        fp8_source, keep_patterns=["*.k_proj.weight"], allow_unmeasured=True
    )
    assert {item.tensor_name for item in kept.allocations if item.method == "fp8"} == {
        "model.layers.0.mlp.down_proj.weight"
    }
    write_data(fp8_source / "tokenizer_config.json", {"model_max_length": 4096})
    with pytest.raises(ArtifactError, match="content changed"):
        cuda_fp8.convert_cuda_fp8(
            fp8_source, plan, fp8_source.parent / "pack", allow_unmeasured=True
        )


def test_fp8_unmeasured_gate_and_protected_schema(fp8_source: Path) -> None:
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        cuda_fp8.plan_cuda_fp8(fp8_source)
    plan = cuda_fp8.plan_cuda_fp8(fp8_source, allow_unmeasured=True)
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        cuda_fp8.convert_cuda_fp8(fp8_source, plan, fp8_source.parent / "pack")
    payload = plan.model_dump(mode="json")
    next(item for item in payload["allocations"] if item["tensor_name"] == "lm_head.weight")[
        "method"
    ] = "fp8"
    with pytest.raises(ValidationError, match="eligible floating text"):
        CudaFp8QuantizationPlan.model_validate(payload)


@pytest.mark.parametrize("device", ["mps", "cuda:abc", "cuda:-1"])
def test_fp8_invalid_device_fails_before_staging(fp8_source: Path, device: str) -> None:
    plan = cuda_fp8.plan_cuda_fp8(fp8_source, allow_unmeasured=True)
    output = fp8_source.parent / "pack"
    with pytest.raises(PlanningError, match="device must"):
        cuda_fp8.convert_cuda_fp8(fp8_source, plan, output, device=device, allow_unmeasured=True)
    assert not output.exists()


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fp8_channel_reconstruction_and_zero_rows(dtype: str) -> None:
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[0, 0, 0, 0], [-0.1, 0.2, 40, -1], [1, -2, 3, 448]])
    weight = weight.to(getattr(torch, dtype))
    encoded, scale = quantize_fp8_channel(weight, device="cpu", rows_per_chunk=1)
    assert encoded.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and scale.shape == (3, 1)
    assert scale[0].item() == 1
    error = (encoded.float() * scale - weight.float()).abs()
    assert bool((error <= weight.float().abs() / 16 + scale / 512).all())
    second, second_scale = quantize_fp8_channel(weight, device="cpu", rows_per_chunk=64)
    assert torch.equal(encoded.view(torch.uint8), second.view(torch.uint8))
    assert torch.equal(scale, second_scale)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e-45])
def test_fp8_rejects_nonfinite_weights_and_scale_underflow(value: float) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ArtifactError):
        quantize_fp8_channel(torch.full((2, 16), value), device="cpu")


@pytest.mark.parametrize("shape", [(0, 16), (16,), (2, 2, 16)])
def test_fp8_rejects_empty_and_nonmatrix_input(shape: tuple[int, ...]) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ArtifactError, match="nonempty floating matrix"):
        quantize_fp8_channel(torch.empty(shape), device="cpu")


def test_fp8_public_layout_preservation_and_decompression(fp8_source: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    original = load_file(str(fp8_source / "model.safetensors"))
    original = {name: tensor.to(torch.bfloat16) for name, tensor in original.items()}
    save_file(original, str(fp8_source / "model.safetensors"))
    plan = cuda_fp8.plan_cuda_fp8(fp8_source, allow_unmeasured=True)
    output = fp8_source.parent / "pack"
    manifest = cuda_fp8.convert_cuda_fp8(
        fp8_source, plan, output, device="cpu", rows_per_chunk=3, allow_unmeasured=True
    )
    assert manifest.backend == "torch-cpu" and manifest.plan_sha256 == stable_sha256(plan)
    assert not manifest.runtime_verified and not manifest.quality_certified
    assert not (output / "credentials.json").exists()
    assert file_sha256(output / "mtp.safetensors") == file_sha256(fp8_source / "mtp.safetensors")
    saved = load_file(str(output / "model.safetensors"))
    for item in plan.allocations:
        if item.source_file != "model.safetensors":
            continue
        weight = saved[item.tensor_name]
        if item.method == "preserve":
            assert weight.dtype == torch.bfloat16
            assert torch.equal(weight, original[item.tensor_name])
        else:
            assert weight.dtype == torch.float8_e4m3fn
            scale = saved[item.tensor_name.removesuffix(".weight") + ".weight_scale"]
            reconstructed = weight.float() * scale
            assert torch.linalg.vector_norm(reconstructed - original[item.tensor_name].float()) < (
                0.04 * torch.linalg.vector_norm(original[item.tensor_name].float())
            )
    assert set(read_data(output / "model.safetensors.index.json")["weight_map"]) == set(saved)
    assert (output / "model.safetensors").stat().st_size < (
        fp8_source / "model.safetensors"
    ).stat().st_size
    for member in manifest.files:
        assert file_sha256(output / member.path) == member.sha256
    config = read_data(output / "config.json")["quantization_config"]
    assert config["config_groups"]["fp8"]["input_activations"]["dynamic"]
    assert "lm_head" in config["ignore"]
    ct = pytest.importorskip("compressed_tensors.quantization")
    parsed = ct.QuantizationConfig.model_validate(config)
    assert parsed.config_groups["fp8"].weights.strategy == "channel"


def test_fp8_forged_plan_and_mutation_roll_back(fp8_source: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    plan = cuda_fp8.plan_cuda_fp8(fp8_source, allow_unmeasured=True)
    output = fp8_source.parent / "pack"
    forged = plan.model_copy(update={"allocations": plan.allocations[:-1]})
    with pytest.raises(PlanningError, match="protection policy"):
        cuda_fp8.convert_cuda_fp8(fp8_source, forged, output, device="cpu", allow_unmeasured=True)
    original = cuda_fp8.quantize_fp8_channel

    def mutate(weight, **kwargs):
        write_data(fp8_source / "tokenizer_config.json", {"model_max_length": 4096})
        return original(weight, **kwargs)

    monkeypatch.setattr(cuda_fp8, "quantize_fp8_channel", mutate)
    with pytest.raises(ArtifactError, match="content changed"):
        cuda_fp8.convert_cuda_fp8(fp8_source, plan, output, device="cpu", allow_unmeasured=True)
    assert not output.exists() and not list(output.parent.glob(".pack.fp8-*"))


def test_fp8_cli_plan_convert_and_mode_mismatch(fp8_source: Path) -> None:
    pytest.importorskip("torch")
    path = fp8_source.parent / "plan.json"
    output = fp8_source.parent / "pack"
    assert (
        main(
            [
                "plan-cuda",
                str(fp8_source),
                "--q-mode",
                "fp8",
                "--allow-unmeasured",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    assert load_model(path, CudaFp8QuantizationPlan).format == "fp8_e4m3"
    command = [
        "convert-cuda",
        str(fp8_source),
        "--plan",
        str(path),
        "--allow-unmeasured",
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    assert main(command) == 2
    assert not output.exists()
    assert main([*command, "--q-mode", "fp8"]) == 0


@pytest.mark.integration
@pytest.mark.parametrize("rows_per_chunk", [1, 17, 256])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fp8_cuda_matches_cpu_bytes(rows_per_chunk: int, dtype: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    generator = torch.Generator().manual_seed(420)
    weight = torch.randn((129, 256), generator=generator).to(getattr(torch, dtype))
    weight[0].zero_()
    reference, scale = quantize_fp8_channel(weight, device="cpu")
    encoded, cuda_scale = quantize_fp8_channel(weight, device="cuda", rows_per_chunk=rows_per_chunk)
    assert torch.equal(reference.view(torch.uint8), encoded.view(torch.uint8))
    assert torch.equal(scale, cuda_scale)
