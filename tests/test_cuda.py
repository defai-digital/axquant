from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import load_file, save_file

from axquant import cuda
from axquant.cli import main
from axquant.errors import ArtifactError, PlanningError
from axquant.nvfp4 import Nvfp4Tensor, dequantize_nvfp4
from axquant.schema.cuda import CudaFileDigest, CudaQuantizationPlan
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data


@pytest.fixture
def cuda_source(tmp_path: Path) -> Path:
    source = tmp_path / "Qwen3-CUDA"
    source.mkdir()
    write_data(source / "config.json", {"model_type": "qwen3", "num_hidden_layers": 1})
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "visual.patch_embed.proj.weight",
        "model.audio_encoder.proj.weight",
        "model.layers.0.mlp.unknown_3d.weight",
        "model.layers.0.mlp.unaligned.weight",
        "model.norm.weight",
    ]
    tensors = {name: np.arange(256, dtype=np.float32).reshape(8, 32) / 20 for name in names}
    tensors[names[-3]] = np.ones((2, 8, 32), dtype=np.float32)
    tensors[names[-2]] = np.ones((8, 17), dtype=np.float32)
    tensors[names[-1]] = np.ones(32, dtype=np.float32)
    save_file(tensors, source / "model.safetensors")
    save_file(
        {"mtp.projection.weight": np.ones((8, 32), dtype=np.float32)}, source / "mtp.safetensors"
    )
    write_data(source / "tokenizer_config.json", {"model_max_length": 1024})
    write_data(source / "credentials.json", {"test_only": "never copy this file"})
    write_data(source / "runtime_check.json", {"test_only": True})
    return source


def build_plan(source: Path, **kwargs) -> CudaQuantizationPlan:
    return cuda.plan_cuda_nvfp4(source, allow_unmeasured=True, **kwargs)


def test_plan_preserves_all_protected_and_ineligible_tensors(cuda_source: Path) -> None:
    plan = build_plan(cuda_source)
    selected = {item.tensor_name for item in plan.allocations if item.method == "nvfp4"}
    assert selected == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    assert plan.activation_precision == 16
    assert plan.algorithm == "rtn"
    assert plan.evidence_kind == "unmeasured"
    assert plan.model_id == "Qwen3-CUDA"
    serialized = plan.model_dump_json()
    assert str(cuda_source.parent) not in serialized
    assert "credentials.json" not in serialized
    assert "runtime_check.json" not in serialized


def test_withdrawn_cuda_fp8_mode_is_rejected(cuda_source: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "plan-cuda",
                str(cuda_source),
                "--q-mode",
                "fp8",
                "--allow-unmeasured",
                "--output",
                str(cuda_source.parent / "plan.json"),
            ]
        )
    assert not (cuda_source.parent / "plan.json").exists()


@pytest.mark.parametrize("spelling", ["view_separator", "view_seperator"])
def test_image_separator_remains_protected(cuda_source: Path, spelling: str) -> None:
    member = cuda_source / "model.safetensors"
    tensors = load_file(member)
    name = f"model.{spelling}"
    tensors[name] = np.ones(32, dtype=np.float32)
    save_file(tensors, member)
    allocation = next(
        item for item in build_plan(cuda_source).allocations if item.tensor_name == name
    )
    assert allocation.role.value == "vision"
    assert allocation.method == "preserve"


def test_runtime_vision_aliases_remain_unquantized(cuda_source: Path) -> None:
    ignored = cuda._quantization_config(build_plan(cuda_source))["ignore"]
    patterns = [item.removeprefix("re:") for item in ignored if item.startswith("re:")]
    for path in (
        "vision_model.encoder.layers.0.mlp.fc1",
        "model.vision_model.transformer.layers.0.mlp.fc1",
        "sam_model.blocks.0.attn.qkv",
        "projector.layers.0",
        "model.audio_encoder.proj",
    ):
        assert any(re.fullmatch(pattern, path) for pattern in patterns)
    assert not any(re.fullmatch(pattern, "model.layers.0.mlp.down_proj") for pattern in patterns)


def test_runtime_mtp_virtual_expert_projections_remain_unquantized(cuda_source: Path) -> None:
    plan = build_plan(cuda_source)
    ignored = cuda._quantization_config(plan)["ignore"]
    patterns = [item.removeprefix("re:") for item in ignored if item.startswith("re:")]
    for prefix in ("mtp", "model.mtp"):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}.layers.1.mixer.experts.0.{projection}"
            assert any(re.fullmatch(pattern, name) for pattern in patterns)
    assert not any(
        re.fullmatch(pattern, "backbone.layers.1.mixer.experts.0.up_proj") for pattern in patterns
    )
    allocation = next(item for item in plan.allocations if item.tensor_name.startswith("mtp."))
    invalid = plan.model_copy(
        update={
            "allocations": [
                item.model_copy(update={"method": "nvfp4"}) if item == allocation else item
                for item in plan.allocations
            ]
        }
    )
    with pytest.raises(PlanningError, match="MTP protection overlaps"):
        cuda._quantization_config(invalid)


def test_keep_policy_and_content_binding(cuda_source: Path) -> None:
    plan = build_plan(cuda_source, keep_patterns=["*.self_attn.*"])
    assert [item.tensor_name for item in plan.allocations if item.method == "nvfp4"] == [
        "model.layers.0.mlp.down_proj.weight"
    ]
    changed = dict(read_data(cuda_source / "config.json"), num_hidden_layers=2)
    write_data(cuda_source / "config.json", changed)
    with pytest.raises(ArtifactError, match="content changed"):
        cuda.convert_cuda_nvfp4(
            cuda_source, plan, cuda_source.parent / "pack", allow_unmeasured=True
        )


def test_both_stages_require_explicit_unmeasured_acknowledgement(cuda_source: Path) -> None:
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        cuda.plan_cuda_nvfp4(cuda_source)
    plan = build_plan(cuda_source)
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        cuda.convert_cuda_nvfp4(cuda_source, plan, cuda_source.parent / "pack")


def test_unknown_architecture_and_prequantized_source_fail_closed(cuda_source: Path) -> None:
    write_data(cuda_source / "config.json", {"model_type": "unknown"})
    with pytest.raises(PlanningError, match="convertible"):
        build_plan(cuda_source)
    write_data(
        cuda_source / "config.json",
        {"model_type": "qwen3", "quantization_config": {"format": "awq"}},
    )
    with pytest.raises(ArtifactError, match="already quantized"):
        build_plan(cuda_source)


@pytest.mark.parametrize("path", ["../weights", "/weights", "dir/../weights", "dir\\weights", "."])
def test_member_paths_cannot_escape_checkpoint(path: str) -> None:
    with pytest.raises(ValidationError):
        CudaFileDigest(path=path, sha256="0" * 64, size_bytes=1)


@pytest.mark.parametrize("identity", ["/private/model", "~/model", "file:model", "C:\\model"])
def test_private_model_id_fails(cuda_source: Path, identity: str) -> None:
    with pytest.raises(ValueError, match="filesystem path"):
        build_plan(cuda_source, model_id=identity)


def test_missing_plan_allocation_is_rejected_before_serialization(cuda_source: Path) -> None:
    plan = build_plan(cuda_source)
    forged = plan.model_copy(update={"allocations": plan.allocations[:-1]})
    with pytest.raises(PlanningError, match="protection policy"):
        cuda.convert_cuda_nvfp4(
            cuda_source, forged, cuda_source.parent / "pack", allow_unmeasured=True
        )


def test_forged_lowered_floor_fails_schema(cuda_source: Path) -> None:
    payload = build_plan(cuda_source).model_dump(mode="json")
    next(item for item in payload["allocations"] if item["tensor_name"] == "lm_head.weight")[
        "method"
    ] = "nvfp4"
    with pytest.raises(ValidationError, match="eligible floating text"):
        CudaQuantizationPlan.model_validate(payload)


def test_same_size_weight_mutation_fails_exact_content_binding(cuda_source: Path) -> None:
    plan = build_plan(cuda_source)
    path = cuda_source / "model.safetensors"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ArtifactError, match="content changed"):
        cuda.convert_cuda_nvfp4(
            cuda_source, plan, cuda_source.parent / "pack", allow_unmeasured=True
        )


def test_symlink_asset_fails(cuda_source: Path) -> None:
    asset = cuda_source / "tokenizer_config.json"
    target = cuda_source.parent / "shared-tokenizer.json"
    asset.rename(target)
    asset.symlink_to(target)
    with pytest.raises(ArtifactError, match="regular files"):
        build_plan(cuda_source)


@pytest.mark.parametrize("device", ["mps", "cuda:abc", "cuda:-1"])
def test_invalid_device_fails_without_output(cuda_source: Path, device: str) -> None:
    output = cuda_source.parent / "pack"
    with pytest.raises(PlanningError, match="device must"):
        cuda.convert_cuda_nvfp4(
            cuda_source, build_plan(cuda_source), output, device=device, allow_unmeasured=True
        )
    assert not output.exists()


def test_existing_output_and_source_overlap_are_rejected(cuda_source: Path) -> None:
    plan = build_plan(cuda_source)
    with pytest.raises(ArtifactError, match="already exists"):
        cuda.convert_cuda_nvfp4(cuda_source, plan, cuda_source, allow_unmeasured=True)
    with pytest.raises(ArtifactError, match="overlap"):
        cuda.convert_cuda_nvfp4(cuda_source, plan, cuda_source / "pack", allow_unmeasured=True)


def test_cpu_export_preserves_bf16_and_public_nvfp4_layout(cuda_source: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    from safetensors.torch import save_file as save_torch

    main_weights = load_file(str(cuda_source / "model.safetensors"))
    main_weights = {name: tensor.to(torch.bfloat16) for name, tensor in main_weights.items()}
    save_torch(main_weights, str(cuda_source / "model.safetensors"))
    plan = build_plan(cuda_source)
    output = cuda_source.parent / "pack"
    manifest = cuda.convert_cuda_nvfp4(
        cuda_source, plan, output, device="cpu", allow_unmeasured=True
    )
    assert manifest.backend == "numpy-reference"
    assert not manifest.runtime_verified and not manifest.quality_certified
    assert manifest.plan_sha256 == stable_sha256(plan)
    assert not (output / "credentials.json").exists()
    assert not (output / "runtime_check.json").exists()
    assert file_sha256(output / "mtp.safetensors") == file_sha256(cuda_source / "mtp.safetensors")
    saved = load_file(str(output / "model.safetensors"))
    for item in plan.allocations:
        if item.source_file != "model.safetensors":
            continue
        if item.method == "preserve":
            assert saved[item.tensor_name].dtype == torch.bfloat16
            assert torch.equal(saved[item.tensor_name], main_weights[item.tensor_name])
        else:
            prefix = item.tensor_name.removesuffix(".weight")
            assert item.tensor_name not in saved
            assert saved[prefix + ".weight_packed"].dtype == torch.uint8
            assert saved[prefix + ".weight_scale"].dtype == torch.float8_e4m3fn
            encoded = Nvfp4Tensor(
                saved[prefix + ".weight_packed"].numpy(),
                saved[prefix + ".weight_scale"].view(torch.uint8).numpy(),
                saved[prefix + ".weight_global_scale"].item(),
                item.shape,
                "numpy-reference",
            )
            original = main_weights[item.tensor_name].float().numpy()
            assert np.linalg.norm(dequantize_nvfp4(encoded) - original) < 0.15 * np.linalg.norm(
                original
            )
    index = read_data(output / "model.safetensors.index.json")
    assert set(index["weight_map"]) == set(saved)
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in saved.values()
    )
    for member in manifest.files:
        assert member.sha256 == file_sha256(output / member.path)
    config = read_data(output / "config.json")["quantization_config"]
    assert config["format"] == "nvfp4-pack-quantized"
    assert config["config_groups"]["nvfp4"]["weights"]["strategy"] == "tensor_group"
    assert "lm_head" in config["ignore"]


def test_export_failure_rolls_back(cuda_source: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    output = cuda_source.parent / "pack"

    def fail_quantizer(_, **kwargs):
        raise ArtifactError("quantizer failure")

    monkeypatch.setattr(cuda, "quantize_nvfp4_reference", fail_quantizer)
    with pytest.raises(ArtifactError, match="quantizer failure"):
        cuda.convert_cuda_nvfp4(
            cuda_source, build_plan(cuda_source), output, device="cpu", allow_unmeasured=True
        )
    assert not output.exists()
    assert not list(output.parent.glob(".pack.nvfp4-*"))


def test_mutation_during_conversion_rolls_back(cuda_source: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    output = cuda_source.parent / "pack"
    original = cuda.quantize_nvfp4_reference

    def mutate_then_quantize(weight, **kwargs):
        write_data(cuda_source / "tokenizer_config.json", {"model_max_length": 2048})
        return original(weight, **kwargs)

    monkeypatch.setattr(cuda, "quantize_nvfp4_reference", mutate_then_quantize)
    with pytest.raises(ArtifactError, match="content changed"):
        cuda.convert_cuda_nvfp4(
            cuda_source, build_plan(cuda_source), output, device="cpu", allow_unmeasured=True
        )
    assert not output.exists()
    assert not list(output.parent.glob(".pack.nvfp4-*"))


def test_fused_qkv_protection_is_atomic(cuda_source: Path) -> None:
    from safetensors.numpy import load_file

    path = cuda_source / "model.safetensors"
    tensors = load_file(path)
    prefix = "model.layers.0.self_attn."
    tensors[prefix + "k_proj.weight"] = np.ones((8, 32), dtype=np.float32)
    tensors[prefix + "v_proj.weight"] = np.ones((8, 32), dtype=np.float32)
    save_file(tensors, path)
    plan = build_plan(cuda_source, keep_patterns=["*.k_proj.weight"])
    assert all(
        item.method == "preserve"
        for item in plan.allocations
        if item.tensor_name.startswith(prefix)
    )
    assert all(
        item.scale_group is None for item in plan.allocations if item.tensor_name.startswith(prefix)
    )


def test_fused_global_scales_match_across_source_shards(cuda_source: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.numpy import load_file
    from safetensors.torch import load_file as load_torch

    path = cuda_source / "model.safetensors"
    tensors = load_file(path)
    prefix = "model.layers.0.self_attn."
    tensors[prefix + "k_proj.weight"] = np.full((8, 32), 2.0, dtype=np.float32)
    save_file(tensors, path)
    v_name = prefix + "v_proj.weight"
    save_file(
        {v_name: np.full((8, 32), 20.0, dtype=np.float32)}, cuda_source / "model-2.safetensors"
    )
    plan = build_plan(cuda_source)
    fused = [item for item in plan.allocations if item.tensor_name.startswith(prefix)]
    assert {item.scale_group for item in fused} == {prefix + "qkv_proj"}
    output = cuda_source.parent / "pack"
    cuda.convert_cuda_nvfp4(cuda_source, plan, output, device="cpu", allow_unmeasured=True)
    saved = {
        **load_torch(str(output / "model.safetensors")),
        **load_torch(str(output / "model-2.safetensors")),
    }
    globals = [
        saved[item.tensor_name.removesuffix(".weight") + ".weight_global_scale"] for item in fused
    ]
    assert all(torch.equal(globals[0], item) for item in globals[1:])
    assert globals[0].item() == pytest.approx(2688 / 20)


def test_cli_plan_and_convert(cuda_source: Path) -> None:
    pytest.importorskip("torch")
    path = cuda_source.parent / "plan.json"
    output = cuda_source.parent / "pack"
    assert main(["plan-cuda", str(cuda_source), "--output", str(path)]) == 2
    assert main(["plan-cuda", str(cuda_source), "--output", str(path), "--allow-unmeasured"]) == 0
    assert load_model(path, CudaQuantizationPlan).format == "nvfp4"
    assert (
        main(
            [
                "convert-cuda",
                str(cuda_source),
                "--plan",
                str(path),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--allow-unmeasured",
            ]
        )
        == 0
    )
    assert (output / cuda.MANIFEST_NAME).exists()


def test_cli_quantize(cuda_source: Path) -> None:
    pytest.importorskip("torch")
    output = cuda_source.parent / "pack"
    assert (
        main(
            [
                "quantize-cuda",
                str(cuda_source),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--allow-unmeasured",
            ]
        )
        == 0
    )
    payload = json.loads((output / cuda.PLAN_NAME).read_text())
    assert payload["algorithm"] == "rtn"


def test_public_compressed_tensors_config_and_decompression(cuda_source: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.nvfp4 import NVFP4PackedCompressor
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from compressed_tensors.quantization import QuantizationConfig
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    from safetensors.torch import load_file

    output = cuda_source.parent / "pack"
    cuda.convert_cuda_nvfp4(
        cuda_source, build_plan(cuda_source), output, device="cpu", allow_unmeasured=True
    )
    config = QuantizationConfig.model_validate(
        read_data(output / "config.json")["quantization_config"]
    )
    saved = load_file(str(output / "model.safetensors"))
    prefix = "model.layers.0.mlp.down_proj."
    state = {
        name.removeprefix(prefix): tensor
        for name, tensor in saved.items()
        if name.startswith(prefix)
    }
    decoded = NVFP4PackedCompressor.decompress(state, config.config_groups["nvfp4"])["weight"]
    expected = Nvfp4Tensor(
        state["weight_packed"].numpy(),
        state["weight_scale"].view(torch.uint8).numpy(),
        state["weight_global_scale"].item(),
        (8, 32),
        "numpy-reference",
    )
    reference = dequantize_nvfp4(expected)
    # The compressor's default BF16 math rounds both scales and reconstruction.
    np.testing.assert_allclose(decoded.float().numpy(), reference, rtol=8e-3, atol=1e-3)
    unpacked = unpack_fp4_from_uint8(state["weight_packed"], 8, 32, dtype=torch.float32)
    fp32 = dequantize(
        unpacked,
        state["weight_scale"].float(),
        args=config.config_groups["nvfp4"].weights,
        global_scale=state["weight_global_scale"],
        dtype=torch.float32,
    )
    np.testing.assert_allclose(fp32.numpy(), reference, rtol=1e-6, atol=1e-6)
