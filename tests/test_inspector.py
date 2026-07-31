from __future__ import annotations

from pathlib import Path

import pytest

from axquant.errors import ArtifactError
from axquant.inspector import classify_tensor, inspect_model
from axquant.schema import (
    ArchitectureSupportLevel,
    OptimizationScope,
    TensorRole,
)


def test_classifies_supported_tensor_roles() -> None:
    assert classify_tensor("model.layers.1.self_attn.q_proj.weight") == TensorRole.ATTENTION
    assert classify_tensor("model.layers.1.mlp.down_proj.weight") == TensorRole.MLP
    assert classify_tensor("model.layers.1.block.weight", "mtp.safetensors") == TensorRole.MTP_BLOCK
    assert classify_tensor("mtp.projection.weight") == TensorRole.MTP_PROJECTION
    assert classify_tensor("mtp.output_head.weight") == TensorRole.MTP_OUTPUT


def test_inventory_detects_mtp_ties_and_protection(tiny_model_dir: Path) -> None:
    inventory = inspect_model(tiny_model_dir, model_id="org/tiny", revision="abc123")
    assert inventory.mtp_present is True
    assert inventory.model.architecture == "TinyForCausalLM"
    assert inventory.quantized_source is False
    assert len(inventory.tensors) == 7
    assert inventory.config_sha256
    assert inventory.tied_weight_groups == [["model.embed_tokens.weight", "lm_head.weight"]]
    mtp = next(tensor for tensor in inventory.tensors if tensor.name == "mtp.projection.weight")
    assert mtp.role == TensorRole.MTP_PROJECTION
    assert mtp.protected_recommendation is True
    assert mtp.current_precision == "f32"


def test_quantized_source_requires_explicit_inventory_permission(
    tiny_model_dir: Path,
) -> None:
    config = tiny_model_dir / "config.json"
    value = __import__("json").loads(config.read_text())
    value["quantization"] = {"bits": 4}
    config.write_text(__import__("json").dumps(value))
    with pytest.raises(ArtifactError, match="already quantized"):
        inspect_model(tiny_model_dir)
    inventory = inspect_model(tiny_model_dir, allow_quantized=True)
    assert inventory.quantized_source is True


def test_qwen36_adapter_sets_text_path_and_protects_vision(
    qwen36_model_dir: Path,
) -> None:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="abc123",
    )
    profile = inventory.architecture_profile
    assert profile.adapter_id == "qwen36-v1"
    assert profile.support_level == ArchitectureSupportLevel.SUPPORTED
    assert profile.optimization_scope == OptimizationScope.TEXT_PATH
    assert profile.dense is True
    assert profile.mtp_declared is True
    assert profile.vision_present is True
    linear_attention = next(tensor for tensor in inventory.tensors if "in_proj_qkvz" in tensor.name)
    convolution = next(tensor for tensor in inventory.tensors if "conv1d" in tensor.name)
    vision = next(tensor for tensor in inventory.tensors if tensor.role == TensorRole.VISION)
    assert linear_attention.role == TensorRole.ATTENTION
    assert convolution.role == TensorRole.ATTENTION
    assert convolution.quantizable is False
    assert vision.protected_recommendation is True


def test_quantized_inventory_reconstructs_logical_parameters(
    packed_model_dir: Path,
) -> None:
    inventory = inspect_model(packed_model_dir, allow_quantized=True)
    weight = next(tensor for tensor in inventory.tensors if tensor.name.endswith(".weight"))
    scales = next(tensor for tensor in inventory.tensors if tensor.name.endswith(".scales"))
    assert weight.physical_elements == 16
    assert weight.parameters == 128
    assert weight.current_precision == "4bit"
    assert weight.current_bits == 4
    assert weight.current_group_size == 64
    assert weight.storage_bytes == 64
    assert scales.quantization_metadata is True
    assert scales.parameters == 0
    assert inventory.total_parameters == 136
    assert inventory.precision_parameters == {"4bit": 128, "f32": 8}
    assert inventory.weight_bytes == (packed_model_dir / "model.safetensors").stat().st_size


def test_inventory_ignores_nested_auxiliary_safetensors(
    tiny_model_dir: Path,
) -> None:
    auxiliary = tiny_model_dir / "auxiliary"
    auxiliary.mkdir()
    (auxiliary / "model.safetensors").write_bytes(
        (tiny_model_dir / "model.safetensors").read_bytes()
    )
    inventory = inspect_model(tiny_model_dir)
    assert len(inventory.tensors) == 7
    assert all(not source.startswith("auxiliary/") for source in inventory.source_files)
