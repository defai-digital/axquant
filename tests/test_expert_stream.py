from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import axquant.expert_stream as expert_stream
from axquant.cli._parser import _build_parser
from axquant.errors import PlanningError
from axquant.expert_stream import (
    build_expert_stream_manifest,
    validate_expert_stream_request,
)
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArchitectureProfile,
    ExpertStreamManifest,
    Inventory,
    ModelIdentity,
    OptimizationScope,
    TensorRole,
    TensorSpec,
)
from axquant.schema.registry import schema_entry
from axquant.serde import load_model, write_data


def _tensor(
    name: str,
    *,
    layer: int,
    role: TensorRole = TensorRole.EXPERT,
    metadata: bool = False,
) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=name.rsplit(".", 1)[0],
        shape=(4, 8, 1 if metadata else 2),
        dtype="F32" if metadata else "U32",
        parameters=0 if metadata else 512,
        physical_elements=32 if metadata else 64,
        storage_bytes=128 if metadata else 256,
        role=role,
        quantizable=not metadata,
        file=f"model-{layer + 1:05d}-of-00002.safetensors",
        current_precision="f32" if metadata else "2bit",
        current_bits=None if metadata else 2,
        current_group_size=None if metadata else 64,
        quantization_metadata=metadata,
        protected_recommendation=False,
        protection_reason=None,
    )


def _packed_inventory() -> Inventory:
    tensors: list[TensorSpec] = []
    for layer in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.switch_mlp.{projection}"
            tensors.extend(
                [
                    _tensor(f"{base}.weight", layer=layer),
                    _tensor(f"{base}.scales", layer=layer, metadata=True),
                    _tensor(f"{base}.biases", layer=layer, metadata=True),
                ]
            )
    tensors.append(
        TensorSpec(
            name="model.layers.0.self_attn.q_proj.weight",
            module_path="model.layers.0.self_attn.q_proj",
            shape=(8, 8),
            dtype="BF16",
            parameters=64,
            physical_elements=64,
            storage_bytes=128,
            role=TensorRole.ATTENTION,
            quantizable=True,
            file="model-00001-of-00002.safetensors",
            current_precision="bf16",
            current_bits=16,
            current_group_size=None,
            quantization_metadata=False,
            protected_recommendation=False,
            protection_reason=None,
        )
    )
    return Inventory(
        model=ModelIdentity(model_id="Qwen/Qwen3.8-2.4T-A95B"),
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors if tensor.quantizable),
        weight_bytes=sum(tensor.storage_bytes for tensor in tensors),
        mtp_weight_bytes=0,
        precision_parameters={},
        mtp_present=False,
        quantized_source=True,
        source_files=sorted({tensor.file for tensor in tensors}),
        architecture_profile=ArchitectureProfile(adapter_id="qwen38-moe-v1"),
        config_sha256="0" * 64,
    )


def test_expert_stream_schema_round_trip_and_registry(tmp_path: Path) -> None:
    manifest = build_expert_stream_manifest(
        _packed_inventory(),
        experts_per_tok=2,
        requirement="required",
        default_group_size=64,
    )
    path = tmp_path / "ax_expert_stream.json"
    write_data(path, manifest)

    assert load_model(path, ExpertStreamManifest) == manifest
    entry = schema_entry("axquant.expert-stream.v1")
    assert entry.model is ExpertStreamManifest
    assert entry.compatibility_class == "operational"
    assert entry.freeze_policy == "additive-ok"


def test_expert_stream_schema_rejects_unknown_mode() -> None:
    payload = build_expert_stream_manifest(
        _packed_inventory(),
        experts_per_tok=2,
        requirement="auto",
        default_group_size=64,
    ).model_dump(mode="json")
    payload["mode"] = "per-expert"

    with pytest.raises(ValidationError):
        ExpertStreamManifest.model_validate(payload)


def test_manifest_from_tiny_packed_inventory_keeps_layer_stack_shards() -> None:
    manifest = build_expert_stream_manifest(
        _packed_inventory(),
        experts_per_tok=2,
        requirement="required",
        default_group_size=64,
    )

    assert manifest.required is True
    assert manifest.mode == "layer-stack"
    assert manifest.num_experts == 4
    assert manifest.experts_per_tok == 2
    assert len(manifest.tensors) == 18
    assert {tensor.layer for tensor in manifest.tensors} == {0, 1}
    assert {tensor.proj for tensor in manifest.tensors} == {"gate", "up", "down"}
    assert {tensor.file for tensor in manifest.tensors} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert manifest.estimated_full_resident_bytes == 3200
    assert manifest.estimated_resident_bytes == 128
    assert manifest.estimated_max_layer_expert_bytes == 1536


def test_gate_projection_without_separate_up_is_declared_gate_up() -> None:
    inventory = _packed_inventory()
    inventory.tensors = [
        tensor for tensor in inventory.tensors if ".switch_mlp.up_proj." not in tensor.name
    ]
    manifest = build_expert_stream_manifest(
        inventory,
        experts_per_tok=2,
        requirement="auto",
        default_group_size=64,
    )

    assert {tensor.proj for tensor in manifest.tensors} == {"gate_up", "down"}


def test_auto_requires_streaming_above_256_gib() -> None:
    small = _packed_inventory()
    assert (
        build_expert_stream_manifest(
            small,
            experts_per_tok=2,
            requirement="auto",
            default_group_size=64,
        ).required
        is False
    )
    large = _packed_inventory()
    large.tensors[0].storage_bytes = expert_stream.AUTO_REQUIRED_BYTES
    assert (
        build_expert_stream_manifest(
            large,
            experts_per_tok=2,
            requirement="auto",
            default_group_size=64,
        ).required
        is True
    )


def test_emit_manifest_and_runtime_pointer_from_synthetic_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen3_5_moe_text","num_experts":4,"num_experts_per_tok":2}',
        encoding="utf-8",
    )
    monkeypatch.setattr(expert_stream, "inspect_model", lambda *args, **kwargs: _packed_inventory())
    plan = SimpleNamespace(
        source_model=ModelIdentity(model_id="Qwen/Qwen3.8-2.4T-A95B"),
        architecture_profile=SimpleNamespace(
            adapter_id="qwen38-moe-v1",
            optimization_scope=OptimizationScope.TEXT_PATH,
        ),
        group_size=64,
        kv_cache=None,
        assignments=[SimpleNamespace(role=TensorRole.EXPERT)],
    )

    emitted = expert_stream.emit_expert_stream_manifest(
        tmp_path,
        plan,
        setting="required",
    )

    manifest_path = tmp_path / "ax_expert_stream.json"
    assert emitted is not None
    assert load_model(manifest_path, ExpertStreamManifest) == emitted
    runtime = build_runtime_metadata(plan, tmp_path)
    assert runtime.memory_policy["expert_stream"] == "required"
    assert runtime.memory_policy["expert_stream_manifest"] == "ax_expert_stream.json"


def test_emit_manifest_accepts_nested_gemma4_top_k_experts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"gemma4","text_config":{"num_experts":4,"top_k_experts":2}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(expert_stream, "inspect_model", lambda *args, **kwargs: _packed_inventory())
    plan = SimpleNamespace(
        source_model=ModelIdentity(model_id="google/gemma-4-26B-A4B-it"),
        architecture_profile=SimpleNamespace(
            adapter_id="gemma4-dense-v1",
            optimization_scope=OptimizationScope.TEXT_PATH,
        ),
        group_size=64,
        kv_cache=None,
        assignments=[SimpleNamespace(role=TensorRole.EXPERT)],
    )

    emitted = expert_stream.emit_expert_stream_manifest(tmp_path, plan, setting="auto")

    assert emitted is not None
    assert emitted.required is False
    assert emitted.experts_per_tok == 2


def test_expert_stream_flags_default_auto_and_super_off_fails_closed() -> None:
    parser = _build_parser()
    staged = parser.parse_args(
        ["convert", "--model", "source", "--plan", "plan.json", "--output", "out"]
    )
    simple = parser.parse_args(["quantize", "Qwen/Qwen3.8-2.4T-A95B"])
    assert staged.expert_stream == "auto"
    assert simple.expert_stream == "auto"

    with pytest.raises(PlanningError, match="512 GB Mac"):
        validate_expert_stream_request("qwen38-moe-v1", "off")

    validate_expert_stream_request("deepseek-v4-v1", "off")
    validate_expert_stream_request("deepseek-v4-v1", "auto")
    validate_expert_stream_request("deepseek-v4-v1", "required")


def _deepseek_flash_inventory() -> Inventory:
    tensors: list[TensorSpec] = []
    for layer in range(2):
        for projection in ("gate_proj", "down_proj"):
            base = f"model.layers.{layer}.ffn.switch_mlp.{projection}"
            tensors.extend(
                [
                    _tensor(f"{base}.weight", layer=layer),
                    _tensor(f"{base}.scales", layer=layer, metadata=True),
                    _tensor(f"{base}.biases", layer=layer, metadata=True),
                ]
            )
        tensors.append(
            TensorSpec(
                name=f"model.layers.{layer}.ffn.shared_experts.w1.weight",
                module_path=f"model.layers.{layer}.ffn.shared_experts.w1",
                shape=(8, 8),
                dtype="BF16",
                parameters=64,
                physical_elements=64,
                storage_bytes=128,
                role=TensorRole.MLP,
                quantizable=True,
                file=f"model-{layer + 1:05d}-of-00002.safetensors",
                current_precision="bf16",
                current_bits=16,
                current_group_size=None,
                quantization_metadata=False,
                protected_recommendation=False,
                protection_reason=None,
            )
        )
    return Inventory(
        model=ModelIdentity(model_id="deepseek-ai/DeepSeek-V4-Flash"),
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors if tensor.quantizable),
        weight_bytes=sum(tensor.storage_bytes for tensor in tensors),
        mtp_weight_bytes=0,
        precision_parameters={},
        mtp_present=False,
        quantized_source=True,
        source_files=sorted({tensor.file for tensor in tensors}),
        architecture_profile=ArchitectureProfile(adapter_id="deepseek-v4-v1"),
        config_sha256="0" * 64,
    )


def test_deepseek_v4_flash_fused_switch_mlp_is_streamable() -> None:
    manifest = build_expert_stream_manifest(
        _deepseek_flash_inventory(),
        experts_per_tok=2,
        requirement="auto",
        default_group_size=64,
    )

    assert manifest.required is False
    assert manifest.mode == "layer-stack"
    assert manifest.num_experts == 4
    assert manifest.experts_per_tok == 2
    assert {tensor.proj for tensor in manifest.tensors} == {"gate_up", "down"}
    assert {tensor.layer for tensor in manifest.tensors} == {0, 1}
    assert all("switch_mlp" in tensor.name for tensor in manifest.tensors)
    assert all("shared_experts" not in tensor.name for tensor in manifest.tensors)


def test_deepseek_v4_source_w1_w2_w3_stacks_map_to_projections() -> None:
    inventory = _deepseek_flash_inventory()
    inventory.tensors = [
        _tensor("model.layers.0.ffn.experts.w1.weight", layer=0),
        _tensor("model.layers.0.ffn.experts.w1.scales", layer=0, metadata=True),
        _tensor("model.layers.0.ffn.experts.w3.weight", layer=0),
        _tensor("model.layers.0.ffn.experts.w3.scales", layer=0, metadata=True),
        _tensor("model.layers.0.ffn.experts.w2.weight", layer=0),
        _tensor("model.layers.0.ffn.experts.w2.scales", layer=0, metadata=True),
    ]
    manifest = build_expert_stream_manifest(
        inventory,
        experts_per_tok=2,
        requirement="required",
        default_group_size=32,
    )
    assert {tensor.proj for tensor in manifest.tensors} == {"gate", "up", "down"}


def test_emit_deepseek_flash_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"deepseek_v4","n_routed_experts":4,"num_experts_per_tok":2}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        expert_stream, "inspect_model", lambda *args, **kwargs: _deepseek_flash_inventory()
    )
    plan = SimpleNamespace(
        source_model=ModelIdentity(model_id="deepseek-ai/DeepSeek-V4-Flash"),
        architecture_profile=SimpleNamespace(
            adapter_id="deepseek-v4-v1",
            optimization_scope=OptimizationScope.TEXT_PATH,
        ),
        group_size=64,
        kv_cache=None,
        assignments=[SimpleNamespace(role=TensorRole.EXPERT)],
    )

    emitted = expert_stream.emit_expert_stream_manifest(tmp_path, plan, setting="required")
    assert emitted is not None
    assert emitted.required is True
    runtime = build_runtime_metadata(plan, tmp_path)
    assert runtime.memory_policy["expert_stream"] == "required"
    assert runtime.memory_policy["expert_stream_manifest"] == "ax_expert_stream.json"
