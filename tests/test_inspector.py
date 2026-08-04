from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.errors import ArtifactError
from axquant.inspector import classify_tensor, inspect_model
from axquant.schema import (
    ArchitectureSupportLevel,
    OptimizationScope,
    SupportTier,
    TensorRole,
)
from axquant.serde import write_data


def test_classifies_supported_tensor_roles() -> None:
    assert classify_tensor("model.layers.1.self_attn.q_proj.weight") == TensorRole.ATTENTION
    assert classify_tensor("model.layers.1.mlp.down_proj.weight") == TensorRole.MLP
    assert classify_tensor("model.layers.1.block.weight", "mtp.safetensors") == TensorRole.MTP_BLOCK
    assert classify_tensor("mtp.projection.weight") == TensorRole.MTP_PROJECTION
    assert classify_tensor("mtp.output_head.weight") == TensorRole.MTP_OUTPUT
    # Mistral3 shells use multi_modal_projector (underscore); must be VISION so
    # MLX-LM sanitize stripping does not break fail-closed plan coverage.
    assert classify_tensor("multi_modal_projector.linear_1.weight") == TensorRole.VISION
    assert classify_tensor("multi_modal_projector.linear_2.weight") == TensorRole.VISION
    assert classify_tensor("vision_tower.transformer.layers.0.attention.q_proj.weight") == (
        TensorRole.VISION
    )
    assert (
        classify_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "norm-shard.safetensors",
        )
        == TensorRole.ATTENTION
    )
    assert (
        classify_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "revision-shard.safetensors",
        )
        == TensorRole.ATTENTION
    )
    assert classify_tensor("encoder.block.weight", "vision.safetensors") == TensorRole.VISION


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


def test_inventory_consumes_immutable_bf16_source_provenance(tiny_model_dir: Path) -> None:
    revision = "a" * 40
    write_data(
        tiny_model_dir / "axquant_source.json",
        {
            "schema_version": "axquant.source-conversion.v1",
            "source_model": "org/source-model",
            "source_revision": revision,
            "dtype": "bfloat16",
            "key_remap_applied": False,
        },
    )

    inventory = inspect_model(tiny_model_dir)
    assert inventory.model.model_id == "org/source-model"
    assert inventory.model.revision == revision

    with pytest.raises(ArtifactError, match="model ID differs"):
        inspect_model(tiny_model_dir, model_id="org/other-model")
    with pytest.raises(ArtifactError, match="revision differs"):
        inspect_model(tiny_model_dir, revision="b" * 40)


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


def test_declared_vision_tower_with_uncovered_naming_disables_conversion(
    qwen36_model_dir: Path,
) -> None:
    """AXQ-018 fail-closed backstop for vision-token classification coverage.

    A declared vision tower (config `vision_config`) with zero VISION-role
    tensors -- e.g. a future family whose vision-tower naming `_VISION_TOKENS`
    does not cover -- must not pass through silently. The misclassified
    tensor still gets some valid generic role and inspection does not
    hard-fail, but the mismatch has to leave a signal.
    """
    # Rebuild model.safetensors with the vision tensor renamed to naming that
    # none of `_VISION_TOKENS` cover, landing it in a generic MLP role instead.
    save_file(
        {
            "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight": np.zeros(
                (8, 8), dtype=np.float32
            ),
            "language_model.model.layers.0.linear_attn.conv1d.weight": np.zeros(
                (8, 4, 1), dtype=np.float32
            ),
            "language_model.model.layers.0.mlp.down_proj.weight": np.zeros(
                (8, 8), dtype=np.float32
            ),
            "language_model.lm_head.weight": np.zeros((16, 8), dtype=np.float32),
            "image_encoder.blocks.0.mlp.fc1.weight": np.zeros((8, 8), dtype=np.float32),
        },
        qwen36_model_dir / "model.safetensors",
    )
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="abc123",
    )
    assert inventory.architecture_profile.vision_present is True
    assert inventory.architecture_profile.support_level == ArchitectureSupportLevel.INVENTORY_ONLY
    assert inventory.architecture_profile.support_tier == SupportTier.INSPECT_ONLY
    assert inventory.architecture_profile.optimization_scope == OptimizationScope.INVENTORY_ONLY
    assert not any(tensor.role == TensorRole.VISION for tensor in inventory.tensors)
    assert any("vision tower" in warning for warning in inventory.warnings)


def test_supported_adapter_downgrades_for_unclassified_tensor(
    qwen36_model_dir: Path,
) -> None:
    save_file(
        {
            "language_model.model.layers.0.self_attn.q_proj.weight": np.zeros(
                (8, 8), dtype=np.float32
            ),
            "language_model.model.layers.0.mlp.down_proj.weight": np.zeros(
                (8, 8), dtype=np.float32
            ),
            "language_model.model.layers.0.new_block.secret_projection.weight": np.zeros(
                (8, 8), dtype=np.float32
            ),
            "language_model.lm_head.weight": np.zeros((16, 8), dtype=np.float32),
            "visual.patch_embed.proj.weight": np.zeros((8, 8), dtype=np.float32),
        },
        qwen36_model_dir / "model.safetensors",
    )

    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="abc123",
    )
    unknown = next(tensor for tensor in inventory.tensors if "secret_projection" in tensor.name)

    assert unknown.role is TensorRole.OTHER
    assert unknown.quantizable is False
    assert inventory.architecture_profile.support_level is (ArchitectureSupportLevel.INVENTORY_ONLY)
    assert inventory.architecture_profile.optimization_scope is (OptimizationScope.INVENTORY_ONLY)
    assert any("classification is incomplete" in warning for warning in inventory.warnings)


def test_nemotron_moegate_is_not_quantizable(tmp_path: Path) -> None:
    """Nemotron-H MoEGate has no MLX to_quantized(); keep it BF16 in plans."""
    import json

    import numpy as np
    from safetensors.numpy import save_file

    model_dir = tmp_path / "Nemotron-3-Nano-30B-A3B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "n_routed_experts": 4,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "backbone.embeddings.weight": np.zeros((32, 16), dtype=np.float32),
            "backbone.layers.0.mixer.gate.weight": np.zeros((4, 16), dtype=np.float32),
            "backbone.layers.0.mixer.gate.e_score_correction_bias": np.zeros(
                (4,), dtype=np.float32
            ),
            "backbone.layers.0.mixer.experts.0.up_proj.weight": np.zeros((8, 16), dtype=np.float32),
            "backbone.layers.0.mixer.out_proj.weight": np.zeros((16, 16), dtype=np.float32),
            "lm_head.weight": np.zeros((32, 16), dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    inventory = inspect_model(
        model_dir,
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        revision="test",
    )
    gate = next(t for t in inventory.tensors if t.name.endswith("mixer.gate.weight"))
    bias = next(t for t in inventory.tensors if "e_score_correction_bias" in t.name)
    expert = next(t for t in inventory.tensors if "experts.0.up_proj" in t.name)
    out_proj = next(t for t in inventory.tensors if t.name.endswith("out_proj.weight"))
    assert gate.role == TensorRole.ROUTER
    assert gate.quantizable is False
    assert bias.quantizable is False
    assert expert.role == TensorRole.EXPERT
    assert expert.quantizable is True
    assert out_proj.quantizable is True


def _write_qwen3_next_fixture(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "num_hidden_layers": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 8), dtype=np.float32),
            "model.layers.0.mlp.gate.weight": np.zeros((4, 8), dtype=np.float32),
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": np.zeros((4, 8, 8), dtype=np.float32),
            "model.layers.0.mlp.switch_mlp.up_proj.weight": np.zeros((4, 8, 8), dtype=np.float32),
            "model.layers.0.mlp.switch_mlp.down_proj.weight": np.zeros((4, 8, 8), dtype=np.float32),
            "model.norm.weight": np.zeros((8,), dtype=np.float32),
            "lm_head.weight": np.zeros((32, 8), dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )


def test_qwen3_next_fused_experts_are_quantizable(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3-Coder-Next"
    _write_qwen3_next_fixture(model_dir)

    inventory = inspect_model(
        model_dir,
        model_id="Qwen/Qwen3-Coder-Next",
        revision="source-revision",
    )

    experts = [tensor for tensor in inventory.tensors if "switch_mlp" in tensor.name]
    assert len(experts) == 3
    assert all(tensor.role == TensorRole.EXPERT for tensor in experts)
    assert all(tensor.quantizable for tensor in experts)
    assert inventory.architecture_profile.support_level == ArchitectureSupportLevel.SUPPORTED
    assert not any("fused expert coverage is incomplete" in item for item in inventory.warnings)


def test_supported_moe_downgrades_when_fused_expert_coverage_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axquant.inspector as inspector_module
    from axquant.architectures.registry import adapter_for

    model_dir = tmp_path / "Qwen3-Coder-Next"
    _write_qwen3_next_fixture(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    original = adapter_for("Qwen/Qwen3-Coder-Next", config)
    assert original is not None

    class BrokenAdapter:
        def profile(self, model_reference: str, model_config: dict[str, object]):
            return original.profile(model_reference, model_config)

        def classify_tensor(self, name: str, source_file: str):
            if "switch_mlp" in name:
                return TensorRole.MLP
            return original.classify_tensor(name, source_file)

    monkeypatch.setattr(inspector_module, "adapter_for", lambda *_args, **_kwargs: BrokenAdapter())
    inventory = inspector_module.inspect_model(
        model_dir,
        model_id="Qwen/Qwen3-Coder-Next",
        revision="source-revision",
    )

    assert inventory.architecture_profile.support_level == ArchitectureSupportLevel.INVENTORY_ONLY
    assert inventory.architecture_profile.optimization_scope == OptimizationScope.INVENTORY_ONLY
    assert any("fused expert coverage is incomplete" in item for item in inventory.warnings)


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


def test_inventory_recognizes_mtp_head_sidecar_filename(tmp_path: Path) -> None:
    """Indexed checkpoints shipping ``mtp_head.safetensors`` (instead of
    ``mtp.safetensors``) must still have that sidecar's tensors picked up —
    converter/probe/release_audit already recognize this alternate filename,
    and inspect must agree or the sidecar is silently dropped from the
    Inventory for any checkpoint that uses a ``model.safetensors.index.json``.
    """
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "model_type": "tiny"}),
        encoding="utf-8",
    )
    save_file(
        {
            "model.embed_tokens.weight": np.zeros((16, 8), dtype=np.float32),
            "lm_head.weight": np.zeros((16, 8), dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "lm_head.weight": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"mtp.projection.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "mtp_head.safetensors",
    )

    inventory = inspect_model(model_dir)

    assert "mtp_head.safetensors" in inventory.source_files
    mtp_tensor = next(t for t in inventory.tensors if t.name == "mtp.projection.weight")
    assert mtp_tensor.role == TensorRole.MTP_PROJECTION


def test_inventory_uses_structured_mtp_sidecar_precision(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "model_type": "tiny"}),
        encoding="utf-8",
    )
    save_file(
        {"model.embed_tokens.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "model.safetensors",
    )
    save_file(
        {"mtp.projection.weight": np.zeros((8, 2), dtype=np.uint32)},
        model_dir / "mtp.safetensors",
    )
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps({"mtp_sidecar_bits": 8}),
        encoding="utf-8",
    )

    inventory = inspect_model(model_dir, allow_quantized=True)

    mtp = next(tensor for tensor in inventory.tensors if tensor.name == "mtp.projection.weight")
    assert mtp.current_bits == 8
    assert mtp.current_precision == "8bit"
    assert mtp.parameters == 64


def test_inventory_rejects_invalid_structured_mtp_sidecar_precision(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "model_type": "tiny"}),
        encoding="utf-8",
    )
    save_file(
        {"mtp.projection.weight": np.zeros((8, 2), dtype=np.uint32)},
        model_dir / "mtp.safetensors",
    )
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps({"mtp_sidecar_bits": True}),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="invalid mtp_sidecar_bits"):
        inspect_model(model_dir, allow_quantized=True)


def test_indexed_inventory_rejects_tensor_membership_drift(tmp_path: Path) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "model_type": "tiny"}),
        encoding="utf-8",
    )
    save_file(
        {"actual.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "model-00001-of-00001.safetensors",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "claimed.weight": "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"missing tensors.*unindexed tensors"):
        inspect_model(model_dir)


def test_indexed_inventory_rejects_tensor_mapped_to_wrong_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "model_type": "tiny"}),
        encoding="utf-8",
    )
    save_file(
        {"first.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"second.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "model-00002-of-00002.safetensors",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first.weight": "model-00002-of-00002.safetensors",
                    "second.weight": "model-00001-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"does not match model\.safetensors\.index\.json"):
        inspect_model(model_dir)


def test_indexed_inventory_rejects_wholly_unindexed_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "indexed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "model_type": "tiny"}),
        encoding="utf-8",
    )
    save_file(
        {"first.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "model-00001-of-00001.safetensors",
    )
    save_file(
        {"hidden.weight": np.zeros((8, 8), dtype=np.float32)},
        model_dir / "unindexed.safetensors",
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first.weight": "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="does not account for Safetensors files"):
        inspect_model(model_dir)


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
