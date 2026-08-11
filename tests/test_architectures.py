from __future__ import annotations

from pathlib import Path

import pytest

from axquant.architectures import registry
from axquant.architectures.dense_family import (
    DENSE_FAMILY_SPECS,
    DenseFamilyAdapter,
    DenseFamilySpec,
)
from axquant.architectures.qwen36 import Qwen36Adapter
from axquant.architectures.registry import adapter_for
from axquant.errors import ArtifactError
from axquant.schema import SupportTier, TensorRole


def _qwen36_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
        },
    }


def test_qwen36_adapter_requires_explicit_product_identity() -> None:
    adapter = Qwen36Adapter()
    config = _qwen36_config()
    assert adapter.matches("Qwen/Qwen3.6-27B", config)
    assert not adapter.matches("Qwen/Qwen3.5-27B", config)
    assert not adapter.matches("/models/anonymous-checkpoint", config)


def test_qwen36_adapter_can_use_pinned_config_identity() -> None:
    adapter = Qwen36Adapter()
    config = {
        "model_type": "qwen3_5",
        "_name_or_path": "Qwen/Qwen3.6-27B",
    }
    assert adapter.matches("/models/revision-snapshot", config)


def test_qwen36_supported_profile_is_convertible() -> None:
    profile = Qwen36Adapter().profile("Qwen/Qwen3.6-27B", _qwen36_config())
    assert profile.support_tier is SupportTier.CONVERTIBLE


def test_qwen36_unsupported_profile_is_inspect_only() -> None:
    config = _qwen36_config()
    text = config["text_config"]
    assert isinstance(text, dict)
    text["num_experts"] = 64
    profile = Qwen36Adapter().profile("Qwen/Qwen3.6-27B", config)
    assert profile.support_tier is SupportTier.INSPECT_ONLY


@pytest.mark.parametrize(
    ("reference", "updates"),
    [
        ("Qwen/Qwen3.6-127B", {}),
        ("Qwen/Qwen3.6-27B", {"num_hidden_layers": 63}),
        ("Qwen/Qwen3.6-27B", {"num_hidden_layers": True}),
        ("Qwen/Qwen3.6-27B", {"hidden_size": 4096}),
        ("Qwen/Qwen3.6-27B", {"intermediate_size": None}),
    ],
)
def test_qwen36_dense_profile_requires_exact_catalog_identity_and_signature(
    reference: str,
    updates: dict[str, object],
) -> None:
    config = _qwen36_config()
    text = config["text_config"]
    assert isinstance(text, dict)
    text.update(updates)

    profile = Qwen36Adapter().profile(reference, config)

    assert profile.support_tier is SupportTier.INSPECT_ONLY


def test_qwen36_null_moe_fields_do_not_downgrade_dense_checkpoint() -> None:
    config = _qwen36_config()
    text = config["text_config"]
    assert isinstance(text, dict)
    text.update(
        {
            "num_experts": None,
            "num_experts_per_tok": 0,
            "moe_intermediate_size": None,
            "enable_moe_block": False,
        }
    )

    profile = Qwen36Adapter().profile("Qwen/Qwen3.6-27B", config)

    assert profile.dense is True
    assert profile.support_tier is SupportTier.CONVERTIBLE


def test_qwen36_structural_signature_supports_renamed_artifact_without_size() -> None:
    profile = Qwen36Adapter().profile(
        "AutomatosX/qwen36-4bit",
        _qwen36_config(),
    )

    assert profile.support_tier is SupportTier.CONVERTIBLE


def test_registry_resolves_qwen36_without_ambiguity() -> None:
    adapter = adapter_for("Qwen/Qwen3.6-27B", _qwen36_config())
    assert adapter is not None
    assert adapter.adapter_id == "qwen36-v1"


def test_registry_resolves_qwen35_family_as_convertible() -> None:
    """AXQ-017 promotion (2026-08-01): real Qwen3.5-9B evidence in the E5 log."""
    config = {
        "model_type": "qwen3_5",
        "_name_or_path": "Qwen/Qwen3.5-9B",
        "text_config": {"num_hidden_layers": 36},
    }
    adapter = adapter_for("Qwen/Qwen3.5-9B", config)
    assert adapter is not None
    assert adapter.adapter_id == "qwen35-dense-v1"
    profile = adapter.profile("Qwen/Qwen3.5-9B", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.product_family == "qwen3.5"
    assert profile.text_layer_count == 36


def test_registry_resolves_qwen3_asr_with_nested_text_and_audio_configs() -> None:
    config = {
        "model_type": "qwen3_asr",
        "thinker_config": {
            "text_config": {"num_hidden_layers": 28},
            "audio_config": {"encoder_layers": 24},
        },
    }
    adapter = adapter_for("Qwen/Qwen3-ASR-1.7B", config)
    assert adapter is not None
    assert adapter.adapter_id == "qwen3-asr-v1"
    profile = adapter.profile("Qwen/Qwen3-ASR-1.7B", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.text_layer_count == 28
    assert profile.audio_present is True
    assert (
        adapter.classify_tensor("audio_tower.layers.0.self_attn.q_proj.weight", "model.safetensors")
        is TensorRole.AUDIO
    )
    assert (
        adapter.classify_tensor("model.layers.0.self_attn.q_proj.weight", "model.safetensors")
        is TensorRole.ATTENTION
    )


def test_registry_resolves_qwen3_vl_and_protects_vision_tower() -> None:
    config = {
        "model_type": "qwen3_vl",
        "text_config": {"num_hidden_layers": 36},
        "vision_config": {"depth": 27},
    }
    adapter = adapter_for("Qwen/Qwen3-VL-8B-Instruct", config)
    assert adapter is not None
    assert adapter.adapter_id == "qwen3-vl-v1"
    profile = adapter.profile("Qwen/Qwen3-VL-8B-Instruct", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.text_layer_count == 36
    assert profile.vision_present is True
    assert (
        adapter.classify_tensor("model.visual.blocks.0.attn.qkv.weight", "model.safetensors")
        is TensorRole.VISION
    )


@pytest.mark.parametrize(
    ("reference", "model_type"),
    [
        ("Qwen/Qwen3-ASR-0.6B", "qwen3_asr"),
        ("Qwen/Qwen3-ASR-17B", "qwen3_asr"),
        ("Qwen/Qwen3-VL-4B-Instruct", "qwen3_vl"),
        ("Qwen/Qwen3-VL-18B-Instruct", "qwen3_vl"),
        ("Qwen/Qwen3-VL-8B-Thinking", "qwen3_vl"),
    ],
)
def test_unpromoted_qwen3_multimodal_variants_remain_inventory_only(
    reference: str,
    model_type: str,
) -> None:
    config = {
        "model_type": model_type,
        "num_hidden_layers": 28,
        "text_config": {"num_hidden_layers": 28},
        "thinker_config": {
            "text_config": {"num_hidden_layers": 28},
            "audio_config": {"encoder_layers": 24},
        },
    }
    assert adapter_for(reference, config) is None


def test_qwen35_moe_or_missing_layers_stays_inspect_only() -> None:
    """The convertible tier applies to dense checkpoints only (fail closed)."""
    moe_config = {
        "model_type": "qwen3_5",
        "_name_or_path": "Qwen/Qwen3.5-35B-A3B",
        "text_config": {"num_hidden_layers": 36, "num_experts": 64},
    }
    adapter = adapter_for("Qwen/Qwen3.5-35B-A3B", moe_config)
    assert adapter is not None
    profile = adapter.profile("Qwen/Qwen3.5-35B-A3B", moe_config)
    assert profile.support_tier is SupportTier.INSPECT_ONLY


def test_qwen35_spec_declines_qwen36_references() -> None:
    spec = next(item for item in DENSE_FAMILY_SPECS if item.adapter_id == "qwen35-dense-v1")
    adapter = DenseFamilyAdapter(spec)
    assert not adapter.matches("Qwen/Qwen3.6-27B", _qwen36_config())


@pytest.mark.parametrize(
    ("reference", "model_type", "expected_adapter", "expected_tier"),
    [
        ("google/gemma-4-12b", "gemma4", "gemma4-dense-v1", SupportTier.CONVERTIBLE),
        ("google/gemma-4-12b", "gemma4_unified", "gemma4-dense-v1", SupportTier.CONVERTIBLE),
        ("openbmb/MiniCPM5-8B", "minicpm5", "minicpm5-dense-v1", SupportTier.CONVERTIBLE),
        ("openbmb/MiniCPM5-1B", "llama", "minicpm5-dense-v1", SupportTier.CONVERTIBLE),
        (
            "mistralai/Devstral-Small-2505",
            "mistral",
            "mistral-devstral-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
        (
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "mistral3",
            "mistral3-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
        (
            "mistralai/Ministral-3-8B-Instruct-2512",
            "mistral3",
            "mistral3-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
        (
            "nvidia/Nemotron-3-Embed-1B-BF16",
            "ministral3",
            "mistral3-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
        (
            "nvidia/Nemotron-3-Embed-8B-BF16",
            "ministral3",
            "mistral3-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
        (
            "Qwen/Qwen3-Embedding-0.6B",
            "qwen3",
            "qwen3-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
        (
            "Qwen/Qwen3-8B",
            "qwen3",
            "qwen3-dense-v1",
            SupportTier.CONVERTIBLE,
        ),
    ],
)
def test_registry_resolves_new_dense_families(
    reference: str, model_type: str, expected_adapter: str, expected_tier: SupportTier
) -> None:
    config: dict[str, object] = {"model_type": model_type, "num_hidden_layers": 40}
    if model_type == "mistral3":
        config = {
            "model_type": "mistral3",
            "text_config": {"model_type": "mistral", "num_hidden_layers": 40},
        }
    # Flat Ministral3 embedding exports (Nemotron-3-Embed) have no text_config.
    if model_type == "ministral3":
        config = {
            "model_type": "ministral3",
            "num_hidden_layers": 40,
            "architectures": ["Ministral3Model"],
            "is_causal": False,
            "pooling": "avg",
        }
    adapter = adapter_for(reference, config)
    assert adapter is not None
    assert adapter.adapter_id == expected_adapter
    profile = adapter.profile(reference, config)
    assert profile.support_tier is expected_tier
    assert profile.dense is True
    assert profile.text_layer_count == 40


def test_qwen3_dense_does_not_claim_qwen35_or_qwen36() -> None:
    """model_type=qwen3 adapter must not match 3.5/3.6 product names."""
    config = {"model_type": "qwen3", "num_hidden_layers": 36}
    assert adapter_for("Qwen/Qwen3.5-9B", config) is None
    assert adapter_for("Qwen/Qwen3.6-27B", config) is None


def test_deepseek_v4_flash_is_convertible_moe() -> None:
    # Mirrors public deepseek-ai/DeepSeek-V4-Flash config keys (n_routed_experts,
    # num_nextn_predict_layers) rather than Qwen-style num_experts / mtp_*.
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 43,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "num_nextn_predict_layers": 1,
    }
    adapter = adapter_for("deepseek-ai/DeepSeek-V4-Flash", config)
    assert adapter is not None
    assert adapter.adapter_id == "deepseek-v4-v1"
    profile = adapter.profile("deepseek-ai/DeepSeek-V4-Flash", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.dense is False
    assert profile.product_family == "deepseek-v4"
    assert profile.text_layer_count == 43
    assert profile.mtp_declared is True
    assert any("deepseek_v4" in note or "MoE" in note for note in profile.notes)
    # CSA / fused-expert style tensor names classify.
    assert (
        adapter.classify_tensor(
            "model.layers.0.attn.indexer.weight",
            "model.safetensors",
        )
        is TensorRole.ATTENTION
    )
    assert (
        adapter.classify_tensor(
            "model.layers.0.ffn.switch_mlp.gate_proj.weight",
            "model.safetensors",
        )
        is TensorRole.EXPERT
    )


def test_gpt_oss_is_convertible_moe() -> None:
    config = {
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "num_hidden_layers": 24,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "quantization_config": {"quant_method": "mxfp4"},
    }
    adapter = adapter_for("openai/gpt-oss-20b", config)
    assert adapter is not None
    assert adapter.adapter_id == "gpt-oss-v1"
    profile = adapter.profile("openai/gpt-oss-20b", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.dense is False
    assert profile.product_family == "gpt-oss"
    assert profile.text_layer_count == 24
    assert profile.mtp_declared is False
    assert (
        adapter.classify_tensor(
            "model.layers.0.mlp.experts.gate_proj.weight",
            "model.safetensors",
        )
        is TensorRole.EXPERT
    )
    assert (
        adapter.classify_tensor(
            "model.layers.0.mlp.router.weight",
            "model.safetensors",
        )
        is TensorRole.ROUTER
    )
    assert (
        adapter.classify_tensor(
            "model.layers.0.self_attn.sinks",
            "model.safetensors",
        )
        is TensorRole.ATTENTION
    )


def test_qwen3_next_coder_is_convertible_moe() -> None:
    config = {
        "model_type": "qwen3_next",
        "architectures": ["Qwen3NextForCausalLM"],
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 512,
        "full_attention_interval": 4,
    }
    adapter = adapter_for("Qwen/Qwen3-Coder-Next", config)
    assert adapter is not None
    assert adapter.adapter_id == "qwen3-next-v1"
    profile = adapter.profile("Qwen/Qwen3-Coder-Next", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.dense is False
    assert profile.text_layer_count == 48
    assert any("MoE" in note or "fused" in note for note in profile.notes)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "model.layers.0.mlp.switch_mlp.up_proj.weight",
        "model.layers.0.mlp.switch_mlp.down_proj.weight",
        "model.layers.0.mlp.switch_glu.fc1.weight",
    ],
)
def test_qwen3_next_fused_switch_weights_classify_as_experts(name: str) -> None:
    config = {
        "model_type": "qwen3_next",
        "num_hidden_layers": 48,
        "num_experts": 512,
    }
    adapter = adapter_for("Qwen/Qwen3-Coder-Next", config)
    assert adapter is not None
    assert adapter.classify_tensor(name, "model.safetensors") is TensorRole.EXPERT


def test_dense_family_tier_fails_closed_for_moe_and_missing_layers() -> None:
    spec = DenseFamilySpec(
        adapter_id="test-dense-v1",
        product_family="testfam",
        model_types=("testfam",),
        reference_pattern=r"testfam",
        support_tier=SupportTier.CONVERTIBLE,
    )
    adapter = DenseFamilyAdapter(spec)
    dense_config = {"model_type": "testfam", "num_hidden_layers": 12}
    assert adapter.profile("org/testfam-7b", dense_config).support_tier is SupportTier.CONVERTIBLE
    moe_config = {"model_type": "testfam", "num_hidden_layers": 12, "num_experts": 8}
    assert adapter.profile("org/testfam-7b", moe_config).support_tier is SupportTier.INSPECT_ONLY
    missing_layers = {"model_type": "testfam"}
    assert (
        adapter.profile("org/testfam-7b", missing_layers).support_tier is SupportTier.INSPECT_ONLY
    )
    for invalid_layers in (True, 0, -1):
        malformed = {"model_type": "testfam", "num_hidden_layers": invalid_layers}
        profile = adapter.profile("org/testfam-7b", malformed)
        assert profile.support_tier is SupportTier.INSPECT_ONLY
        assert profile.text_layer_count is None


def test_dense_family_allow_moe_opt_in() -> None:
    spec = DenseFamilySpec(
        adapter_id="test-moe-v1",
        product_family="testmoe",
        model_types=("testmoe",),
        reference_pattern=r"testmoe",
        support_tier=SupportTier.CONVERTIBLE,
        allow_moe=True,
    )
    adapter = DenseFamilyAdapter(spec)
    moe_config = {"model_type": "testmoe", "num_hidden_layers": 12, "num_experts": 8}
    profile = adapter.profile("org/testmoe-x", moe_config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.dense is False


def test_registry_rejects_ambiguous_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    first = DenseFamilyAdapter(
        DenseFamilySpec(
            adapter_id="test-a-v1",
            product_family="testfam",
            model_types=("testfam",),
            reference_pattern=r"testfam",
            support_tier=SupportTier.INSPECT_ONLY,
        )
    )
    second = DenseFamilyAdapter(
        DenseFamilySpec(
            adapter_id="test-b-v1",
            product_family="testfam",
            model_types=("testfam",),
            reference_pattern=r"testfam",
            support_tier=SupportTier.INSPECT_ONLY,
        )
    )
    monkeypatch.setattr(registry, "_ADAPTERS", (first, second))
    config = {"model_type": "testfam", "num_hidden_layers": 12}
    with pytest.raises(ArtifactError, match="ambiguous"):
        adapter_for("org/testfam-7b", config)


def test_shared_classifier_matches_qwen36_conventions() -> None:
    adapter = Qwen36Adapter()
    cases = {
        "model.language_model.layers.4.mlp.up_proj.weight": TensorRole.MLP,
        "model.language_model.layers.4.self_attn.q_proj.weight": TensorRole.ATTENTION,
        "model.language_model.layers.4.input_layernorm.weight": TensorRole.NORM,
        "lm_head.weight": TensorRole.LM_HEAD,
        "model.language_model.embed_tokens.weight": TensorRole.EMBEDDING,
        "model.visual.patch_embed.proj.weight": TensorRole.VISION,
        "mtp.layers.0.fc.weight": TensorRole.MTP_PROJECTION,
    }
    for name, expected in cases.items():
        assert adapter.classify_tensor(name, "model.safetensors") is expected
    assert adapter.classify_tensor("totally.unknown.tensor", "model.safetensors") is None
    assert (
        adapter.classify_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "norm-shard.safetensors",
        )
        is TensorRole.ATTENTION
    )
    assert adapter.classify_tensor("model.layers.0.block.weight", "mtp.safetensors") is (
        TensorRole.MTP_BLOCK
    )
    assert adapter.classify_tensor("encoder.block.weight", "vision.safetensors") is (
        TensorRole.VISION
    )


def test_extra_role_patterns_take_precedence() -> None:
    spec = DenseFamilySpec(
        adapter_id="test-dense-v1",
        product_family="testfam",
        model_types=("testfam",),
        reference_pattern=r"testfam",
        support_tier=SupportTier.INSPECT_ONLY,
        extra_role_patterns=(("special_head", TensorRole.LM_HEAD),),
    )
    adapter = DenseFamilyAdapter(spec)
    assert (
        adapter.classify_tensor("model.special_head.weight", "model.safetensors")
        is TensorRole.LM_HEAD
    )


def test_support_matrix_lists_every_registered_family(tmp_path: Path) -> None:
    from axquant.architectures.registry import support_matrix
    from axquant.cli import main
    from axquant.schema import SupportMatrix
    from axquant.serde import load_model

    matrix = support_matrix()
    tiers = {entry.adapter_id: entry.support_tier for entry in matrix.entries}
    assert tiers == {
        "qwen36-v1": SupportTier.CONVERTIBLE,
        "nemotron3-v1": SupportTier.CONVERTIBLE,
        "qwen35-dense-v1": SupportTier.CONVERTIBLE,
        "qwen3-next-v1": SupportTier.CONVERTIBLE,
        "qwen3-dense-v1": SupportTier.CONVERTIBLE,
        "qwen3-asr-v1": SupportTier.CONVERTIBLE,
        "qwen3-vl-v1": SupportTier.CONVERTIBLE,
        # gemma4_unified converts via prepared gemma4 text-path (source_prep).
        "gemma4-dense-v1": SupportTier.CONVERTIBLE,
        "minicpm5-dense-v1": SupportTier.CONVERTIBLE,
        "mistral-devstral-dense-v1": SupportTier.CONVERTIBLE,
        "mistral3-dense-v1": SupportTier.CONVERTIBLE,
        "deepseek-v4-v1": SupportTier.CONVERTIBLE,
        "gpt-oss-v1": SupportTier.CONVERTIBLE,
    }
    families = [entry.product_family for entry in matrix.entries]
    assert families[0] == "qwen3.6"

    output = tmp_path / "support-matrix.json"
    assert main(["support-matrix", "--output", str(output)]) == 0
    loaded = load_model(output, SupportMatrix)
    assert loaded.entries == matrix.entries


def test_moe_router_and_expert_classification() -> None:
    """Qwen-style MoE names: `mlp.gate` is the router, `experts.*` the experts."""
    adapter = Qwen36Adapter()
    cases = {
        "model.language_model.layers.3.mlp.gate.weight": TensorRole.ROUTER,
        "model.language_model.layers.3.mlp.experts.17.gate_proj.weight": TensorRole.EXPERT,
        "model.language_model.layers.3.mlp.experts.17.down_proj.weight": TensorRole.EXPERT,
        "model.language_model.layers.3.mlp.shared_expert.up_proj.weight": TensorRole.EXPERT,
        "model.language_model.layers.3.mlp.shared_expert_gate.weight": TensorRole.ROUTER,
        "model.language_model.layers.3.mlp.gate_proj.weight": TensorRole.MLP,
    }
    for name, expected in cases.items():
        assert adapter.classify_tensor(name, "model.safetensors") is expected


def test_nemotron3_catalog_moe_is_convertible() -> None:
    from axquant.architectures.nemotron3 import Nemotron3Adapter

    config = {
        "model_type": "nemotron_h",
        "_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "num_hidden_layers": 52,
        "hidden_size": 2688,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "moe_intermediate_size": 1856,
    }
    adapter = Nemotron3Adapter()
    assert adapter.matches("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", config)
    profile = adapter.profile("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.dense is False
    assert profile.text_layer_count == 52
    # Non-Nano catalog / experimental refs stay fail-closed (thin-support policy).
    other_cfg = {**config, "_name_or_path": "nvidia/Nemotron-3-experimental"}
    other = adapter.profile("nvidia/Nemotron-3-experimental", other_cfg)
    assert other.support_tier is SupportTier.INSPECT_ONLY
    super_cfg = {
        **config,
        "_name_or_path": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    }
    super_profile = adapter.profile("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", super_cfg)
    assert super_profile.support_tier is SupportTier.INSPECT_ONLY
    conflicting_identity = adapter.profile(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        config,
    )
    assert conflicting_identity.support_tier is SupportTier.INSPECT_ONLY
    wrong_signature = adapter.profile(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        {**config, "hidden_size": 4096},
    )
    assert wrong_signature.support_tier is SupportTier.INSPECT_ONLY
    for invalid_layers in (True, 0, -1):
        malformed = {**config, "num_hidden_layers": invalid_layers}
        malformed_profile = adapter.profile(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            malformed,
        )
        assert malformed_profile.support_tier is SupportTier.INSPECT_ONLY
        assert malformed_profile.text_layer_count is None
    # Classification for hybrid MoE tensors.
    cases = {
        "backbone.layers.3.mixer.experts.14.up_proj.weight": TensorRole.EXPERT,
        "backbone.layers.3.mixer.experts.14.down_proj.weight": TensorRole.EXPERT,
        "backbone.layers.3.mixer.shared_experts.up_proj.weight": TensorRole.MLP,
        "backbone.layers.3.mixer.gate.weight": TensorRole.ROUTER,
        "backbone.layers.3.mixer.conv1d.weight": TensorRole.ATTENTION,
        "backbone.layers.3.norm.weight": TensorRole.NORM,
        "backbone.embeddings.weight": TensorRole.EMBEDDING,
    }
    for name, expected in cases.items():
        assert adapter.classify_tensor(name, "model.safetensors") is expected


def test_nemotron_expert_fuses_to_switch_mlp_fc() -> None:
    from axquant.module_paths import fused_expert_module, mlx_module_aliases

    path = "backbone.layers.8.mixer.experts.20.up_proj"
    assert fused_expert_module(path) == "backbone.layers.8.mixer.switch_mlp.fc1"
    assert fused_expert_module("backbone.layers.8.mixer.experts.20.down_proj") == (
        "backbone.layers.8.mixer.switch_mlp.fc2"
    )
    aliases = mlx_module_aliases(path)
    assert "backbone.layers.8.mixer.switch_mlp.fc1" in aliases


def test_mistral_devstral_not_confused_with_minicpm() -> None:
    config = {"model_type": "llama", "num_hidden_layers": 16}
    adapter = adapter_for("openbmb/MiniCPM5-1B", config)
    assert adapter is not None
    assert adapter.adapter_id == "minicpm5-dense-v1"
    # Unscoped llama without mistral/devstral/minicpm name does not match a family.
    assert adapter_for("meta-llama/Llama-3.1-8B", config) is None


def test_qwen36_moe_catalog_size_is_supported() -> None:
    config = {
        "model_type": "qwen3_5_moe",
        "_name_or_path": "Qwen/Qwen3.6-35B-A3B",
        "text_config": {
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_experts": 256,
            "num_experts_per_tok": 8,
        },
    }
    profile = Qwen36Adapter().profile("Qwen/Qwen3.6-35B-A3B", config)
    assert profile.support_tier is SupportTier.CONVERTIBLE
    assert profile.dense is False
    assert profile.config_model_type == "qwen3_5_moe"
    # A non-catalog MoE reference stays fail-closed.
    other = Qwen36Adapter().profile("Qwen/Qwen3.6-99B-A9B", config)
    assert other.support_tier is SupportTier.INSPECT_ONLY
    malformed = {
        **config,
        "text_config": {**config["text_config"], "num_experts": 128},
    }
    malformed_profile = Qwen36Adapter().profile("Qwen/Qwen3.6-35B-A3B", malformed)
    assert malformed_profile.support_tier is SupportTier.INSPECT_ONLY
