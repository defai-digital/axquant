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
    ("reference", "model_type", "expected_adapter"),
    [
        ("google/gemma-4-12b", "gemma4", "gemma4-dense-v1"),
        ("openbmb/MiniCPM5-8B", "minicpm5", "minicpm5-dense-v1"),
        ("nvidia/Nemotron-3-22B", "nemotron3", "nemotron3-dense-v1"),
    ],
)
def test_registry_resolves_new_dense_families(
    reference: str, model_type: str, expected_adapter: str
) -> None:
    config = {"model_type": model_type, "num_hidden_layers": 40}
    adapter = adapter_for(reference, config)
    assert adapter is not None
    assert adapter.adapter_id == expected_adapter
    profile = adapter.profile(reference, config)
    assert profile.support_tier is SupportTier.INSPECT_ONLY
    assert profile.dense is True
    assert profile.text_layer_count == 40


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
        "qwen35-dense-v1": SupportTier.CONVERTIBLE,
        "gemma4-dense-v1": SupportTier.INSPECT_ONLY,
        "minicpm5-dense-v1": SupportTier.INSPECT_ONLY,
        "nemotron3-dense-v1": SupportTier.INSPECT_ONLY,
    }
    families = [entry.product_family for entry in matrix.entries]
    assert families[0] == "qwen3.6"

    output = tmp_path / "support-matrix.json"
    assert main(["support-matrix", "--output", str(output)]) == 0
    loaded = load_model(output, SupportMatrix)
    assert loaded.entries == matrix.entries
