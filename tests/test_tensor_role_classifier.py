"""Tensor-role classification hardening.

Locks three properties: role words never match inside an unrelated identifier, the Qwen3.8
Flash-Next family classifies through a closed grammar, and the real Flash-Next checkpoint index
keeps classifying exactly as recorded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from axquant.architectures.dense_family import (
    DENSE_FAMILY_SPECS,
    DenseFamilyAdapter,
    classify_dense_tensor,
)
from axquant.architectures.registry import adapter_for
from axquant.schema import TensorRole

_FIXTURE = Path(__file__).parent / "data" / "qwen4_exp_flash_next_roles.json"
_SHARD = "model-00001-of-00131.safetensors"


def _expand(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        low, _, high = part.partition("-")
        values.extend(range(int(low), int(high or low) + 1))
    return values


def _golden() -> dict[str, TensorRole]:
    fixture = json.loads(_FIXTURE.read_text())
    names: dict[str, TensorRole] = {}
    for entry in fixture["entries"]:
        layers: list[int | None] = list(_expand(entry["layers"])) if "layers" in entry else [None]
        shards: list[int | None] = list(_expand(entry["shards"])) if "shards" in entry else [None]
        for layer in layers:
            for shard in shards:
                name = entry["pattern"]
                if layer is not None:
                    name = name.replace("{L}", str(layer))
                if shard is not None:
                    name = name.replace("{S}", str(shard))
                assert name not in names, name
                names[name] = TensorRole(entry["role"])
    return names


@pytest.fixture(scope="module")
def flash_next() -> DenseFamilyAdapter:
    spec = next(s for s in DENSE_FAMILY_SPECS if s.adapter_id == "qwen4-exp-v1")
    return DenseFamilyAdapter(spec)


def test_flash_next_golden_index_classifies_exactly_as_recorded(
    flash_next: DenseFamilyAdapter,
) -> None:
    golden = _golden()
    source = json.loads(_FIXTURE.read_text())["source"]
    assert len(golden) == source["tensor_count"] == 1658
    actual = {name: flash_next.classify_tensor(name, _SHARD) for name in golden}
    unclassified = sorted(name for name, role in actual.items() if role is None)
    assert unclassified == []
    mismatched = {
        name: (golden[name], actual[name]) for name in golden if actual[name] != golden[name]
    }
    assert mismatched == {}


def test_flash_next_golden_index_resolves_to_the_qwen4_adapter() -> None:
    config = {"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 48}}
    adapter = adapter_for("Qwen/Qwen3.8-Flash-Next", config)
    assert adapter is not None and adapter.adapter_id == "qwen4-exp-v1"


def test_flash_next_protected_roles_are_never_quantizable() -> None:
    protected = {
        TensorRole.NORM,
        TensorRole.ROUTER,
        TensorRole.LM_HEAD,
        TensorRole.EMBEDDING,
        TensorRole.VISION,
        TensorRole.MTP_BLOCK,
        TensorRole.MTP_PROJECTION,
    }
    for name, role in _golden().items():
        if any(marker in name for marker in ("hyper_connection", ".ple.", "visual.", "mtp.")):
            assert role in protected, name
        # The MTP copy is protected as mtp_block because the MTP path is checked first.
        if not name.startswith("mtp.") and name.endswith(
            ("mlp.gate.weight", "shared_expert_gate.weight")
        ):
            assert role is TensorRole.ROUTER, name


def test_only_the_qwen4_family_uses_a_closed_grammar() -> None:
    assert [s.adapter_id for s in DENSE_FAMILY_SPECS if s.role_grammar] == ["qwen4-exp-v1"]


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.seq_proj.weight",
        "model.layers.0.conv_proj.weight",
        "model.layers.0.block_proj.weight",
        "model.layers.0.to_proj.weight",
        "model.layers.3.shortcut_w1_scale",
        "model.layers.3.csa_scale",
        "model.layers.3.tx_mlpx.weight",
        "model.layers.3.gr.mixer.weight",
        "model.layers.3.mystery_block.weight",
    ],
)
def test_role_words_do_not_match_inside_other_identifiers(name: str) -> None:
    assert classify_dense_tensor(name, "model.safetensors") is None


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("model.layers.0.attn.wq_a.weight", TensorRole.ATTENTION),
        ("model.layers.0.attn.wq_b.weight", TensorRole.ATTENTION),
        ("model.layers.0.attn.wkv.weight", TensorRole.ATTENTION),
        ("model.layers.0.attn.qkv_proj.weight", TensorRole.ATTENTION),
        ("model.layers.0.attn.kv_proj.weight", TensorRole.ATTENTION),
        ("model.layers.0.attn.q_a_proj.weight", TensorRole.ATTENTION),
        ("model.layers.0.mlp1.weight", TensorRole.MLP),
        ("model.layers.0.w1.weight", TensorRole.MLP),
        ("model.layers.0.w3.weight", TensorRole.MLP),
        ("backbone.layers.3.mixer.in_proj.weight", TensorRole.ATTENTION),
        ("backbone.layers.3.mixer.conv1d.weight", TensorRole.ATTENTION),
        # Protective stages are unchanged: a norm-like name stays at the norm floor.
        ("model.layers.0.state_norm_proj.weight", TensorRole.NORM),
    ],
)
def test_established_role_spellings_are_preserved(name: str, role: TensorRole) -> None:
    assert classify_dense_tensor(name, "model.safetensors") is role


@pytest.mark.parametrize(
    "name",
    [
        # A known container with an unlisted member fails closed.
        "model.language_model.layers.3.mlp.moe_gate.weight",
        "model.language_model.layers.3.linear_attn.new_gate.weight",
        "model.language_model.layers.3.self_attn.rope_proj.weight",
        # An unknown container fails closed even when it embeds a role word.
        "model.language_model.layers.3.mystery_mlp_block.weight",
        "model.language_model.layers.3.ngram.conv1d.weight",
        "model.language_model.layers.3.residual_mix.input_mix_weight_up.weight",
        "model.language_model.layers.3.mlp_residual_mix.block_inject_weight.weight",
    ],
)
def test_flash_next_unknown_components_fail_closed(
    flash_next: DenseFamilyAdapter, name: str
) -> None:
    assert flash_next.classify_tensor(name, _SHARD) is None


_RENAMES = {
    "drop the model.language_model prefix": lambda n: n.replace("model.language_model.", "model."),
    "self_attn to attn": lambda n: n.replace("self_attn", "attn"),
    "linear_attn to gdn": lambda n: n.replace("linear_attn", "gdn"),
    "mlp to moe": lambda n: re.sub(r"\.mlp\.", ".moe.", n),
    "hyper_connection to residual_mix": lambda n: n.replace("hyper_connection", "residual_mix"),
    "ple to ngram": lambda n: n.replace(".ple.", ".ngram."),
    "layers to blocks": lambda n: n.replace(".layers.", ".blocks."),
}
# A rename may only keep a tensor's role, fail closed, or raise its protection. These are the
# permitted protection upgrades: a PLE norm vector renamed away from the .ple. extra is then
# recognized as the norm it is named after.
_PROTECTIVE_DRIFT = {(TensorRole.EMBEDDING, TensorRole.NORM)}


@pytest.mark.parametrize("label", list(_RENAMES))
def test_flash_next_renames_never_yield_a_different_quantizable_role(
    flash_next: DenseFamilyAdapter, label: str
) -> None:
    rename = _RENAMES[label]
    drift: dict[str, tuple[TensorRole, TensorRole | None]] = {}
    for name, role in _golden().items():
        if name.startswith("mtp.") or ".visual." in name:
            continue  # protected sidecar paths are covered by the golden test
        renamed = flash_next.classify_tensor(rename(name), _SHARD)
        if renamed is None or renamed is role:
            continue
        if (role, renamed) not in _PROTECTIVE_DRIFT:
            drift[name] = (role, renamed)
    assert drift == {}


def test_flash_next_rename_matrix_exercises_both_outcomes(
    flash_next: DenseFamilyAdapter,
) -> None:
    unchanged = failed_closed = 0
    for name, role in _golden().items():
        if name.startswith("mtp.") or ".visual." in name:
            continue
        renamed = flash_next.classify_tensor(
            _RENAMES["hyper_connection to residual_mix"](name), _SHARD
        )
        if renamed is role:
            unchanged += 1
        elif renamed is None:
            failed_closed += 1
    assert unchanged > 0 and failed_closed > 0


def test_shared_classifier_matches_the_recorded_flash_next_roles_without_the_grammar() -> None:
    """The grammar must not be what makes the real checkpoint classify."""
    spec = next(s for s in DENSE_FAMILY_SPECS if s.adapter_id == "qwen4-exp-v1")
    names: Any = _golden()
    differing = {
        name: (role, classify_dense_tensor(name, _SHARD, spec.extra_role_patterns))
        for name, role in names.items()
        if classify_dense_tensor(name, _SHARD, spec.extra_role_patterns) != role
    }
    assert differing == {}
