"""Family support investment policy (thin Nemotron, primary Qwen, etc.)."""

from __future__ import annotations

from pathlib import Path

from axquant.architectures.nemotron3 import Nemotron3Adapter
from axquant.architectures.registry import support_matrix
from axquant.cli import main
from axquant.schema import SupportMatrix, SupportTier
from axquant.serde import load_model
from axquant.support_policy import (
    InvestmentPosture,
    ordered_policies,
    policy_for_adapter,
    support_policy_markdown,
)


def test_qwen_is_primary_cert_track() -> None:
    policy = policy_for_adapter("qwen36-v1")
    assert policy is not None
    assert policy.investment_posture is InvestmentPosture.PRIMARY
    assert policy.cert_track is True
    assert policy.priority == 1


def test_nemotron_is_thin_and_nano_only_convertible() -> None:
    policy = policy_for_adapter("nemotron3-v1")
    assert policy is not None
    assert policy.investment_posture is InvestmentPosture.THIN
    assert policy.cert_track is False
    adapter = Nemotron3Adapter()
    nano = {
        "model_type": "nemotron_h",
        "_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "num_hidden_layers": 52,
        "hidden_size": 2688,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "moe_intermediate_size": 1856,
    }
    assert (
        adapter.profile("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", nano).support_tier
        is SupportTier.CONVERTIBLE
    )
    super_cfg = {
        **nano,
        "_name_or_path": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    }
    super_profile = adapter.profile("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", super_cfg)
    assert super_profile.support_tier is SupportTier.INSPECT_ONLY
    assert any("Super/Ultra" in note for note in super_profile.notes)
    ultra_cfg = {
        **nano,
        "_name_or_path": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    }
    assert (
        adapter.profile("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", ultra_cfg).support_tier
        is SupportTier.INSPECT_ONLY
    )


def test_support_matrix_includes_posture_and_sorted_priority() -> None:
    matrix = support_matrix()
    assert matrix.policy_version == "axquant.support-policy.v1"
    priorities = [entry.priority for entry in matrix.entries]
    assert priorities == sorted(priorities)
    by_id = {entry.adapter_id: entry for entry in matrix.entries}
    assert by_id["qwen36-v1"].investment_posture == "primary"
    assert by_id["qwen36-v1"].cert_track is True
    assert by_id["nemotron3-v1"].investment_posture == "thin"
    assert by_id["mistral-devstral-dense-v1"].investment_posture == "secondary"


def test_convertible_adapters_match_conversion_host_smoke_coverage() -> None:
    """Every declared-convertible adapter is in the conversion-host smoke set.

    Keeps the remote family matrix and registry from drifting. Qwen3-ASR and
    Qwen3-VL use their architecture runtimes on df-macbookpro-m5; the text families also
    remain in the macstudio-m2u coverage set.
    """
    matrix = support_matrix()
    convertible = {
        entry.adapter_id
        for entry in matrix.entries
        if entry.support_tier is SupportTier.CONVERTIBLE
    }
    # One representative smoke per adapter (Qwen 3.6 covers dense + MoE path).
    expected = {
        "qwen36-v1",
        "qwen35-dense-v1",
        "qwen3-next-v1",
        "qwen3-dense-v1",
        "qwen3-asr-v1",
        "qwen3-vl-v1",
        "minicpm5-dense-v1",
        "gemma4-dense-v1",
        "mistral-devstral-dense-v1",
        "mistral3-dense-v1",
        "nemotron3-v1",
        "deepseek-v4-v1",
    }
    assert convertible == expected


def test_support_policy_cli(tmp_path: Path) -> None:
    out = tmp_path / "policy.md"
    assert main(["support-policy", "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "thin" in text
    assert "Nano" in text
    assert support_policy_markdown() == text
    assert ordered_policies()[0].adapter_id == "qwen36-v1"


def test_support_matrix_cli_json(tmp_path: Path) -> None:
    out = tmp_path / "matrix.json"
    assert main(["support-matrix", "--output", str(out)]) == 0
    loaded = load_model(out, SupportMatrix)
    assert any(entry.adapter_id == "nemotron3-v1" for entry in loaded.entries)
    nemo = next(entry for entry in loaded.entries if entry.adapter_id == "nemotron3-v1")
    assert nemo.investment_posture == "thin"
