"""Registered architecture and family policy consistency."""

from axquant.architectures.registry import support_matrix
from axquant.schema import SupportTier
from axquant.support_policy import InvestmentPosture, policy_for_adapter


def test_registered_adapters_have_consistent_policies_except_existing_gaps() -> None:
    # These pre-existing adapters use registry defaults; new adapters need policies.
    existing_gaps = {"qwen3-asr-v1", "qwen3-vl-v1"}
    missing = set()
    for entry in support_matrix().entries:
        policy = policy_for_adapter(entry.adapter_id)
        if policy is None:
            missing.add(entry.adapter_id)
            continue
        assert policy.product_family == entry.product_family
        assert policy.declared_tier is entry.support_tier
    assert missing == existing_gaps


def test_hy_v3_support_matrix_is_development_only() -> None:
    entry = next(entry for entry in support_matrix().entries if entry.adapter_id == "hy-v3-v1")
    assert entry.product_family == "hy-v3"
    assert entry.support_tier is SupportTier.CONVERTIBLE
    assert entry.investment_posture == InvestmentPosture.SECONDARY.value
    assert entry.priority == 40
    assert entry.cert_track is False
    assert any("no translation-quality claim" in note for note in entry.notes)
    assert any("expert_bias stays with the router" in note for note in entry.notes)
