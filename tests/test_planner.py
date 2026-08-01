from __future__ import annotations

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.planner import plan_quantization, storage_bpw
from axquant.schema import (
    Inventory,
    ModelIdentity,
    MtpPolicy,
    PlanRequest,
    ProfileName,
    TensorRole,
    TensorSpec,
)


def _tensor(
    name: str,
    parameters: int,
    role: TensorRole,
    *,
    file: str = "model.safetensors",
    quantizable: bool = True,
) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=name.removesuffix(".weight"),
        shape=(parameters, 1),
        dtype="BF16",
        parameters=parameters,
        role=role,
        quantizable=quantizable,
        file=file,
        current_precision="bf16",
    )


def _inventory() -> Inventory:
    tensors = [
        _tensor("model.layers.0.mlp.down_proj.weight", 10_000, TensorRole.MLP),
        _tensor("model.layers.0.self_attn.q_proj.weight", 1_000, TensorRole.ATTENTION),
        _tensor("lm_head.weight", 100, TensorRole.LM_HEAD),
        _tensor(
            "mtp.projection.weight",
            100,
            TensorRole.MTP_PROJECTION,
            file="mtp.safetensors",
        ),
    ]
    return Inventory(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors),
        mtp_present=True,
        quantized_source=False,
        source_files=["model.safetensors", "mtp.safetensors"],
        config_sha256="a" * 64,
    )


def _request(**overrides: object) -> PlanRequest:
    values: dict[str, object] = {
        "profile": ProfileName.AGENT_CODING,
        "target_bpw": 6.5,
        "allow_unmeasured": True,
        "mtp": MtpPolicy(mode="protected"),
    }
    values.update(overrides)
    return PlanRequest.model_validate(values)


def test_storage_cost_includes_affine_metadata() -> None:
    assert storage_bpw(4, 64) == 4.5
    assert storage_bpw(6, 64) == 6.5
    assert storage_bpw(16, None) == 16.0


def test_planner_respects_budget_and_external_mtp_protection() -> None:
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    plan = plan_quantization(report, _request())
    assert plan.effective_bpw <= 6.5
    assert plan.constraints.minimum_quality_retention == 0.98
    assert plan.global_validation_required is True
    mtp = next(allocation for allocation in plan.assignments if allocation.role.is_mtp)
    head = next(
        allocation for allocation in plan.assignments if allocation.role == TensorRole.LM_HEAD
    )
    assert mtp.bits == 16
    assert "byte-for-byte" in mtp.reason
    assert head.bits == 16
    assert plan.mtp_distribution["bf16"].fraction == 1.0


def test_sidecar_policy_preserves_logical_mtp_from_integrated_source_shards() -> None:
    inventory = _inventory()
    mtp = next(tensor for tensor in inventory.tensors if tensor.role.is_mtp)
    mtp.file = "model-00013-of-00015.safetensors"
    inventory.source_files = ["model-00013-of-00015.safetensors"]
    plan = plan_quantization(
        architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING),
        _request(),
    )
    allocation = next(item for item in plan.assignments if item.role.is_mtp)
    assert allocation.bits == 16
    assert "byte-for-byte" in allocation.reason


def test_unmeasured_plan_requires_explicit_override() -> None:
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    with pytest.raises(PlanningError, match="not release quality"):
        plan_quantization(report, _request(allow_unmeasured=False))


def test_top_n_request_accepted() -> None:
    """Top-N candidate generation is supported since v0.4."""
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    # Should not raise - top-N is now accepted
    plan = plan_quantization(report, _request(candidate_count=8))
    assert plan is not None
    assert plan.effective_bpw > 0


def test_infeasible_protection_budget_fails() -> None:
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    with pytest.raises(PlanningError, match="infeasible"):
        plan_quantization(report, _request(target_bpw=4.0))


def test_kv_cache_allocator_covers_layers_with_boundary_floor() -> None:
    from axquant.planner import allocate_kv_cache

    plan = allocate_kv_cache(20, default_bits=4, group_size=64)
    assert plan.allocation_basis == "architecture-prior"
    assert [layer.layer_index for layer in plan.layers] == list(range(20))
    boundary = 2  # ceil(20 / 10)
    for layer in plan.layers:
        expected = 8 if layer.layer_index < boundary or layer.layer_index >= 20 - boundary else 4
        assert layer.bits == expected
        assert layer.group_size == 64
        assert layer.reason
    assert all(layer.bits >= plan.min_bits for layer in plan.layers)


def test_kv_cache_allocator_rejects_invalid_inputs() -> None:
    from axquant.planner import allocate_kv_cache

    with pytest.raises(PlanningError, match="positive text layer count"):
        allocate_kv_cache(0)
    with pytest.raises(PlanningError, match="policy floor"):
        allocate_kv_cache(8, default_bits=4, min_bits=6)


def test_kv_cache_plan_schema_rejects_gaps_and_floor_violations() -> None:
    from axquant.schema import KvCachePlan, KvLayerAllocation

    def layer(index: int, bits: int = 8) -> KvLayerAllocation:
        return KvLayerAllocation(layer_index=index, bits=bits, group_size=64, reason="test")

    with pytest.raises(ValueError, match="without gaps"):
        KvCachePlan(
            allocation_basis="architecture-prior",
            default_bits=4,
            layers=[layer(0), layer(2)],
        )
    with pytest.raises(ValueError, match="policy floor"):
        KvCachePlan(
            allocation_basis="architecture-prior",
            min_bits=6,
            default_bits=6,
            layers=[layer(0, bits=4)],
        )


def test_plan_refreshes_stale_tier_from_current_registry_policy() -> None:
    from axquant.schema import ArchitectureSupportLevel, SupportTier

    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    stale_profile = report.architecture_profile.model_copy(
        update={
            "adapter_id": "qwen36-v1",
            "support_level": ArchitectureSupportLevel.SUPPORTED,
            "support_tier": SupportTier.INSPECT_ONLY,
        }
    )
    stale_report = report.model_copy(update={"architecture_profile": stale_profile})
    plan = plan_quantization(stale_report, _request())
    assert plan.architecture_profile.support_tier is SupportTier.CONVERTIBLE

    unknown_profile = stale_profile.model_copy(update={"adapter_id": "unknown-adapter"})
    unknown_report = report.model_copy(update={"architecture_profile": unknown_profile})
    unknown_plan = plan_quantization(unknown_report, _request())
    assert unknown_plan.architecture_profile.support_tier is SupportTier.INSPECT_ONLY


def test_axq026_lm_head_floor_default_stays_bf16_and_fails_closed() -> None:
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    # 4.70 BPW is below the BF16-LM-head policy minimum for this inventory
    # (4.7054), so the default floor must fail closed rather than downgrade.
    with pytest.raises(PlanningError, match="policy minimum"):
        plan_quantization(report, _request(target_bpw=4.7))


def test_axq026_lowered_lm_head_floor_is_explicit_and_recorded() -> None:
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
    )
    plan = plan_quantization(report, _request(target_bpw=4.7, lm_head_min_bits=8))
    head = next(
        allocation for allocation in plan.assignments if allocation.role == TensorRole.LM_HEAD
    )
    assert head.bits == 8
    assert "AXQ-026" in head.reason
    assert "measured quality evidence" in head.reason
    assert plan.constraints.lm_head_min_bits == 8
    # The lowered floor never drops below 8-bit even under budget pressure.
    assert all(
        allocation.bits >= 8
        for allocation in plan.assignments
        if allocation.role == TensorRole.LM_HEAD
    )
