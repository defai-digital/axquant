from __future__ import annotations

import pytest
from pydantic import ValidationError

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.planner import plan_quantization, storage_bpw
from axquant.schema import (
    Inventory,
    ModelIdentity,
    MtpPolicy,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
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
    assert storage_bpw(4, 32) == 5.0
    assert storage_bpw(4, 128) == 4.25


def test_multi_group_prior_emits_group_grid_and_strategy_metadata() -> None:
    """AXQ-028: multi-group priors + strategy labels on allocations."""
    from axquant.schema import OutlierStrategy, ScaleStrategy

    inventory = _inventory()
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
        candidate_group_sizes=(32, 64, 128),
    )
    mlp = next(entry for entry in report.entries if entry.tensor.role == TensorRole.MLP)
    groups = {candidate.group_size for candidate in mlp.candidates if candidate.bits == 4}
    assert groups == {32, 64, 128}
    kl_by_gs = {
        candidate.group_size: candidate.metrics.output_kl
        for candidate in mlp.candidates
        if candidate.bits == 4
    }
    assert kl_by_gs[32] < kl_by_gs[64] < kl_by_gs[128]

    # Near the policy floor: cheapest storage options (largest group at 4-bit for trunk).
    plan = plan_quantization(
        report,
        _request(target_bpw=4.5, candidate_group_sizes=(32, 64, 128)),
    )
    assert plan.candidate_group_sizes == (32, 64, 128)
    mlp_alloc = next(item for item in plan.assignments if item.role == TensorRole.MLP)
    assert mlp_alloc.bits == 4
    assert mlp_alloc.group_size == 128  # coarsest group = lowest storage BPW
    assert mlp_alloc.scale_strategy == ScaleStrategy.GROUP_AFFINE
    assert mlp_alloc.outlier_strategy == OutlierStrategy.NONE
    assert "storage_bpw" in mlp_alloc.strategy_metadata
    head = next(item for item in plan.assignments if item.role == TensorRole.LM_HEAD)
    assert head.scale_strategy == ScaleStrategy.NONE
    assert head.bits == 16


def test_extra_budget_upgrades_toward_finer_group() -> None:
    """Extra BPW budget upgrades trunk tensors off the coarsest 4-bit/gs128 option."""
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.AGENT_CODING,
        candidate_group_sizes=(32, 64, 128),
    )
    floor = plan_quantization(
        report,
        _request(target_bpw=4.5, candidate_group_sizes=(32, 64, 128)),
    )
    richer = plan_quantization(
        report,
        _request(target_bpw=6.0, candidate_group_sizes=(32, 64, 128)),
    )
    assert richer.target_class == "6bit"
    floor_mlp = next(item for item in floor.assignments if item.role == TensorRole.MLP)
    rich_mlp = next(item for item in richer.assignments if item.role == TensorRole.MLP)
    assert floor_mlp.group_size == 128
    assert rich_mlp.predicted_loss <= floor_mlp.predicted_loss
    assert (rich_mlp.bits, rich_mlp.group_size) != (floor_mlp.bits, floor_mlp.group_size) or (
        rich_mlp.bits > floor_mlp.bits
    )


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


def test_mtp_policy_rejects_a_floor_below_the_protected_minimum() -> None:
    # AXQ-007: protected floors may be raised by hardware capability but must
    # never be silently lowered below the documented 8-bit MTP minimum.
    with pytest.raises(ValidationError):
        MtpPolicy(min_bits=2)


def test_plan_rejects_a_loaded_allocation_below_its_protection_floor() -> None:
    # `plan_quantization` always emits a compliant plan, so this exercises
    # the defense-in-depth backstop on `QuantizationPlan` itself: a plan
    # loaded from disk (e.g. hand-edited JSON fed to `axquant convert`) with
    # a protected role pushed below its floor must fail to validate, not
    # silently reach conversion.
    report = architecture_prior_report(_inventory(), profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(report, _request(target_bpw=6.5))
    lm_head_index = next(
        index
        for index, allocation in enumerate(plan.assignments)
        if allocation.role == TensorRole.LM_HEAD
    )
    payload = plan.model_dump(mode="json")
    payload["assignments"][lm_head_index]["bits"] = 4

    with pytest.raises(ValidationError, match="violates protection floors"):
        QuantizationPlan.model_validate(payload)
