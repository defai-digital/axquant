from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

import axquant.planner as planner_module
from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.planner import plan_quantization, storage_bpw
from axquant.schema import (
    CandidateMeasurement,
    HardwareProfile,
    Inventory,
    MetricVector,
    ModelIdentity,
    MtpPolicy,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    QuantMethod,
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


@pytest.mark.parametrize("target_bpw", [4.5, 5.0, 6.0, 8.0])
def test_priority_queue_budget_upgrades_match_legacy_full_scan(target_bpw: float) -> None:
    tensors = [
        _tensor(
            f"model.layers.{index}.mlp.down_proj.weight",
            1_000 + index * 137,
            TensorRole.MLP,
        )
        for index in range(12)
    ]
    inventory = Inventory(
        model=ModelIdentity(model_id="org/upgrade-order", revision="abc"),
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors),
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
        candidate_group_sizes=(32, 64, 128),
    )
    request = _request(
        target_bpw=target_bpw,
        candidate_group_sizes=(32, 64, 128),
    )
    weights = planner_module.objective_for(request.profile).normalized()
    choices = []
    for entry in report.entries:
        options, reason = planner_module._options_for(
            entry,
            request,
            weights,
            evidence_kind=report.evidence_kind,
        )
        choices.append(
            planner_module._Choice(
                entry=entry,
                options=options,
                policy_reason=reason,
            )
        )
    legacy_choices = copy.deepcopy(choices)
    queued_choices = copy.deepcopy(choices)
    total_parameters = sum(choice.entry.tensor.parameters for choice in choices)
    target_storage_bits = target_bpw * total_parameters
    initial_storage_bits = sum(
        choice.selected.storage_bpw * choice.entry.tensor.parameters for choice in choices
    )

    legacy_storage_bits = initial_storage_bits
    while True:
        best: tuple[float, int, float] | None = None
        for choice_index, choice in enumerate(legacy_choices):
            candidate = planner_module._next_upgrade_candidate(choice_index, choice)
            if candidate is None:
                continue
            if legacy_storage_bits + candidate[2] > target_storage_bits + 1e-6:
                continue
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        legacy_choice = legacy_choices[best[1]]
        legacy_choice.index += 1
        legacy_choice.upgraded = True
        legacy_storage_bits += best[2]

    queued_storage_bits = planner_module._apply_budget_upgrades(
        queued_choices,
        running_storage_bits=initial_storage_bits,
        target_storage_bits=target_storage_bits,
    )

    assert queued_storage_bits == pytest.approx(legacy_storage_bits)
    assert [(choice.index, choice.upgraded) for choice in queued_choices] == [
        (choice.index, choice.upgraded) for choice in legacy_choices
    ]


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


def test_planner_harmonizes_tied_weights_at_one_executable_signature() -> None:
    trunk = _tensor("model.layers.0.mlp.down_proj.weight", 10_000, TensorRole.MLP)
    embedding = _tensor("model.embed_tokens.weight", 100, TensorRole.EMBEDDING)
    head = _tensor("lm_head.weight", 100, TensorRole.LM_HEAD)
    embedding.tied_to = head.name
    head.tied_to = embedding.name
    inventory = Inventory(
        model=ModelIdentity(model_id="org/tied-model", revision="abc"),
        tensors=[trunk, embedding, head],
        total_parameters=10_200,
        quantizable_parameters=10_200,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
        tied_weight_groups=[[embedding.name, head.name]],
    )
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(report, _request(target_bpw=5.0))
    tied = {
        allocation.tensor: (allocation.bits, allocation.method, allocation.group_size)
        for allocation in plan.assignments
        if allocation.tensor in {embedding.name, head.name}
    }
    assert len(set(tied.values())) == 1
    assert next(iter(tied.values())) == (16, QuantMethod.BF16, None)
    assert all(
        "tied-weight group" in allocation.reason
        for allocation in plan.assignments
        if allocation.tensor in tied
    )


def test_planner_rejects_tied_weight_precision_outside_budget() -> None:
    trunk = _tensor("model.layers.0.mlp.down_proj.weight", 10_000, TensorRole.MLP)
    embedding = _tensor("model.embed_tokens.weight", 100, TensorRole.EMBEDDING)
    head = _tensor("lm_head.weight", 100, TensorRole.LM_HEAD)
    embedding.tied_to = head.name
    head.tied_to = embedding.name
    inventory = Inventory(
        model=ModelIdentity(model_id="org/tied-model", revision="abc"),
        tensors=[trunk, embedding, head],
        total_parameters=10_200,
        quantizable_parameters=10_200,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    with pytest.raises(PlanningError, match=r"tied-weight group.*within the budget"):
        plan_quantization(report, _request(target_bpw=4.7))


def test_planner_rejects_dangling_tied_weight_reference() -> None:
    tensor = _tensor("model.embed_tokens.weight", 100, TensorRole.EMBEDDING)
    tensor.tied_to = "missing.weight"
    inventory = Inventory(
        model=ModelIdentity(model_id="org/tied-model", revision="abc"),
        tensors=[tensor],
        total_parameters=100,
        quantizable_parameters=100,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    with pytest.raises(PlanningError, match="tied to missing tensor"):
        plan_quantization(report, _request(target_bpw=16.0))


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
    with pytest.raises(PlanningError, match=r"bit widths.*5"):
        allocate_kv_cache(8, default_bits=5)
    with pytest.raises(PlanningError, match=r"bit widths.*5"):
        allocate_kv_cache(8, min_bits=5)
    with pytest.raises(PlanningError, match=r"group sizes.*7"):
        allocate_kv_cache(8, group_size=7)


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


@pytest.mark.parametrize(
    "payload",
    [
        {
            "allocation_basis": "architecture-prior",
            "min_bits": 4,
            "default_bits": 5,
            "default_group_size": 64,
            "layers": [{"layer_index": 0, "bits": 4, "group_size": 64, "reason": "test"}],
        },
        {
            "allocation_basis": "architecture-prior",
            "min_bits": 5,
            "default_bits": 6,
            "default_group_size": 64,
            "layers": [{"layer_index": 0, "bits": 6, "group_size": 64, "reason": "test"}],
        },
        {
            "allocation_basis": "architecture-prior",
            "min_bits": 4,
            "default_bits": 4,
            "default_group_size": 7,
            "layers": [{"layer_index": 0, "bits": 4, "group_size": 64, "reason": "test"}],
        },
        {
            "allocation_basis": "architecture-prior",
            "min_bits": 4,
            "default_bits": 4,
            "default_group_size": 64,
            "layers": [{"layer_index": 0, "bits": 5, "group_size": 64, "reason": "test"}],
        },
        {
            "allocation_basis": "architecture-prior",
            "min_bits": 4,
            "default_bits": 4,
            "default_group_size": 64,
            "layers": [{"layer_index": 0, "bits": 4, "group_size": 7, "reason": "test"}],
        },
    ],
)
def test_kv_cache_plan_schema_rejects_non_executable_grid(
    payload: dict[str, object],
) -> None:
    from axquant.schema import KvCachePlan

    with pytest.raises(ValidationError, match="AX Engine KV cache does not support"):
        KvCachePlan.model_validate(payload)


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
    with pytest.raises(ValidationError, match="executable grid"):
        MtpPolicy(candidate_bits=(5, 8))


def test_loaded_plan_rejects_inconsistent_summary_and_identity_fields() -> None:
    report = architecture_prior_report(_inventory(), profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(report, _request(target_bpw=6.5))

    duplicate = plan.model_dump(mode="json")
    duplicate["assignments"].append(dict(duplicate["assignments"][0]))
    with pytest.raises(ValidationError, match="tensor names must be unique"):
        QuantizationPlan.model_validate(duplicate)

    wrong_budget = plan.model_dump(mode="json")
    wrong_budget["effective_bpw"] = plan.target_bpw + 0.1
    with pytest.raises(ValidationError, match="exceeds its target"):
        QuantizationPlan.model_validate(wrong_budget)

    wrong_constraint = plan.model_dump(mode="json")
    wrong_constraint["constraints"]["effective_bpw_limit"] = plan.target_bpw + 0.1
    with pytest.raises(ValidationError, match="does not match"):
        QuantizationPlan.model_validate(wrong_constraint)

    disabled_validation = plan.model_dump(mode="json")
    disabled_validation["global_validation_required"] = False
    with pytest.raises(ValidationError, match="Input should be True"):
        QuantizationPlan.model_validate(disabled_validation)

    noncanonical_grid = plan.model_dump(mode="json")
    noncanonical_grid["candidate_bits"] = [6, 4, 6, 8, 16]
    with pytest.raises(ValidationError, match="sorted and unique"):
        QuantizationPlan.model_validate(noncanonical_grid)

    stale_distribution = plan.model_dump(mode="json")
    stale_distribution["assignments"][0]["parameters"] += 1
    with pytest.raises(ValidationError, match=r"distribution.*does not match"):
        QuantizationPlan.model_validate(stale_distribution)


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


@pytest.mark.parametrize(
    "module_path",
    [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ],
)
def test_planner_only_selects_executable_affine_method_for_fused_experts(
    module_path: str,
) -> None:
    tensor = _tensor(f"{module_path}.weight", 10_000, TensorRole.EXPERT)
    inventory = Inventory(
        model=ModelIdentity(model_id="org/moe-model", revision="abc"),
        tensors=[tensor],
        total_parameters=tensor.parameters,
        quantizable_parameters=tensor.parameters,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    entry = report.entries[0]
    entry.candidates.append(
        CandidateMeasurement(
            bits=4,
            method=QuantMethod.DWQ,
            group_size=64,
            metrics=MetricVector(),
            note="lower-loss refinement candidate",
        )
    )

    plan = plan_quantization(
        report,
        _request(
            target_bpw=4.5,
            candidate_methods=(QuantMethod.AFFINE, QuantMethod.DWQ),
        ),
    )

    assert plan.assignments[0].bits == 4
    assert plan.assignments[0].method is QuantMethod.AFFINE


def test_planner_rejects_fused_experts_without_a_common_budgeted_signature() -> None:
    tensors = [
        _tensor(
            f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
            10_000,
            TensorRole.EXPERT,
        )
        for index in range(2)
    ]
    inventory = Inventory(
        model=ModelIdentity(model_id="org/moe-model", revision="abc"),
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors),
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
        candidate_group_sizes=(64, 128),
    )
    report.entries[0].candidates = [
        candidate
        for candidate in report.entries[0].candidates
        if candidate.bits == 16 or (candidate.bits == 4 and candidate.group_size == 64)
    ]
    report.entries[1].candidates = [
        candidate
        for candidate in report.entries[1].candidates
        if candidate.bits == 16 or (candidate.bits == 4 and candidate.group_size == 128)
    ]

    with pytest.raises(PlanningError, match="no common precision within the budget"):
        plan_quantization(
            report,
            _request(
                target_bpw=4.5,
                candidate_group_sizes=(64, 128),
            ),
        )


def test_hardware_profile_rejects_non_executable_precision_grid() -> None:
    with pytest.raises(ValidationError, match=r"does not support bits.*5"):
        HardwareProfile(supported_bits=(4, 5, 16))
    with pytest.raises(ValidationError, match=r"does not support group sizes.*7"):
        HardwareProfile(supported_group_sizes=(7, 64))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bits", 5),
        ("group_size", 7),
    ],
)
def test_plan_rejects_loaded_allocation_outside_declared_hardware_grid(
    field: str,
    value: int,
) -> None:
    report = architecture_prior_report(_inventory(), profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(report, _request(target_bpw=6.5))
    assignment_index = next(
        index
        for index, allocation in enumerate(plan.assignments)
        if allocation.role == TensorRole.MLP
    )
    payload = plan.model_dump(mode="json")
    payload["assignments"][assignment_index][field] = value

    with pytest.raises(ValidationError, match="non-executable allocations"):
        QuantizationPlan.model_validate(payload)
