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
