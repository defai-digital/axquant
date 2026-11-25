"""QP3: experimental 2/3-bit labels; AXQ-070 retires 2/3-bit planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.experimental_bits import (
    EXPERIMENTAL_WARNING,
    annotate_experimental_low_bit_plan,
    is_experimental_low_bit,
    plan_uses_experimental_low_bits,
)
from axquant.manual import manual_quantization_plan
from axquant.planner import plan_quantization
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    HardwareProfile,
    Inventory,
    ManualPlanRecipe,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QuantMethod,
    TensorRole,
    TensorSpec,
)


def _inventory() -> Inventory:
    tensors = [
        TensorSpec(
            name="model.layers.0.mlp.down_proj.weight",
            module_path="model.layers.0.mlp.down_proj",
            shape=(128, 128),
            dtype="BF16",
            parameters=16384,
            role=TensorRole.MLP,
            quantizable=True,
            file="model.safetensors",
            current_precision="bf16",
        ),
        TensorSpec(
            name="model.layers.0.self_attn.q_proj.weight",
            module_path="model.layers.0.self_attn.q_proj",
            shape=(64, 64),
            dtype="BF16",
            parameters=4096,
            role=TensorRole.ATTENTION,
            quantizable=True,
            file="model.safetensors",
            current_precision="bf16",
        ),
        TensorSpec(
            name="model.norm.weight",
            module_path="model.norm",
            shape=(64,),
            dtype="BF16",
            parameters=64,
            role=TensorRole.NORM,
            quantizable=False,
            file="model.safetensors",
            current_precision="bf16",
        ),
        TensorSpec(
            name="lm_head.weight",
            module_path="lm_head",
            shape=(16, 16),
            dtype="BF16",
            parameters=256,
            role=TensorRole.LM_HEAD,
            quantizable=True,
            file="model.safetensors",
            current_precision="bf16",
        ),
    ]
    return Inventory(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        tensors=tensors,
        total_parameters=sum(t.parameters for t in tensors),
        quantizable_parameters=sum(t.parameters for t in tensors if t.quantizable),
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
        architecture_profile=ArchitectureProfile(
            support_level=ArchitectureSupportLevel.SUPPORTED,
            product_family="qwen3.6",
            optimization_scope=OptimizationScope.TEXT_PATH,
            adapter_id="qwen36-v1",
            text_layer_count=1,
        ),
    )


def test_is_experimental_low_bit() -> None:
    assert is_experimental_low_bit(2)
    assert is_experimental_low_bit(3)
    assert not is_experimental_low_bit(4)


def test_planner_never_plans_experimental_low_bits() -> None:
    # AXQ-070: 2- and 3-bit affine storage is not written. A request naming
    # those widths plans on the MXFP4 rung instead and carries no
    # experimental low-bit labels.
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 16),
        group_size=32,
    )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            target_bpw=5.5,
            candidate_bits=(2, 3, 4, 16),
            group_size=32,
            allow_unmeasured=True,
            hardware=HardwareProfile(),
        ),
    )
    assert not plan_uses_experimental_low_bits(plan)
    assert "experimental" not in plan.target_class
    assert EXPERIMENTAL_WARNING not in plan.warnings
    assert not any("robust trunk only" in warning for warning in plan.warnings)
    quantized = [item for item in plan.assignments if item.bits < 16]
    assert quantized
    assert all(
        (item.bits, item.method, item.group_size) == (4, QuantMethod.MXFP4, 32)
        for item in quantized
    )
    # Norm / LM-head floors still apply.
    assert all(
        item.bits == 16
        for item in plan.assignments
        if item.role in {TensorRole.NORM, TensorRole.LM_HEAD}
    )


def test_planner_refuses_a_low_bit_only_grid() -> None:
    # Without the MXFP4 rung a 2-bit grid leaves only BF16 for quantizable
    # tensors, so a sub-16 BPW target is infeasible rather than silently 2-bit.
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 16),
        group_size=32,
    )
    with pytest.raises(PlanningError, match="infeasible"):
        plan_quantization(
            report,
            PlanRequest(
                profile=ProfileName.GENERAL,
                target_bpw=6.0,
                candidate_bits=(2, 16),
                group_size=32,
                allow_unmeasured=True,
            ),
        )


def test_manual_experimental_low_bit_recipe_is_refused(qwen36_model_dir: Path) -> None:
    """AXQ-070: a 2-bit manual recipe fails closed instead of planning."""
    from axquant.inspector import inspect_model

    recipe = ManualPlanRecipe(
        target_bpw=12.0,
        default_bits=2,
        default_method=QuantMethod.AFFINE,
        group_size=32,
        allow_unmatched_rules=True,
    )
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    with pytest.raises(PlanningError, match="AXQ-070"):
        manual_quantization_plan(inventory, recipe)


def test_annotate_is_idempotent() -> None:
    # Historical plans with 2-bit assignments still load and annotate; build
    # one by rewriting the MXFP4 trunk of a current plan.
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.GENERAL,
        candidate_bits=(4, 16),
        group_size=32,
    )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            target_bpw=6.0,
            candidate_bits=(4, 16),
            group_size=32,
            allow_unmeasured=True,
        ),
    )
    historical = plan.model_copy(
        update={
            "assignments": [
                item.model_copy(update={"bits": 2, "method": QuantMethod.AFFINE})
                if item.role == TensorRole.MLP
                else item
                for item in plan.assignments
            ]
        }
    )
    once = annotate_experimental_low_bit_plan(historical)
    assert once.warnings.count(EXPERIMENTAL_WARNING) == 1
    assert "experimental" in once.target_class
    again = annotate_experimental_low_bit_plan(once)
    assert again.warnings.count(EXPERIMENTAL_WARNING) == 1
    assert again.warnings == once.warnings
