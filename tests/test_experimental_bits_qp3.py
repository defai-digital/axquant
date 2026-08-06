"""QP3: experimental 2/3-bit planning labels and fine-group recipes."""

from __future__ import annotations

from pathlib import Path

from axquant.analyzer import architecture_prior_report
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


def test_planner_labels_experimental_2bit_plan() -> None:
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
            target_bpw=4.0,
            candidate_bits=(2, 3, 4, 16),
            group_size=32,
            allow_unmeasured=True,
            hardware=HardwareProfile(),
        ),
    )
    assert plan_uses_experimental_low_bits(plan)
    assert "experimental" in plan.target_class
    assert any(EXPERIMENTAL_WARNING[:40] in warning for warning in plan.warnings)
    # Norm / LM-head floors still apply.
    assert all(
        item.bits == 16
        for item in plan.assignments
        if item.role in {TensorRole.NORM, TensorRole.LM_HEAD}
    )


def test_experimental_low_bits_restricted_to_robust_trunk() -> None:
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
            target_bpw=4.0,
            candidate_bits=(2, 3, 4, 16),
            group_size=32,
            allow_unmeasured=True,
            hardware=HardwareProfile(),
        ),
    )
    experimental = [item for item in plan.assignments if item.bits in (2, 3)]
    # The tight budget must still place experimental bits — but only on trunk.
    assert experimental
    assert all(item.role in {TensorRole.MLP, TensorRole.EXPERT} for item in experimental)
    attention = [item for item in plan.assignments if item.role == TensorRole.ATTENTION]
    assert attention and all(item.bits >= 4 for item in attention)
    # RM-42 annotation names the affected tensors for diagnosability.
    assert any("robust trunk only" in warning for warning in plan.warnings)
    assert any(
        experimental[0].tensor.split(".")[-2] in warning
        for warning in plan.warnings
        if "robust trunk only" in warning
    )


def test_manual_experimental_recipe_example_loads(qwen36_model_dir: Path) -> None:
    """Drive the shipped YAML through load_model + manual_quantization_plan (QP3)."""
    from axquant.inspector import inspect_model
    from axquant.serde import load_model

    recipe_path = Path("examples/qwen36-experimental-2bit-v0.1.yaml")
    assert recipe_path.is_file()
    recipe = load_model(recipe_path, ManualPlanRecipe)
    assert recipe.target_bpw >= 12.0
    assert recipe.allow_unmatched_rules is True
    assert recipe.default_bits == 2
    assert recipe.group_size == 32

    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    plan = manual_quantization_plan(inventory, recipe)
    assert plan_uses_experimental_low_bits(plan)
    assert "experimental" in plan.target_class
    assert EXPERIMENTAL_WARNING in plan.warnings
    assert plan.effective_bpw <= recipe.target_bpw + 1e-6
    trunk = [
        item
        for item in plan.assignments
        if item.role in {TensorRole.MLP, TensorRole.ATTENTION, TensorRole.EXPERT} and item.bits < 16
    ]
    assert trunk, "expected at least one experimental low-bit trunk allocation"
    assert all(item.bits == 2 and item.group_size == 32 for item in trunk)
    for item in plan.assignments:
        if item.role in {TensorRole.LM_HEAD, TensorRole.VISION, TensorRole.NORM}:
            assert item.bits == 16


def test_annotate_is_idempotent() -> None:
    report = architecture_prior_report(
        _inventory(),
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 16),
        group_size=32,
    )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            # RM-42 keeps attention at >= 4 bits, so the policy minimum for
            # this inventory is ~5.76 BPW (attention rides at BF16 on a
            # (2, 16) grid); 6.0 still forces the MLP trunk down to 2-bit.
            target_bpw=6.0,
            candidate_bits=(2, 16),
            group_size=32,
            allow_unmeasured=True,
        ),
    )
    again = annotate_experimental_low_bit_plan(plan)
    assert again.warnings.count(EXPERIMENTAL_WARNING) == 1
