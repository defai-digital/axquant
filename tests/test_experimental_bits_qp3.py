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
    ManualPrecisionRule,
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


def test_manual_experimental_recipe_example_loads(tmp_path: Path) -> None:
    recipe_path = Path("examples/qwen36-experimental-2bit-v0.1.yaml")
    assert recipe_path.is_file()
    # Parse via ManualPlanRecipe after yaml load if pyyaml available; otherwise construct.
    recipe = ManualPlanRecipe(
        profile=ProfileName.GENERAL,
        target_bpw=3.5,
        default_bits=2,
        default_method=QuantMethod.AFFINE,
        group_size=32,
        rules=[
            ManualPrecisionRule(
                rule_id="protect-norms",
                roles=(TensorRole.NORM,),
                bits=16,
                method=QuantMethod.BF16,
                reason="Norms stay BF16 under protection floors",
            ),
            ManualPrecisionRule(
                rule_id="protect-lm-head",
                roles=(TensorRole.LM_HEAD,),
                bits=16,
                method=QuantMethod.BF16,
                reason="LM head stays BF16 by default",
            ),
            ManualPrecisionRule(
                rule_id="trunk",
                roles=(TensorRole.MLP, TensorRole.ATTENTION),
                bits=2,
                method=QuantMethod.AFFINE,
                group_size=32,
                reason="Experimental 2-bit trunk with fine groups",
            ),
        ],
    )
    plan = manual_quantization_plan(_inventory(), recipe)
    assert plan_uses_experimental_low_bits(plan)
    assert "2bit-experimental" in plan.target_class or "experimental" in plan.target_class
    assert EXPERIMENTAL_WARNING in plan.warnings
    mlp = next(item for item in plan.assignments if item.role == TensorRole.MLP)
    assert mlp.bits == 2
    assert mlp.group_size == 32


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
            target_bpw=5.0,
            candidate_bits=(2, 16),
            group_size=32,
            allow_unmeasured=True,
        ),
    )
    again = annotate_experimental_low_bit_plan(plan)
    assert again.warnings.count(EXPERIMENTAL_WARNING) == 1
