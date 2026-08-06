from __future__ import annotations

from pathlib import Path

import pytest

from axquant.cli import main
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.manual import manual_quantization_plan
from axquant.schema import (
    EvidenceKind,
    ManualPlanRecipe,
    ManualPrecisionRule,
    ProfileName,
    QuantizationPlan,
    QuantMethod,
    TensorRole,
    TensorSpec,
)
from axquant.serde import load_model, write_data


def _inventory(model_dir: Path):
    return inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )


def _recipe(**overrides: object) -> ManualPlanRecipe:
    values: dict[str, object] = {
        "profile": ProfileName.AGENT_CODING,
        "target_bpw": 16.0,
        "default_bits": 4,
        "default_method": QuantMethod.AFFINE,
        "rules": [
            ManualPrecisionRule(
                rule_id="attention-6bit",
                bits=6,
                method=QuantMethod.AFFINE,
                roles=(TensorRole.ATTENTION,),
                reason="manual attention protection",
            )
        ],
    }
    values.update(overrides)
    return ManualPlanRecipe.model_validate(values)


def test_manual_plan_applies_rules_and_mandatory_protection(
    qwen36_model_dir: Path,
) -> None:
    plan = manual_quantization_plan(_inventory(qwen36_model_dir), _recipe())
    assert plan.target_class == "16p0bpw"
    attention = next(
        allocation
        for allocation in plan.assignments
        if allocation.role == TensorRole.ATTENTION and allocation.bits < 16
    )
    mtp = next(allocation for allocation in plan.assignments if allocation.role.is_mtp)
    vision = next(
        allocation for allocation in plan.assignments if allocation.role == TensorRole.VISION
    )
    head = next(
        allocation for allocation in plan.assignments if allocation.role == TensorRole.LM_HEAD
    )
    assert attention.bits == 6
    assert attention.group_size == 64
    assert "attention-6bit" in attention.reason
    assert mtp.bits == 16
    assert "byte-for-byte" in mtp.reason
    assert vision.bits == 16
    assert head.bits == 16
    assert plan.evidence_kind == EvidenceKind.ARCHITECTURE_PRIOR
    assert plan.global_validation_required is True
    assert plan.candidate_bits == (4, 6, 16)


def test_manual_plan_rejects_unmatched_and_unsafe_rules(
    qwen36_model_dir: Path,
) -> None:
    unmatched = ManualPrecisionRule(
        rule_id="typo",
        bits=6,
        method=QuantMethod.AFFINE,
        tensor_glob="missing.*",
        reason="should not silently pass",
    )
    with pytest.raises(PlanningError, match="matched no tensors"):
        manual_quantization_plan(
            _inventory(qwen36_model_dir),
            _recipe(rules=[unmatched]),
        )

    unsafe = ManualPrecisionRule(
        rule_id="mtp-4bit",
        bits=4,
        method=QuantMethod.AFFINE,
        roles=(TensorRole.MTP_PROJECTION,),
        reason="unsafe test",
    )
    with pytest.raises(PlanningError, match="protected minimum"):
        manual_quantization_plan(
            _inventory(qwen36_model_dir),
            _recipe(rules=[unsafe]),
        )


def test_manual_plan_enforces_declared_bpw_limit(
    qwen36_model_dir: Path,
) -> None:
    with pytest.raises(PlanningError, match="above its"):
        manual_quantization_plan(
            _inventory(qwen36_model_dir),
            _recipe(target_bpw=4.8),
        )


def test_manual_plan_rejects_mutable_revision_alias(qwen36_model_dir: Path) -> None:
    inventory = _inventory(qwen36_model_dir)
    inventory.model.revision = "main"

    with pytest.raises(PlanningError, match="revision-pinned"):
        manual_quantization_plan(inventory, _recipe())


def test_plan_manual_cli_emits_plan_and_report(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.json"
    recipe_path = tmp_path / "recipe.yaml"
    plan_path = tmp_path / "plan.json"
    markdown_path = tmp_path / "plan.md"
    write_data(inventory_path, _inventory(qwen36_model_dir))
    write_data(recipe_path, _recipe())
    result = main(
        [
            "plan-manual",
            "--inventory",
            str(inventory_path),
            "--recipe",
            str(recipe_path),
            "--output",
            str(plan_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )
    assert result == 0
    plan = load_model(plan_path, QuantizationPlan)
    assert plan.evidence_kind == EvidenceKind.ARCHITECTURE_PRIOR
    assert "# AXQuant Plan Report" in markdown_path.read_text(encoding="utf-8")


def test_manual_axq026_lm_head_floor_requires_explicit_opt_in(
    qwen36_model_dir: Path,
) -> None:
    head_rule = ManualPrecisionRule(
        rule_id="lm-head-8bit",
        bits=8,
        method=QuantMethod.AFFINE,
        roles=(TensorRole.LM_HEAD,),
        reason="AXQ-026 size-gate path",
    )
    # Without the governed opt-in the BF16 floor still rejects the rule.
    with pytest.raises(PlanningError, match="protected minimum"):
        manual_quantization_plan(
            _inventory(qwen36_model_dir),
            _recipe(rules=[head_rule]),
        )

    plan = manual_quantization_plan(
        _inventory(qwen36_model_dir),
        _recipe(rules=[head_rule], lm_head_min_bits=8),
    )
    head = next(
        allocation for allocation in plan.assignments if allocation.role == TensorRole.LM_HEAD
    )
    assert head.bits == 8
    assert plan.constraints.lm_head_min_bits == 8


def test_manual_tied_weight_harmonization_preserves_explicit_method(
    qwen36_model_dir: Path,
) -> None:
    inventory = _inventory(qwen36_model_dir)
    unprotected = [
        tensor
        for tensor in inventory.tensors
        if tensor.quantizable
        and not tensor.role.is_mtp
        and tensor.role not in {TensorRole.LM_HEAD, TensorRole.VISION}
    ]
    left, right = unprotected[:2]
    inventory.tied_weight_groups = [[left.name, right.name]]
    shared_rule = ManualPrecisionRule(
        rule_id="shared-awq",
        bits=8,
        method=QuantMethod.AWQ,
        tensor_glob=f"*{left.name.split('.')[-2]}*",
        group_size=32,
        reason="shared tied-weight strategy",
    )
    shared_rule_right = shared_rule.model_copy(
        update={
            "rule_id": "shared-awq-right",
            "tensor_glob": f"*{right.name.split('.')[-2]}*",
        }
    )
    plan = manual_quantization_plan(
        inventory,
        _recipe(rules=[shared_rule, shared_rule_right]),
    )
    tied = [
        allocation
        for allocation in plan.assignments
        if allocation.tensor in {left.name, right.name}
    ]
    assert {(allocation.bits, allocation.method, allocation.group_size) for allocation in tied} == {
        (8, QuantMethod.AWQ, 32)
    }


def test_manual_tied_weights_reject_conflicting_quantizers(
    qwen36_model_dir: Path,
) -> None:
    inventory = _inventory(qwen36_model_dir)
    unprotected = [
        tensor
        for tensor in inventory.tensors
        if tensor.quantizable
        and not tensor.role.is_mtp
        and tensor.role not in {TensorRole.LM_HEAD, TensorRole.VISION}
    ]
    left, right = unprotected[:2]
    inventory.tied_weight_groups = [[left.name, right.name]]
    rules = [
        ManualPrecisionRule(
            rule_id="left-awq",
            bits=8,
            method=QuantMethod.AWQ,
            tensor_glob=f"*{left.name.split('.')[-2]}*",
            reason="left strategy",
        ),
        ManualPrecisionRule(
            rule_id="right-gptq",
            bits=8,
            method=QuantMethod.GPTQ,
            tensor_glob=f"*{right.name.split('.')[-2]}*",
            reason="right strategy",
        ),
    ]
    with pytest.raises(PlanningError, match="conflicting manual methods"):
        manual_quantization_plan(inventory, _recipe(rules=rules))


def _expert_tensor(name: str, module_path: str) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=module_path,
        shape=(64, 64),
        dtype="BF16",
        parameters=4096,
        role=TensorRole.EXPERT,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )


@pytest.mark.parametrize(
    ("module_path", "method"),
    [
        ("model.layers.0.mlp.experts.0.gate_proj", QuantMethod.AWQ),
        ("model.layers.0.mlp.experts.gate_up_proj", QuantMethod.DWQ),
    ],
)
def test_manual_rejects_non_affine_method_on_fused_and_packed_experts(
    qwen36_model_dir: Path,
    module_path: str,
    method: QuantMethod,
) -> None:
    inventory = _inventory(qwen36_model_dir)
    inventory.tensors.append(_expert_tensor(f"{module_path}.weight", module_path))
    rule = ManualPrecisionRule(
        rule_id="expert-refinement",
        bits=4,
        method=method,
        module_glob=f"*{module_path.removeprefix('model.')}",
        group_size=64,
        reason="unexecutable refinement on a fused/packed expert module",
    )
    with pytest.raises(PlanningError, match="affine"):
        manual_quantization_plan(inventory, _recipe(rules=[rule]))


def test_manual_rejects_mixed_precisions_within_fused_expert_group(
    qwen36_model_dir: Path,
) -> None:
    inventory = _inventory(qwen36_model_dir)
    for index in range(2):
        module_path = f"model.layers.0.mlp.experts.{index}.gate_proj"
        inventory.tensors.append(_expert_tensor(f"{module_path}.weight", module_path))
    rule = ManualPrecisionRule(
        rule_id="expert0-6bit",
        bits=6,
        method=QuantMethod.AFFINE,
        module_glob="*experts.0.gate_proj",
        reason="asymmetric precision inside one switch module",
    )
    with pytest.raises(PlanningError, match="fused expert module"):
        manual_quantization_plan(inventory, _recipe(rules=[rule]))
