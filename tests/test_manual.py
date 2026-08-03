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
