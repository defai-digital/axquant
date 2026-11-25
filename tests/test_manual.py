from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.manual import manual_quantization_plan
from axquant.schema import (
    CalibrationEvidence,
    EvidenceKind,
    HardwareProfile,
    ManualPlanRecipe,
    ManualPrecisionRule,
    ProfileName,
    QuantizationPlan,
    QuantMethod,
    SensitivityReport,
    TensorRole,
    TensorSpec,
)
from axquant.serde import load_model, write_data
from axquant.source_binding import (
    binding_path_beside,
    load_source_plan_binding,
    source_binding_issues,
)


def _inventory(model_dir: Path):
    return inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )


# AXQ-070 product storage: 4-bit MXFP4 and 8-bit MXFP8 at group 32, or BF16.
_PRODUCT_HARDWARE = HardwareProfile(
    supported_bits=(4, 8, 16),
    supported_methods=(QuantMethod.AFFINE, QuantMethod.MXFP4, QuantMethod.BF16),
    supported_group_sizes=(32,),
)


def _recipe(**overrides: object) -> ManualPlanRecipe:
    values: dict[str, object] = {
        "profile": ProfileName.AGENT_CODING,
        "target_bpw": 16.0,
        "default_bits": 4,
        "default_method": QuantMethod.MXFP4,
        "group_size": 32,
        "hardware": _PRODUCT_HARDWARE,
        "rules": [
            ManualPrecisionRule(
                rule_id="attention-8bit",
                bits=8,
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
    assert attention.bits == 8
    assert attention.method == QuantMethod.AFFINE
    assert attention.group_size == 32
    assert "attention-8bit" in attention.reason
    assert mtp.bits == 16
    assert "byte-for-byte" in mtp.reason
    assert vision.bits == 16
    assert head.bits == 16
    assert plan.evidence_kind == EvidenceKind.ARCHITECTURE_PRIOR
    assert plan.global_validation_required is True
    assert plan.candidate_bits == (4, 8, 16)


def test_deepseek_v4_mxfp4_recipe_is_a_valid_manual_recipe() -> None:
    path = Path(__file__).resolve().parents[1] / (
        "examples/deepseek-v4-experimental-mxfp4-v0.1.yaml"
    )
    recipe = ManualPlanRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert recipe.default_bits == 4
    assert recipe.group_size == 32
    trunk = next(rule for rule in recipe.rules if rule.rule_id == "trunk-mxfp4")
    attention = next(rule for rule in recipe.rules if rule.rule_id == "attention-mxfp4")
    assert trunk.bits == 4 and attention.bits == 4
    assert set(trunk.roles) == {TensorRole.MLP, TensorRole.EXPERT}


def test_qwen38_flash_next_recipes_keep_ple_on_group_size_32() -> None:
    """PLE shards are 160-wide; 160 % 64 != 0, so embeddings must stay gs32."""

    recipes = (
        ("examples/qwen38-flash-next-axq-mxfp4-v0.1.yaml", 4, 32),
        ("examples/qwen38-flash-next-axq-mxfp4-attn8-v0.1.yaml", 4, 32),
        ("examples/qwen38-flash-next-axq-mxfp8-mtp-v0.1.yaml", 8, 32),
    )
    for relative, default_bits, global_group in recipes:
        path = Path(__file__).resolve().parents[1] / relative
        recipe = ManualPlanRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert recipe.default_bits == default_bits
        assert recipe.group_size == global_group
        embedding = next(rule for rule in recipe.rules if rule.rule_id == "embedding-8")
        assert TensorRole.EMBEDDING in embedding.roles
        assert embedding.bits == 8
        assert embedding.group_size == 32
        assert 160 % embedding.group_size == 0
        vision = next(rule for rule in recipe.rules if rule.rule_id == "protect-vision")
        assert vision.bits == 16


def test_qwen3_coder_next_mxfp4_recipe_is_a_valid_manual_recipe() -> None:
    path = Path(__file__).resolve().parents[1] / ("examples/qwen3-coder-next-axq-mxfp4-v0.1.yaml")
    recipe = ManualPlanRecipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert recipe.default_bits == 4
    assert recipe.group_size == 32
    attention = next(rule for rule in recipe.rules if rule.rule_id == "attention-mxfp4")
    experts = next(rule for rule in recipe.rules if rule.rule_id == "experts-mxfp4")
    routers = next(rule for rule in recipe.rules if rule.rule_id == "routers-8")
    assert attention.bits == 4 and attention.group_size == 32
    assert set(experts.roles) == {TensorRole.EXPERT, TensorRole.MLP}
    assert routers.bits == 8 and routers.method == QuantMethod.AFFINE
    assert all(rule.rule_id != "protect-vision" for rule in recipe.rules)


def test_manual_plan_rejects_unmatched_and_unsafe_rules(
    qwen36_model_dir: Path,
) -> None:
    unmatched = ManualPrecisionRule(
        rule_id="typo",
        bits=8,
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
        method=QuantMethod.MXFP4,
        roles=(TensorRole.MTP_PROJECTION,),
        group_size=32,
        reason="unsafe test",
    )
    with pytest.raises(PlanningError, match="protected minimum"):
        manual_quantization_plan(
            _inventory(qwen36_model_dir),
            _recipe(rules=[unsafe]),
        )


@pytest.mark.parametrize(
    ("bits", "method", "group_size", "detail"),
    [
        (6, QuantMethod.AFFINE, 32, "6-bit affine group 32"),
        (2, QuantMethod.AFFINE, 32, "2-bit affine group 32"),
        (4, QuantMethod.AFFINE, 32, "4-bit affine group 32"),
        (8, QuantMethod.AFFINE, 64, "8-bit affine group 64"),
        (4, QuantMethod.DWQ, 32, "4-bit dwq group 32"),
    ],
)
def test_manual_plan_refuses_storage_outside_axq070(
    qwen36_model_dir: Path,
    bits: int,
    method: QuantMethod,
    group_size: int,
    detail: str,
) -> None:
    rule = ManualPrecisionRule(
        rule_id="retired-storage",
        bits=bits,
        method=method,
        roles=(TensorRole.ATTENTION,),
        group_size=group_size,
        reason="AXQ-070 retired storage",
    )
    with pytest.raises(PlanningError, match=f"AXQ-070: manual recipe assigns {detail}"):
        manual_quantization_plan(_inventory(qwen36_model_dir), _recipe(rules=[rule]))


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
        rule_id="shared-mxfp4",
        bits=4,
        method=QuantMethod.MXFP4,
        tensor_glob=f"*{left.name.split('.')[-2]}*",
        group_size=32,
        reason="shared tied-weight strategy",
    )
    shared_rule_right = shared_rule.model_copy(
        update={
            "rule_id": "shared-mxfp4-right",
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
        (4, QuantMethod.MXFP4, 32)
    }


def test_manual_tied_weights_harmonize_to_the_widest_product_lane(
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
            rule_id="left-mxfp4",
            bits=4,
            method=QuantMethod.MXFP4,
            tensor_glob=f"*{left.name.split('.')[-2]}*",
            group_size=32,
            reason="left strategy",
        ),
        ManualPrecisionRule(
            rule_id="right-mxfp8",
            bits=8,
            method=QuantMethod.AFFINE,
            tensor_glob=f"*{right.name.split('.')[-2]}*",
            group_size=32,
            reason="right strategy",
        ),
    ]
    plan = manual_quantization_plan(inventory, _recipe(rules=rules))
    tied = [
        allocation
        for allocation in plan.assignments
        if allocation.tensor in {left.name, right.name}
    ]
    assert {(allocation.bits, allocation.method, allocation.group_size) for allocation in tied} == {
        (8, QuantMethod.AFFINE, 32)
    }


def test_manual_tied_weights_refuse_refined_quantizers(
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
    with pytest.raises(PlanningError, match="AXQ-070: manual recipe assigns 8-bit awq"):
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


@pytest.mark.parametrize("method", [QuantMethod.AWQ, QuantMethod.DWQ, QuantMethod.GPTQ])
@pytest.mark.parametrize(
    "module_path",
    [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ],
)
def test_manual_refuses_refinement_on_fused_and_packed_experts(
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
        group_size=32,
        reason="AXQ-070 retired refined affine packing",
    )
    with pytest.raises(PlanningError, match=f"AXQ-070: manual recipe assigns 4-bit {method.value}"):
        manual_quantization_plan(inventory, _recipe(rules=[rule]))


@pytest.mark.parametrize(
    "module_path",
    [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ],
)
def test_manual_accepts_mxfp4_on_fused_and_packed_experts(
    qwen36_model_dir: Path,
    module_path: str,
) -> None:
    inventory = _inventory(qwen36_model_dir)
    inventory.tensors.append(_expert_tensor(f"{module_path}.weight", module_path))
    rule = ManualPrecisionRule(
        rule_id="expert-mxfp4",
        bits=4,
        method=QuantMethod.MXFP4,
        module_glob=f"*{module_path.removeprefix('model.')}",
        group_size=32,
        reason="native MXFP4 pack executes on a fused switch",
    )
    plan = manual_quantization_plan(inventory, _recipe(rules=[rule]))
    experts = [item for item in plan.assignments if item.role == TensorRole.EXPERT]
    assert experts
    assert all(
        item.method == QuantMethod.MXFP4 and item.bits == 4 and item.group_size == 32
        for item in experts
    )


def test_manual_rejects_mixed_precisions_within_fused_expert_group(
    qwen36_model_dir: Path,
) -> None:
    inventory = _inventory(qwen36_model_dir)
    for index in range(2):
        module_path = f"model.layers.0.mlp.experts.{index}.gate_proj"
        inventory.tensors.append(_expert_tensor(f"{module_path}.weight", module_path))
    rule = ManualPrecisionRule(
        rule_id="expert0-8bit",
        bits=8,
        method=QuantMethod.AFFINE,
        module_glob="*experts.0.gate_proj",
        group_size=32,
        reason="asymmetric precision inside one switch module",
    )
    with pytest.raises(PlanningError, match="fused expert module"):
        manual_quantization_plan(inventory, _recipe(rules=[rule]))


def _measured_sensitivity(model_dir: Path) -> SensitivityReport:
    """Measured report whose MTP entries carry only the probe's zero marker."""
    inventory = _inventory(model_dir)
    prior = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    for entry in prior.entries:
        if not entry.tensor.role.is_mtp:
            continue
        for candidate in entry.candidates:
            candidate.metrics = candidate.metrics.model_copy(update={"mtp_acceptance_loss": 0.0})
    return SensitivityReport(
        model=prior.model,
        architecture_profile=prior.architecture_profile,
        profile=prior.profile,
        evidence_kind=EvidenceKind.MEASURED,
        inventory_sha256=prior.inventory_sha256,
        entries=prior.entries,
        calibration=CalibrationEvidence(
            dataset_id="mh1-test",
            dataset_sha256="d" * 64,
            samples=4,
            domains=["coding"],
            sequence_length=32,
            backend="test",
            reference="unit",
        ),
        warnings=[],
    )


def test_manual_plan_fail_closed_on_zero_mtp_signal(qwen36_model_dir: Path) -> None:
    with pytest.raises(PlanningError, match="--allow-mtp-unmeasured"):
        manual_quantization_plan(
            _inventory(qwen36_model_dir),
            _recipe(),
            sensitivity=_measured_sensitivity(qwen36_model_dir),
        )


def test_manual_plan_allow_mtp_unmeasured_warns(qwen36_model_dir: Path) -> None:
    from axquant.planner import MTP_UNMEASURED_WARNING

    plan = manual_quantization_plan(
        _inventory(qwen36_model_dir),
        _recipe(),
        sensitivity=_measured_sensitivity(qwen36_model_dir),
        allow_mtp_unmeasured=True,
    )
    assert MTP_UNMEASURED_WARNING in plan.warnings


def test_plan_manual_cli_writes_a_source_binding(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    """AXQ-048: a locally planned manual plan carries its source fingerprint.

    The plan is built from the inventory, so the inventory's checkpoint is the
    directory the binding must describe — never a directory a later convert is
    handed, which would only compare against itself.
    """

    inventory_path = tmp_path / "inventory.json"
    recipe_path = tmp_path / "recipe.yaml"
    plan_path = tmp_path / "plan.json"
    inventory = _inventory(qwen36_model_dir)
    write_data(inventory_path, inventory)
    write_data(recipe_path, _recipe())

    assert (
        main(
            [
                "plan-manual",
                "--inventory",
                str(inventory_path),
                "--recipe",
                str(recipe_path),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )

    plan = load_model(plan_path, QuantizationPlan)
    binding = load_source_plan_binding(binding_path_beside(plan_path))
    assert binding.source_model.local_path is None
    assert (
        source_binding_issues(
            binding=binding,
            plan=plan,
            source_dir=qwen36_model_dir,
        )
        == []
    )


def test_plan_manual_cli_writes_no_binding_without_a_local_source(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    """A plan whose source is not present locally gets no binding.

    Converting it from a local directory stays fail-closed rather than comparing
    a fingerprint the consumer minted from the directory it is about to convert.
    """

    inventory_path = tmp_path / "inventory.json"
    recipe_path = tmp_path / "recipe.yaml"
    plan_path = tmp_path / "plan.json"
    inventory = _inventory(qwen36_model_dir)
    inventory.model = inventory.model.model_copy(update={"local_path": None})
    write_data(inventory_path, inventory)
    write_data(recipe_path, _recipe())

    assert (
        main(
            [
                "plan-manual",
                "--inventory",
                str(inventory_path),
                "--recipe",
                str(recipe_path),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )

    assert not binding_path_beside(plan_path).exists()
