"""Measured 2/3/4 trunk mix: fused switch modules upgrade as one unit."""

from __future__ import annotations

from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.errors import PlanningError
from axquant.experimental_mix import (
    EXPERIMENTAL_MIX_WARNING,
    plan_experimental_mix,
)
from axquant.module_paths import fused_expert_module
from axquant.planner import allocation_unit_key, plan_quantization
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CalibrationEvidence,
    CandidateMeasurement,
    EvidenceKind,
    Inventory,
    MetricVector,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    QuantMethod,
    SensitivityReport,
    TensorRole,
    TensorSpec,
)
from axquant.serde import load_model, write_data


def _tensor(name: str, parameters: int, role: TensorRole) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=name.removesuffix(".weight"),
        shape=(parameters, 1),
        dtype="BF16",
        parameters=parameters,
        role=role,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )


def _inventory(tensors: list[TensorSpec]) -> Inventory:
    return Inventory(
        model=ModelIdentity(model_id="org/flash-mix", revision="abc"),
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors),
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
        architecture_profile=ArchitectureProfile(
            support_level=ArchitectureSupportLevel.SUPPORTED,
            product_family="deepseek-v4",
            optimization_scope=OptimizationScope.TEXT_PATH,
            adapter_id="generic",
            text_layer_count=2,
        ),
    )


def _request(**overrides: object) -> PlanRequest:
    values: dict[str, object] = {
        "profile": ProfileName.GENERAL,
        "target_bpw": 4.1,
        "candidate_bits": (2, 3, 4, 8, 16),
        "group_size": 32,
        "allow_unmeasured": True,
    }
    values.update(overrides)
    return PlanRequest.model_validate(values)


def _affine_ladder(*kl_by_bits: tuple[int, float]) -> list[CandidateMeasurement]:
    candidates = [
        CandidateMeasurement(
            bits=bits,
            method=QuantMethod.AFFINE,
            group_size=32,
            metrics=MetricVector(output_kl=kl),
        )
        for bits, kl in kl_by_bits
    ]
    candidates.append(
        CandidateMeasurement(
            bits=16,
            method=QuantMethod.BF16,
            group_size=None,
            metrics=MetricVector(),
        )
    )
    return candidates


def _deepseek_mix_tensors() -> list[TensorSpec]:
    tensors = []
    for layer in (0, 1):
        for expert in (0, 1):
            tensors.append(
                _tensor(
                    f"layers.{layer}.ffn.experts.{expert}.w1.weight",
                    1_000,
                    TensorRole.EXPERT,
                )
            )
    tensors.append(_tensor("layers.0.attn.q_proj.weight", 1_000, TensorRole.ATTENTION))
    tensors.append(_tensor("lm_head.weight", 100, TensorRole.LM_HEAD))
    return tensors


def _apply_kl(
    report: SensitivityReport,
    kl_by_tensor: dict[str, tuple[float, float, float]],
) -> None:
    for entry in report.entries:
        if entry.tensor.role == TensorRole.LM_HEAD:
            entry.candidates = [
                CandidateMeasurement(
                    bits=16,
                    method=QuantMethod.BF16,
                    group_size=None,
                    metrics=MetricVector(),
                )
            ]
            continue
        if entry.tensor.role == TensorRole.ATTENTION:
            entry.candidates = _affine_ladder((4, 0.05), (8, 0.02))
            continue
        kl2, kl3, kl4 = kl_by_tensor[entry.tensor.name]
        entry.candidates = _affine_ladder((2, kl2), (3, kl3), (4, kl4), (8, kl4 * 0.5))


def test_allocation_unit_key_groups_deepseek_fused_switch() -> None:
    left = "layers.0.ffn.experts.0.w1"
    right = "layers.0.ffn.experts.1.w1"
    assert fused_expert_module(left) == "layers.0.ffn.switch_mlp.gate_proj"
    assert allocation_unit_key(left, f"{left}.weight") == allocation_unit_key(
        right, f"{right}.weight"
    )
    assert allocation_unit_key(left, f"{left}.weight") != allocation_unit_key(
        "layers.1.ffn.experts.0.w1",
        "layers.1.ffn.experts.0.w1.weight",
    )


def test_plan_experimental_mix_upgrades_highest_kl_fused_unit_first() -> None:
    inventory = _inventory(_deepseek_mix_tensors())
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 8, 16),
        group_size=32,
    )
    sensitive = (1.0, 0.2, 0.1)
    dull = (0.4, 0.3, 0.25)
    kl_by_tensor = {
        "layers.0.ffn.experts.0.w1.weight": sensitive,
        "layers.0.ffn.experts.1.w1.weight": sensitive,
        "layers.1.ffn.experts.0.w1.weight": dull,
        "layers.1.ffn.experts.1.w1.weight": dull,
    }
    _apply_kl(report, kl_by_tensor)

    plan = plan_experimental_mix(report, _request())
    by_tensor = {item.tensor: item for item in plan.assignments}

    assert by_tensor["layers.0.ffn.experts.0.w1.weight"].bits == 3
    assert by_tensor["layers.0.ffn.experts.1.w1.weight"].bits == 3
    assert by_tensor["layers.1.ffn.experts.0.w1.weight"].bits == 2
    assert by_tensor["layers.1.ffn.experts.1.w1.weight"].bits == 2
    assert by_tensor["layers.0.attn.q_proj.weight"].bits >= 4
    assert by_tensor["lm_head.weight"].bits == 16
    assert all(item.method == QuantMethod.AFFINE for item in plan.assignments if item.bits < 16)
    assert EXPERIMENTAL_MIX_WARNING in plan.warnings
    assert any("fused switch modules" in warning for warning in plan.warnings)
    assert "experimental" in plan.target_class


def test_plan_experimental_mix_fail_closed_when_fused_unit_has_no_common_signature() -> None:
    inventory = _inventory(
        [
            _tensor("layers.0.ffn.experts.0.w1.weight", 1_000, TensorRole.EXPERT),
            _tensor("layers.0.ffn.experts.1.w1.weight", 1_000, TensorRole.EXPERT),
            _tensor("lm_head.weight", 100, TensorRole.LM_HEAD),
        ]
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 16),
        group_size=32,
    )
    for entry in report.entries:
        if entry.tensor.role == TensorRole.LM_HEAD:
            continue
        bits = 2 if ".experts.0." in entry.tensor.name else 3
        # No shared BF16 rung: a fused stack with only disjoint quantized
        # candidates must fail closed instead of collapsing to 16-bit.
        entry.candidates = [
            CandidateMeasurement(
                bits=bits,
                method=QuantMethod.AFFINE,
                group_size=32,
                metrics=MetricVector(output_kl=0.5),
            )
        ]

    with pytest.raises(PlanningError, match="no common precision"):
        plan_experimental_mix(report, _request(target_bpw=16.0))


def test_plan_experimental_mix_ignores_awq_on_fused_experts() -> None:
    inventory = _inventory(
        [
            _tensor("layers.0.ffn.experts.0.w1.weight", 1_000, TensorRole.EXPERT),
            _tensor("layers.0.ffn.experts.1.w1.weight", 1_000, TensorRole.EXPERT),
            _tensor("lm_head.weight", 100, TensorRole.LM_HEAD),
        ]
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 16),
        group_size=32,
    )
    for entry in report.entries:
        if entry.tensor.role == TensorRole.LM_HEAD:
            continue
        entry.candidates = [
            CandidateMeasurement(
                bits=2,
                method=QuantMethod.AWQ,
                group_size=32,
                metrics=MetricVector(output_kl=0.01),
            ),
            *_affine_ladder((2, 0.9), (3, 0.4), (4, 0.2)),
        ]

    plan = plan_experimental_mix(report, _request(target_bpw=4.0))
    experts = [item for item in plan.assignments if item.role == TensorRole.EXPERT]
    assert experts
    assert all(item.method == QuantMethod.AFFINE for item in experts)
    assert len({item.bits for item in experts}) == 1


def test_plan_experimental_mix_accepts_measured_evidence_without_unmeasured_flag() -> None:
    inventory = _inventory(_deepseek_mix_tensors())
    prior = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 8, 16),
        group_size=32,
    )
    kl = {
        "layers.0.ffn.experts.0.w1.weight": (0.8, 0.3, 0.2),
        "layers.0.ffn.experts.1.w1.weight": (0.8, 0.3, 0.2),
        "layers.1.ffn.experts.0.w1.weight": (0.5, 0.4, 0.3),
        "layers.1.ffn.experts.1.w1.weight": (0.5, 0.4, 0.3),
    }
    _apply_kl(prior, kl)
    measured = SensitivityReport(
        model=prior.model,
        architecture_profile=prior.architecture_profile,
        profile=prior.profile,
        evidence_kind=EvidenceKind.MEASURED,
        inventory_sha256=prior.inventory_sha256,
        entries=prior.entries,
        calibration=CalibrationEvidence(
            dataset_id="axquant-test-mix",
            dataset_sha256="b" * 64,
            samples=8,
            domains=["general"],
            sequence_length=32,
            backend="test",
            reference="unit",
        ),
        warnings=[],
    )
    plan = plan_experimental_mix(measured, _request(allow_unmeasured=False, target_bpw=5.0))
    assert plan.evidence_kind == EvidenceKind.MEASURED
    experts = [item for item in plan.assignments if item.role == TensorRole.EXPERT]
    bits = {item.bits for item in experts}
    assert bits <= {2, 3, 4, 8}
    fused_bits: dict[str, set[int]] = {}
    for item in experts:
        fused = fused_expert_module(item.module_path)
        assert fused is not None
        fused_bits.setdefault(fused, set()).add(item.bits)
    assert all(len(values) == 1 for values in fused_bits.values())


def test_default_plan_quantization_still_uses_tensor_then_harmonize() -> None:
    inventory = _inventory(_deepseek_mix_tensors())
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 8, 16),
        group_size=32,
    )
    plan = plan_quantization(report, _request(target_bpw=16.0))
    assert not any("fused switch modules" in warning for warning in plan.warnings)
    assert EXPERIMENTAL_MIX_WARNING not in plan.warnings


def test_plan_experimental_mix_requires_2bit_grid() -> None:
    inventory = _inventory(_deepseek_mix_tensors())
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(4, 8, 16),
        group_size=32,
    )
    with pytest.raises(PlanningError, match="requires 2-bit"):
        plan_experimental_mix(report, _request(candidate_bits=(4, 8, 16)))


def test_plan_experimental_mix_cli_writes_plan(tmp_path: Path) -> None:
    inventory = _inventory(_deepseek_mix_tensors())
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(2, 3, 4, 8, 16),
        group_size=32,
    )
    sensitivity_path = tmp_path / "sensitivity.json"
    plan_path = tmp_path / "mix-plan.json"
    markdown_path = tmp_path / "mix-plan.md"
    write_data(sensitivity_path, report)
    exit_code = main(
        [
            "plan-experimental-mix",
            "--sensitivity",
            str(sensitivity_path),
            "--target-bpw",
            "16",
            "--allow-unmeasured",
            "--output",
            str(plan_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )
    assert exit_code == 0
    plan = load_model(plan_path, QuantizationPlan)
    assert EXPERIMENTAL_MIX_WARNING in plan.warnings
    assert 2 in plan.candidate_bits
    assert "# AXQuant Plan Report" in markdown_path.read_text(encoding="utf-8")
