"""MXFP4 first-class planning method (AXQ-046 MH5 / ADR-0015).

Locks the method validation (bits=4 / group_size=32 fail-closed everywhere a
method-bearing model is constructed), the frozen-contract version dispatch
(old plan.v1 / sensitivity.v1 / refinement.v2 artifacts keep loading with
their original meaning), and the CLI method whitelists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.schema import (
    Allocation,
    CandidateEntry,
    CandidateMeasurement,
    EvidenceKind,
    HardwareKernelCoverage,
    Inventory,
    KernelLatencyEntry,
    ManualPlanRecipe,
    ManualPrecisionRule,
    MetricVector,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    QuantizerExecutionRecord,
    QuantMethod,
    RefinementConfig,
    RefinementResult,
    RuntimeName,
    SensitivityReport,
    TensorRole,
    TensorSpec,
)
from axquant.schema.loading import (
    load_plan_request,
    load_quantization_plan,
    load_refinement_result,
    load_sensitivity_report,
)
from axquant.serde import stable_sha256, write_data


def _allocation(method: QuantMethod = QuantMethod.MXFP4, *, bits: int = 4, group_size: int = 32):
    return Allocation(
        tensor="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        role=TensorRole.MLP,
        parameters=64,
        bits=bits,
        method=method,
        group_size=group_size,
        predicted_loss=0.1,
        metrics=MetricVector(output_kl=0.1),
        reason="mxfp4 validation fixture",
    )


def test_allocation_mxfp4_requires_4bit_group32() -> None:
    plan = _allocation()
    assert plan.method == QuantMethod.MXFP4
    with pytest.raises(ValueError, match="mxfp4 allocations require"):
        _allocation(bits=6)
    with pytest.raises(ValueError, match="mxfp4 allocations require"):
        _allocation(group_size=64)
    # Non-mxfp4 methods keep the existing grid.
    assert _allocation(method=QuantMethod.AFFINE, group_size=64).group_size == 64


def test_candidate_measurement_mxfp4_requires_4bit_group32() -> None:
    CandidateMeasurement(
        bits=4,
        method=QuantMethod.MXFP4,
        group_size=32,
        metrics=MetricVector(output_kl=0.1),
    )
    with pytest.raises(ValueError, match="mxfp4 candidates require"):
        CandidateMeasurement(
            bits=4,
            method=QuantMethod.MXFP4,
            group_size=64,
            metrics=MetricVector(output_kl=0.1),
        )


def test_manual_rule_and_recipe_mxfp4_require_4bit_group32() -> None:
    ManualPrecisionRule(
        rule_id="mxfp4",
        bits=4,
        method=QuantMethod.MXFP4,
        roles=(TensorRole.MLP,),
        group_size=32,
        reason="mxfp4 rule",
    )
    with pytest.raises(ValueError, match="mxfp4 manual rules require"):
        ManualPrecisionRule(
            rule_id="mxfp4",
            bits=4,
            method=QuantMethod.MXFP4,
            roles=(TensorRole.MLP,),
            group_size=64,
            reason="mxfp4 rule",
        )
    ManualPlanRecipe(default_bits=4, default_method=QuantMethod.MXFP4, group_size=32)
    with pytest.raises(ValueError, match="mxfp4 manual default"):
        ManualPlanRecipe(default_bits=4, default_method=QuantMethod.MXFP4, group_size=64)


def test_tensor_spec_mxfp4_current_precision_is_4bit_group32() -> None:
    def spec(**overrides: object) -> TensorSpec:
        values: dict[str, object] = {
            "name": "model.layers.0.mlp.down_proj.weight",
            "module_path": "model.layers.0.mlp.down_proj",
            "shape": (32, 32),
            "dtype": "FLOAT4",
            "parameters": 1024,
            "role": TensorRole.MLP,
            "quantizable": True,
            "file": "model.safetensors",
            "current_precision": "mxfp4",
            "current_bits": 4,
            "current_group_size": 32,
            "current_method": QuantMethod.MXFP4,
        }
        values.update(overrides)
        return TensorSpec.model_validate(values)

    spec()
    with pytest.raises(ValueError, match="mxfp4 tensors record"):
        spec(current_group_size=64)


def test_execution_records_mxfp4_require_4bit_group32() -> None:
    QuantizerExecutionRecord(
        method=QuantMethod.MXFP4,
        module_path="model.layers.0.mlp.down_proj",
        bits=4,
        group_size=32,
        success=True,
    )
    with pytest.raises(ValueError, match="mxfp4 quantizer records require"):
        QuantizerExecutionRecord(
            method=QuantMethod.MXFP4,
            module_path="model.layers.0.mlp.down_proj",
            bits=4,
            group_size=64,
            success=True,
        )
    HardwareKernelCoverage(
        bits=4,
        group_size=32,
        method=QuantMethod.MXFP4,
        roles=[TensorRole.MLP],
        shapes=[(32, 32)],
        module_count=1,
        parameter_count=1024,
        quantizer_execution_records=1,
        kernel_evidence="measured",
    )
    with pytest.raises(ValueError, match="mxfp4 hardware coverage requires"):
        HardwareKernelCoverage(
            bits=4,
            group_size=64,
            method=QuantMethod.MXFP4,
            roles=[TensorRole.MLP],
            shapes=[(32, 32)],
            module_count=1,
            parameter_count=1024,
            quantizer_execution_records=1,
            kernel_evidence="measured",
        )
    KernelLatencyEntry(
        runtime=RuntimeName.AX_ENGINE,
        bits=4,
        group_size=32,
        method=QuantMethod.MXFP4,
        hidden_size=32,
        decode_median_us=1.0,
        prefill_median_us=1.0,
        dispersion=0.1,
        iterations=3,
    )
    with pytest.raises(ValueError, match="mxfp4 kernel entries require"):
        KernelLatencyEntry(
            runtime=RuntimeName.AX_ENGINE,
            bits=4,
            group_size=64,
            method=QuantMethod.MXFP4,
            hidden_size=32,
            decode_median_us=1.0,
            prefill_median_us=1.0,
            dispersion=0.1,
            iterations=3,
        )


def _planned() -> QuantizationPlan:
    tensor = TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(64, 64),
        dtype="BF16",
        parameters=64 * 64,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )
    inventory = Inventory(
        model=ModelIdentity(model_id="org/model"),
        tensors=[tensor],
        total_parameters=64 * 64,
        quantizable_parameters=64 * 64,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    return plan_quantization(
        architecture_prior_report(inventory, profile=ProfileName.GENERAL),
        PlanRequest(profile=ProfileName.GENERAL, target_bpw=4.5, allow_unmeasured=True),
    )


def test_plan_v1_on_disk_still_loads(tmp_path: Path) -> None:
    plan = _planned()
    assert plan.schema_version == "axquant.plan.v2"
    path = tmp_path / "plan.json"
    write_data(path, plan)
    loaded = load_quantization_plan(path)
    assert isinstance(loaded, QuantizationPlan)
    assert loaded.schema_version == "axquant.plan.v2"
    assert stable_sha256(loaded) == stable_sha256(plan)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "axquant.plan.v1"
    legacy_path = tmp_path / "plan-v1.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = load_quantization_plan(legacy_path)
    # Old-version instances remain valid current-type instances (subclass).
    assert isinstance(legacy, QuantizationPlan)
    assert legacy.schema_version == "axquant.plan.v1"
    assert [assignment.tensor for assignment in legacy.assignments] == [
        assignment.tensor for assignment in plan.assignments
    ]
    # The frozen envelope rejects values the version never defined.
    payload["assignments"][0]["method"] = "mxfp4"
    mxfp4_path = tmp_path / "plan-v1-mxfp4.json"
    mxfp4_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match=r"mxfp4|Input should be"):
        load_quantization_plan(mxfp4_path)

    payload["assignments"][0]["method"] = "affine"
    payload["schema_version"] = "axquant.plan.v9"
    unknown_path = tmp_path / "plan-v9.json"
    unknown_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactError, match="unsupported schema_version"):
        load_quantization_plan(unknown_path)


def test_sensitivity_v1_on_disk_still_loads(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.GENERAL)
    assert report.schema_version == "axquant.sensitivity.v2"
    path = tmp_path / "sensitivity.json"
    write_data(path, report)
    assert load_sensitivity_report(path).schema_version == "axquant.sensitivity.v2"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "axquant.sensitivity.v1"
    legacy_path = tmp_path / "sensitivity-v1.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = load_sensitivity_report(legacy_path)
    assert isinstance(legacy, SensitivityReport)
    assert legacy.schema_version == "axquant.sensitivity.v1"


def test_refinement_v2_on_disk_still_loads(tmp_path: Path) -> None:
    plan = _planned()
    refinement = RefinementResult(
        config=RefinementConfig(top_n=1, max_iterations=1),
        history=[
            CandidateEntry(
                candidate_id="candidate-a",
                plan_sha256=stable_sha256(plan),
                change_description="fixture candidate",
                reason="fixture",
                predicted_bpw=plan.effective_bpw,
                predicted_loss=0.1,
                budget_impact=0.0,
                state="selected",
            )
        ],
        candidate_plans={"candidate-a": plan},
        selected_candidate_id="candidate-a",
        selected_plan=plan,
        selected_plan_sha256=stable_sha256(plan),
        selection_basis="proxy",
        iterations_used=0,
        evaluations_used=0,
        converged=True,
    )
    assert refinement.schema_version == "axquant.refinement.v3"
    path = tmp_path / "refinement.json"
    write_data(path, refinement)
    assert load_refinement_result(path).schema_version == "axquant.refinement.v3"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "axquant.refinement.v2"
    for embedded in payload["candidate_plans"].values():
        embedded["schema_version"] = "axquant.plan.v1"
    payload["selected_plan"]["schema_version"] = "axquant.plan.v1"
    legacy_path = tmp_path / "refinement-v2.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = load_refinement_result(legacy_path)
    assert legacy.schema_version == "axquant.refinement.v2"
    assert set(legacy.candidate_plans) == {"candidate-a"}


def test_plan_request_v1_on_disk_still_loads(tmp_path: Path) -> None:
    request = PlanRequest(profile=ProfileName.GENERAL, target_bpw=4.8)
    assert request.schema_version == "axquant.plan-request.v2"
    path = tmp_path / "request.json"
    write_data(path, request)
    assert load_plan_request(path).schema_version == "axquant.plan-request.v2"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "axquant.plan-request.v1"
    legacy_path = tmp_path / "request-v1.json"
    legacy_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_plan_request(legacy_path).schema_version == "axquant.plan-request.v1"


def test_cli_method_parsers_accept_mxfp4() -> None:
    from axquant.cli._parser import _methods, _probe_methods

    assert _methods("affine,mxfp4") == (QuantMethod.AFFINE, QuantMethod.MXFP4)
    assert _probe_methods("mxfp4") == (QuantMethod.MXFP4,)
    with pytest.raises(Exception, match="measured probing cannot execute"):
        _probe_methods("bf16")


def test_probe_config_v2_emitted_and_mxfp4_selectable() -> None:
    from axquant.schema import ProbeConfig

    config = ProbeConfig(
        model=ModelIdentity(model_id="m"),
        calibration_cache="/tmp",
        candidate_methods=(QuantMethod.MXFP4,),
    )
    assert config.schema_version == "axquant.probe-config.v2"
    assert config.candidate_methods == (QuantMethod.MXFP4,)
    assert ProbeConfig.model_fields["schema_version"].default == "axquant.probe-config.v2"


def test_evidence_kind_release_quality_unchanged() -> None:
    # The MXFP4 chain bump must not move evidence gating semantics.
    assert EvidenceKind.ARCHITECTURE_PRIOR.release_quality is False
    assert EvidenceKind.MEASURED.release_quality is True
