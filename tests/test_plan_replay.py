from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.errors import PlanningError
from axquant.plan_replay import replay_measured_plan
from axquant.planner import plan_quantization
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CalibrationEvidence,
    EvidenceKind,
    Inventory,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    SensitivityReport,
    SupportTier,
    TensorRole,
    TensorSpec,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


def _report_and_plan() -> tuple[SensitivityReport, QuantizationPlan]:
    tensors = [
        TensorSpec(
            name="model.layers.0.mlp.down_proj.weight",
            module_path="model.layers.0.mlp.down_proj",
            shape=(1024, 1024),
            dtype="BF16",
            parameters=1_048_576,
            role=TensorRole.MLP,
            quantizable=True,
            file="model.safetensors",
            current_precision="bf16",
        ),
        TensorSpec(
            name="lm_head.weight",
            module_path="lm_head",
            shape=(256, 1024),
            dtype="BF16",
            parameters=262_144,
            role=TensorRole.LM_HEAD,
            quantizable=True,
            file="model.safetensors",
            current_precision="bf16",
        ),
    ]
    profile = ArchitectureProfile(
        adapter_id="qwen36-v1",
        product_family="qwen3.6",
        config_model_type="qwen3_5",
        support_level=ArchitectureSupportLevel.SUPPORTED,
        support_tier=SupportTier.CONVERTIBLE,
        optimization_scope=OptimizationScope.TEXT_PATH,
        dense=True,
        text_layer_count=1,
    )
    inventory = Inventory(
        model=ModelIdentity(model_id="Qwen/Qwen3.6-test", revision="a" * 40),
        architecture_profile=profile,
        tensors=tensors,
        total_parameters=sum(tensor.parameters for tensor in tensors),
        quantizable_parameters=sum(tensor.parameters for tensor in tensors),
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="b" * 64,
    )
    prior = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    report = SensitivityReport.model_validate(
        {
            **prior.model_dump(mode="python"),
            "evidence_kind": EvidenceKind.MEASURED,
            "calibration": CalibrationEvidence(
                dataset_id="clean-room-test",
                dataset_sha256="c" * 64,
                samples=1,
                domains=["coding"],
                sequence_length=128,
                backend="test-measured-probe",
                reference="unit-test",
            ),
        }
    )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=8.0,
        ),
    )
    return report, plan


def _legacy_payload(plan: QuantizationPlan) -> dict[str, object]:
    payload = plan.model_dump(mode="json")
    payload["target_class"] = "4bit"
    payload["weight_distribution"] = {"4bit": {"parameters": 1, "fraction": 1.0}}
    payload["mtp_distribution"] = {"bf16": {"parameters": 1, "fraction": 1.0}}
    return payload


def test_replay_repairs_legacy_summaries_and_rebinds_exact_measured_evidence() -> None:
    report, plan = _report_and_plan()
    replayed = replay_measured_plan(
        report,
        _legacy_payload(plan),
        source_file_sha256="c" * 64,
    )

    assert replayed.target_class == "8bit"
    assert replayed.analysis_sha256 == stable_sha256(report)
    assert replayed.evidence_kind is EvidenceKind.MEASURED
    assert replayed.assignments == [
        assignment.model_copy(
            update={
                "reason": "exact measured-plan replay from checksum-bound source",
                "strategy_metadata": {
                    **replayed.assignments[index].strategy_metadata,
                },
            }
        )
        for index, assignment in enumerate(plan.assignments)
    ]
    assert sum(share.fraction for share in replayed.weight_distribution.values()) == pytest.approx(
        1.0
    )
    assert replayed.mtp_distribution == {}
    assert any("cccccccccccc" in warning for warning in replayed.warnings)


def test_replay_fails_closed_on_metric_or_identity_drift() -> None:
    report, plan = _report_and_plan()
    payload = _legacy_payload(plan)
    changed_metrics = deepcopy(payload)
    assignments = changed_metrics["assignments"]
    assert isinstance(assignments, list)
    first = assignments[0]
    assert isinstance(first, dict)
    metrics = first["metrics"]
    assert isinstance(metrics, dict)
    metrics["output_kl"] = float(metrics["output_kl"]) + 0.01
    with pytest.raises(PlanningError, match="metrics differ"):
        replay_measured_plan(report, changed_metrics, source_file_sha256="d" * 64)

    changed_model = report.model_copy(
        update={"model": report.model.model_copy(update={"model_id": "Qwen/other"})}
    )
    with pytest.raises(PlanningError, match="source model differs"):
        replay_measured_plan(changed_model, payload, source_file_sha256="d" * 64)


def test_plan_replay_cli_writes_checksum_bound_plan(tmp_path: Path) -> None:
    report, plan = _report_and_plan()
    report_path = tmp_path / "sensitivity.json"
    source_path = tmp_path / "legacy-plan.json"
    output_path = tmp_path / "replayed-plan.json"
    write_data(report_path, report)
    write_data(source_path, _legacy_payload(plan))

    assert (
        main(
            [
                "plan-replay",
                "--sensitivity",
                str(report_path),
                "--source-plan",
                str(source_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    replayed = load_model(output_path, QuantizationPlan)
    assert replayed.analysis_sha256 == stable_sha256(report)
    assert any(file_sha256(source_path) in warning for warning in replayed.warnings)
