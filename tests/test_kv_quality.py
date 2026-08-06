"""Measured-KV serving-quality report (RM-21 / WS-6, report-only)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from axquant.errors import ArtifactError
from axquant.kv_quality import build_kv_serving_quality_report
from axquant.planner import allocate_kv_cache
from axquant.schema import (
    KvServingQualityProfileResult,
    KvServingQualityReport,
    ModelIdentity,
    ProfileName,
)
from axquant.serde import stable_sha256


def _result(
    profile: ProfileName,
    context_tokens: int,
    *,
    quantized: float = 0.97,
    bf16: float = 1.0,
) -> KvServingQualityProfileResult:
    return KvServingQualityProfileResult(
        profile=profile,
        context_tokens=context_tokens,
        bf16_kv_score=bf16,
        quantized_kv_score=quantized,
        retention=quantized / bf16,
        evaluation_sha256="e" * 64,
    )


def _dual_profile_matrix() -> list[KvServingQualityProfileResult]:
    return [
        _result(ProfileName.GENERAL, 512),
        _result(ProfileName.GENERAL, 8192),
        _result(ProfileName.AGENT_CODING, 512),
        _result(ProfileName.AGENT_CODING, 8192),
    ]


def _passing_execution() -> dict[str, object]:
    return {
        "ok": True,
        "per_layer_execution": True,
        "quantized_layers_active": 16,
    }


def test_report_binds_plan_digest_and_dual_profile_matrix() -> None:
    kv_plan = allocate_kv_cache(20, default_bits=4, group_size=64)
    report = build_kv_serving_quality_report(
        model=ModelIdentity(model_id="org/model", revision="a" * 40),
        kv_plan=kv_plan,
        execution_summary=_passing_execution(),
        results=_dual_profile_matrix(),
    )
    assert report.kv_plan_sha256 == stable_sha256(kv_plan)
    assert report.report_only is True
    assert report.runtime == "mlx-lm-kv"
    assert len(report.results) == 4


def test_report_requires_exact_per_layer_execution() -> None:
    kv_plan = allocate_kv_cache(8, default_bits=4, group_size=64)
    with pytest.raises(ArtifactError, match="exact per-layer execution"):
        build_kv_serving_quality_report(
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            kv_plan=kv_plan,
            execution_summary={**_passing_execution(), "per_layer_execution": False},
            results=_dual_profile_matrix(),
        )
    with pytest.raises(ArtifactError, match="passing mlx-lm-kv execution"):
        build_kv_serving_quality_report(
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            kv_plan=kv_plan,
            execution_summary={**_passing_execution(), "ok": False},
            results=_dual_profile_matrix(),
        )
    with pytest.raises(ArtifactError, match="active quantized KV layer"):
        build_kv_serving_quality_report(
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            kv_plan=kv_plan,
            execution_summary={**_passing_execution(), "quantized_layers_active": 0},
            results=_dual_profile_matrix(),
        )


def test_schema_requires_short_and_long_context_for_both_profiles() -> None:
    kv_plan = allocate_kv_cache(8, default_bits=4, group_size=64)
    single_context = [
        _result(ProfileName.GENERAL, 512),
        _result(ProfileName.AGENT_CODING, 512),
    ]
    with pytest.raises(ValidationError, match="short and long context"):
        build_kv_serving_quality_report(
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            kv_plan=kv_plan,
            execution_summary=_passing_execution(),
            results=single_context,
        )


def test_schema_rejects_inconsistent_retention() -> None:
    with pytest.raises(ValidationError, match="retention is inconsistent"):
        KvServingQualityReport(
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            kv_plan_sha256="f" * 64,
            results=[
                *_dual_profile_matrix()[:3],
                _dual_profile_matrix()[3].model_copy(update={"retention": 0.5}),
            ],
        )


def test_kv_serving_quality_cli_binds_the_executed_plan(tmp_path) -> None:
    import json

    from axquant.cli import main
    from axquant.planner import plan_quantization
    from axquant.schema import (
        ArchitectureProfile,
        ArchitectureSupportLevel,
        CandidateMeasurement,
        EvidenceKind,
        HardwareProfile,
        MetricVector,
        OptimizationScope,
        PlanRequest,
        QuantMethod,
        SensitivityReport,
        TensorRole,
        TensorSensitivity,
        TensorSpec,
    )
    from axquant.serde import load_model, write_data

    tensor = TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(16, 1024),
        dtype="BF16",
        parameters=16384,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )
    report = SensitivityReport(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        architecture_profile=ArchitectureProfile(
            support_level=ArchitectureSupportLevel.SUPPORTED,
            product_family="qwen3.6",
            optimization_scope=OptimizationScope.TEXT_PATH,
            adapter_id="qwen36-v1",
            text_layer_count=1,
        ),
        profile=ProfileName.GENERAL,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
        inventory_sha256="a" * 64,
        entries=[
            TensorSensitivity(
                tensor=tensor,
                candidates=[
                    CandidateMeasurement(bits=16, method=QuantMethod.BF16, metrics=MetricVector()),
                    CandidateMeasurement(
                        bits=4,
                        method=QuantMethod.AFFINE,
                        group_size=64,
                        metrics=MetricVector(output_kl=0.1),
                    ),
                ],
            )
        ],
    )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            target_bpw=5.0,
            candidate_bits=(4, 16),
            allow_unmeasured=True,
            hardware=HardwareProfile(),
        ),
    )
    kv_plan = allocate_kv_cache(4, default_bits=4, group_size=64)
    plan = plan.model_copy(update={"kv_cache": kv_plan})
    plan_path = tmp_path / "axquant_plan.json"
    write_data(plan_path, plan)
    summary_path = tmp_path / "kv_execution_summary.json"
    summary_path.write_text(json.dumps(_passing_execution()), encoding="utf-8")
    results_path = tmp_path / "kv_results.json"
    results_path.write_text(
        json.dumps([result.model_dump(mode="json") for result in _dual_profile_matrix()]),
        encoding="utf-8",
    )
    output_path = tmp_path / "kv_serving_quality.json"

    assert (
        main(
            [
                "kv-serving-quality",
                "--plan",
                str(plan_path),
                "--execution-summary",
                str(summary_path),
                "--results",
                str(results_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    written = load_model(output_path, KvServingQualityReport)
    assert written.kv_plan_sha256 == stable_sha256(kv_plan)
    assert written.model.model_id == "org/model"


def test_kv_serving_quality_cli_requires_a_kv_plan(tmp_path) -> None:
    import json

    from axquant.cli import main

    plan_path = tmp_path / "plan_without_kv.json"
    plan_path.write_text(json.dumps({"schema_version": "axquant.plan.v1"}), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_passing_execution()), encoding="utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_text("[]", encoding="utf-8")
    # An unloadable plan surfaces as the CLI failure exit code.
    assert (
        main(
            [
                "kv-serving-quality",
                "--plan",
                str(plan_path),
                "--execution-summary",
                str(summary_path),
                "--results",
                str(results_path),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
