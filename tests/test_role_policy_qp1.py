"""QP1: role-aware preferences and refinement holdout binding."""

from __future__ import annotations

from pathlib import Path

import pytest

from axquant.cli import main
from axquant.errors import RefinementError
from axquant.planner import plan_quantization
from axquant.refinement import (
    _complete_objective_loss,
    refine_candidates,
    select_complete_candidate,
)
from axquant.role_policy import (
    method_preference_rank,
    prefer_method_on_tie,
    ranking_loss,
    role_preferences_active,
)
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CalibrationEvidence,
    CandidateMeasurement,
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    EvidenceKind,
    HardwareProfile,
    MetricVector,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QuantMethod,
    RefinementConfig,
    RefinementMeasurementSet,
    RefinementResult,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
)
from axquant.serde import load_model, stable_sha256, write_data


def _tensor(name: str, role: TensorRole, parameters: int) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=name.removesuffix(".weight"),
        shape=(parameters // 8, 8),
        dtype="BF16",
        parameters=parameters,
        role=role,
        quantizable=role not in {TensorRole.NORM, TensorRole.LM_HEAD},
        file="model.safetensors",
        current_precision="bf16",
    )


def _measured_report() -> SensitivityReport:
    """Measured multi-method multi-group candidates for attention vs mlp."""
    attention = _tensor("model.layers.0.self_attn.q_proj.weight", TensorRole.ATTENTION, 4096)
    mlp = _tensor("model.layers.0.mlp.down_proj.weight", TensorRole.MLP, 8192)
    norm = _tensor("model.layers.0.input_layernorm.weight", TensorRole.NORM, 256)
    entries = []
    for tensor in (attention, mlp, norm):
        if not tensor.quantizable:
            entries.append(
                TensorSensitivity(
                    tensor=tensor,
                    candidates=[
                        CandidateMeasurement(
                            bits=16,
                            method=QuantMethod.BF16,
                            group_size=None,
                            metrics=MetricVector(),
                        )
                    ],
                )
            )
            continue
        candidates = [
            CandidateMeasurement(
                bits=16,
                method=QuantMethod.BF16,
                group_size=None,
                metrics=MetricVector(),
            )
        ]
        for bits in (4, 6, 8):
            for group_size in (32, 64, 128):
                base = (16 - bits) * 0.1 + (group_size / 64.0) * 0.02
                # AWQ slightly worse than affine for attention (within 5% margin).
                for method, bump in (
                    (QuantMethod.AFFINE, 0.0),
                    (QuantMethod.AWQ, 0.03 * base),
                    (QuantMethod.DWQ, 0.01 * base),
                ):
                    candidates.append(
                        CandidateMeasurement(
                            bits=bits,
                            method=method,
                            group_size=group_size,
                            metrics=MetricVector(
                                output_kl=base + bump,
                                hidden_state_error=base * 0.5,
                                token_disagreement=base * 0.3,
                            ),
                        )
                    )
        entries.append(TensorSensitivity(tensor=tensor, candidates=candidates))
    return SensitivityReport(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        architecture_profile=ArchitectureProfile(
            support_level=ArchitectureSupportLevel.SUPPORTED,
            product_family="qwen3.6",
            optimization_scope=OptimizationScope.TEXT_PATH,
            adapter_id="qwen36-v1",
            text_layer_count=2,
        ),
        profile=ProfileName.AGENT_CODING,
        evidence_kind=EvidenceKind.MEASURED,
        inventory_sha256="a" * 64,
        calibration=CalibrationEvidence(
            dataset_id="ref",
            dataset_sha256="b" * 64,
            samples=128,
            domains=["general"],
            sequence_length=128,
            backend="test",
            reference="test",
        ),
        entries=entries,
    )


def test_role_preferences_active_only_for_measured() -> None:
    assert role_preferences_active(EvidenceKind.MEASURED)
    assert not role_preferences_active(EvidenceKind.ARCHITECTURE_PRIOR)


def test_prefer_awq_within_margin_for_attention() -> None:
    assert prefer_method_on_tie(
        TensorRole.ATTENTION,
        current_method=QuantMethod.AFFINE,
        current_loss=1.0,
        candidate_method=QuantMethod.AWQ,
        candidate_loss=1.04,
        evidence_kind=EvidenceKind.MEASURED,
    )
    assert not prefer_method_on_tie(
        TensorRole.ATTENTION,
        current_method=QuantMethod.AFFINE,
        current_loss=1.0,
        candidate_method=QuantMethod.AWQ,
        candidate_loss=1.04,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
    )


def test_gptq_preference_rank_follows_awq() -> None:
    assert method_preference_rank(TensorRole.ATTENTION, QuantMethod.AWQ) == 0
    assert method_preference_rank(TensorRole.ATTENTION, QuantMethod.GPTQ) == 1
    assert method_preference_rank(TensorRole.ATTENTION, QuantMethod.AFFINE) == 2
    assert method_preference_rank(TensorRole.MLP, QuantMethod.GPTQ) == 3


def test_measured_plan_prefers_awq_for_attention_when_within_margin() -> None:
    report = _measured_report()
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=5.5,
            candidate_bits=(4, 6, 8, 16),
            candidate_group_sizes=(32, 64, 128),
            hardware=HardwareProfile(),
            allow_unmeasured=False,
        ),
    )
    attention = next(item for item in plan.assignments if item.role == TensorRole.ATTENTION)
    # At equal storage key, measured attention should pick AWQ within margin.
    assert attention.method == QuantMethod.AWQ
    assert attention.scale_strategy.value == "channel-awq"
    # Protection floor still holds for norms.
    norm = next(item for item in plan.assignments if item.role == TensorRole.NORM)
    assert norm.bits == 16


def test_measured_plan_selects_gptq_when_clearly_better() -> None:
    report = _measured_report()
    mlp_entry = next(entry for entry in report.entries if entry.tensor.role == TensorRole.MLP)
    for candidate in list(mlp_entry.candidates):
        if candidate.method != QuantMethod.AFFINE or candidate.bits == 16:
            continue
        mlp_entry.candidates.append(
            CandidateMeasurement(
                bits=candidate.bits,
                method=QuantMethod.GPTQ,
                group_size=candidate.group_size,
                metrics=candidate.metrics.model_copy(
                    update={
                        "output_kl": candidate.metrics.output_kl * 0.5,
                        "hidden_state_error": candidate.metrics.hidden_state_error * 0.5,
                        "token_disagreement": candidate.metrics.token_disagreement * 0.5,
                    }
                ),
            )
        )
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=5.5,
            candidate_bits=(4, 6, 8, 16),
            candidate_group_sizes=(32, 64, 128),
            hardware=HardwareProfile(),
            allow_unmeasured=False,
        ),
    )
    mlp = next(item for item in plan.assignments if item.role == TensorRole.MLP)
    # GPTQ loss is far outside the AFFINE preference margin, so it wins its key.
    assert mlp.method == QuantMethod.GPTQ
    assert mlp.scale_strategy.value == "gptq-hessian"
    assert mlp.outlier_strategy.value == "none"


def test_ranking_loss_discounts_preferred_group() -> None:
    base = 1.0
    discounted = ranking_loss(
        loss=base,
        role=TensorRole.ATTENTION,
        method=QuantMethod.AWQ,
        group_size=32,
        evidence_kind=EvidenceKind.MEASURED,
    )
    assert discounted < base
    prior = ranking_loss(
        loss=base,
        role=TensorRole.ATTENTION,
        method=QuantMethod.AWQ,
        group_size=32,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
    )
    assert prior == base


def test_refine_proxy_labels_development_evidence() -> None:
    report = _measured_report()
    result = refine_candidates(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=6.0,
            allow_unmeasured=False,
            hardware=HardwareProfile(),
        ),
        RefinementConfig(top_n=1, max_iterations=1, evaluation_budget=5),
    )
    assert result.selection_basis == "proxy"
    assert result.evidence_label == "proxy-development"
    assert any("development evidence" in warning for warning in result.warnings)


def test_holdout_digest_mismatch_fails_closed() -> None:
    report = _measured_report()
    result = refine_candidates(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=6.0,
            hardware=HardwareProfile(),
        ),
        RefinementConfig(
            top_n=1,
            max_iterations=1,
            evaluation_budget=5,
            holdout_measurement_set_sha256="c" * 64,
        ),
    )
    hardware = CompleteCandidateHardware(
        device_name="test",
        chip="M-test",
        unified_memory_bytes=64 * 1024**3,
        os_version="15.0",
        ax_engine_version="0.0.0",
        mlx_version="0.0.0",
        mlx_lm_version="0.0.0",
        power_mode="default",
        kernel_fallbacks=0,
    )
    measurement = CompleteCandidateMeasurement(
        candidate_id=result.selected_candidate_id,
        candidate_model=report.model,
        profile=report.profile,
        plan_sha256=stable_sha256(result.selected_plan),
        artifact_manifest_sha256="d" * 64,
        quality_comparison_sha256="e" * 64,
        validation_sha256="f" * 64,
        measured_bpw=5.0,
        objective_loss=_complete_objective_loss(
            result.selected_plan,
            quality_retention=0.99,
            perplexity_ratio=1.0,
            mtp_acceptance_retention=1.0,
            mtp_speedup=1.2,
            peak_memory_ratio=0.5,
        ),
        quality_retention=0.99,
        perplexity_ratio=1.0,
        mtp_acceptance_retention=1.0,
        mtp_speedup=1.2,
        peak_memory_ratio=0.5,
        hardware=hardware,
        validation_passed=True,
    )
    measurements = RefinementMeasurementSet(
        refinement_sha256=stable_sha256(result),
        evaluator_version="test",
        measurements=[measurement],
    )
    assert stable_sha256(measurements) != "c" * 64
    with pytest.raises(RefinementError, match="holdout measurement set digest"):
        select_complete_candidate(result, measurements)


def test_holdout_digest_match_binds_selection(tmp_path: Path) -> None:
    report = _measured_report()
    result = refine_candidates(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=6.0,
            hardware=HardwareProfile(),
        ),
        RefinementConfig(top_n=1, max_iterations=1, evaluation_budget=5),
    )
    hardware = CompleteCandidateHardware(
        device_name="test",
        chip="M-test",
        unified_memory_bytes=64 * 1024**3,
        os_version="15.0",
        ax_engine_version="0.0.0",
        mlx_version="0.0.0",
        mlx_lm_version="0.0.0",
        power_mode="default",
        kernel_fallbacks=0,
    )
    measurement = CompleteCandidateMeasurement(
        candidate_id=result.selected_candidate_id,
        candidate_model=report.model,
        profile=report.profile,
        plan_sha256=stable_sha256(result.selected_plan),
        artifact_manifest_sha256="d" * 64,
        quality_comparison_sha256="e" * 64,
        validation_sha256="f" * 64,
        measured_bpw=5.0,
        objective_loss=_complete_objective_loss(
            result.selected_plan,
            quality_retention=0.99,
            perplexity_ratio=1.0,
            mtp_acceptance_retention=1.0,
            mtp_speedup=1.2,
            peak_memory_ratio=0.5,
        ),
        quality_retention=0.99,
        perplexity_ratio=1.0,
        mtp_acceptance_retention=1.0,
        mtp_speedup=1.2,
        peak_memory_ratio=0.5,
        hardware=hardware,
        validation_passed=True,
    )
    measurements = RefinementMeasurementSet(
        refinement_sha256=stable_sha256(result),
        evaluator_version="test",
        measurements=[measurement],
    )
    digest = stable_sha256(measurements)
    bound_config = result.config.model_copy(update={"holdout_measurement_set_sha256": digest})
    bound = result.model_copy(update={"config": bound_config})
    selected = select_complete_candidate(bound, measurements)
    assert selected.selection_basis == "complete-model"
    assert selected.evidence_label == "holdout-bound"
    assert selected.holdout_measurement_set_sha256 == digest

    refinement_path = tmp_path / "refinement.json"
    measurements_path = tmp_path / "measurements.json"
    output_path = tmp_path / "selected.json"
    write_data(refinement_path, bound)
    write_data(measurements_path, measurements)
    assert (
        main(
            [
                "refine-select",
                "--refinement",
                str(refinement_path),
                "--measurements",
                str(measurements_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    cli_selected = load_model(output_path, RefinementResult)
    assert cli_selected.holdout_measurement_set_sha256 == digest
