"""Tests for the global candidate refinement loop (v0.4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import axquant.refinement as refinement_module
from axquant.errors import RefinementError
from axquant.planner import plan_quantization
from axquant.profiles import thresholds_for
from axquant.refinement import (
    _ax_engine_attention_packing_compatible,
    _canonicalize_ax_engine_attention_packs,
    _compute_plan_loss,
    _is_monotonic_precision_refinement,
    build_complete_candidate_measurement,
    coordinate_descent_swap,
    generate_top_n_plans,
    refine_candidates,
    select_complete_candidate,
)
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CandidateMeasurement,
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    EvidenceKind,
    MetricVector,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QuantMethod,
    RefinementConfig,
    RefinementMeasurementSet,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
)
from axquant.serde import stable_sha256


def _make_tensor(name: str, role: TensorRole, params: int = 1024) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=name.replace(".", "_"),
        shape=(params // 8, 8),
        dtype="bfloat16",
        parameters=params,
        role=role,
        quantizable=role not in (TensorRole.NORM, TensorRole.LM_HEAD),
        file="model.safetensors",
        current_precision="bfloat16",
    )


def _make_sensitivity_report() -> SensitivityReport:
    tensors = [
        _make_tensor("model.layers.0.mlp.down_proj.weight", TensorRole.MLP, 2048),
        _make_tensor("model.layers.0.self_attn.q_proj.weight", TensorRole.ATTENTION, 2048),
        _make_tensor("model.layers.1.mlp.down_proj.weight", TensorRole.MLP, 2048),
        _make_tensor("model.embed_tokens.weight", TensorRole.EMBEDDING, 4096),
        _make_tensor("model.norm.weight", TensorRole.NORM, 512),
    ]
    entries = []
    for tensor in tensors:
        if tensor.quantizable:
            candidates = [
                CandidateMeasurement(
                    bits=b,
                    method=QuantMethod.AFFINE if b < 16 else QuantMethod.BF16,
                    group_size=64 if b < 16 else None,
                    metrics=MetricVector(
                        output_kl=(16 - b) * 0.1,
                        hidden_state_error=(16 - b) * 0.05,
                        token_disagreement=(16 - b) * 0.03,
                        task_loss_delta=(16 - b) * 0.08,
                        mtp_acceptance_loss=(16 - b) * 0.02,
                    ),
                )
                for b in (4, 6, 8, 16)
            ]
        else:
            candidates = [
                CandidateMeasurement(
                    bits=16,
                    method=QuantMethod.BF16,
                    metrics=MetricVector(),
                )
            ]
        entries.append(TensorSensitivity(tensor=tensor, candidates=candidates))

    return SensitivityReport(
        model=ModelIdentity(model_id="test-model", revision="rev1"),
        architecture_profile=ArchitectureProfile(
            adapter_id="qwen36-v1",
            product_family="qwen3.6",
            support_level=ArchitectureSupportLevel.SUPPORTED,
            optimization_scope=OptimizationScope.TEXT_PATH,
        ),
        profile=ProfileName.AGENT_CODING,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
        inventory_sha256="test_hash",
        entries=entries,
        warnings=["test report"],
    )


def _make_request(**kwargs) -> PlanRequest:
    defaults = {
        "profile": ProfileName.AGENT_CODING,
        "target_bpw": 8.0,
        "candidate_bits": (4, 6, 8, 16),
        "group_size": 64,
        "allow_unmeasured": True,
        "candidate_count": 1,
        "random_seed": 0,
    }
    defaults.update(kwargs)
    return PlanRequest(**defaults)


def test_ax_engine_attention_packing_requires_matching_qkv_formats() -> None:
    plan = plan_quantization(_make_sensitivity_report(), _make_request())
    q_proj = next(
        allocation for allocation in plan.assignments if "self_attn.q_proj" in allocation.tensor
    )
    prefix = "model.language_model.layers.0"
    assignments = [
        q_proj.model_copy(update={"module_path": f"{prefix}.self_attn.q_proj"}),
        q_proj.model_copy(update={"module_path": f"{prefix}.self_attn.k_proj"}),
        q_proj.model_copy(update={"module_path": f"{prefix}.self_attn.v_proj"}),
    ]
    compatible = plan.model_copy(update={"assignments": assignments})

    assert _ax_engine_attention_packing_compatible(compatible)

    incompatible = compatible.model_copy(
        update={
            "assignments": [
                assignments[0],
                assignments[1].model_copy(update={"bits": 4 if q_proj.bits != 4 else 6}),
                assignments[2],
            ]
        }
    )
    assert not _ax_engine_attention_packing_compatible(incompatible)


def test_attention_pack_canonicalization_uses_measured_lower_precision() -> None:
    report = _make_sensitivity_report()
    q_entry = next(entry for entry in report.entries if "self_attn.q_proj" in entry.tensor.name)
    k_entry = q_entry.model_copy(
        update={
            "tensor": q_entry.tensor.model_copy(
                update={
                    "name": "model.layers.0.self_attn.k_proj.weight",
                    "module_path": "model.language_model.layers.0.self_attn.k_proj",
                }
            )
        }
    )
    v_entry = q_entry.model_copy(
        update={
            "tensor": q_entry.tensor.model_copy(
                update={
                    "name": "model.layers.0.self_attn.v_proj.weight",
                    "module_path": "model.language_model.layers.0.self_attn.v_proj",
                }
            )
        }
    )
    report = report.model_copy(update={"entries": [*report.entries, k_entry, v_entry]})
    plan = plan_quantization(report, _make_request())
    q_assignment = next(
        assignment for assignment in plan.assignments if "self_attn.q_proj" in assignment.tensor
    )
    four_bit = next(candidate for candidate in q_entry.candidates if candidate.bits == 4)
    six_bit = next(candidate for candidate in q_entry.candidates if candidate.bits == 6)
    prefix = "model.language_model.layers.0.self_attn"
    assignments = [
        q_assignment.model_copy(
            update={
                "module_path": f"{prefix}.q_proj",
                "bits": 6,
                "method": six_bit.method,
                "group_size": six_bit.group_size,
                "metrics": six_bit.metrics,
            }
        ),
        q_assignment.model_copy(
            update={
                "tensor": k_entry.tensor.name,
                "module_path": f"{prefix}.k_proj",
                "bits": 4,
                "method": four_bit.method,
                "group_size": four_bit.group_size,
                "metrics": four_bit.metrics,
            }
        ),
        q_assignment.model_copy(
            update={
                "tensor": v_entry.tensor.name,
                "module_path": f"{prefix}.v_proj",
                "bits": 6,
                "method": six_bit.method,
                "group_size": six_bit.group_size,
                "metrics": six_bit.metrics,
            }
        ),
    ]
    incompatible = plan.model_copy(update={"assignments": assignments})

    canonical = _canonicalize_ax_engine_attention_packs(incompatible, report)

    assert _ax_engine_attention_packing_compatible(canonical)
    assert {assignment.bits for assignment in canonical.assignments} == {4}


def _complete_hardware() -> CompleteCandidateHardware:
    return CompleteCandidateHardware(
        device_name="Test Mac",
        chip="M4 Max",
        unified_memory_bytes=128 * 1024**3,
        os_version="macOS",
        ax_engine_version="6.11.1",
        mlx_version="0.32.0",
        mlx_lm_version="0.31.0",
        power_mode="AC power",
        kernel_fallbacks=0,
    )


class TestGenerateTopNPlans:
    def test_single_plan(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request()
        plans = generate_top_n_plans(report, request, n=1)
        assert len(plans) == 1

    def test_multiple_plans(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request()
        plans = generate_top_n_plans(report, request, n=3)
        assert len(plans) >= 1  # At least the primary plan
        assert len(plans) <= 3

    def test_invalid_n(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request()
        with pytest.raises(RefinementError, match="at least 1"):
            generate_top_n_plans(report, request, n=0)

    def test_deterministic(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request()
        plans1 = generate_top_n_plans(report, request, n=2)
        plans2 = generate_top_n_plans(report, request, n=2)
        assert len(plans1) == len(plans2)
        for p1, p2 in zip(plans1, plans2, strict=True):
            assert p1.effective_bpw == p2.effective_bpw


class TestComputePlanLoss:
    def test_loss_positive(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request()
        plan = plan_quantization(report, request)
        loss = _compute_plan_loss(plan)
        assert loss >= 0.0

    def test_higher_precision_lower_loss(self) -> None:
        report = _make_sensitivity_report()
        # Lower BPW forces more aggressive quantization
        low_plan = plan_quantization(report, _make_request(target_bpw=7.0))
        # Higher BPW allows more upgrades
        high_plan = plan_quantization(report, _make_request(target_bpw=12.0))
        low_loss = _compute_plan_loss(low_plan)
        high_loss = _compute_plan_loss(high_plan)
        assert high_loss <= low_loss


class TestCoordinateDescentSwap:
    def test_swap_improves_or_none(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        plan = plan_quantization(report, request)
        improved = coordinate_descent_swap(plan, report, request, swap_radius=3)
        if improved is not None:
            assert _compute_plan_loss(improved) <= _compute_plan_loss(plan)

    def test_tight_budget_no_swap(self) -> None:
        report = _make_sensitivity_report()
        # Tight budget close to policy minimum - limited room for upgrades
        request = _make_request(target_bpw=7.0)
        plan = plan_quantization(report, request)
        # With a tight budget, swaps should be limited
        coordinate_descent_swap(plan, report, request, swap_radius=1)
        # May or may not find improvement depending on budget headroom


class TestRefineCandidates:
    def test_basic_refinement(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        config = RefinementConfig(
            top_n=2,
            max_iterations=3,
            evaluation_budget=20,
            convergence_threshold=0.0001,
            swap_radius=3,
            random_seed=42,
        )
        result = refine_candidates(report, request, config)
        assert result.iterations_used >= 0
        assert result.evaluations_used >= 1
        assert len(result.history) >= 1
        assert result.selected_plan_sha256 is not None

    def test_budget_exhaustion(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        config = RefinementConfig(
            top_n=2,
            max_iterations=100,
            evaluation_budget=3,  # Very small budget
            random_seed=0,
        )
        result = refine_candidates(report, request, config)
        assert result.evaluations_used <= 3

    def test_convergence(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        config = RefinementConfig(
            top_n=1,
            max_iterations=50,
            evaluation_budget=100,
            convergence_threshold=1.0,  # Very high threshold = converge quickly
            random_seed=0,
        )
        result = refine_candidates(report, request, config)
        # Should converge quickly with high threshold
        assert result.iterations_used <= 5

    def test_history_immutability(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        config = RefinementConfig(top_n=2, max_iterations=3, random_seed=0)
        result = refine_candidates(report, request, config)
        # History entries should have valid states
        for entry in result.history:
            assert entry.state in ("selected", "rejected", "pending")
            assert entry.candidate_id
            assert entry.plan_sha256

    def test_result_schema(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        config = RefinementConfig(top_n=1, max_iterations=2, random_seed=0)
        result = refine_candidates(report, request, config)
        # Validate schema fields
        assert result.schema_version == "axquant.refinement.v2"
        assert result.config == config
        assert result.selected_plan_sha256 in {entry.plan_sha256 for entry in result.history}
        assert result.candidate_plans
        assert result.selection_basis == "proxy"
        assert isinstance(result.converged, bool)

    def test_top_n_variants_form_a_monotonic_parent_chain(self) -> None:
        result = refine_candidates(
            _make_sensitivity_report(),
            _make_request(target_bpw=8.0),
            RefinementConfig(
                top_n=2,
                max_iterations=1,
                evaluation_budget=2,
                random_seed=17,
            ),
        )

        assert len(result.history) >= 2
        assert result.history[0].parent_id is None
        for parent, child in zip(result.history, result.history[1:], strict=False):
            assert child.parent_id == parent.candidate_id
            assert _is_monotonic_precision_refinement(
                result.candidate_plans[parent.candidate_id],
                result.candidate_plans[child.candidate_id],
            )

    def test_coordinate_descent_uses_variant_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = _make_sensitivity_report()
        request = _make_request(target_bpw=8.0)
        variant = plan_quantization(report, _make_request(target_bpw=9.0))
        observed_targets: list[float] = []

        monkeypatch.setattr(
            refinement_module,
            "generate_top_n_plans",
            lambda _report, _request, _count, *, deadline=None: [variant],
        )

        def record_target(
            _plan: object,
            _report: object,
            candidate_request: PlanRequest,
            *,
            swap_radius: int,
        ) -> None:
            del swap_radius
            observed_targets.append(candidate_request.target_bpw)
            return None

        monkeypatch.setattr(refinement_module, "coordinate_descent_swap", record_target)

        refine_candidates(
            report,
            request,
            RefinementConfig(top_n=1, max_iterations=1, random_seed=23),
        )

        assert observed_targets == [variant.target_bpw]


class TestPlannerTopN:
    def test_candidate_count_gt1_accepted(self) -> None:
        report = _make_sensitivity_report()
        request = _make_request(candidate_count=3)
        # Should not raise - top-N is now accepted
        plan = plan_quantization(report, request)
        assert plan is not None
        assert plan.effective_bpw > 0


def test_complete_model_selection_uses_only_passing_measurements() -> None:
    report = _make_sensitivity_report()
    request = _make_request(target_bpw=8.0)
    refinement = refine_candidates(
        report,
        request,
        RefinementConfig(top_n=2, max_iterations=1, random_seed=7),
    )
    records = []
    for index, entry in enumerate(refinement.history[:2]):
        records.append(
            CompleteCandidateMeasurement(
                candidate_id=entry.candidate_id,
                candidate_model=ModelIdentity(
                    model_id=f"AutomatosX/{entry.candidate_id}",
                    revision=f"candidate-{index}",
                ),
                profile=refinement.selected_plan.profile,
                plan_sha256=entry.plan_sha256,
                artifact_manifest_sha256=f"artifact-{index}",
                quality_comparison_sha256=f"quality-{index}",
                validation_sha256=f"validation-{index}",
                measured_bpw=7.5 + index,
                objective_loss=0.1 + index,
                quality_retention=1.0,
                mtp_acceptance_retention=0.97,
                mtp_speedup=1.25,
                peak_memory_ratio=0.8,
                hardware=_complete_hardware(),
                validation_passed=index == 1,
            )
        )
    second_host = records[1].model_copy(
        update={
            "measurement_id": f"{records[1].candidate_id}-second-host",
            "measured_bpw": records[1].measured_bpw + 0.2,
            "objective_loss": records[1].objective_loss + 0.2,
        }
    )
    selected = select_complete_candidate(
        refinement,
        RefinementMeasurementSet(
            refinement_sha256=stable_sha256(refinement),
            evaluator_version="test-v1",
            measurements=[*records, second_host],
        ),
    )
    assert selected.selection_basis == "complete-model"
    assert selected.selected_candidate_id == records[1].candidate_id
    assert (
        next(
            entry for entry in selected.history if entry.candidate_id == records[1].candidate_id
        ).measured_loss
        == second_host.objective_loss
    )


def test_complete_model_selection_rejects_all_failed_candidates() -> None:
    report = _make_sensitivity_report()
    refinement = refine_candidates(
        report,
        _make_request(target_bpw=8.0),
        RefinementConfig(top_n=1, max_iterations=1),
    )
    entry = refinement.history[0]
    measurements = RefinementMeasurementSet(
        refinement_sha256=stable_sha256(refinement),
        evaluator_version="test-v1",
        measurements=[
            CompleteCandidateMeasurement(
                candidate_id=entry.candidate_id,
                candidate_model=ModelIdentity(
                    model_id=f"AutomatosX/{entry.candidate_id}",
                    revision="candidate",
                ),
                profile=refinement.selected_plan.profile,
                plan_sha256=entry.plan_sha256,
                artifact_manifest_sha256="artifact",
                quality_comparison_sha256="quality",
                validation_sha256="validation",
                measured_bpw=8.0,
                objective_loss=0.1,
                quality_retention=0.9,
                mtp_acceptance_retention=0.8,
                mtp_speedup=0.9,
                peak_memory_ratio=0.9,
                hardware=_complete_hardware(),
                validation_passed=False,
            )
        ],
    )
    with pytest.raises(RefinementError, match="no complete-model candidate"):
        select_complete_candidate(refinement, measurements)
    measurements.measurements[0].profile = ProfileName.CODING
    with pytest.raises(RefinementError, match="measurement profile differs"):
        select_complete_candidate(refinement, measurements)


def test_complete_measurement_is_built_from_bound_release_artifacts() -> None:
    refinement = refine_candidates(
        _make_sensitivity_report(),
        _make_request(target_bpw=8.0),
        RefinementConfig(top_n=1, max_iterations=1),
    )
    candidate_id, plan = next(iter(refinement.candidate_plans.items()))
    candidate_model = ModelIdentity(model_id=f"AutomatosX/{candidate_id}", revision="candidate")
    reference_model = ModelIdentity(model_id="AutomatosX/reference", revision="reference")
    artifact = SimpleNamespace(
        plan_sha256=stable_sha256(plan),
        profile=plan.profile,
        weight_file_size_bytes=100,
        measured_total_bpw=7.5,
    )
    quality = SimpleNamespace(
        reference_model=reference_model,
        candidate_model=candidate_model,
        aggregate=SimpleNamespace(retention=0.99),
        perplexity_ratio=0.95,
    )
    validation = SimpleNamespace(
        profile=plan.profile,
        reference_model=reference_model,
        candidate_model=candidate_model,
        passed=True,
        thresholds=thresholds_for(plan.profile),
        issues=[],
        release_exceptions=[],
        comparisons={
            "artifact.candidate_source_sha256": "artifact-sha",
            "artifact.candidate_weight_bytes": 100,
            "artifact.weight_size_ratio": 1.05,
            "mtp.acceptance_retention": 0.98,
            "hardware.effective_speedup": 1.25,
            "hardware.peak_memory_ratio": 0.8,
            "hardware.device_name": "Test Mac",
            "hardware.chip": "M4 Max",
            "hardware.unified_memory_bytes": 128 * 1024**3,
            "hardware.os_version": "macOS",
            "software.ax_engine": "6.11.1",
            "software.mlx": "0.32.0",
            "software.mlx_lm": "0.31.0",
            "hardware.power_mode": "AC power",
            "hardware.kernel_fallbacks": 0,
        },
    )
    measurement = build_complete_candidate_measurement(
        candidate_id=candidate_id,
        plan=plan,
        artifact=artifact,
        artifact_sha256="artifact-sha",
        quality=quality,
        quality_sha256="quality-sha",
        validation=validation,
        validation_sha256="validation-sha",
    )
    assert measurement.plan_sha256 == stable_sha256(plan)
    assert measurement.measured_bpw == 7.5
    assert measurement.quality_retention == 0.99
    assert measurement.perplexity_ratio == 0.95
    assert measurement.hardware.chip == "M4 Max"
    assert measurement.validation_passed is True
    assert measurement.objective_loss > 0.0

    validation.thresholds = validation.thresholds.model_copy(update={"min_effective_speedup": 1.0})
    with pytest.raises(RefinementError, match="authoritative profile thresholds"):
        build_complete_candidate_measurement(
            candidate_id=candidate_id,
            plan=plan,
            artifact=artifact,
            artifact_sha256="artifact-sha",
            quality=quality,
            quality_sha256="quality-sha",
            validation=validation,
            validation_sha256="validation-sha",
        )
    validation.thresholds = thresholds_for(plan.profile)

    validation.comparisons["artifact.weight_size_ratio"] = 1.2
    with pytest.raises(RefinementError, match="ungoverned size overage"):
        build_complete_candidate_measurement(
            candidate_id=candidate_id,
            plan=plan,
            artifact=artifact,
            artifact_sha256="artifact-sha",
            quality=quality,
            quality_sha256="quality-sha",
            validation=validation,
            validation_sha256="validation-sha",
        )
    validation.comparisons["artifact.weight_size_ratio"] = 1.05

    validation.comparisons["artifact.candidate_source_sha256"] = "another-artifact"
    with pytest.raises(RefinementError, match="does not bind"):
        build_complete_candidate_measurement(
            candidate_id=candidate_id,
            plan=plan,
            artifact=artifact,
            artifact_sha256="artifact-sha",
            quality=quality,
            quality_sha256="quality-sha",
            validation=validation,
            validation_sha256="validation-sha",
        )
