from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from axquant.cli import main
from axquant.cli._parser import _build_parser
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.joint import (
    _crossover,
    _inventory_digest,
    _kv_candidate_matches,
    _same_model_id,
    coupled_joint_loss,
    diagnose_joint_interaction,
    plan_joint_allocation,
    select_coupled_candidate,
)
from axquant.schema import (
    CalibrationEvidence,
    CandidateMeasurement,
    EvidenceKind,
    JointBudgetCandidate,
    JointInteractionReport,
    JointProxyScores,
    JointSelectionReport,
    KvLayerAllocation,
    KvLayerSensitivity,
    KvSensitivityReport,
    MetricVector,
    ProfileName,
    QualityEvaluationResult,
    QualityGenerationConfig,
    QualityMetrics,
    QualityTaskResult,
    QuantizationPlan,
    QuantMethod,
    SoftwareVersions,
)
from axquant.serde import load_model, stable_sha256, write_data


def _enable_kv_accounting(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def _versions() -> SoftwareVersions:
    return SoftwareVersions(
        axquant="1.9.0b1",
        python="3.12",
        safetensors="0.6",
        pydantic="2.11",
    )


def _quality(model_dir: Path, score: float, path: Path) -> Path:
    inventory = inspect_model(model_dir)
    write_data(
        path,
        QualityEvaluationResult(
            model=inventory.model,
            dataset_sha256="a" * 64,
            generation=QualityGenerationConfig(
                prompt_format="raw",
                max_sequence_length=32,
                max_generation_tokens=8,
            ),
            metrics=QualityMetrics(task_scores={"general": score}),
            task_results=[
                QualityTaskResult(
                    task_id="t1",
                    category="general",
                    output="ok",
                    score=score,
                    check_scores={"exact": score},
                )
            ],
            samples=1,
            evaluated_tokens=4,
            random_seed=0,
            software_versions=_versions(),
        ),
    )
    return path


def test_diagnose_joint_cli_help_is_development_only() -> None:
    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    help_text = subparsers.choices["diagnose-joint"].format_help()
    assert "cannot authorize a Hub pack or certificate" in help_text
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["diagnose-joint", "--help"])
    assert exc_info.value.code == 0


def test_diagnose_joint_without_quality_is_not_certified(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    output = tmp_path / "joint"

    exit_code = main(
        [
            "diagnose-joint",
            "--model",
            str(tiny_model_dir),
            "--max-memory",
            "2GB",
            "--contexts",
            "8,64",
            "--weight-bpws",
            "16",
            "--kv-bits",
            "4,8,16",
            "--allow-unmeasured",
            "--reserve-memory",
            "0B",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = load_model(output / "joint-interaction.json", JointInteractionReport)
    assert report.experimental is True
    assert report.certification_eligible is False
    assert report.verdict == "insufficient-measured-interaction"
    assert report.evidence_kind == "architecture_prior"
    assert report.interaction is None
    assert report.candidates
    assert (output / "joint-interaction.md").is_file()
    markdown = (output / "joint-interaction.md").read_text(encoding="utf-8")
    assert "not a certification claim" in markdown
    assert "## Estimated feasible cells" in markdown
    assert report.crossover.detected is False
    assert report.crossover.ranking_complete is False
    assert all(not candidate.ranking_available for candidate in report.candidates)
    assert all(winner.target_bpw is None for winner in report.crossover.winners)
    assert len(report.inventory_sha256) == 64
    assert len(report.weight_sensitivity_sha256) == 64
    assert report.kv_sensitivity_sha256 is None
    assert any(output.glob("weight-plan-*.json"))
    assert any(output.glob("kv-plan-*.json"))


def test_supplied_inventory_must_match_model(tiny_model_dir: Path, tmp_path: Path) -> None:
    _enable_kv_accounting(tiny_model_dir)
    inventory = inspect_model(tiny_model_dir)
    foreign = inventory.model_copy(
        update={"model": inventory.model.model_copy(update={"model_id": "not-this-model"})}
    )
    path = tmp_path / "foreign-inventory.json"
    write_data(path, foreign)
    with pytest.raises(PlanningError, match="inventory"):
        diagnose_joint_interaction(
            model_dir=tiny_model_dir,
            max_memory_bytes=2_000_000_000,
            contexts=(8,),
            weight_bpws=(16.0,),
            kv_bits=(8,),
            profile=ProfileName.GENERAL,
            output_dir=tmp_path / "inv",
            inventory_path=path,
            allow_unmeasured=True,
            reserve_bytes=0,
        )


def test_partial_quality_inputs_fail_closed(tiny_model_dir: Path, tmp_path: Path) -> None:
    _enable_kv_accounting(tiny_model_dir)
    quality = _quality(tiny_model_dir, 0.9, tmp_path / "weight-only.json")
    with pytest.raises(PlanningError, match="together"):
        diagnose_joint_interaction(
            model_dir=tiny_model_dir,
            max_memory_bytes=2_000_000_000,
            contexts=(8,),
            weight_bpws=(16.0,),
            kv_bits=(8,),
            profile=ProfileName.GENERAL,
            output_dir=tmp_path / "partial",
            allow_unmeasured=True,
            reserve_bytes=0,
            quality_weight_only_path=quality,
        )


def test_measured_quadruple_computes_signed_interaction(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    report = diagnose_joint_interaction(
        model_dir=tiny_model_dir,
        max_memory_bytes=2_000_000_000,
        contexts=(8, 64),
        weight_bpws=(16.0,),
        kv_bits=(8, 16),
        profile=ProfileName.GENERAL,
        output_dir=tmp_path / "measured",
        allow_unmeasured=True,
        reserve_bytes=0,
        interaction_threshold=0.02,
        quality_baseline_path=_quality(tiny_model_dir, 1.00, tmp_path / "b.json"),
        quality_weight_only_path=_quality(tiny_model_dir, 0.90, tmp_path / "w.json"),
        quality_kv_only_path=_quality(tiny_model_dir, 0.95, tmp_path / "k.json"),
        quality_joint_path=_quality(tiny_model_dir, 0.80, tmp_path / "j.json"),
    )

    assert report.interaction is not None
    assert report.interaction.baseline_score == pytest.approx(1.00)
    assert report.interaction.weight_only_delta == pytest.approx(0.10)
    assert report.interaction.kv_only_delta == pytest.approx(0.05)
    assert report.interaction.joint_delta == pytest.approx(0.20)
    assert report.interaction.interaction == pytest.approx(0.05)
    assert report.interaction.material is True
    assert report.verdict == "interaction-material"
    assert report.certification_eligible is False
    assert report.evidence_kind == "architecture_prior"


def test_identical_treatments_have_zero_interaction(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    report = diagnose_joint_interaction(
        model_dir=tiny_model_dir,
        max_memory_bytes=2_000_000_000,
        contexts=(8,),
        weight_bpws=(16.0,),
        kv_bits=(8,),
        profile=ProfileName.GENERAL,
        output_dir=tmp_path / "zero",
        allow_unmeasured=True,
        reserve_bytes=0,
        quality_baseline_path=_quality(tiny_model_dir, 0.80, tmp_path / "b.json"),
        quality_weight_only_path=_quality(tiny_model_dir, 0.80, tmp_path / "w.json"),
        quality_kv_only_path=_quality(tiny_model_dir, 0.80, tmp_path / "k.json"),
        quality_joint_path=_quality(tiny_model_dir, 0.80, tmp_path / "j.json"),
    )
    assert report.verdict == "interaction-small"
    assert report.interaction is not None
    assert report.interaction.interaction == pytest.approx(0.0)
    assert report.interaction.material is False


def test_mismatched_quality_suites_fail_closed(tiny_model_dir: Path, tmp_path: Path) -> None:
    _enable_kv_accounting(tiny_model_dir)
    other = _quality(tiny_model_dir, 0.90, tmp_path / "other.json")
    payload = json.loads(other.read_text(encoding="utf-8"))
    payload["dataset_sha256"] = "b" * 64
    other.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PlanningError, match="dataset_sha256"):
        diagnose_joint_interaction(
            model_dir=tiny_model_dir,
            max_memory_bytes=2_000_000_000,
            contexts=(8,),
            weight_bpws=(16.0,),
            kv_bits=(8,),
            profile=ProfileName.GENERAL,
            output_dir=tmp_path / "mismatch",
            allow_unmeasured=True,
            reserve_bytes=0,
            quality_baseline_path=_quality(tiny_model_dir, 1.00, tmp_path / "b.json"),
            quality_weight_only_path=other,
            quality_kv_only_path=_quality(tiny_model_dir, 0.95, tmp_path / "k.json"),
            quality_joint_path=_quality(tiny_model_dir, 0.80, tmp_path / "j.json"),
        )


def test_crossover_detection_requires_distinct_feasible_winners() -> None:
    def cell(
        *,
        context: int,
        bpw: float,
        bits: int,
        feasible: bool,
        proxy: float,
    ) -> JointBudgetCandidate:
        weight_bytes = 100
        kv_bytes = 20
        reserve = 0
        limit = 200 if feasible else 50
        remainder = limit - weight_bytes - kv_bytes - reserve
        return JointBudgetCandidate(
            target_bpw=bpw,
            kv_default_bits=bits,
            context_length=context,
            weight_bytes=weight_bytes,
            kv_bytes=kv_bytes,
            reserve_bytes=reserve,
            limit_bytes=limit,
            remainder_bytes=remainder,
            feasible=feasible,
            ranking_available=True,
            estimated_main_bpw=bpw,
            proxy=JointProxyScores(
                weight_output_kl=proxy,
                kv_output_kl=0.0,
                additive_output_kl=proxy,
            ),
            plan_sha256="a" * 64,
            kv_plan_sha256="b" * 64,
        )

    summary = _crossover(
        [
            cell(context=8, bpw=6.0, bits=8, feasible=True, proxy=0.1),
            cell(context=8, bpw=4.0, bits=4, feasible=True, proxy=0.3),
            cell(context=128, bpw=6.0, bits=8, feasible=False, proxy=0.1),
            cell(context=128, bpw=4.0, bits=4, feasible=True, proxy=0.3),
        ]
    )
    assert summary.detected is True
    assert summary.ranking_complete is True
    assert summary.winners[0].target_bpw == 6.0
    assert summary.winners[1].target_bpw == 4.0


def test_equal_proxy_prefers_lower_kv_bits_and_more_remainder() -> None:
    def cell(*, bits: int, remainder: int) -> JointBudgetCandidate:
        weight_bytes = 100
        kv_bytes = 20
        reserve = 0
        limit = weight_bytes + kv_bytes + reserve + remainder
        return JointBudgetCandidate(
            target_bpw=6.0,
            kv_default_bits=bits,
            context_length=8,
            weight_bytes=weight_bytes,
            kv_bytes=kv_bytes,
            reserve_bytes=reserve,
            limit_bytes=limit,
            remainder_bytes=remainder,
            feasible=True,
            ranking_available=True,
            estimated_main_bpw=6.0,
            proxy=JointProxyScores(
                weight_output_kl=0.1,
                kv_output_kl=0.0,
                additive_output_kl=0.1,
            ),
            plan_sha256="a" * 64,
            kv_plan_sha256="b" * 64,
        )

    summary = _crossover(
        [
            cell(bits=16, remainder=10),
            cell(bits=8, remainder=40),
        ]
    )
    assert summary.detected is False
    assert summary.winners[0].kv_default_bits == 8


def test_same_model_id_accepts_equivalent_paths(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    assert _same_model_id(str(target), str(alias)) is True
    assert _same_model_id(str(target), str(tmp_path / "other")) is False


def test_quality_bound_to_symlink_path_matches_resolved_model(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    alias = tmp_path / "alias-model"
    alias.symlink_to(tiny_model_dir)
    quality = _quality(alias, 0.90, tmp_path / "alias-w.json")
    report = diagnose_joint_interaction(
        model_dir=tiny_model_dir,
        max_memory_bytes=2_000_000_000,
        contexts=(8,),
        weight_bpws=(16.0,),
        kv_bits=(8,),
        profile=ProfileName.GENERAL,
        output_dir=tmp_path / "alias-ok",
        allow_unmeasured=True,
        reserve_bytes=0,
        quality_baseline_path=_quality(tiny_model_dir, 1.00, tmp_path / "alias-b.json"),
        quality_weight_only_path=quality,
        quality_kv_only_path=_quality(tiny_model_dir, 0.95, tmp_path / "alias-k.json"),
        quality_joint_path=_quality(tiny_model_dir, 0.85, tmp_path / "alias-j.json"),
    )
    assert report.verdict == "interaction-small"
    assert report.interaction is not None
    assert report.interaction.material is False


def test_kv16_probe_matches_plan_group_size() -> None:
    layer16 = KvLayerAllocation(layer_index=0, bits=16, group_size=64, reason="bf16")
    bf16 = CandidateMeasurement(
        bits=16,
        method=QuantMethod.BF16,
        group_size=None,
        metrics=MetricVector(),
    )
    assert _kv_candidate_matches(bf16, layer16) is True

    layer4 = KvLayerAllocation(layer_index=0, bits=4, group_size=64, reason="kv4")
    wrong_group = CandidateMeasurement(
        bits=4,
        method=QuantMethod.AFFINE,
        group_size=32,
        metrics=MetricVector(),
    )
    right_group = CandidateMeasurement(
        bits=4,
        method=QuantMethod.AFFINE,
        group_size=64,
        metrics=MetricVector(),
    )
    assert _kv_candidate_matches(wrong_group, layer4) is False
    assert _kv_candidate_matches(right_group, layer4) is True


def test_swapped_task_categories_fail_closed(tiny_model_dir: Path, tmp_path: Path) -> None:
    _enable_kv_accounting(tiny_model_dir)
    inventory = inspect_model(tiny_model_dir)

    def write_pair(path: Path, categories: tuple[str, str]) -> Path:
        write_data(
            path,
            QualityEvaluationResult(
                model=inventory.model,
                dataset_sha256="a" * 64,
                generation=QualityGenerationConfig(
                    prompt_format="raw",
                    max_sequence_length=32,
                    max_generation_tokens=8,
                ),
                metrics=QualityMetrics(task_scores={categories[0]: 1.0, categories[1]: 0.8}),
                task_results=[
                    QualityTaskResult(
                        task_id="t1",
                        category=categories[0],
                        output="ok",
                        score=1.0,
                        check_scores={"exact": 1.0},
                    ),
                    QualityTaskResult(
                        task_id="t2",
                        category=categories[1],
                        output="ok",
                        score=0.8,
                        check_scores={"exact": 0.8},
                    ),
                ],
                samples=2,
                evaluated_tokens=4,
                random_seed=0,
                software_versions=_versions(),
            ),
        )
        return path

    with pytest.raises(PlanningError, match="task id/category"):
        diagnose_joint_interaction(
            model_dir=tiny_model_dir,
            max_memory_bytes=2_000_000_000,
            contexts=(8,),
            weight_bpws=(16.0,),
            kv_bits=(8,),
            profile=ProfileName.GENERAL,
            output_dir=tmp_path / "swap",
            allow_unmeasured=True,
            reserve_bytes=0,
            quality_baseline_path=write_pair(tmp_path / "b.json", ("general", "coding")),
            quality_weight_only_path=write_pair(tmp_path / "w.json", ("coding", "general")),
            quality_kv_only_path=write_pair(tmp_path / "k.json", ("general", "coding")),
            quality_joint_path=write_pair(tmp_path / "j.json", ("general", "coding")),
        )


def _candidate(*, bpw: float, bits: int, weight_kl: float, kv_kl: float) -> JointBudgetCandidate:
    return JointBudgetCandidate(
        target_bpw=bpw,
        kv_default_bits=bits,
        context_length=4096,
        weight_bytes=100,
        kv_bytes=20,
        reserve_bytes=0,
        limit_bytes=200,
        remainder_bytes=80,
        feasible=True,
        ranking_available=True,
        estimated_main_bpw=bpw,
        proxy=JointProxyScores(
            weight_output_kl=weight_kl,
            kv_output_kl=kv_kl,
            additive_output_kl=weight_kl + kv_kl,
        ),
        plan_sha256="a" * 64,
        kv_plan_sha256="b" * 64,
    )


def test_zero_interaction_keeps_additive_ranking() -> None:
    even = _candidate(bpw=4.8, bits=4, weight_kl=0.4, kv_kl=0.4)
    skewed = _candidate(bpw=6.0, bits=16, weight_kl=0.7, kv_kl=0.1)
    zero_even = coupled_joint_loss(even, interaction=0.0, max_weight_kl=0.7, max_kv_kl=0.4)
    zero_skewed = coupled_joint_loss(skewed, interaction=0.0, max_weight_kl=0.7, max_kv_kl=0.4)
    assert zero_even == pytest.approx(0.8)
    assert zero_skewed == pytest.approx(0.8)


def test_positive_interaction_penalizes_compressing_both_sides() -> None:
    even = _candidate(bpw=4.8, bits=4, weight_kl=0.4, kv_kl=0.4)
    skewed = _candidate(bpw=6.0, bits=16, weight_kl=0.7, kv_kl=0.1)
    winner, scored = select_coupled_candidate(
        [even, skewed],
        context_length=4096,
        interaction=0.2,
    )
    assert winner.target_bpw == 6.0
    assert winner.kv_default_bits == 16
    assert scored[0].selected is True
    assert scored[0].coupled_loss < scored[1].coupled_loss


def _write_kv_analysis(model_dir: Path, path: Path) -> Path:
    inventory = inspect_model(model_dir)
    write_data(
        path,
        KvSensitivityReport(
            model=inventory.model,
            architecture_profile=inventory.architecture_profile,
            profile=ProfileName.GENERAL,
            evidence_kind=EvidenceKind.MEASURED_DEVELOPMENT,
            inventory_sha256=_inventory_digest(inventory),
            probe_backend="test",
            group_size=64,
            text_layer_count=1,
            entries=[
                KvLayerSensitivity(
                    layer_index=0,
                    candidates=[
                        CandidateMeasurement(
                            bits=4,
                            method=QuantMethod.AFFINE,
                            group_size=64,
                            metrics=MetricVector(output_kl=0.20),
                        ),
                        CandidateMeasurement(
                            bits=8,
                            method=QuantMethod.AFFINE,
                            group_size=64,
                            metrics=MetricVector(output_kl=0.10),
                        ),
                        CandidateMeasurement(
                            bits=16,
                            method=QuantMethod.BF16,
                            group_size=None,
                            metrics=MetricVector(output_kl=0.0),
                        ),
                    ],
                )
            ],
            calibration=CalibrationEvidence(
                dataset_id="test",
                dataset_sha256="c" * 64,
                samples=8,
                domains=["general"],
                sequence_length=32,
                backend="test",
                reference="test",
            ),
        ),
    )
    return path


def test_plan_joint_requires_quality_and_kv_analysis(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    with pytest.raises(PlanningError, match="kv-analysis"):
        plan_joint_allocation(
            model_dir=tiny_model_dir,
            max_memory_bytes=2_000_000_000,
            context_length=8,
            profile=ProfileName.GENERAL,
            output_dir=tmp_path / "no-kv",
            allow_unmeasured=True,
        )


def test_plan_joint_small_i_emits_independent_plan(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    kv = _write_kv_analysis(tiny_model_dir, tmp_path / "kv.json")
    selection = plan_joint_allocation(
        model_dir=tiny_model_dir,
        max_memory_bytes=2_000_000_000,
        context_length=8,
        profile=ProfileName.GENERAL,
        output_dir=tmp_path / "small-i",
        contexts=(8,),
        weight_bpws=(16.0,),
        kv_bits=(4, 8, 16),
        kv_analysis_path=kv,
        allow_unmeasured=True,
        reserve_bytes=0,
        independent_bpw=16.0,
        quality_baseline_path=_quality(tiny_model_dir, 0.80, tmp_path / "b.json"),
        quality_weight_only_path=_quality(tiny_model_dir, 0.80, tmp_path / "w.json"),
        quality_kv_only_path=_quality(tiny_model_dir, 0.80, tmp_path / "k.json"),
        quality_joint_path=_quality(tiny_model_dir, 0.80, tmp_path / "j.json"),
    )
    assert selection.selection_basis == "independent"
    assert selection.differs_from_independent is False
    assert (tmp_path / "small-i" / "axquant_plan.json").is_file()


def test_plan_joint_material_i_writes_coupled_plan(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    kv = _write_kv_analysis(tiny_model_dir, tmp_path / "kv.json")
    selection = plan_joint_allocation(
        model_dir=tiny_model_dir,
        max_memory_bytes=2_000_000_000,
        context_length=8,
        profile=ProfileName.GENERAL,
        output_dir=tmp_path / "material-i",
        contexts=(8,),
        weight_bpws=(16.0,),
        kv_bits=(4, 8, 16),
        kv_analysis_path=kv,
        allow_unmeasured=True,
        reserve_bytes=0,
        independent_bpw=16.0,
        quality_baseline_path=_quality(tiny_model_dir, 1.00, tmp_path / "b.json"),
        quality_weight_only_path=_quality(tiny_model_dir, 0.90, tmp_path / "w.json"),
        quality_kv_only_path=_quality(tiny_model_dir, 0.95, tmp_path / "k.json"),
        quality_joint_path=_quality(tiny_model_dir, 0.80, tmp_path / "j.json"),
    )
    assert selection.selection_basis == "coupled-interaction"
    assert selection.interaction_material is True
    assert selection.scored_cells
    assert selection.differs_from_independent is False
    assert (tmp_path / "material-i" / "axquant_plan.json").is_file()
    plan = load_model(tmp_path / "material-i" / "axquant_plan.json", QuantizationPlan)
    assert plan.kv_cache is not None
    assert plan.kv_cache.allocation_basis == "measured"
    assert selection.plan_sha256 == stable_sha256(plan)
    assert plan.schema_version == "axquant.plan.v1"


def test_plan_joint_negative_i_can_select_a_different_kv_cell(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    kv = _write_kv_analysis(tiny_model_dir, tmp_path / "kv.json")
    selection = plan_joint_allocation(
        model_dir=tiny_model_dir,
        max_memory_bytes=2_000_000_000,
        context_length=8,
        profile=ProfileName.GENERAL,
        output_dir=tmp_path / "neg-i",
        contexts=(8,),
        weight_bpws=(16.0,),
        kv_bits=(4, 8, 16),
        kv_analysis_path=kv,
        allow_unmeasured=True,
        reserve_bytes=0,
        independent_bpw=16.0,
        quality_baseline_path=_quality(tiny_model_dir, 1.00, tmp_path / "b.json"),
        quality_weight_only_path=_quality(tiny_model_dir, 0.70, tmp_path / "w.json"),
        quality_kv_only_path=_quality(tiny_model_dir, 0.70, tmp_path / "k.json"),
        quality_joint_path=_quality(tiny_model_dir, 0.90, tmp_path / "j.json"),
    )
    assert selection.selection_basis == "coupled-interaction"
    assert selection.interaction is not None and selection.interaction < 0
    assert selection.differs_from_independent is True
    assert selection.selected_kv_bits < 16
    plan = load_model(tmp_path / "neg-i" / "axquant_plan.json", QuantizationPlan)
    assert plan.kv_cache is not None
    assert any(layer.bits < 16 for layer in plan.kv_cache.layers)


def test_plan_joint_cli_writes_convert_ready_plan(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    _enable_kv_accounting(tiny_model_dir)
    kv = _write_kv_analysis(tiny_model_dir, tmp_path / "kv.json")
    output = tmp_path / "cli-joint"
    exit_code = main(
        [
            "plan-joint",
            "--model",
            str(tiny_model_dir),
            "--max-memory",
            "2GB",
            "--context",
            "8",
            "--contexts",
            "8",
            "--weight-bpws",
            "16",
            "--kv-bits",
            "4,8,16",
            "--kv-analysis",
            str(kv),
            "--allow-unmeasured",
            "--reserve-memory",
            "0B",
            "--independent-bpw",
            "16",
            "--quality-baseline",
            str(_quality(tiny_model_dir, 0.80, tmp_path / "b.json")),
            "--quality-weight-only",
            str(_quality(tiny_model_dir, 0.80, tmp_path / "w.json")),
            "--quality-kv-only",
            str(_quality(tiny_model_dir, 0.80, tmp_path / "k.json")),
            "--quality-joint",
            str(_quality(tiny_model_dir, 0.80, tmp_path / "j.json")),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    plan = load_model(output / "axquant_plan.json", QuantizationPlan)
    selection = load_model(output / "joint-selection.json", JointSelectionReport)
    assert plan.schema_version == "axquant.plan.v1"
    assert selection.certification_eligible is False
    assert selection.selection_basis == "independent"
    assert selection.plan_sha256 == stable_sha256(plan)
