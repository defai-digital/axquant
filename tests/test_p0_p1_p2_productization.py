"""P0/P1/P2 productization: ladders, probe capacity, scoreboard, deferred, recovery rank."""

from __future__ import annotations

from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.deferred import deferred_feature_matrix, require_feature
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.ladders import get_ladder, list_ladders, plan_request_for_ladder
from axquant.planner import allocate_kv_cache, plan_quantization
from axquant.probe_capacity import assess_probe_capacity, assess_probe_capacity_from_inventory
from axquant.recovery import rank_recovery_targets
from axquant.schema import (
    ConvertLadderName,
    DeferredFeature,
    EvidenceKind,
    ProbeMode,
    ProfileName,
    QuantMethod,
)
from axquant.scoreboard import build_scoreboard, scoreboard_markdown
from axquant.serde import load_model, write_data
from axquant.unified_sensitivity import bind_unified_sensitivity


def test_ladders_progress_and_multi_group() -> None:
    ladders = list_ladders()
    assert [item.name for item in ladders] == [
        ConvertLadderName.PRIOR,
        ConvertLadderName.MEASURED_LITE,
        ConvertLadderName.MEASURED_FULL,
        ConvertLadderName.REFINE_AWQ_DWQ,
    ]
    prior = get_ladder("prior")
    assert prior.candidate_group_sizes == (32, 64)
    assert prior.allow_unmeasured
    full = get_ladder(ConvertLadderName.MEASURED_FULL)
    assert QuantMethod.DWQ in full.candidate_methods
    assert 32 in full.candidate_group_sizes
    refine = get_ladder("refine-awq-dwq")
    assert QuantMethod.AWQ in refine.candidate_methods
    assert QuantMethod.GPTQ in refine.candidate_methods
    request = plan_request_for_ladder(prior, profile=ProfileName.GENERAL)
    assert request.candidate_group_sizes == (32, 64)
    assert request.target_bpw == 4.8


def test_prior_ladder_plans_multi_group(qwen36_model_dir: Path) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    request = plan_request_for_ladder(
        ConvertLadderName.PRIOR,
        profile=ProfileName.GENERAL,
        target_bpw=14.0,
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=request.candidate_bits,
        group_size=request.group_size,
        candidate_group_sizes=request.candidate_group_sizes,
    )
    # At least one quantizable tensor should expose multi-group candidates.
    multi = [
        entry
        for entry in report.entries
        if entry.tensor.quantizable
        and len({c.group_size for c in entry.candidates if c.group_size is not None}) > 1
    ]
    assert multi
    plan = plan_quantization(report, request)
    assert plan.candidate_group_sizes == (32, 64)
    assert plan.evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR


def test_probe_capacity_recommends_modes() -> None:
    # Huge model + tiny memory → prior-only.
    tight = assess_probe_capacity(
        parameter_count=70_000_000_000,
        available_memory_bytes=8 * 1024**3,
        headroom_fraction=0.70,
    )
    assert tight.recommended_mode is ProbeMode.PRIOR_ONLY
    assert not next(m for m in tight.modes if m.mode is ProbeMode.BF16_FULL).feasible

    # Small model + large memory → bf16-full.
    roomy = assess_probe_capacity(
        parameter_count=1_000_000,
        available_memory_bytes=64 * 1024**3,
        headroom_fraction=0.70,
    )
    assert roomy.recommended_mode is ProbeMode.BF16_FULL
    assert next(m for m in roomy.modes if m.mode is ProbeMode.BF16_FULL).release_quality_eligible


def test_probe_capacity_from_inventory(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    path = tmp_path / "inventory.json"
    write_data(path, inventory)
    report = assess_probe_capacity_from_inventory(
        path,
        available_memory_bytes=512 * 1024**3,
    )
    assert report.parameter_count == inventory.total_parameters
    assert report.recommended_mode is ProbeMode.BF16_FULL


def test_probe_capacity_rejects_boolean_and_non_finite_inputs() -> None:
    with pytest.raises(PlanningError, match="parameter_count"):
        assess_probe_capacity(parameter_count=True, available_memory_bytes=1024)
    with pytest.raises(PlanningError, match="available memory"):
        assess_probe_capacity(parameter_count=1, available_memory_bytes=True)
    with pytest.raises(PlanningError, match="headroom_fraction"):
        assess_probe_capacity(
            parameter_count=1,
            available_memory_bytes=1024,
            headroom_fraction=float("nan"),
        )


def test_probe_capacity_never_rounds_measured_modes_to_zero_bytes() -> None:
    report = assess_probe_capacity(
        parameter_count=1,
        available_memory_bytes=1024,
    )
    measured = [mode for mode in report.modes if mode.mode is not ProbeMode.PRIOR_ONLY]
    assert measured
    assert all(mode.estimated_bytes >= 1 for mode in measured)


def test_probe_capacity_rejects_inconsistent_inventory(qwen36_model_dir: Path) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    inconsistent = inventory.model_copy(update={"total_parameters": inventory.total_parameters + 1})
    with pytest.raises(PlanningError, match="does not match"):
        assess_probe_capacity_from_inventory(
            inconsistent,
            available_memory_bytes=64 * 1024**3,
        )


def test_scoreboard_lists_missing_and_engine_mtp(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    request = plan_request_for_ladder(
        ConvertLadderName.PRIOR,
        profile=ProfileName.GENERAL,
        target_bpw=14.0,
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=request.candidate_bits,
        candidate_group_sizes=request.candidate_group_sizes,
        group_size=request.group_size,
    )
    plan = plan_quantization(report, request)
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)
    board = build_scoreboard(plan=plan_path)
    assert board.overall_status == "incomplete"
    assert "size_ratio_vs_uniform4" in board.missing_mandatory
    assert "mtp_speedup" in board.missing_mandatory
    mtp_row = next(row for row in board.rows if row.metric_id == "mtp_speedup")
    assert mtp_row.owner == "ax-engine"
    markdown = scoreboard_markdown(board)
    assert "AX Engine" in markdown or "ax-engine" in markdown


def test_unified_sensitivity_binding(qwen36_model_dir: Path) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    report = architecture_prior_report(inventory, profile=ProfileName.GENERAL, group_size=64)
    request = plan_request_for_ladder(
        ConvertLadderName.PRIOR,
        profile=ProfileName.GENERAL,
        target_bpw=14.0,
    )
    plan = plan_quantization(report, request)
    plan.kv_cache = allocate_kv_cache(inventory.architecture_profile.text_layer_count or 1)
    binding = bind_unified_sensitivity(report, plan=plan)
    assert binding.weight_evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR
    assert binding.kv_allocation_basis == "architecture-prior"
    assert binding.weight_sensitivity_sha256


def test_deferred_features_fail_closed() -> None:
    matrix = deferred_feature_matrix()
    assert {item["feature"] for item in matrix} == {item.value for item in DeferredFeature}
    with pytest.raises(PlanningError, match="VLM optimization"):
        require_feature(DeferredFeature.VLM_OPTIMIZATION)
    with pytest.raises(PlanningError, match="Per-expert"):
        require_feature(DeferredFeature.PER_EXPERT_UNFUSED)


def test_recovery_rank_orders_by_loss(qwen36_model_dir: Path) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    request = plan_request_for_ladder(
        ConvertLadderName.PRIOR,
        profile=ProfileName.GENERAL,
        target_bpw=14.0,
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=request.candidate_bits,
        candidate_group_sizes=request.candidate_group_sizes,
        group_size=request.group_size,
    )
    plan = plan_quantization(report, request)
    ranking = rank_recovery_targets(plan, sensitivity=report, limit=5)
    assert ranking.targets
    assert len(ranking.targets) <= 5
    # Scores should be non-increasing.
    scores = [ranking.scores[name] for name in ranking.targets]
    assert scores == sorted(scores, reverse=True)


def test_cli_ladders_probe_scoreboard_deferred(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    inventory = inspect_model(str(qwen36_model_dir), model_id="Qwen/Qwen3.6-27B")
    inv_path = tmp_path / "inventory.json"
    write_data(inv_path, inventory)
    request = plan_request_for_ladder(
        ConvertLadderName.PRIOR,
        profile=ProfileName.GENERAL,
        target_bpw=14.0,
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=request.candidate_bits,
        candidate_group_sizes=request.candidate_group_sizes,
        group_size=request.group_size,
    )
    plan = plan_quantization(report, request)
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)

    assert (
        main(
            [
                "ladders",
                "--output",
                str(tmp_path / "ladders.json"),
                "--markdown-output",
                str(tmp_path / "ladders.md"),
            ]
        )
        == 0
    )
    assert (tmp_path / "ladders.md").is_file()

    assert (
        main(
            [
                "probe-capacity",
                "--inventory",
                str(inv_path),
                "--available-memory-bytes",
                "100",  # tiny budget forces prior-only even on fixture inventories
                "--output",
                str(tmp_path / "capacity.json"),
                "--markdown-output",
                str(tmp_path / "capacity.md"),
            ]
        )
        == 0
    )
    from axquant.schema import ProbeCapacityReport

    capacity = load_model(tmp_path / "capacity.json", ProbeCapacityReport)
    assert capacity.recommended_mode is ProbeMode.PRIOR_ONLY

    assert (
        main(
            [
                "scoreboard",
                "--plan",
                str(plan_path),
                "--output",
                str(tmp_path / "scoreboard.json"),
                "--markdown-output",
                str(tmp_path / "scoreboard.md"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "deferred-features",
                "--output",
                str(tmp_path / "deferred.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "recovery-rank",
                "--plan",
                str(plan_path),
                "--output",
                str(tmp_path / "recovery.json"),
            ]
        )
        == 0
    )
    sens_path = tmp_path / "sens.json"
    write_data(sens_path, report)
    assert (
        main(
            [
                "bind-sensitivity",
                "--sensitivity",
                str(sens_path),
                "--plan",
                str(plan_path),
                "--output",
                str(tmp_path / "binding.json"),
            ]
        )
        == 0
    )


def test_expert_memory_recipe_loads(qwen36_model_dir: Path) -> None:
    from axquant.manual import manual_quantization_plan
    from axquant.schema import ManualPlanRecipe
    from axquant.serde import load_model as load

    recipe_path = Path("examples/expert-memory-tier-v0.1.yaml")
    recipe = load(recipe_path, ManualPlanRecipe)
    inventory = inspect_model(
        str(qwen36_model_dir),
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    plan = manual_quantization_plan(inventory, recipe)
    experts = [a for a in plan.assignments if a.role.value == "expert"]
    routers = [a for a in plan.assignments if a.role.value == "router"]
    # Dense Qwen fixture may have no experts/routers; recipe still validates.
    for assignment in experts:
        assert assignment.bits == 2
    for assignment in routers:
        assert assignment.bits == 8
