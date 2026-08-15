from __future__ import annotations

from pathlib import Path

import pytest

from axquant.cli import main
from axquant.cli._parser import _build_parser
from axquant.errors import PlanningError
from axquant.optimizer import optimize_deployment, parse_memory_bytes
from axquant.schema import DeploymentPlan, ProfileName, RuntimeName
from axquant.serde import load_model


def test_optimize_cli_help_lists_joint_budget_options() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["optimize", "--help"])
    assert exc_info.value.code == 0


def test_memory_parser_requires_explicit_units() -> None:
    assert parse_memory_bytes("18GB") == 18_000_000_000
    assert parse_memory_bytes("1GiB") == 1 << 30
    with pytest.raises(ValueError, match="unit"):
        parse_memory_bytes("18")


def test_infeasible_optimize_fails_with_breakdown(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanningError, match=r"weights=.*kv=.*reserve=.*limit="):
        optimize_deployment(
            model_dir=tiny_model_dir,
            max_memory_bytes=100,
            context_length=128,
            profile=ProfileName.GENERAL,
            runtime=RuntimeName.AX_ENGINE,
            minimum_quality_retention=0.98,
            mode="balanced",
            output_dir=tmp_path / "infeasible",
            allow_unmeasured=True,
            target_bpw=16.0,
            reserve_bytes=0,
        )


def test_mode_overlay_changes_recorded_objective(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    balanced = optimize_deployment(
        model_dir=tiny_model_dir,
        max_memory_bytes=10_000,
        context_length=128,
        profile=ProfileName.GENERAL,
        runtime=RuntimeName.AX_ENGINE,
        minimum_quality_retention=0.98,
        mode="balanced",
        output_dir=tmp_path / "balanced",
        allow_unmeasured=True,
        target_bpw=16.0,
        reserve_bytes=0,
    )
    quality = optimize_deployment(
        model_dir=tiny_model_dir,
        max_memory_bytes=10_000,
        context_length=128,
        profile=ProfileName.GENERAL,
        runtime=RuntimeName.AX_ENGINE,
        minimum_quality_retention=0.98,
        mode="quality",
        output_dir=tmp_path / "quality",
        allow_unmeasured=True,
        target_bpw=16.0,
        reserve_bytes=0,
    )

    assert quality.objective.task_loss_delta > balanced.objective.task_loss_delta
    assert quality.objective.output_kl > balanced.objective.output_kl
    assert quality.objective.peak_memory_cost < balanced.objective.peak_memory_cost


def test_allow_unmeasured_cli_emits_architecture_prior_not_certified(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "deployment"

    exit_code = main(
        [
            "optimize",
            "--model",
            str(tiny_model_dir),
            "--max-memory",
            "2GB",
            "--context",
            "128",
            "--profile",
            "general",
            "--runtime",
            "ax-engine",
            "--allow-unmeasured",
            "--target-bpw",
            "16",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    deployment = load_model(output / "deployment-plan.json", DeploymentPlan)
    assert deployment.evidence_kind == "architecture_prior"
    assert "certified" not in deployment.evidence_kind
    assert (output / "deployment-plan.md").is_file()
    assert (output / "axquant_plan.json").is_file()
