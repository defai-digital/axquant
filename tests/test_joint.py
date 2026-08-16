from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from axquant.cli import main
from axquant.cli._parser import _build_parser
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.joint import _crossover, _same_model_id, diagnose_joint_interaction
from axquant.schema import (
    JointBudgetCandidate,
    JointInteractionReport,
    JointProxyScores,
    ProfileName,
    QualityEvaluationResult,
    QualityGenerationConfig,
    QualityMetrics,
    QualityTaskResult,
    SoftwareVersions,
)
from axquant.serde import load_model, write_data


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
    assert report.evidence_kind.value == "architecture_prior"
    assert report.interaction is None
    assert report.candidates
    assert (output / "joint-interaction.md").is_file()
    markdown = (output / "joint-interaction.md").read_text(encoding="utf-8")
    assert "not a certification claim" in markdown
    assert report.crossover.detected is False
    assert report.crossover.ranking_complete is False
    assert all(not candidate.ranking_available for candidate in report.candidates)
    assert all(winner.target_bpw is None for winner in report.crossover.winners)


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
    assert report.evidence_kind.value == "architecture_prior"


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
