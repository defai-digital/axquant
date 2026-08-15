"""Simple convert UX: OptiQ-like ergonomics with two-door evidence rules."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# Reuse the fake MLX harness from test_quantize without package import issues.
import test_quantize as _tq

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.recipes import export_recipe_bundle
from axquant.schema import (
    EvidenceKind,
    PlanRequest,
    ProfileName,
    QuickConversionSummary,
    SupportTier,
)
from axquant.serde import load_model, write_data
from axquant.simple_convert import (
    default_output_dir,
    infer_model_id,
    looks_like_hub_id,
    resolve_download_policy,
    simple_convert,
    simple_convert_help_markdown,
    target_class_for_bpw,
)

_install_fake_mlx = _tq._install_fake_mlx


def test_looks_like_hub_id() -> None:
    assert looks_like_hub_id("Qwen/Qwen3.6-27B")
    assert not looks_like_hub_id("models/Qwen/Qwen3.6-27B")
    assert not looks_like_hub_id("/models/local")
    assert not looks_like_hub_id("./relative")
    assert not looks_like_hub_id("solo-name")


def test_target_class_and_default_output() -> None:
    assert target_class_for_bpw(4.8) == "4bit"
    assert target_class_for_bpw(6.0) == "6bit"
    assert target_class_for_bpw(5.5) == "5p5bpw"
    out = default_output_dir("Qwen/Qwen3.6-27B", target_bpw=4.8, parent="/tmp")
    assert out.name == "AX-Qwen3.6-27B-MLX-AXQ-4bit"
    with pytest.raises(PlanningError, match="finite"):
        default_output_dir("Qwen/Qwen3.6-27B", target_bpw=math.nan)


def test_download_policy() -> None:
    allow, notes = resolve_download_policy("Qwen/Qwen3.6-27B", allow_download=False)
    assert allow is False
    assert any("cache" in note.lower() or "download" in note.lower() for note in notes)
    allow, notes = resolve_download_policy("Qwen/Qwen3.6-27B", allow_download=True)
    assert allow is True


def test_simple_convert_default_output(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    monkeypatch.chdir(tmp_path)
    summary = simple_convert(
        str(qwen36_model_dir),
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
        target_bpw=14.0,
        ax_engine_manifest="skip",
    )
    assert summary.development_evidence
    assert summary.evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR
    assert summary.support_tier is SupportTier.CONVERTIBLE
    assert Path(summary.output_path).is_dir()
    assert "AX-Qwen3.6-27B-MLX-AXQ" in summary.output_path
    assert any("Simple convert path" in note for note in summary.notes)
    assert any("development evidence" in note.lower() for note in summary.notes)


def test_quantize_positional_minimal_cli(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    monkeypatch.chdir(tmp_path)
    summary_path = tmp_path / "summary.json"
    code = main(
        [
            "quantize",
            str(qwen36_model_dir),
            "--model-id",
            "Qwen/Qwen3.6-27B",
            "--revision",
            "source-revision",
            "--target-bpw",
            "14.0",
            "--ax-engine-manifest",
            "skip",
            "--json",
            str(summary_path),
        ]
    )
    assert code == 0
    summary = load_model(summary_path, QuickConversionSummary)
    assert summary.development_evidence
    assert Path(summary.output_path).is_dir()


def test_simple_convert_help_cli(tmp_path: Path) -> None:
    out = tmp_path / "help.md"
    assert main(["simple-convert-help", "--output", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Two doors" in text
    assert "axquant quantize" in text
    assert "python -m pip install 'axquant[mlx]'" in text
    assert "Do not clone this repository" in text
    assert simple_convert_help_markdown() == text


def test_infer_model_id() -> None:
    assert infer_model_id("Qwen/Qwen3.6-27B") == "Qwen/Qwen3.6-27B"
    assert infer_model_id("/local/path", model_id="org/name") == "org/name"
    assert infer_model_id("/local/path") is None


def test_simple_convert_refuses_measured_ladder_without_recipe(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanningError, match="measured sensitivity"):
        simple_convert(
            str(qwen36_model_dir),
            output=tmp_path / "out",
            ladder="measured-full",
            ax_engine_manifest="skip",
        )


def test_simple_convert_rejects_nonfinite_target(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanningError, match="finite"):
        simple_convert(
            str(qwen36_model_dir),
            output=tmp_path / "out",
            target_bpw=math.nan,
            ax_engine_manifest="skip",
        )


def test_simple_convert_rejects_output_inside_source(
    qwen36_model_dir: Path,
) -> None:
    with pytest.raises(PlanningError, match="must not overlap"):
        simple_convert(
            str(qwen36_model_dir),
            output=qwen36_model_dir / "converted",
            target_bpw=14.0,
            ax_engine_manifest="skip",
        )


def test_simple_convert_recipe_uses_its_target_and_rejects_override(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision=revision,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)
    bundle_path = export_recipe_bundle(
        plan=plan_path,
        output_dir=tmp_path / "bundle",
        bundle_id="recipe",
    )
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    monkeypatch.chdir(tmp_path)

    summary = simple_convert(
        str(qwen36_model_dir),
        model_id="Qwen/Qwen3.6-27B",
        revision=revision,
        recipe=bundle_path,
        ax_engine_manifest="skip",
    )

    assert summary.target_bpw == 14.0
    assert Path(summary.output_path).name.endswith("14p0bpw")
    with pytest.raises(PlanningError, match="cannot be combined with --recipe"):
        simple_convert(
            str(qwen36_model_dir),
            output=tmp_path / "override",
            model_id="Qwen/Qwen3.6-27B",
            revision=revision,
            recipe=bundle_path,
            target_bpw=8.0,
            ax_engine_manifest="skip",
        )
    with pytest.raises(PlanningError, match="profile does not match"):
        simple_convert(
            str(qwen36_model_dir),
            output=tmp_path / "profile-override",
            model_id="Qwen/Qwen3.6-27B",
            revision=revision,
            recipe=bundle_path,
            profile=ProfileName.GENERAL,
            ax_engine_manifest="skip",
        )
    with pytest.raises(PlanningError, match="--kv-cache cannot be combined"):
        simple_convert(
            str(qwen36_model_dir),
            output=tmp_path / "kv-override",
            model_id="Qwen/Qwen3.6-27B",
            revision=revision,
            recipe=bundle_path,
            kv_cache="prior",
            ax_engine_manifest="skip",
        )
