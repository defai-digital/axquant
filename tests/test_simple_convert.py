"""Simple convert UX: OptiQ-like ergonomics with two-door evidence rules."""

from __future__ import annotations

from pathlib import Path

import pytest

# Reuse the fake MLX harness from test_quantize without package import issues.
import test_quantize as _tq

from axquant.cli import main
from axquant.errors import PlanningError
from axquant.schema import EvidenceKind, QuickConversionSummary, SupportTier
from axquant.serde import load_model
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
    assert not looks_like_hub_id("/models/local")
    assert not looks_like_hub_id("./relative")
    assert not looks_like_hub_id("solo-name")


def test_target_class_and_default_output() -> None:
    assert target_class_for_bpw(4.8) == "4bit"
    assert target_class_for_bpw(6.0) == "6bit"
    assert target_class_for_bpw(5.5) == "5p5bpw"
    out = default_output_dir("Qwen/Qwen3.6-27B", target_bpw=4.8, parent="/tmp")
    assert out.name == "AX-Qwen3.6-27B-MLX-AXQ-4bit"


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
