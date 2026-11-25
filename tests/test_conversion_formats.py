"""Format presets exercise the public conversion path with synthetic weights."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_quantize import _install_fake_mlx

from axquant.cli import main
from axquant.conversion_formats import MLX_FORMATS
from axquant.errors import PlanningError
from axquant.schema import QuickConversionSummary
from axquant.serde import load_model
from axquant.simple_convert import default_output_dir, simple_convert


@pytest.mark.parametrize("format", MLX_FORMATS)
def test_format_conversion(
    format: str, qwen36_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use complete 32-value blocks; the generic fixture has 8-wide matrices.
    source = qwen36_model_dir / "model.safetensors"
    tensors = load_file(source)
    for name, value in tensors.items():
        if value.ndim == 2:
            tensors[name] = np.zeros((32, 128), dtype=np.float32)
    save_file(tensors, source)
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    output = tmp_path / format
    summary_path = tmp_path / "summary.json"
    assert (
        main(
            [
                "quantize",
                str(qwen36_model_dir),
                # Same shapes as the admitted Qwen3.8-27B DFlash2 target (AXQ-067).
                "--model-id",
                "Qwen/Qwen3.6-27B",
                "--format",
                format,
                "--output",
                str(output),
                "--ax-engine-manifest",
                "skip",
                "--json",
                str(summary_path),
            ]
        )
        == 0
    )
    summary = load_model(summary_path, QuickConversionSummary)
    declaration = json.loads((output / "axquant_dflash2_target.json").read_text())
    assert declaration["status"] == "non_target"
    assert declaration["source_model_id"] == "Qwen/Qwen3.6-27B"
    config = json.loads((output / "config.json").read_text())
    entries = [v for v in config["quantization"].values() if isinstance(v, dict)]
    bits = {"mxfp4": 4, "mxfp8": 8}[format]
    assert any(e["bits"] == bits and e.get("mode") == format for e in entries)
    # AXQ-070: every quantized entry is MXFP4 or MXFP8 at group 32.
    assert all(
        (e["bits"], e.get("mode"), e["group_size"]) in {(4, "mxfp4", 32), (8, "mxfp8", 32)}
        for e in entries
    )
    assert "language_model.lm_head" not in config["quantization"]
    attention = config["quantization"].get(
        "language_model.model.layers.0.linear_attn.in_proj_qkvz", {}
    )
    if format == "mxfp8":
        assert attention.get("bits", 16) >= 8
    assert summary.development_evidence
    assert summary.runtime_smoke_passed is None
    assert (output / "mtp.safetensors").read_bytes() == (
        qwen36_model_dir / "mtp.safetensors"
    ).read_bytes()


@pytest.mark.parametrize(
    "options",
    [
        {"recipe": "missing-recipe"},
        {"ladder": "measured-lite"},
    ],
)
def test_format_conflicts_before_source_resolution(options: dict) -> None:
    with pytest.raises(PlanningError, match="--format"):
        simple_convert("missing/model", format="mxfp4", **options)


def test_unknown_format_before_source_resolution() -> None:
    with pytest.raises(PlanningError, match="unknown MLX format"):
        simple_convert("missing/model", format="nvfp4")


@pytest.mark.parametrize("format", MLX_FORMATS)
def test_format_default_name(format: str, tmp_path: Path) -> None:
    name = default_output_dir("org/model", format=format, parent=tmp_path).name
    assert name.endswith(f"-MLX-AXQ-{format.upper()}")


def test_mlx_formats_are_the_axq070_presets() -> None:
    assert MLX_FORMATS == ("mxfp4", "mxfp8")
    for retired in ("2bit", "3bit", "4bit", "6bit", "8bit"):
        with pytest.raises(PlanningError, match="unknown MLX format"):
            simple_convert("missing/model", format=retired)


def test_no_eligible_base_tensor_fails_before_conversion(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    source = qwen36_model_dir / "model.safetensors"
    tensors = load_file(source)
    # Only protected or non-quantizable tensors remain: no MXFP4 base allocation.
    save_file(
        {k: v for k, v in tensors.items() if ".mlp." not in k and "in_proj_qkvz" not in k},
        source,
    )
    output = tmp_path / "no-output"
    with pytest.raises(PlanningError, match="produced no"):
        simple_convert(
            str(qwen36_model_dir),
            output=output,
            model_id="Qwen/Qwen3.6-27B",
            format="mxfp4",
        )
    assert not output.exists()
