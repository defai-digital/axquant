from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

import axquant.converter as converter
from axquant.cli import main
from axquant.errors import PlanningError
from axquant.quantize import DEVELOPMENT_NOTE, _validate_runtime_smoke, quick_convert
from axquant.schema import (
    ArtifactManifest,
    EvidenceKind,
    QuickConversionSummary,
    SupportTier,
)
from axquant.serde import load_model


def _install_fake_mlx(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    def _module_paths() -> list[str]:
        with safe_open(model_dir / "model.safetensors", framework="numpy") as source:
            return [
                name.rsplit(".", 1)[0]
                for name in list(source.keys())
                if not name.startswith("visual.")
            ]

    class FakeModel:
        def named_modules(self):
            return [(path, object()) for path in _module_paths()]

    def fake_load(*args, **kwargs):
        return FakeModel(), {}, {}

    def fake_convert(model, *, mlx_path, quant_predicate, **kwargs):
        del model, kwargs
        output = Path(mlx_path)
        output.mkdir()
        converted_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        converted_config.pop("vision_config", None)
        quantization: dict[str, dict[str, int | str]] = {}
        for path, module in FakeModel().named_modules():
            config = quant_predicate(path, module)
            if isinstance(config, dict):
                quantization[path] = config
        with safe_open(model_dir / "model.safetensors", framework="numpy") as source:
            tensors: dict[str, np.ndarray] = {}
            for name in list(source.keys()):
                if name.startswith("visual."):
                    continue
                value = source.get_tensor(name)
                path = name.rsplit(".", 1)[0]
                config = quantization.get(path)
                if config is None:
                    tensors[name] = value
                    continue
                bits = int(config["bits"])
                group_size = int(config["group_size"])
                assert value.shape[-1] * bits % 32 == 0
                tensors[name] = np.zeros(
                    (*value.shape[:-1], value.shape[-1] * bits // 32),
                    dtype=np.uint32,
                )
                metadata_shape = (
                    *value.shape[:-1],
                    max(1, (value.shape[-1] + group_size - 1) // group_size),
                )
                tensors[f"{path}.scales"] = np.ones(metadata_shape, dtype=np.float32)
                tensors[f"{path}.biases"] = np.zeros(metadata_shape, dtype=np.float32)
        converted_config["quantization"] = quantization
        (output / "config.json").write_text(json.dumps(converted_config), encoding="utf-8")
        save_file(tensors, output / "model.safetensors")

    monkeypatch.setattr(converter, "_mlx_api", lambda: (fake_convert, fake_load))


def test_quick_convert_produces_development_artifact(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    output = tmp_path / "quick-candidate"
    summary = quick_convert(
        model=str(qwen36_model_dir),
        output=output,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
        target_bpw=14.0,
        ax_engine_manifest="skip",
    )
    assert summary.support_tier is SupportTier.CONVERTIBLE
    assert summary.evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR
    assert summary.plan_source == "architecture-prior"
    assert summary.development_evidence
    assert DEVELOPMENT_NOTE in summary.notes
    assert summary.runtime_smoke == "none"
    assert summary.runtime_smoke_passed is None
    assert summary.output_path == str(output.resolve())
    manifest = load_model(output / "axquant_manifest.json", ArtifactManifest)
    assert manifest.measured_total_bpw == pytest.approx(summary.measured_total_bpw)
    # The external MTP sidecar was auto-discovered from the source checkpoint.
    assert (output / "mtp.safetensors").read_bytes() == (
        qwen36_model_dir / "mtp.safetensors"
    ).read_bytes()


def test_quick_convert_refuses_inspect_only_families(tiny_model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(PlanningError, match="inspect-only"):
        quick_convert(
            model=str(tiny_model_dir),
            output=tmp_path / "refused",
        )


def test_quick_convert_runtime_smoke_requires_matching_multimodal_backend(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.wav"
    image = tmp_path / "sample.png"
    audio.write_bytes(b"audio")
    image.write_bytes(b"image")

    _validate_runtime_smoke(
        "mlx-audio",
        adapter_id="qwen3-asr-v1",
        audio_input=audio,
        image_input=None,
    )
    _validate_runtime_smoke(
        "mlx-vlm",
        adapter_id="qwen3-vl-v1",
        audio_input=None,
        image_input=image,
    )
    with pytest.raises(PlanningError, match="wrong runtime"):
        _validate_runtime_smoke(
            "mlx-lm",
            adapter_id="qwen3-asr-v1",
            audio_input=audio,
            image_input=None,
        )
    with pytest.raises(PlanningError, match="requires --image-input"):
        _validate_runtime_smoke(
            "mlx-vlm",
            adapter_id="qwen3-vl-v1",
            audio_input=None,
            image_input=None,
        )
    with pytest.raises(PlanningError, match="only valid"):
        _validate_runtime_smoke(
            "mlx-audio",
            adapter_id="qwen3-dense-v1",
            audio_input=audio,
            image_input=None,
        )


def test_quick_convert_raises_infeasible_target_bpw(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected floors can make a low target infeasible; simple convert raises once."""
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    output = tmp_path / "raised-bpw"
    # Extremely low target forces a raise to the policy minimum.
    summary = quick_convert(
        model=str(qwen36_model_dir),
        output=output,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
        target_bpw=0.5,
        ax_engine_manifest="skip",
    )
    assert summary.development_evidence
    # Plan target must be above the impossible 0.5 request.
    from axquant.schema import QuantizationPlan
    from axquant.serde import load_model

    plan = load_model(output / "axquant_plan.json", QuantizationPlan)
    assert plan.target_bpw > 0.5
    assert any("raised from" in warning for warning in plan.warnings)
    assert any("protection floors" in warning for warning in plan.warnings)


def test_quantize_cli_writes_summary_json(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    output = tmp_path / "cli-candidate"
    summary_path = tmp_path / "quantize-summary.json"
    exit_code = main(
        [
            "quantize",
            "--model",
            str(qwen36_model_dir),
            "--model-id",
            "Qwen/Qwen3.6-27B",
            "--revision",
            "source-revision",
            "--target-bpw",
            "14.0",
            "--output",
            str(output),
            "--ax-engine-manifest",
            "skip",
            "--json",
            str(summary_path),
        ]
    )
    assert exit_code == 0
    summary = load_model(summary_path, QuickConversionSummary)
    assert summary.development_evidence
    assert summary.source_model.model_id == "Qwen/Qwen3.6-27B"
    assert (output / "axquant_manifest.json").is_file()


def test_quick_convert_binds_recipe_bundle(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from axquant.analyzer import architecture_prior_report
    from axquant.inspector import inspect_model
    from axquant.planner import plan_quantization
    from axquant.recipes import export_recipe_bundle
    from axquant.schema import PlanRequest, ProfileName
    from axquant.serde import write_data

    source_revision = "a" * 40
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision=source_revision,
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
    plan_path = tmp_path / "bundle-plan.json"
    write_data(plan_path, plan)
    bundle_path = export_recipe_bundle(
        plan=plan_path,
        output_dir=tmp_path / "bundle",
        bundle_id="qwen36-27b-prior-r1",
    )

    _install_fake_mlx(monkeypatch, qwen36_model_dir)
    output = tmp_path / "recipe-candidate"
    summary = quick_convert(
        model=str(qwen36_model_dir),
        output=output,
        model_id="Qwen/Qwen3.6-27B",
        revision=source_revision,
        recipe=bundle_path,
        ax_engine_manifest="skip",
    )
    assert summary.plan_source == "recipe-bundle"
    assert summary.recipe_bundle_id == "qwen36-27b-prior-r1"
    assert summary.profile is ProfileName.AGENT_CODING
    assert summary.target_bpw == pytest.approx(14.0)
    assert any("recipe bundle qwen36-27b-prior-r1" in note for note in summary.notes)
    assert (output / "axquant_manifest.json").is_file()
