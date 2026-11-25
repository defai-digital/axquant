"""Real-hardware end-to-end integration tests (stability release v1.1.1, Phase C).

These tests run the full measured pipeline on a tiny real llama-family
checkpoint with the actual MLX/MLX-LM backend: tokenized calibration cache →
activation capture → measured MXFP4 sensitivity probe → planner → real
``mlx_lm.convert`` conversion into AXQ-070 product storage (MXFP4 / MXFP8 /
BF16). They are gated to Apple Silicon macOS and carry
the ``integration`` marker so ``-m "not integration"`` deselects them.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import axquant.capture as capture_module

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import test_capture as _tc
from safetensors import safe_open

from axquant.activation_cache import tokenize_calibration
from axquant.converter import convert_model
from axquant.errors import CaptureError, PlanningError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.probe import probe_tensor_sensitivity
from axquant.schema import (
    ActivationCaptureManifest,
    HardwareProfile,
    ModelIdentity,
    PlanRequest,
    ProbeConfig,
    ProfileName,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantMethod,
    SupportTier,
)
from axquant.schema.sensitivity import SensitivityReport
from axquant.serde import load_model, write_data
from axquant.source_binding import build_source_plan_binding

CAPTURE_ACTIVATIONS_DIR = capture_module.CAPTURE_ACTIVATIONS_DIR
CAPTURE_MANIFEST_NAME = capture_module.CAPTURE_MANIFEST_NAME
capture_calibration_activations = capture_module.capture_calibration_activations
load_capture_activations = capture_module.load_capture_activations

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="real-hardware integration tests require Apple Silicon macOS",
    ),
]

# The tiny llama checkpoint is inspected under a MiniCPM5 reference: the
# MiniCPM5 dense family is a plain llama-arch export promoted to the
# convertible tier, so planning and conversion pass the fail-closed scope
# gates without weakening them.
_MODEL_ID = "openbmb/MiniCPM5-1B"
_REVISION = "a" * 40
_GROUP_SIZE = 32
_BITS = (4,)

_TEXTS = (
    "def sort_list(items): return sorted(items) and filter them all",
    "Fix the bug in this function and add regression coverage now",
    "Generate a JSON response with every field populated fully",
    "Refactor the parser into smaller composable helper pieces",
    "Write unit tests for the tokenizer edge cases today please",
    "Document the public API surface with accurate examples here",
    "Profile the hot loop and remove the extra allocations soon",
    "Validate all checksums before publishing the artifact bundle",
)


def _build_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-llama"
    _tc._write_tiny_llama(model_dir)
    return model_dir


def _tokenize(tmp_path: Path, *, sequence_length: int, name: str = "cache") -> Path:
    dataset = tmp_path / f"{name}.jsonl"
    dataset.write_text(
        "\n".join(json.dumps({"text": text}) for text in _TEXTS),
        encoding="utf-8",
    )
    cache_dir = tmp_path / name
    tokenize_calibration(
        model=ModelIdentity(model_id=_MODEL_ID, revision=_REVISION),
        dataset_path=dataset,
        output_dir=cache_dir,
        profile=ProfileName.AGENT_CODING,
        sequence_length=sequence_length,
        random_seed=7,
        tokenizer=_tc._SmallTokenizer(),
        separation_attested=True,
    )
    return cache_dir


def _candidate(report: SensitivityReport, tensor: str, method: QuantMethod):
    entry = next(entry for entry in report.entries if entry.tensor.name == tensor)
    return next(
        candidate
        for candidate in entry.candidates
        if candidate.bits == _BITS[0]
        and candidate.method == method
        and candidate.group_size == _GROUP_SIZE
    )


def test_end_to_end_capture_probe_plan_convert(tmp_path: Path) -> None:
    model_dir = _build_model(tmp_path)
    cache_dir = _tokenize(tmp_path, sequence_length=32)

    # --- capture: real replay of the verified cache through the BF16 model ---
    capture_dir = tmp_path / "capture"
    capture_manifest = capture_calibration_activations(
        model_dir=model_dir,
        cache_dir=cache_dir,
        output_dir=capture_dir,
        max_rows=16,
        token_budget=256,
    )
    assert capture_manifest.model == _MODEL_ID
    assert capture_manifest.entries
    assert (capture_dir / CAPTURE_MANIFEST_NAME).is_file()
    assert (capture_dir / "completion.json").is_file()
    loaded = load_capture_activations(capture_dir, model=_MODEL_ID, revision=_REVISION)
    assert set(loaded) == {entry.module_path for entry in capture_manifest.entries}
    assert "model.layers.0.self_attn.q_proj" in loaded

    # --- inspect + measured MXFP4 probe on real hardware ---
    inventory = inspect_model(model_dir, model_id=_MODEL_ID, revision=_REVISION)
    assert inventory.architecture_profile.support_tier is SupportTier.CONVERTIBLE

    report = probe_tensor_sensitivity(
        inventory,
        config=ProbeConfig(
            model=inventory.model,
            calibration_cache=str(cache_dir),
            candidate_bits=_BITS,
            candidate_methods=(QuantMethod.MXFP4,),
            group_size=_GROUP_SIZE,
            token_budget_per_candidate=128,
            metric_positions_per_sample=8,
            # The MLX probe backend extracts hidden states only through the
            # multimodal ``language_model.model`` wrapper; a plain llama-family
            # backbone yields logits only, so probe the output capture point.
            capture_points=("output",),
        ),
    )
    assert report.evidence_kind.release_quality or report.calibration is not None

    probed = [
        entry.tensor.name
        for entry in report.entries
        if any(candidate.bits < 16 for candidate in entry.candidates)
    ]
    assert probed, "expected at least one tensor with quantized candidates"
    for tensor in probed:
        candidate = _candidate(report, tensor, QuantMethod.MXFP4)
        # Loose, non-flaky sanity bound on the random tiny-weight fixture.
        assert np.isfinite(candidate.metrics.output_kl)
        assert np.isfinite(candidate.metrics.task_loss_delta)

    # --- plan: MXFP4 trunk from the measured report ---
    request = PlanRequest(
        profile=ProfileName.AGENT_CODING,
        target_bpw=8.0,
        candidate_bits=(4, 8, 16),
        group_size=_GROUP_SIZE,
        candidate_methods=(QuantMethod.MXFP4,),
        allow_unmeasured=True,
        # AXQ-070 product hardware: MXFP4 / MXFP8 slot / BF16 at group 32.
        hardware=HardwareProfile(
            supported_bits=(4, 8, 16),
            supported_methods=(QuantMethod.AFFINE, QuantMethod.MXFP4, QuantMethod.BF16),
            supported_group_sizes=(_GROUP_SIZE,),
        ),
    )
    plan = plan_quantization(report, request)
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)
    plan = load_model(plan_path, QuantizationPlan)
    quantized = [allocation for allocation in plan.assignments if allocation.bits < 16]
    assert quantized, "expected quantized allocations at 4-bit"
    assert all(allocation.method == QuantMethod.MXFP4 for allocation in quantized)
    assert all(allocation.bits == _BITS[0] for allocation in quantized)

    # --- convert: real mlx_lm.convert into MXFP4 storage ---
    output_dir = tmp_path / "converted"
    artifact = convert_model(
        source_binding=_source_binding(plan, model_dir),
        model=str(model_dir),
        plan=plan,
        output=output_dir,
        allow_unmeasured=True,
        ax_engine_manifest="skip",
    )
    assert artifact.plan_sha256
    assert output_dir.is_dir()

    weight_files = sorted(output_dir.glob("*.safetensors"))
    assert weight_files, "converted checkpoint has no safetensors weights"
    keys: list[str] = []
    for weight_file in weight_files:
        with safe_open(str(weight_file), framework="numpy") as handle:
            keys.extend(handle.keys())
    quantized_scales = {key[: -len(".scales")] for key in keys if key.endswith(".scales")}
    assert "model.layers.0.self_attn.q_proj" in quantized_scales
    # MXFP4 carries E8M0 scales only; affine bias metadata must not appear.
    assert not any(key.endswith(".biases") for key in keys)
    # Protected embedding stays BF16: no quantized scales sidecar for it.
    assert "model.embed_tokens.weight" in keys
    assert "model.embed_tokens" not in quantized_scales
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["quantization"]["mode"] == "mxfp4"
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["group_size"] == _GROUP_SIZE

    execution = load_model(
        output_dir / "axquant_quantizer_execution.json",
        QuantizerExecutionManifest,
    )
    assert len(execution.records) == len(quantized)
    for record in execution.records:
        assert record.success
        assert record.method == QuantMethod.MXFP4
        assert record.note is not None and record.note.endswith("mxfp4 packing")
        assert record.metadata == {}

    # --- fail closed: refined affine storage is outside AXQ-070 ---
    gptq_plan = plan.model_copy(
        update={
            "assignments": [
                allocation.model_copy(update={"method": QuantMethod.GPTQ})
                if allocation.bits < 16
                else allocation
                for allocation in plan.assignments
            ]
        }
    )
    with pytest.raises(PlanningError, match="AXQ-070"):
        convert_model(
            source_binding=_source_binding(plan, model_dir),
            model=str(model_dir),
            plan=gptq_plan,
            output=tmp_path / "converted-gptq",
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )
    assert not (tmp_path / "converted-gptq").exists()


def test_capture_resume_matches_uninterrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted segmented capture resumes to the single-shot artifact."""
    model_dir = _build_model(tmp_path)
    cache_dir = _tokenize(tmp_path, sequence_length=16, name="cache-multi")
    output_dir = tmp_path / "capture"
    control_dir = tmp_path / "capture-control"

    real_replay = capture_module._replay_segment
    calls = {"count": 0}

    def boom(model: object, mlx: object, batches: list[object]) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        real_replay(model, mlx, batches)

    monkeypatch.setattr(capture_module, "_replay_segment", boom)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        capture_calibration_activations(
            model_dir=model_dir,
            cache_dir=cache_dir,
            output_dir=output_dir,
            max_rows=8,
            segment_batches=1,
        )
    assert (output_dir / "capture_progress.json").is_file()
    assert (output_dir / CAPTURE_ACTIVATIONS_DIR / ".partial").is_dir()
    assert not (output_dir / "completion.json").exists()
    with pytest.raises(CaptureError, match="incomplete"):
        load_capture_activations(output_dir, model=_MODEL_ID)

    monkeypatch.undo()
    resumed = capture_calibration_activations(
        model_dir=model_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        max_rows=8,
        segment_batches=1,
    )
    assert not (output_dir / "capture_progress.json").exists()
    assert (output_dir / "completion.json").is_file()

    control = capture_calibration_activations(
        model_dir=model_dir,
        cache_dir=cache_dir,
        output_dir=control_dir,
        max_rows=8,
        segment_batches=1,
    )
    assert isinstance(resumed, ActivationCaptureManifest)
    assert resumed.model_dump(mode="json", exclude={"created_at"}) == control.model_dump(
        mode="json", exclude={"created_at"}
    )
    resumed_rows = load_capture_activations(output_dir, model=_MODEL_ID, revision=_REVISION)
    control_rows = load_capture_activations(control_dir, model=_MODEL_ID, revision=_REVISION)
    assert set(resumed_rows) == set(control_rows)
    for name in resumed_rows:
        np.testing.assert_array_equal(resumed_rows[name], control_rows[name])


def _source_binding(plan: QuantizationPlan, source_dir: Path) -> object:
    """The binding production writes beside the plan (AXQ-048)."""

    return build_source_plan_binding(plan, source_dir)
