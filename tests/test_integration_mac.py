"""Real-hardware end-to-end integration tests (stability release v1.1.1, Phase C).

These tests run the full measured pipeline on a tiny real llama-family
checkpoint with the actual MLX/MLX-LM backend: tokenized calibration cache →
activation capture → measured AWQ/GPTQ sensitivity probe → planner → real
``mlx_lm.convert`` conversion. They are gated to Apple Silicon macOS and carry
the ``integration`` marker so ``-m "not integration"`` deselects them.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import test_capture as _tc
from safetensors import safe_open

from axquant.activation_cache import tokenize_calibration
from axquant.capture import (
    CAPTURE_ACTIVATIONS_DIR,
    CAPTURE_MANIFEST_NAME,
    capture_calibration_activations,
    load_capture_activations,
)
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
_REVISION = "rev0"
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

    # --- inspect + measured probe over AFFINE/AWQ/GPTQ on real hardware ---
    inventory = inspect_model(model_dir, model_id=_MODEL_ID, revision=_REVISION)
    assert inventory.architecture_profile.support_tier is SupportTier.CONVERTIBLE

    report = probe_tensor_sensitivity(
        inventory,
        config=ProbeConfig(
            model=inventory.model,
            calibration_cache=str(cache_dir),
            candidate_bits=_BITS,
            candidate_methods=(QuantMethod.AFFINE, QuantMethod.AWQ, QuantMethod.GPTQ),
            group_size=_GROUP_SIZE,
            token_budget_per_candidate=128,
            metric_positions_per_sample=8,
            # The MLX probe backend extracts hidden states only through the
            # multimodal ``language_model.model`` wrapper; a plain llama-family
            # backbone yields logits only, so probe the output capture point.
            capture_points=("output",),
        ),
        calibration_activations=loaded,
    )
    assert report.evidence_kind.release_quality or report.calibration is not None

    probed = [
        entry.tensor.name
        for entry in report.entries
        if any(candidate.bits < 16 for candidate in entry.candidates)
    ]
    assert probed, "expected at least one tensor with quantized candidates"
    for tensor in probed:
        affine = _candidate(report, tensor, QuantMethod.AFFINE)
        for method in (QuantMethod.AWQ, QuantMethod.GPTQ):
            refined = _candidate(report, tensor, method)
            # Refinement methods share the identical affine packing, so the
            # hardware-cost fields must reuse the AFFINE control exactly.
            assert refined.metrics.peak_memory_cost == affine.metrics.peak_memory_cost
            assert refined.metrics.prefill_latency_cost == affine.metrics.prefill_latency_cost
            assert refined.metrics.decode_latency_cost == affine.metrics.decode_latency_cost

    # Quality gate (loose, non-flaky): hidden_state_error is unavailable here
    # (the backend returns hidden states only for multimodal wrapper models),
    # so the primary gate is the specified output_kl bound. On this random
    # tiny-weight fixture output_kl saturates near zero for every method, so
    # the discriminating check is task_loss_delta under the same loose bound:
    # GPTQ refines each layer against its measured inputs and must stay within
    # reach of the AFFINE control. This catches a catastrophically broken
    # implementation; it cannot prove superiority on random weights.
    for tensor in probed:
        affine = _candidate(report, tensor, QuantMethod.AFFINE)
        gptq = _candidate(report, tensor, QuantMethod.GPTQ)
        kl_bound = max(
            affine.metrics.output_kl * 1.5,
            affine.metrics.output_kl + 1e-3,
        )
        assert gptq.metrics.output_kl <= kl_bound, (
            f"{tensor}: GPTQ output_kl {gptq.metrics.output_kl} "
            f"exceeds loose AFFINE bound {kl_bound}"
        )
        loss_bound = max(
            affine.metrics.task_loss_delta * 1.5,
            affine.metrics.task_loss_delta + 1e-3,
        )
        assert gptq.metrics.task_loss_delta <= loss_bound, (
            f"{tensor}: GPTQ task_loss_delta {gptq.metrics.task_loss_delta} "
            f"exceeds loose AFFINE bound {loss_bound}"
        )

    # --- plan: force the refinement method via the candidate-method filter ---
    request = PlanRequest(
        profile=ProfileName.AGENT_CODING,
        target_bpw=8.0,
        group_size=_GROUP_SIZE,
        candidate_methods=(QuantMethod.GPTQ,),
        allow_unmeasured=True,
        hardware=HardwareProfile(),
    )
    plan = plan_quantization(report, request)
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)
    plan = load_model(plan_path, QuantizationPlan)
    quantized = [allocation for allocation in plan.assignments if allocation.bits < 16]
    assert quantized, "expected quantized allocations at 4-bit"
    assert all(allocation.method == QuantMethod.GPTQ for allocation in quantized)
    assert all(allocation.bits == _BITS[0] for allocation in quantized)

    # --- convert: real mlx_lm.convert with GPTQ refinement from the capture ---
    output_dir = tmp_path / "converted"
    artifact = convert_model(
        model=str(model_dir),
        plan=plan,
        output=output_dir,
        calibration_activations=loaded,
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
    # Protected embedding stays BF16: no quantized scales sidecar for it.
    assert "model.embed_tokens.weight" in keys
    assert "model.embed_tokens" not in quantized_scales

    execution = load_model(
        output_dir / "axquant_quantizer_execution.json",
        QuantizerExecutionManifest,
    )
    gptq_records = [record for record in execution.records if record.method == QuantMethod.GPTQ]
    assert len(gptq_records) == len(quantized)
    for record in gptq_records:
        assert record.success
        assert record.note is not None and "GPTQ" in record.note
        assert "gptq_damping" in record.metadata
        assert record.metadata["bits"] == _BITS[0]
        assert record.metadata["group_size"] == _GROUP_SIZE

    # --- fail closed: refinement plan without calibration activations ---
    # build_quant_predicate rejects the plan before any conversion work.
    with pytest.raises(PlanningError, match="calibration activations"):
        convert_model(
            model=str(model_dir),
            plan=plan,
            output=tmp_path / "converted-no-calibration",
            calibration_activations=None,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )
    assert not (tmp_path / "converted-no-calibration").exists()


def test_capture_resume_matches_uninterrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted segmented capture resumes to the single-shot artifact."""
    model_dir = _build_model(tmp_path)
    cache_dir = _tokenize(tmp_path, sequence_length=16, name="cache-multi")
    output_dir = tmp_path / "capture"
    control_dir = tmp_path / "capture-control"

    import axquant.capture as capture_module

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
