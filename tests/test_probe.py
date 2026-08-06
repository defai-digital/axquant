"""Tests for the measured sensitivity probe backend (v0.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from _capture_helpers import load_test_activation_capture

from axquant.activation_cache import tokenize_calibration
from axquant.capture_binding import (
    CAPTURE_MANIFEST_SHA256_KEY,
    LoadedActivationCapture,
)
from axquant.errors import BackendUnavailableError, ProbeError
from axquant.inspector import inspect_model
from axquant.probe import (
    ForwardResult,
    MlxProbeBackend,
    ProbeState,
    _candidate_bits_for_tensor,
    _module_group_for_tensor,
    compute_cosine_distance,
    compute_hidden_state_error,
    compute_kl_divergence,
    compute_token_disagreement,
    probe_tensor_sensitivity,
)
from axquant.schema import (
    ActivationCaptureManifest,
    CalibrationManifest,
    EvidenceKind,
    ModelIdentity,
    ProbeConfig,
    ProbeProgress,
    ProfileName,
    QuantMethod,
    SensitivityReport,
    SupportTier,
    TensorRole,
    TokenizedCacheManifest,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    special_tokens_map: ClassVar[dict[str, str]] = {}

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "</s>": 2}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> list[int]:
        del add_special_tokens, truncation
        return ([ord(character) % 13 + 3 for character in text] + [2])[:max_length]


class _MeasuredFakeBackend:
    def __init__(self) -> None:
        self.bits: int | None = None
        self.method: QuantMethod | None = None
        self.methods: list[QuantMethod] = []
        self.restorations = 0

    def load_model(self, model_dir: Path) -> None:
        assert model_dir.is_dir()

    def quantize_module(
        self,
        module_path: str,
        bits: int,
        group_size: int,
        method: QuantMethod = QuantMethod.AFFINE,
    ) -> None:
        assert module_path
        assert group_size == 64
        self.bits = bits
        self.method = method
        self.methods.append(method)

    def restore_module(self, module_path: str) -> None:
        assert module_path
        if self.bits is not None:
            self.restorations += 1
        self.bits = None
        self.method = None

    def forward(self, input_ids: object) -> ForwardResult:
        ids = np.asarray(input_ids, dtype=np.int64)
        positions = max(1, len(ids) - 1)
        logits = np.zeros((1, positions, 16), dtype=np.float32)
        targets = ids[1 : positions + 1] % 16
        for position, target in enumerate(targets):
            logits[0, position, target] = 4.0
        perturbation = 0.0 if self.bits is None else (16 - self.bits) / 16
        if self.method == QuantMethod.DWQ:
            perturbation *= 0.9
        logits[..., 0] += perturbation
        hidden = np.broadcast_to(ids[:positions][None, :, None], (1, positions, 4)).astype(
            np.float32
        )
        hidden = hidden + perturbation
        return ForwardResult(
            logits=logits,
            hidden_states=hidden,
            loss=1.0 + perturbation,
            token_count=len(ids),
            peak_memory_bytes=1024 + int(perturbation * 100),
            latency_seconds=0.01 + perturbation,
        )


@pytest.fixture
def probe_config() -> ProbeConfig:
    return ProbeConfig(
        model=ModelIdentity(model_id="test-model", revision="abc123", local_path="/tmp/model"),
        calibration_cache="/tmp/cache",
        candidate_bits=(4, 6, 8, 16),
        group_size=64,
        token_budget_per_candidate=1024,
        module_group_probing=True,
        early_termination_factor=3.0,
    )


class TestProbeConfigValidation:
    def test_valid_config(self, probe_config: ProbeConfig) -> None:
        assert probe_config.candidate_bits == (4, 6, 8, 16)
        assert probe_config.group_size == 64
        assert probe_config.early_termination_factor == 3.0

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError, match="probe candidate bits"):
            ProbeConfig(
                model=ModelIdentity(model_id="m"),
                calibration_cache="/tmp",
                candidate_bits=(1, 32),
            )

    def test_bits_normalized(self) -> None:
        config = ProbeConfig(
            model=ModelIdentity(model_id="m"),
            calibration_cache="/tmp",
            candidate_bits=(8, 4, 16, 6, 4),
        )
        assert config.candidate_bits == (4, 6, 8, 16)

    def test_invalid_early_termination(self) -> None:
        with pytest.raises(ValueError):
            ProbeConfig(
                model=ModelIdentity(model_id="m"),
                calibration_cache="/tmp",
                early_termination_factor=0.5,
            )

    def test_probe_methods_are_executable_and_normalized(self) -> None:
        config = ProbeConfig(
            model=ModelIdentity(model_id="m"),
            calibration_cache="/tmp",
            candidate_methods=(QuantMethod.DWQ, QuantMethod.AFFINE, QuantMethod.DWQ),
        )
        assert config.candidate_methods == (QuantMethod.AFFINE, QuantMethod.DWQ)
        refined = ProbeConfig(
            model=ModelIdentity(model_id="m"),
            calibration_cache="/tmp",
            candidate_methods=(QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.AFFINE),
        )
        assert refined.candidate_methods == (
            QuantMethod.AFFINE,
            QuantMethod.AWQ,
            QuantMethod.GPTQ,
        )
        with pytest.raises(ValueError, match="probe methods"):
            ProbeConfig(
                model=ModelIdentity(model_id="m"),
                calibration_cache="/tmp",
                candidate_methods=(QuantMethod.BF16,),
            )

    def test_capture_points_are_validated_and_deduplicated(self) -> None:
        config = ProbeConfig(
            model=ModelIdentity(model_id="m"),
            calibration_cache="/tmp",
            capture_points=("output", "hidden", "output"),
        )
        assert config.capture_points == ("output", "hidden")

        with pytest.raises(ValueError, match="probe capture points"):
            ProbeConfig(
                model=ModelIdentity(model_id="m"),
                calibration_cache="/tmp",
                capture_points=(),
            )
        with pytest.raises(ValueError, match="probe capture points"):
            ProbeConfig(
                model=ModelIdentity(model_id="m"),
                calibration_cache="/tmp",
                capture_points=("logits",),
            )


class TestModuleGrouping:
    def test_attention_group(self) -> None:
        result = _module_group_for_tensor("model.layers.5.self_attn.q_proj.weight")
        assert result == "model.layers.5.self_attn"

    def test_mlp_group(self) -> None:
        result = _module_group_for_tensor("model.layers.12.mlp.down_proj.weight")
        assert result == "model.layers.12.mlp"

    def test_no_group_for_embedding(self) -> None:
        result = _module_group_for_tensor("model.embed_tokens.weight")
        assert result is None

    def test_no_group_for_norm(self) -> None:
        result = _module_group_for_tensor("model.norm.weight")
        assert result is None


class TestMetricComputation:
    def test_kl_divergence_identical(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_kl_divergence(p, p)
        assert abs(result) < 1e-10

    def test_kl_divergence_different(self) -> None:
        p = np.array([0.9, 0.05, 0.03, 0.02])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_kl_divergence(p, q)
        assert result > 0.0

    def test_kl_divergence_2d(self) -> None:
        p = np.array([[0.7, 0.3], [0.5, 0.5]])
        q = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = compute_kl_divergence(p, q)
        assert result > 0.0

    def test_hidden_state_error_identical(self) -> None:
        h = np.random.randn(10, 64)
        result = compute_hidden_state_error(h, h)
        assert abs(result) < 1e-10

    def test_hidden_state_error_different(self) -> None:
        h1 = np.zeros((10, 64))
        h2 = np.ones((10, 64))
        result = compute_hidden_state_error(h1, h2)
        assert abs(result - 1.0) < 1e-10

    def test_cosine_distance_identical(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        result = compute_cosine_distance(v, v)
        assert abs(result) < 1e-10

    def test_cosine_distance_identical_high_dimension_clamps_nonnegative(self) -> None:
        # At high dimension, dot()/  (norm * norm) can round to fractionally above 1.0 for
        # a vector compared with itself, which would otherwise make 1.0 - similarity
        # negative (observed in production: -1.665e-16 on a real hidden-state probe).
        # MetricVector.cosine_distance requires ge=0.0, so the result must clamp to 0.0.
        np.random.seed(1)
        v = np.random.randn(8192)
        result = compute_cosine_distance(v, v)
        assert result >= 0.0

    def test_cosine_distance_orthogonal(self) -> None:
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        result = compute_cosine_distance(v1, v2)
        assert abs(result - 1.0) < 1e-10

    def test_cosine_distance_zero_vector(self) -> None:
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 1.0])
        result = compute_cosine_distance(v1, v2)
        assert result == 1.0

    def test_token_disagreement_identical(self) -> None:
        tokens = np.array([1, 5, 3, 7, 2])
        result = compute_token_disagreement(tokens, tokens)
        assert result == 0.0

    def test_token_disagreement_all_different(self) -> None:
        t1 = np.array([1, 2, 3, 4])
        t2 = np.array([5, 6, 7, 8])
        result = compute_token_disagreement(t1, t2)
        assert result == 1.0

    def test_token_disagreement_partial(self) -> None:
        t1 = np.array([1, 2, 3, 4])
        t2 = np.array([1, 2, 7, 8])
        result = compute_token_disagreement(t1, t2)
        assert abs(result - 0.5) < 1e-10

    def test_token_disagreement_shape_mismatch(self) -> None:
        t1 = np.array([1, 2, 3])
        t2 = np.array([1, 2])
        with pytest.raises(ProbeError, match="shape mismatch"):
            compute_token_disagreement(t1, t2)


class TestProbeState:
    def test_initial_state(self) -> None:
        state = ProbeState()
        assert state.total_tensors == 0
        assert not state.is_tensor_complete("any_tensor")

    def test_record_and_check(self) -> None:
        state = ProbeState()
        state.record_tensor("tensor_a", [])
        assert state.is_tensor_complete("tensor_a")
        assert not state.is_tensor_complete("tensor_b")


class TestMlxProbeBackend:
    def test_backend_unavailable_without_mlx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        def unavailable(name: str) -> None:
            raise ImportError(f"test has no optional backend: {name}")

        backend = MlxProbeBackend()
        monkeypatch.setattr(importlib, "import_module", unavailable)
        with pytest.raises(BackendUnavailableError, match="requires mlx and mlx-lm"):
            backend._ensure_mlx()

    def test_quantize_without_load(self) -> None:
        backend = MlxProbeBackend()
        backend._mlx = True  # Fake MLX availability
        with pytest.raises(ProbeError, match="model not loaded"):
            backend.quantize_module("some.module", 4, 64)

    def test_forward_without_load(self) -> None:
        backend = MlxProbeBackend()
        backend._mlx = True  # Fake MLX availability
        with pytest.raises(ProbeError, match="model not loaded"):
            backend.forward(None)


def test_probe_replays_verified_tokens_and_emits_measured_evidence(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    identity = ModelIdentity(
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
        local_path=str(qwen36_model_dir),
    )
    inventory = inspect_model(
        qwen36_model_dir,
        model_id=identity.model_id,
        revision=identity.revision,
    )
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"text": "repair this function"}),
                json.dumps({"text": "return valid JSON"}),
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    calibration = CalibrationManifest(
        model=identity,
        profile=ProfileName.AGENT_CODING,
        dataset_id=str(dataset),
        dataset_sha256=file_sha256(dataset),
        samples=2,
        domains=[],
        sequence_length=32,
        random_seed=11,
        calibration_evaluation_separation_attested=True,
    )
    cache.mkdir()
    write_data(cache / "calibration_manifest.json", calibration)
    cache_manifest = tokenize_calibration(
        model=identity,
        dataset_path=dataset,
        output_dir=cache,
        profile=ProfileName.AGENT_CODING,
        sequence_length=32,
        random_seed=11,
        tokenizer=_FakeTokenizer(),
        calibration_manifest_sha256=stable_sha256(
            calibration.model_dump(mode="json", exclude={"created_at"})
        ),
        separation_attested=True,
    )
    backend = _MeasuredFakeBackend()
    progress_path = tmp_path / "probe-progress.json"
    config = ProbeConfig(
        model=identity,
        calibration_cache=str(cache),
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 16),
        group_size=64,
        token_budget_per_candidate=32,
    )
    report = probe_tensor_sensitivity(
        inventory,
        config=config,
        backend=backend,
        state_path=progress_path,
    )

    assert report.evidence_kind == EvidenceKind.MEASURED_DEVELOPMENT
    assert not report.evidence_kind.release_quality
    assert report.calibration is not None
    assert report.calibration.dataset_id == str(dataset)
    assert report.calibration.dataset_sha256 == cache_manifest.dataset_sha256
    assert report.calibration.samples == cache_manifest.samples
    assert report.calibration.metadata["cache_key_sha256"] == cache_manifest.cache_key_sha256
    quantized_entry = next(
        entry
        for entry in report.entries
        if any(candidate.bits == 4 for candidate in entry.candidates)
    )
    four_bit = next(candidate for candidate in quantized_entry.candidates if candidate.bits == 4)
    assert four_bit.measured_tokens > 0
    assert four_bit.metrics.output_kl > 0
    assert four_bit.metrics.hidden_state_error > 0
    assert four_bit.metrics.task_loss_delta > 0
    assert 1.0 <= four_bit.metrics.peak_memory_cost < 2.0
    assert four_bit.metrics.prefill_latency_cost > 1.0
    bf16 = next(candidate for candidate in quantized_entry.candidates if candidate.bits == 16)
    assert bf16.metrics.peak_memory_cost == 1.0
    assert bf16.metrics.prefill_latency_cost == 1.0
    assert backend.restorations > 0
    assert load_model(progress_path, ProbeProgress).complete

    resumed_backend = _MeasuredFakeBackend()
    resumed = probe_tensor_sensitivity(
        inventory,
        config=config,
        backend=resumed_backend,
        state_path=progress_path,
    )
    assert resumed.inventory_sha256 == report.inventory_sha256
    assert resumed_backend.restorations == 0

    target_tensor = quantized_entry.tensor.name
    refined_backend = _MeasuredFakeBackend()
    refined = probe_tensor_sensitivity(
        inventory,
        config=config.model_copy(
            update={
                "candidate_bits": (4,),
                "candidate_methods": (QuantMethod.DWQ,),
                "target_tensors": (target_tensor,),
            }
        ),
        backend=refined_backend,
        state_path=tmp_path / "dwq-progress.json",
        base_report=report,
    )
    refined_entry = next(entry for entry in refined.entries if entry.tensor.name == target_tensor)
    dwq = next(
        candidate
        for candidate in refined_entry.candidates
        if candidate.bits == 4 and candidate.method == QuantMethod.DWQ
    )
    assert dwq.measured_tokens == four_bit.measured_tokens
    assert dwq.metrics.output_kl < four_bit.metrics.output_kl
    assert dwq.metrics.peak_memory_cost == four_bit.metrics.peak_memory_cost
    assert dwq.metrics.prefill_latency_cost == four_bit.metrics.prefill_latency_cost
    assert refined.calibration is not None
    assert refined.calibration.metadata["base_sensitivity_sha256"] == stable_sha256(report)
    assert refined.calibration.metadata["candidate_methods"] == "dwq"
    assert refined_backend.methods == [QuantMethod.DWQ]
    untouched = next(entry for entry in refined.entries if entry.tensor.name != target_tensor)
    base_untouched = next(
        entry for entry in report.entries if entry.tensor.name == untouched.tensor.name
    )
    assert untouched == base_untouched

    # AXQ-017: a tier promotion after the base probe must not invalidate the
    # measured contract — the tier is registry policy, not evidence. Rebuild
    # the base as it would have been recorded before promotion and extend it
    # against the current (promoted) inventory.
    stale_profile = report.architecture_profile.model_copy(
        update={"support_tier": SupportTier.INSPECT_ONLY}
    )
    stale_inventory = inventory.model_copy(
        update={
            "architecture_profile": stale_profile,
            "warnings": list(stale_profile.notes),
        }
    )
    stale_report = report.model_copy(
        update={
            "architecture_profile": stale_profile,
            "inventory_sha256": stable_sha256(
                stale_inventory.model_dump(mode="json", exclude={"created_at"})
            ),
        }
    )
    promoted = probe_tensor_sensitivity(
        inventory,
        config=config.model_copy(
            update={
                "candidate_bits": (4,),
                "candidate_methods": (QuantMethod.DWQ,),
                "target_tensors": (target_tensor,),
            }
        ),
        backend=_MeasuredFakeBackend(),
        state_path=tmp_path / "stale-tier-progress.json",
        base_report=stale_report,
    )
    promoted_entry = next(entry for entry in promoted.entries if entry.tensor.name == target_tensor)
    assert any(
        candidate.bits == 4 and candidate.method == QuantMethod.DWQ
        for candidate in promoted_entry.candidates
    )

    # Reports recorded before AXQ-017 hashed an inventory serialization with
    # no support_tier key at all; the legacy reconstruction must accept them.
    legacy_dump = stale_inventory.model_dump(mode="json", exclude={"created_at"})
    legacy_dump["architecture_profile"].pop("support_tier", None)
    legacy_report = stale_report.model_copy(update={"inventory_sha256": stable_sha256(legacy_dump)})
    legacy = probe_tensor_sensitivity(
        inventory,
        config=config.model_copy(
            update={
                "candidate_bits": (4,),
                "candidate_methods": (QuantMethod.DWQ,),
                "target_tensors": (target_tensor,),
            }
        ),
        backend=_MeasuredFakeBackend(),
        state_path=tmp_path / "legacy-hash-progress.json",
        base_report=legacy_report,
    )
    assert any(
        candidate.bits == 4 and candidate.method == QuantMethod.DWQ
        for entry in legacy.entries
        if entry.tensor.name == target_tensor
        for candidate in entry.candidates
    )


def _base_probe_report(
    tmp_path: Path,
    qwen36_model_dir: Path,
) -> tuple[object, object, object]:
    """Build a tiny tokenized cache and a base AFFINE probe report for refinement tests."""
    identity = ModelIdentity(
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
        local_path=str(qwen36_model_dir),
    )
    inventory = inspect_model(
        qwen36_model_dir,
        model_id=identity.model_id,
        revision=identity.revision,
    )
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"text": "repair this function"}),
                json.dumps({"text": "return valid JSON"}),
            ]
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    calibration = CalibrationManifest(
        model=identity,
        profile=ProfileName.AGENT_CODING,
        dataset_id=str(dataset),
        dataset_sha256=file_sha256(dataset),
        samples=2,
        domains=[],
        sequence_length=32,
        random_seed=11,
        calibration_evaluation_separation_attested=True,
    )
    cache.mkdir()
    write_data(cache / "calibration_manifest.json", calibration)
    tokenize_calibration(
        model=identity,
        dataset_path=dataset,
        output_dir=cache,
        profile=ProfileName.AGENT_CODING,
        sequence_length=32,
        random_seed=11,
        tokenizer=_FakeTokenizer(),
        calibration_manifest_sha256=stable_sha256(
            calibration.model_dump(mode="json", exclude={"created_at"})
        ),
        separation_attested=True,
    )
    config = ProbeConfig(
        model=identity,
        calibration_cache=str(cache),
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 16),
        group_size=64,
        token_budget_per_candidate=32,
    )
    report = probe_tensor_sensitivity(
        inventory,
        config=config,
        backend=_MeasuredFakeBackend(),
        state_path=tmp_path / "probe-progress.json",
    )
    return inventory, config, report


def _bound_probe_capture(
    tmp_path: Path,
    config: ProbeConfig,
    report: SensitivityReport,
    activations: dict[str, np.ndarray],
) -> LoadedActivationCapture:
    cache_manifest = load_model(
        Path(config.calibration_cache) / "tokenized_cache_manifest.json",
        TokenizedCacheManifest,
    )
    assert report.calibration is not None
    manifest = ActivationCaptureManifest(
        model=config.model.model_id,
        revision=config.model.revision,
        tokenized_cache_manifest_sha256=stable_sha256(cache_manifest),
        cache_key_sha256=cache_manifest.cache_key_sha256,
        calibration_dataset_id=report.calibration.dataset_id,
        max_rows=max(rows.shape[0] for rows in activations.values()),
    )
    return load_test_activation_capture(
        tmp_path / "capture",
        manifest=manifest,
        activations=activations,
    )


def test_probe_requires_calibration_activations_for_awq_gptq(
    qwen36_model_dir: Path,
) -> None:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    for method in (QuantMethod.AWQ, QuantMethod.GPTQ):
        config = ProbeConfig(
            model=inventory.model,
            calibration_cache="/nonexistent-cache",
            candidate_methods=(method,),
        )
        with pytest.raises(ProbeError, match="capture-activations"):
            probe_tensor_sensitivity(inventory, config=config)


def test_probe_rejects_mutable_revision_alias(qwen36_model_dir: Path) -> None:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="main",
    )
    config = ProbeConfig(
        model=inventory.model,
        calibration_cache="/nonexistent-cache",
    )

    with pytest.raises(ProbeError, match="revision-pinned"):
        probe_tensor_sensitivity(inventory, config=config)


def test_probe_rejects_model_identity_mismatch(qwen36_model_dir: Path) -> None:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    mismatched_id = inventory.model.model_copy(update={"model_id": "org/other-model"})
    config = ProbeConfig(model=mismatched_id, calibration_cache="/nonexistent-cache")
    with pytest.raises(ProbeError, match="does not match the inventory model"):
        probe_tensor_sensitivity(inventory, config=config)

    mismatched_path = inventory.model.model_copy(
        update={"local_path": str(qwen36_model_dir.parent)}
    )
    config = ProbeConfig(model=mismatched_path, calibration_cache="/nonexistent-cache")
    with pytest.raises(ProbeError, match="does not match the inventory source"):
        probe_tensor_sensitivity(inventory, config=config)


def test_probe_awq_gptq_refinement_normalizes_hardware_costs(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    inventory, config, report = _base_probe_report(tmp_path, qwen36_model_dir)
    quantized_entry = next(
        entry
        for entry in report.entries
        if any(candidate.bits == 4 for candidate in entry.candidates)
    )
    four_bit = next(candidate for candidate in quantized_entry.candidates if candidate.bits == 4)
    target_tensor = quantized_entry.tensor.name
    activations = {target_tensor.removesuffix(".weight"): np.zeros((4, 8), dtype=np.float32)}
    bound_activations = _bound_probe_capture(tmp_path, config, report, activations)

    with pytest.raises(ProbeError, match="unbound activation mapping"):
        probe_tensor_sensitivity(
            inventory,
            config=config.model_copy(
                update={
                    "candidate_bits": (4,),
                    "candidate_methods": (QuantMethod.AWQ,),
                    "target_tensors": (target_tensor,),
                }
            ),
            backend=_MeasuredFakeBackend(),
            base_report=report,
            calibration_activations=activations,
        )

    for method, note_text in (
        (QuantMethod.AWQ, "activation-aware channel scaling"),
        (QuantMethod.GPTQ, "hessian error-compensated rounding"),
    ):
        refined = probe_tensor_sensitivity(
            inventory,
            config=config.model_copy(
                update={
                    "candidate_bits": (4,),
                    "candidate_methods": (method,),
                    "target_tensors": (target_tensor,),
                }
            ),
            backend=_MeasuredFakeBackend(),
            state_path=tmp_path / f"{method.value}-progress.json",
            base_report=report,
            calibration_activations=bound_activations,
        )
        refined_entry = next(
            entry for entry in refined.entries if entry.tensor.name == target_tensor
        )
        candidate = next(
            candidate
            for candidate in refined_entry.candidates
            if candidate.bits == 4 and candidate.method == method
        )
        assert candidate.supported
        assert candidate.measured_tokens == four_bit.measured_tokens
        assert candidate.metrics.peak_memory_cost == four_bit.metrics.peak_memory_cost
        assert candidate.metrics.prefill_latency_cost == four_bit.metrics.prefill_latency_cost
        assert candidate.note is not None
        assert note_text in candidate.note
        assert "hardware costs normalized" in candidate.note
        assert refined.calibration is not None
        assert (
            refined.calibration.metadata[CAPTURE_MANIFEST_SHA256_KEY]
            == bound_activations.manifest_sha256
        )


def test_probe_refinement_skips_failed_affine_packing_control(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    """A failed AFFINE placeholder must not become the refinement hardware-cost control."""
    inventory, config, _report = _base_probe_report(tmp_path, qwen36_model_dir)

    class _AffineFailingBackend(_MeasuredFakeBackend):
        def quantize_module(
            self,
            module_path: str,
            bits: int,
            group_size: int,
            method: QuantMethod = QuantMethod.AFFINE,
        ) -> None:
            if method == QuantMethod.AFFINE:
                raise ProbeError("simulated affine packing failure")
            super().quantize_module(module_path, bits, group_size, method)

    report = probe_tensor_sensitivity(
        inventory,
        config=config.model_copy(
            update={
                "candidate_bits": (4,),
                "candidate_methods": (QuantMethod.AFFINE, QuantMethod.DWQ),
            }
        ),
        backend=_AffineFailingBackend(),
    )
    quantized_entry = next(
        entry
        for entry in report.entries
        if any(candidate.bits == 4 for candidate in entry.candidates)
    )
    affine = next(
        candidate
        for candidate in quantized_entry.candidates
        if candidate.bits == 4 and candidate.method == QuantMethod.AFFINE
    )
    assert not affine.supported
    dwq = next(
        candidate
        for candidate in quantized_entry.candidates
        if candidate.bits == 4 and candidate.method == QuantMethod.DWQ
    )
    assert dwq.supported
    assert dwq.metrics.peak_memory_cost > 0.0
    assert dwq.metrics.prefill_latency_cost > 0.0
    assert dwq.note is not None
    assert "hardware costs normalized" not in dwq.note


def test_probe_rejects_capture_bound_to_another_cache(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    inventory, config, report = _base_probe_report(tmp_path, qwen36_model_dir)
    target = next(tensor for tensor in inventory.tensors if tensor.quantizable)
    activations = {
        target.module_path: np.zeros((4, target.shape[-1]), dtype=np.float32),
    }
    capture = _bound_probe_capture(tmp_path, config, report, activations)
    wrong_manifest = capture.manifest.model_copy(update={"cache_key_sha256": "0" * 64})
    wrong_capture = load_test_activation_capture(
        tmp_path / "wrong-capture",
        manifest=wrong_manifest,
        activations=activations,
    )

    with pytest.raises(ProbeError, match="cache key does not match"):
        probe_tensor_sensitivity(
            inventory,
            config=config.model_copy(
                update={
                    "candidate_bits": (4,),
                    "candidate_methods": (QuantMethod.AWQ,),
                    "target_tensors": (target.name,),
                }
            ),
            backend=_MeasuredFakeBackend(),
            base_report=report,
            calibration_activations=wrong_capture,
        )


class TestMlxProbeBackendRefinement:
    def test_awq_gptq_quantize_forward_restore(self, tmp_path: Path) -> None:
        pytest.importorskip("mlx_lm")
        import test_capture as _tc

        model_dir = tmp_path / "tiny-llama"
        _tc._write_tiny_llama(model_dir)
        module_path = "model.layers.0.self_attn.q_proj"
        rng = np.random.default_rng(0)
        activations = {module_path: rng.standard_normal((16, 32)).astype(np.float32)}
        tokens = np.array([[3, 4, 5, 6, 7, 8]], dtype=np.int32)

        for method in (QuantMethod.AWQ, QuantMethod.GPTQ):
            backend = MlxProbeBackend(calibration_activations=activations)
            backend.load_model(model_dir)
            _, _, module = backend._get_parent_and_module(module_path)
            original_weight = np.asarray(module.weight)
            reference = backend.forward(tokens)
            backend.quantize_module(module_path, 4, 32, method)
            _, _, quantized = backend._get_parent_and_module(module_path)
            assert quantized is not module
            measured = backend.forward(tokens)
            assert measured.loss is not None
            assert reference.loss is not None
            backend.restore_module(module_path)
            _, _, restored = backend._get_parent_and_module(module_path)
            assert restored is module
            assert np.array_equal(np.asarray(restored.weight), original_weight)

    def test_awq_gptq_missing_activations_fail_closed(self, tmp_path: Path) -> None:
        pytest.importorskip("mlx_lm")
        import test_capture as _tc

        model_dir = tmp_path / "tiny-llama"
        _tc._write_tiny_llama(model_dir)
        for method in (QuantMethod.AWQ, QuantMethod.GPTQ):
            backend = MlxProbeBackend()
            backend.load_model(model_dir)
            with pytest.raises(ProbeError, match="capture-activations"):
                backend.quantize_module("model.layers.0.self_attn.q_proj", 4, 32, method)
            backend.restore_module("model.layers.0.self_attn.q_proj")


def test_probe_role_floors_keep_embedding_measurable(
    qwen36_model_dir: Path,
) -> None:
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    config = ProbeConfig(
        model=inventory.model,
        calibration_cache="/tmp/cache",
        candidate_bits=(4, 6, 8, 16),
    )
    mlp = next(
        tensor
        for tensor in inventory.tensors
        if tensor.quantizable and tensor.role == TensorRole.MLP
    )
    embedding = mlp.model_copy(
        update={"role": TensorRole.EMBEDDING, "protected_recommendation": True}
    )
    head = mlp.model_copy(update={"role": TensorRole.LM_HEAD, "protected_recommendation": True})
    assert embedding.protected_recommendation is True
    assert _candidate_bits_for_tensor(embedding, config) == (8, 16)
    # AXQ-026: the probe measures the LM head down to 8-bit so the governed
    # lowered floor is backed by measurement; the planner default stays BF16.
    assert _candidate_bits_for_tensor(head, config) == (8, 16)
    assert _candidate_bits_for_tensor(mlp, config) == (4, 6, 8, 16)
