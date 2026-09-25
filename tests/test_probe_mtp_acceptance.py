"""MH6 (AXQ-046): measured MTP acceptance proxy losses in the probe.

Locks the optional ProbeBackend MTP-forward capability: capable backends
produce strictly-positive mtp_acceptance_loss values for MTP-role candidates
(epsilon floor on perfect agreement), capability-less backends keep the
honest 0.0 unmeasured marker, and the MH1 gate (planner.enforce_mtp_acceptance_measured)
interacts with both as specified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from axquant.activation_cache import tokenize_calibration
from axquant.errors import PlanningError, ProbeError
from axquant.inspector import inspect_model
from axquant.planner import enforce_mtp_acceptance_measured, objective_for
from axquant.probe import (
    MTP_ACCEPTANCE_LOSS_EPSILON,
    ForwardResult,
    MlxProbeBackend,
    MtpForwardResult,
    _mtp_forward_capability,
    probe_tensor_sensitivity,
)
from axquant.schema import (
    CalibrationManifest,
    EvidenceKind,
    ModelIdentity,
    ProbeConfig,
    ProfileName,
    QuantMethod,
    SensitivityReport,
)
from axquant.serde import file_sha256, stable_sha256, write_data


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


class _MtpCapableFakeBackend:
    """Fake backend with the MH6 MTP-forward capability.

    Draft logits always argmax to token 0 for the reference model; a
    quantized MTP module disagrees on a ``disagreement_fraction`` of
    positions.
    """

    supports_mtp_forward = True

    def __init__(self, disagreement_fraction: float = 0.5) -> None:
        self.bits: int | None = None
        self.disagreement_fraction = disagreement_fraction
        self.mtp_forwards = 0

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

    def restore_module(self, module_path: str) -> None:
        assert module_path
        self.bits = None

    def forward(self, input_ids: object) -> ForwardResult:
        ids = np.asarray(input_ids, dtype=np.int64)
        positions = max(1, len(ids) - 1)
        logits = np.zeros((1, positions, 16), dtype=np.float32)
        targets = ids[1 : positions + 1] % 16
        for position, target in enumerate(targets):
            logits[0, position, target] = 4.0
        perturbation = 0.0 if self.bits is None else (16 - self.bits) / 16
        logits[..., 0] += perturbation
        hidden = np.broadcast_to(ids[:positions][None, :, None], (1, positions, 4)).astype(
            np.float32
        )
        return ForwardResult(
            logits=logits,
            hidden_states=hidden,
            loss=1.0 + perturbation,
            token_count=len(ids),
            peak_memory_bytes=1024,
            latency_seconds=0.01,
        )

    def forward_mtp(self, input_ids: object, hidden_states: object) -> MtpForwardResult:
        del input_ids
        self.mtp_forwards += 1
        hidden = np.asarray(hidden_states)
        draft = np.zeros((hidden.shape[0], hidden.shape[1], 16), dtype=np.float32)
        if self.bits is not None and self.disagreement_fraction > 0:
            every = max(1, round(1.0 / self.disagreement_fraction))
            draft[:, ::every, 1] = 1.0
        return MtpForwardResult(draft_logits=draft)


class _CapabilityLessFakeBackend:
    """Fake backend without the optional MH6 capability (probe-v6 behavior)."""

    def __init__(self) -> None:
        self.bits: int | None = None

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

    def restore_module(self, module_path: str) -> None:
        assert module_path
        self.bits = None

    def forward(self, input_ids: object) -> ForwardResult:
        ids = np.asarray(input_ids, dtype=np.int64)
        positions = max(1, len(ids) - 1)
        logits = np.zeros((1, positions, 16), dtype=np.float32)
        targets = ids[1 : positions + 1] % 16
        for position, target in enumerate(targets):
            logits[0, position, target] = 4.0
        perturbation = 0.0 if self.bits is None else (16 - self.bits) / 16
        logits[..., 0] += perturbation
        hidden = np.broadcast_to(ids[:positions][None, :, None], (1, positions, 4)).astype(
            np.float32
        )
        return ForwardResult(
            logits=logits,
            hidden_states=hidden,
            loss=1.0 + perturbation,
            token_count=len(ids),
            peak_memory_bytes=1024,
            latency_seconds=0.01,
        )


def _run_probe(
    model_dir: Path,
    tmp_path: Path,
    backend: object,
    *,
    candidate_bits: tuple[int, ...] = (4, 8, 16),
) -> SensitivityReport:
    identity = ModelIdentity(
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
        local_path=str(model_dir),
    )
    inventory = inspect_model(
        model_dir,
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
    cache = tmp_path / f"cache-{candidate_bits!r}"
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
        candidate_bits=candidate_bits,
        group_size=64,
        token_budget_per_candidate=32,
    )
    return probe_tensor_sensitivity(
        inventory,
        config=config,
        backend=backend,  # type: ignore[arg-type]
    )


def _mtp_entries(report: SensitivityReport) -> list[Any]:
    entries = [entry for entry in report.entries if entry.tensor.role.is_mtp]
    assert entries, "probe report must contain MTP-role entries"
    return entries


def test_capability_helper_detects_only_fully_declared_backends() -> None:
    assert _mtp_forward_capability(_CapabilityLessFakeBackend()) is None
    incomplete = _MtpCapableFakeBackend()
    incomplete.forward_mtp = None  # type: ignore[method-assign,assignment]
    assert _mtp_forward_capability(incomplete) is None  # type: ignore[arg-type]
    capable = _MtpCapableFakeBackend()
    assert _mtp_forward_capability(capable) is capable  # type: ignore[arg-type]


def test_mtp_forward_measures_positive_acceptance_loss_and_gate_passes(
    qwen36_integrated_mtp_model_dir: Path,
    tmp_path: Path,
) -> None:
    backend = _MtpCapableFakeBackend(disagreement_fraction=0.5)
    report = _run_probe(qwen36_integrated_mtp_model_dir, tmp_path, backend)

    mtp_losses = [
        candidate.metrics.mtp_acceptance_loss
        for entry in _mtp_entries(report)
        for candidate in entry.candidates
    ]
    assert all(value > 0 for value in mtp_losses)
    quantized = [
        candidate
        for entry in _mtp_entries(report)
        for candidate in entry.candidates
        if candidate.bits < 16
    ]
    assert quantized, "integrated MTP tensors must enter the candidate grid"
    assert all(0.4 < candidate.metrics.mtp_acceptance_loss < 0.6 for candidate in quantized)
    bf16 = [
        candidate
        for entry in _mtp_entries(report)
        for candidate in entry.candidates
        if candidate.bits == 16
    ]
    # Reference precision is the untouched MTP module: perfect draft agreement
    # floors at epsilon instead of the zero unmeasured marker.
    assert all(
        candidate.metrics.mtp_acceptance_loss == MTP_ACCEPTANCE_LOSS_EPSILON for candidate in bf16
    )
    # Per-candidate provenance is recorded on the candidate notes.
    assert all(
        candidate.note is not None and "teacher-forced MTP forward" in candidate.note
        for candidate in quantized + bf16
    )
    # The field stays 0.0 outside the MTP scope.
    non_mtp_losses = [
        candidate.metrics.mtp_acceptance_loss
        for entry in report.entries
        if not entry.tensor.role.is_mtp
        for candidate in entry.candidates
    ]
    assert non_mtp_losses and all(value == 0.0 for value in non_mtp_losses)
    assert report.evidence_kind == EvidenceKind.MEASURED_DEVELOPMENT
    assert report.calibration is not None
    assert report.calibration.metadata["mtp_acceptance_provenance"] == "mtp-forward"
    assert any("measured by teacher-forced MTP forward probes" in w for w in report.warnings)
    assert backend.mtp_forwards > 0

    # MH1 interplay: measured non-zero MTP losses must not trip the gate.
    weights = objective_for(ProfileName.AGENT_CODING)
    assert weights.mtp_acceptance_loss > 0
    assert enforce_mtp_acceptance_measured(report, weights) == []


def test_capability_less_backend_keeps_zero_marker_and_gate_fails_closed(
    qwen36_integrated_mtp_model_dir: Path,
    tmp_path: Path,
) -> None:
    report = _run_probe(qwen36_integrated_mtp_model_dir, tmp_path, _CapabilityLessFakeBackend())

    mtp_candidates = [candidate for entry in _mtp_entries(report) for candidate in entry.candidates]
    assert all(candidate.metrics.mtp_acceptance_loss == 0.0 for candidate in mtp_candidates)
    assert all(
        candidate.note is not None and "unmeasured" in candidate.note
        for candidate in mtp_candidates
    )
    assert report.calibration is not None
    assert report.calibration.metadata["mtp_acceptance_provenance"] == "unmeasured"

    weights = objective_for(ProfileName.AGENT_CODING)
    with pytest.raises(PlanningError, match="--allow-mtp-unmeasured"):
        enforce_mtp_acceptance_measured(report, weights)
    warnings = enforce_mtp_acceptance_measured(report, weights, allow_mtp_unmeasured=True)
    assert warnings


def test_perfect_agreement_floors_at_epsilon(
    qwen36_integrated_mtp_model_dir: Path,
    tmp_path: Path,
) -> None:
    backend = _MtpCapableFakeBackend(disagreement_fraction=0.0)
    report = _run_probe(qwen36_integrated_mtp_model_dir, tmp_path, backend)

    mtp_losses = [
        candidate.metrics.mtp_acceptance_loss
        for entry in _mtp_entries(report)
        for candidate in entry.candidates
    ]
    assert mtp_losses
    # Strictly positive by construction: the MH1 gate reads any non-zero loss
    # as measured, so a perfect candidate floors at epsilon rather than 0.0.
    assert all(value == MTP_ACCEPTANCE_LOSS_EPSILON for value in mtp_losses)
    assert enforce_mtp_acceptance_measured(report, objective_for(ProfileName.AGENT_CODING)) == []


def test_preserved_mtp_tensor_records_epsilon_when_capable(
    qwen36_integrated_mtp_model_dir: Path,
    tmp_path: Path,
) -> None:
    # candidate_bits without an MTP-legal width preserves integrated MTP
    # tensors at reference precision.
    report = _run_probe(
        qwen36_integrated_mtp_model_dir,
        tmp_path,
        _MtpCapableFakeBackend(),
        candidate_bits=(4, 16),
    )

    mtp_candidates = [candidate for entry in _mtp_entries(report) for candidate in entry.candidates]
    assert all(candidate.bits == 16 for candidate in mtp_candidates)
    assert all(
        candidate.metrics.mtp_acceptance_loss == MTP_ACCEPTANCE_LOSS_EPSILON
        for candidate in mtp_candidates
    )
    assert all(
        candidate.note is not None and "teacher-forced MTP forward" in candidate.note
        for candidate in mtp_candidates
    )
    assert enforce_mtp_acceptance_measured(report, objective_for(ProfileName.AGENT_CODING)) == []


def test_mlx_backend_mtp_capability_detection() -> None:
    class _Block:
        def __call__(self, hidden: object) -> object:
            return hidden

    class _Embed:
        def __call__(self, tokens: object) -> object:
            return tokens

    class _Model:
        def __init__(self, named: list[tuple[str, object]]) -> None:
            self._named = named

        def named_modules(self) -> list[tuple[str, object]]:
            return list(self._named)

    backend = MlxProbeBackend()
    backend._model = _Model(
        [
            ("language_model.model.embed_tokens", _Embed()),
            ("language_model.mtp", _Block()),
            ("language_model.mtp.layers.0", _Block()),
        ]
    )
    backend._detect_mtp_forward_capability()
    # Only the outermost MTP container is kept as a draft block.
    assert backend.supports_mtp_forward
    assert len(backend._mtp_blocks) == 1

    without_mtp = MlxProbeBackend()
    without_mtp._model = _Model([("model.embed_tokens", _Embed())])
    without_mtp._detect_mtp_forward_capability()
    assert not without_mtp.supports_mtp_forward

    without_embedding = MlxProbeBackend()
    without_embedding._model = _Model([("mtp", _Block())])
    without_embedding._detect_mtp_forward_capability()
    assert not without_embedding.supports_mtp_forward


def test_mlx_backend_forward_mtp_fails_closed() -> None:
    backend = MlxProbeBackend()
    with pytest.raises(ProbeError, match="model not loaded"):
        backend.forward_mtp(None, None)
    backend = MlxProbeBackend()
    backend.supports_mtp_forward = False
    backend._model = object()
    with pytest.raises(ProbeError, match="no usable MTP modules"):
        backend.forward_mtp(None, None)
