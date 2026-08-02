from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from axquant.activation_cache import tokenize_calibration
from axquant.errors import PlanningError, ProbeError
from axquant.inspector import inspect_model
from axquant.kv_probe import _kv_candidate_metrics, measure_kv_sensitivity
from axquant.planner import allocate_kv_cache_measured
from axquant.schema import (
    CalibrationManifest,
    EvidenceKind,
    KvSensitivityReport,
    ModelIdentity,
    ProfileName,
)
from axquant.serde import stable_sha256, write_data


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


class _FakeKvBackend:
    """Deterministic backend: layer 0 is far more KV-sensitive than the rest."""

    backend_id = "fake-kv"

    def __init__(self, *, quantizable: set[int] | None = None) -> None:
        self.loaded: Path | None = None
        self._quantizable = quantizable

    def load_model(self, model_dir: Path) -> None:
        assert model_dir.is_dir()
        self.loaded = model_dir

    def quantizable_layers(self) -> set[int]:
        return set(range(64)) if self._quantizable is None else self._quantizable

    def forward_logits(
        self,
        input_ids: Any,
        *,
        layer_bits: dict[int, int] | None,
        group_size: int,
    ) -> Any:
        assert group_size == 64
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        positions = max(1, len(ids) - 1)
        logits = np.zeros((1, positions, 16), dtype=np.float32)
        targets = ids[1 : positions + 1] % 16
        for position, target in enumerate(targets):
            logits[0, position, target] = 4.0
        if layer_bits:
            ((layer, bits),) = layer_bits.items()
            weight = 4.0 if layer == 0 else 0.01
            logits[..., 0] += weight * (16 - bits) / 16
        return logits


def _calibration_cache(tmp_path: Path, identity: ModelIdentity) -> Path:
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
        dataset_sha256="",
        samples=2,
        domains=[],
        sequence_length=32,
        random_seed=11,
        calibration_evaluation_separation_attested=True,
    )
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
    calibration.dataset_sha256 = cache_manifest.dataset_sha256
    calibration.domains = cache_manifest.domains
    write_data(cache / "calibration_manifest.json", calibration)
    cache_manifest.calibration_manifest_sha256 = stable_sha256(
        calibration.model_dump(mode="json", exclude={"created_at"})
    )
    write_data(cache / "tokenized_cache_manifest.json", cache_manifest)
    return cache


def _measured_report(qwen36_model_dir: Path, tmp_path: Path) -> KvSensitivityReport:
    identity = ModelIdentity(
        model_id="Qwen/Qwen3.6-27B",
        revision="revision-pinned",
        local_path=str(qwen36_model_dir),
    )
    inventory = inspect_model(
        qwen36_model_dir,
        model_id=identity.model_id,
        revision=identity.revision,
    )
    cache = _calibration_cache(tmp_path, identity)
    return measure_kv_sensitivity(
        inventory,
        model_dir=qwen36_model_dir,
        calibration_cache=cache,
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 8),
        token_budget=32,
        backend=_FakeKvBackend(),
    )


def test_measure_kv_sensitivity_covers_layers_with_monotone_metrics(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    report = _measured_report(qwen36_model_dir, tmp_path)
    assert report.evidence_kind is EvidenceKind.MEASURED_DEVELOPMENT
    assert report.probe_backend == "fake-kv"
    assert report.calibration is not None
    assert report.text_layer_count == 64
    assert len(report.entries) == 64
    sensitive = next(entry for entry in report.entries if entry.layer_index == 0)
    four_bit = next(candidate for candidate in sensitive.candidates if candidate.bits == 4)
    eight_bit = next(candidate for candidate in sensitive.candidates if candidate.bits == 8)
    bf16 = next(candidate for candidate in sensitive.candidates if candidate.bits == 16)
    assert four_bit.metrics.output_kl > eight_bit.metrics.output_kl > 0.0
    assert bf16.metrics.output_kl == 0.0
    assert four_bit.measured_tokens > 0
    interior = next(entry for entry in report.entries if entry.layer_index == 5)
    interior_four = next(candidate for candidate in interior.candidates if candidate.bits == 4)
    assert interior_four.metrics.output_kl < four_bit.metrics.output_kl


def test_measured_kv_allocation_spends_bits_on_sensitive_layers(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    report = _measured_report(qwen36_model_dir, tmp_path)
    interior = next(entry for entry in report.entries if entry.layer_index == 5)
    interior_four = next(candidate for candidate in interior.candidates if candidate.bits == 4)
    budget = interior_four.metrics.output_kl * 2
    plan = allocate_kv_cache_measured(report, max_output_kl=budget)
    assert plan.allocation_basis == "measured"
    assert plan.sensitivity_sha256 == stable_sha256(report)
    by_layer = {layer.layer_index: layer for layer in plan.layers}
    assert by_layer[0].bits == 16  # sensitive layer exceeds the budget at 4 and 8 bit
    assert by_layer[5].bits == 4
    assert "measured output KL" in by_layer[5].reason
    assert "no quantized KV candidate met" in by_layer[0].reason
    assert len(plan.layers) == report.text_layer_count


def test_measured_kv_allocation_rejects_prior_reports_and_bad_budget(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    report = _measured_report(qwen36_model_dir, tmp_path)
    with pytest.raises(PlanningError, match="budget must be positive"):
        allocate_kv_cache_measured(report, max_output_kl=0.0)
    prior = report.model_copy(update={"evidence_kind": EvidenceKind.ARCHITECTURE_PRIOR})
    with pytest.raises(PlanningError, match="measured sensitivity report"):
        allocate_kv_cache_measured(prior)


def test_measure_kv_sensitivity_requires_pinned_revision(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    inventory = inspect_model(qwen36_model_dir, model_id="Qwen/Qwen3.6-27B")
    with pytest.raises(ProbeError, match="revision-pinned"):
        measure_kv_sensitivity(
            inventory,
            model_dir=qwen36_model_dir,
            calibration_cache=tmp_path / "missing",
            profile=ProfileName.AGENT_CODING,
            backend=_FakeKvBackend(),
        )


def test_converter_binds_measured_kv_report(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    from axquant.analyzer import architecture_prior_report
    from axquant.converter import _validated_kv_sensitivity_source
    from axquant.planner import plan_quantization
    from axquant.schema import PlanRequest

    report = _measured_report(qwen36_model_dir, tmp_path)
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="revision-pinned",
    )
    prior = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(
        prior,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )
    plan.kv_cache = allocate_kv_cache_measured(report, max_output_kl=10.0)
    report_path = tmp_path / "kv_sensitivity.json"
    write_data(report_path, report)

    with pytest.raises(PlanningError, match="requires --kv-sensitivity"):
        _validated_kv_sensitivity_source(plan, None)
    assert _validated_kv_sensitivity_source(plan, report_path) == report_path.resolve()

    tampered = report.model_copy(update={"probe_backend": "someone-else"})
    tampered_path = tmp_path / "tampered.json"
    write_data(tampered_path, tampered)
    with pytest.raises(PlanningError, match="digest does not match"):
        _validated_kv_sensitivity_source(plan, tampered_path)

    plan.kv_cache = None
    with pytest.raises(PlanningError, match="no measured KV-cache section"):
        _validated_kv_sensitivity_source(plan, report_path)


def test_publication_gate_reproduces_measured_kv_allocation(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    from axquant.analyzer import architecture_prior_report
    from axquant.errors import ValidationGateError
    from axquant.planner import plan_quantization
    from axquant.reporting import _verify_measured_kv_plan
    from axquant.schema import PlanRequest

    report = _measured_report(qwen36_model_dir, tmp_path)
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="revision-pinned",
    )
    prior = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(
        prior,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )
    plan.kv_cache = allocate_kv_cache_measured(report, max_output_kl=10.0)
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    with pytest.raises(ValidationGateError, match=r"packaged kv_sensitivity\.json"):
        _verify_measured_kv_plan(artifact, plan)

    write_data(artifact / "kv_sensitivity.json", report)
    _verify_measured_kv_plan(artifact, plan)

    assert plan.kv_cache is not None
    drifted = plan.kv_cache.model_copy(
        update={
            "layers": [
                layer.model_copy(update={"bits": 16, "reason": "hand-edited"})
                for layer in plan.kv_cache.layers
            ]
        }
    )
    plan.kv_cache = drifted
    with pytest.raises(ValidationGateError, match="cannot be reproduced"):
        _verify_measured_kv_plan(artifact, plan)

    plan.kv_cache = allocate_kv_cache_measured(report, max_output_kl=10.0).model_copy(
        update={"max_output_kl": None}
    )
    with pytest.raises(ValidationGateError, match="selection budget"):
        _verify_measured_kv_plan(artifact, plan)


def test_hybrid_architecture_marks_non_kv_layers_unsupported(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    identity = ModelIdentity(
        model_id="Qwen/Qwen3.6-27B",
        revision="revision-pinned",
        local_path=str(qwen36_model_dir),
    )
    inventory = inspect_model(
        qwen36_model_dir,
        model_id=identity.model_id,
        revision=identity.revision,
    )
    cache = _calibration_cache(tmp_path, identity)
    # Only layers 3 and 7 have standard KV caches (hybrid linear-attention model).
    report = measure_kv_sensitivity(
        inventory,
        model_dir=qwen36_model_dir,
        calibration_cache=cache,
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 8),
        token_budget=32,
        backend=_FakeKvBackend(quantizable={3, 7}),
    )
    recurrent = next(entry for entry in report.entries if entry.layer_index == 0)
    assert all(not candidate.supported for candidate in recurrent.candidates if candidate.bits < 16)
    attention = next(entry for entry in report.entries if entry.layer_index == 3)
    assert all(candidate.supported for candidate in attention.candidates)

    plan = allocate_kv_cache_measured(report, max_output_kl=10.0)
    by_layer = {layer.layer_index: layer for layer in plan.layers}
    assert by_layer[0].bits == 16
    assert "not quantizable" in by_layer[0].reason
    assert by_layer[3].bits == 4
    assert by_layer[7].bits == 4


def test_kv_candidate_metrics_applies_softmax_before_kl() -> None:
    # Real logits are unbounded and signed. `_kv_candidate_metrics` must
    # soft-max them into probabilities before computing KL — feeding raw
    # logits into a clip-and-renormalize KL (the pre-fix behavior) silently
    # floors every negative logit to the same value and produces a
    # numerically meaningless "measured" sensitivity score.
    reference_logits = np.array([[[2.0, -1.0, 0.5]]], dtype=np.float32)
    candidate_logits = np.array([[[2.0, -1.0, -3.0]]], dtype=np.float32)

    metrics = _kv_candidate_metrics(
        [reference_logits],
        [candidate_logits],
        metric_positions=1,
    )

    def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> float:
        ref_p = np.exp(ref - ref.max())
        ref_p = ref_p / ref_p.sum()
        cand_p = np.exp(cand - cand.max())
        cand_p = cand_p / cand_p.sum()
        return float(np.sum(ref_p * np.log(ref_p / cand_p)))

    expected = _softmax_kl(reference_logits[0, 0], candidate_logits[0, 0])
    assert metrics.output_kl == pytest.approx(expected, rel=1e-5)
