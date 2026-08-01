"""End-to-end integration test: inspect -> calibrate -> tokenize -> probe -> plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from axquant.activation_cache import tokenize_calibration
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.probe import ForwardResult, probe_tensor_sensitivity
from axquant.schema import (
    CalibrationManifest,
    EvidenceKind,
    ModelIdentity,
    MtpPolicy,
    PlanRequest,
    ProbeConfig,
    ProfileName,
    QuantMethod,
    TensorRole,
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
        return ([ord(c) % 13 + 3 for c in text] + [2])[:max_length]


class _FakeBackend:
    def __init__(self) -> None:
        self.bits: int | None = None
        self.method: QuantMethod | None = None
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
        self.bits = bits
        self.method = method

    def restore_module(self, module_path: str) -> None:
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


def test_full_measured_pipeline_closes_loop(tiny_model_dir: Path, tmp_path: Path) -> None:
    identity = ModelIdentity(
        model_id="test/tiny-model",
        revision="abc123",
        local_path=str(tiny_model_dir),
    )

    inventory = inspect_model(
        tiny_model_dir,
        model_id=identity.model_id,
        revision=identity.revision,
    )
    inventory_sha = stable_sha256(inventory.model_dump(mode="json", exclude={"created_at"}))

    dataset = tmp_path / "calibration.jsonl"
    samples = [
        {"domain": "coding", "text": "Implement a binary search tree with insert and delete."},
        {"domain": "json", "text": "Parse this JSON schema and validate the payload structure."},
        {"domain": "tool", "text": "Call the weather API with location parameter set to Tokyo."},
        {"domain": "multilingual", "text": "Translate the following paragraph into Japanese."},
        {"domain": "long-context", "text": "A" * 2100},
    ]
    dataset.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")

    cache = tmp_path / "cache"
    calibration = CalibrationManifest(
        model=identity,
        profile=ProfileName.AGENT_CODING,
        dataset_id=str(dataset),
        dataset_sha256="",
        samples=len(samples),
        domains=[],
        sequence_length=64,
        random_seed=42,
        calibration_evaluation_separation_attested=True,
    )
    cache_manifest = tokenize_calibration(
        model=identity,
        dataset_path=dataset,
        output_dir=cache,
        profile=ProfileName.AGENT_CODING,
        sequence_length=64,
        random_seed=42,
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

    backend = _FakeBackend()
    config = ProbeConfig(
        model=identity,
        calibration_cache=str(cache),
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 8, 16),
        group_size=64,
        token_budget_per_candidate=64,
    )
    report = probe_tensor_sensitivity(
        inventory,
        config=config,
        backend=backend,
        state_path=tmp_path / "progress.json",
    )

    assert report.evidence_kind in (EvidenceKind.MEASURED, EvidenceKind.MEASURED_DEVELOPMENT)
    assert report.calibration is not None
    assert report.inventory_sha256 == inventory_sha
    assert backend.restorations > 0

    quantizable_entries = [e for e in report.entries if e.tensor.quantizable]
    assert len(quantizable_entries) > 0
    entries_with_low_bits = [
        e for e in quantizable_entries if any(c.bits < 16 for c in e.candidates)
    ]
    assert len(entries_with_low_bits) >= 2
    for entry in entries_with_low_bits:
        measured = [c for c in entry.candidates if c.bits < 16]
        assert all(c.metrics.output_kl > 0 for c in measured)

    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
            mtp=MtpPolicy(mode="protected"),
        ),
    )

    assert plan.effective_bpw <= 14.0
    assert plan.evidence_kind == report.evidence_kind

    plan_tensors = {a.tensor for a in plan.assignments}
    inventory_tensors = {t.name for t in inventory.tensors}
    assert plan_tensors == inventory_tensors

    for allocation in plan.assignments:
        if allocation.role == TensorRole.NORM:
            assert allocation.bits >= 16
        if allocation.role == TensorRole.LM_HEAD:
            assert allocation.bits >= 16
