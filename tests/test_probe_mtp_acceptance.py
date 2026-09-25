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
    _canonical_mtp_entries,
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


class _LateCapableFakeBackend(_MtpCapableFakeBackend):
    """Capability that only appears once ``load_model`` inspected the tree.

    Mirrors ``MlxProbeBackend``, which initializes ``supports_mtp_forward`` to
    False and only sets it inside ``load_model``.
    """

    supports_mtp_forward = False

    def load_model(self, model_dir: Path) -> None:
        super().load_model(model_dir)
        self.supports_mtp_forward = True


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


def test_mtp_capability_resolves_after_load_model(
    qwen36_integrated_mtp_model_dir: Path,
    tmp_path: Path,
) -> None:
    """Regression: read the capability only after the backend has loaded.

    ``MlxProbeBackend`` initializes ``supports_mtp_forward`` to False and only
    sets it inside ``load_model``. Resolving the capability before the load
    therefore reported every probe as capability-less: all MTP candidates kept
    the 0.0 marker and the provenance claimed the backend had no MTP forward,
    silently disabling the measured-acceptance path.
    """

    backend = _LateCapableFakeBackend()
    assert not backend.supports_mtp_forward
    report = _run_probe(qwen36_integrated_mtp_model_dir, tmp_path, backend)

    assert backend.mtp_forwards > 0
    assert report.calibration is not None
    assert report.calibration.metadata["mtp_acceptance_provenance"] == "mtp-forward"
    mtp_losses = [
        candidate.metrics.mtp_acceptance_loss
        for entry in _mtp_entries(report)
        for candidate in entry.candidates
    ]
    assert any(value > 0.0 for value in mtp_losses)
    assert enforce_mtp_acceptance_measured(report, objective_for(ProfileName.AGENT_CODING)) == []


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
    pytest.importorskip("mlx.core")

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
    pytest.importorskip("mlx.core")

    backend = MlxProbeBackend()
    with pytest.raises(ProbeError, match="model not loaded"):
        backend.forward_mtp(None, None)
    backend = MlxProbeBackend()
    backend.supports_mtp_forward = False
    backend._model = object()
    with pytest.raises(ProbeError, match="no usable MTP modules"):
        backend.forward_mtp(None, None)


# --- Qwen3.5/Qwen3Next sidecar reconstruction (real-checkpoint convention) ---
#
# MLX-LM sanitizers strip integrated ``mtp.*`` weights at load, so on these
# architectures the loaded model tree exposes no MTP modules. The probe
# rebuilds the block from the checkpoint directory: dedicated mtp.safetensors
# or mtp.* keys inside the indexed shards. The layout below mirrors the real
# AutomatosX AX-Ornith-1.5-9B-MLX-AXQ-6bit-MTP pack (Qwen3_5 arch, 1 MTP
# layer) validated on hardware on 2026-09-24: norm weights are stored shifted
# (gamma - 1, +1.0 added back at reconstruction), and the fusion order is
# fc(cat([pre_fc_norm_embedding(embed(t+1)), pre_fc_norm_hidden(hidden_t)])).

_SIDE_HIDDEN = 128


def _sidecar_tensor_names() -> list[str]:
    names = [
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
    ]
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        names.append(f"mtp.layers.0.self_attn.{projection}.weight")
    for norm in ("q_norm", "k_norm"):
        names.append(f"mtp.layers.0.self_attn.{norm}.weight")
    for part in ("input_layernorm", "post_attention_layernorm"):
        names.append(f"mtp.layers.0.{part}.weight")
    for projection in ("gate_proj", "up_proj", "down_proj"):
        names.append(f"mtp.layers.0.mlp.{projection}.weight")
    return names


def _write_synthetic_sidecar(
    model_dir: Path,
    *,
    hidden: int = _SIDE_HIDDEN,
    omit: tuple[str, ...] = (),
    rename: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Write a synthetic mtp.safetensors with the real Qwen3.5 key layout."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(7)
    arrays: dict[str, np.ndarray] = {}
    for name in _sidecar_tensor_names():
        if name in omit:
            continue
        key = (rename or {}).get(name, name)
        if name == "mtp.fc.weight":
            arrays[key] = (rng.standard_normal((hidden, 2 * hidden)) * 0.02).astype(np.float16)
        elif name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
            arrays[key] = (rng.standard_normal((2 * hidden, hidden)) * 0.02).astype(np.float16)
        elif name.endswith("mlp.down_proj.weight"):
            arrays[key] = (rng.standard_normal((hidden, 2 * hidden)) * 0.02).astype(np.float16)
        elif name.endswith("self_attn.q_proj.weight"):
            arrays[key] = (rng.standard_normal((2 * hidden, hidden)) * 0.02).astype(np.float16)
        elif name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
            arrays[key] = (rng.standard_normal((hidden, hidden)) * 0.02).astype(np.float16)
        elif name.endswith("self_attn.o_proj.weight"):
            arrays[key] = (rng.standard_normal((hidden, 2 * hidden)) * 0.02).astype(np.float16)
        elif name.endswith("q_norm.weight"):
            arrays[key] = (0.5 + rng.standard_normal((2 * hidden,)) * 0.01).astype(np.float16)
        elif name.endswith("k_norm.weight"):
            arrays[key] = (0.5 + rng.standard_normal((hidden,)) * 0.01).astype(np.float16)
        else:  # 1-D RMSNorm weights, stored in the shifted (gamma - 1) convention
            arrays[key] = (0.5 + rng.standard_normal((hidden,)) * 0.01).astype(np.float16)
    save_file(arrays, str(model_dir / "mtp.safetensors"))
    return arrays


def test_sidecar_entries_canonicalize_real_qwen35_layout(tmp_path: Path) -> None:
    arrays = _write_synthetic_sidecar(tmp_path)
    entries = _canonical_mtp_entries(tmp_path)
    assert set(entries) == {name.removeprefix("mtp.") for name in arrays}
    assert entries["fc.weight"].shape == (_SIDE_HIDDEN, 2 * _SIDE_HIDDEN)
    assert entries["pre_fc_norm_embedding.weight"].shape == (_SIDE_HIDDEN,)
    assert entries["layers.0.self_attn.q_proj.weight"].shape == (2 * _SIDE_HIDDEN, _SIDE_HIDDEN)


def test_sidecar_entries_find_mtp_keys_inside_indexed_shards(tmp_path: Path) -> None:
    """HF-native exports keep mtp.* keys in the indexed main shards."""
    from safetensors.numpy import save_file

    shard = {
        "model.language_model.model.layers.0.mlp.gate_proj.weight": np.zeros(
            (8, 8), dtype=np.float16
        ),
        "model.language_model.mtp.fc.weight": np.zeros((8, 16), dtype=np.float16),
        "model.language_model.mtp.layers.0.mlp.down_proj.weight": np.zeros(
            (8, 16), dtype=np.float16
        ),
    }
    save_file(shard, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in shard}}),
        encoding="utf-8",
    )
    entries = _canonical_mtp_entries(tmp_path)
    assert set(entries) == {"fc.weight", "layers.0.mlp.down_proj.weight"}
    assert all(ref.path.name == "model-00001-of-00001.safetensors" for ref in entries.values())


def test_sidecar_entries_ignore_checkpoints_without_mtp(tmp_path: Path) -> None:
    from safetensors.numpy import save_file

    save_file(
        {"model.language_model.model.layers.0.mlp.gate_proj.weight": np.zeros((8, 8), np.float16)},
        str(tmp_path / "model.safetensors"),
    )
    assert _canonical_mtp_entries(tmp_path) == {}


def _fake_sidecar_model(vocab: int = 64, hidden: int = _SIDE_HIDDEN) -> Any:
    """Tiny Qwen3.5-shaped nn.Module tree for reconstruction tests (needs mlx)."""
    mlx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class _Args:
        def __init__(self) -> None:
            self.hidden_size = hidden
            self.rms_norm_eps = 1e-6
            self.tie_word_embeddings = False

    offsets_seen: list[Any] = []

    class _Attn(nn.Module):
        def __init__(self, args: _Args) -> None:
            dim = args.hidden_size
            self.q_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(2 * dim, dim, bias=False)
            self.q_norm = nn.RMSNorm(2 * dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)

        def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
            del mask
            offsets_seen.append(getattr(cache, "offset", None))
            queries = mlx.tanh(self.q_norm(self.q_proj(x)))
            gate = mlx.concatenate([self.k_norm(self.k_proj(x)), self.v_proj(x)], axis=-1)
            return self.o_proj(queries * mlx.sigmoid(gate))

    class _Mlp(nn.Module):
        def __init__(self, args: _Args) -> None:
            dim = args.hidden_size
            self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.down_proj = nn.Linear(2 * dim, dim, bias=False)

        def __call__(self, x: Any) -> Any:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class _Layer(nn.Module):
        def __init__(self, args: _Args, layer_idx: int) -> None:
            del layer_idx
            self.self_attn = _Attn(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mlp = _Mlp(args)

        def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
            hidden = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
            return hidden + self.mlp(self.post_attention_layernorm(hidden))

    class _Backbone(nn.Module):
        def __init__(self, args: _Args) -> None:
            self.embed_tokens = nn.Embedding(vocab, args.hidden_size)
            self.layers = [_Layer(args, 0), _Layer(args, 1)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, tokens: Any) -> Any:
            hidden = self.embed_tokens(tokens)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.norm(hidden)

    class _TextModel(nn.Module):
        def __init__(self, args: _Args) -> None:
            self.args = args
            self.model = _Backbone(args)
            self.lm_head = nn.Linear(args.hidden_size, vocab, bias=False)

    class _Model(nn.Module):
        def __init__(self) -> None:
            self.language_model = _TextModel(_Args())

    return _Model(), offsets_seen


def _detect_on_fake_model(tmp_path: Path, model: Any) -> MlxProbeBackend:
    backend = MlxProbeBackend()
    backend._model = model
    backend._model_dir = tmp_path
    backend._detect_mtp_forward_capability()
    return backend


def test_sidecar_capability_reconstructs_and_teacher_forces(tmp_path: Path) -> None:
    mlx = pytest.importorskip("mlx.core")
    stored = _write_synthetic_sidecar(tmp_path)
    model, offsets_seen = _fake_sidecar_model()
    backend = _detect_on_fake_model(tmp_path, model)

    assert backend.supports_mtp_forward
    # Every reconstructed 1-D norm weight must carry the +1.0 storage shift.
    norm_weight = np.asarray(
        backend._mtp_sidecar["pre_fc_norm_embedding"].weight.astype(mlx.float32)
    )
    expected = stored["mtp.pre_fc_norm_embedding.weight"].astype(np.float32) + 1.0
    assert np.allclose(norm_weight, expected, atol=1e-3)
    fc_weight = backend._mtp_sidecar["fc"].weight
    assert fc_weight.shape == (_SIDE_HIDDEN, 2 * _SIDE_HIDDEN)

    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 64, size=(1, 10)).astype(np.int32)
    hidden = rng.standard_normal((1, 4, _SIDE_HIDDEN)).astype(np.float32)
    result = backend.forward_mtp(tokens, hidden)
    assert result.draft_logits.shape == (1, 4, 64)
    assert np.isfinite(result.draft_logits).all()
    # The rope-offset shim must hand the trailing window its true position.
    assert offsets_seen and all(offset == 6 for offset in offsets_seen)


def test_sidecar_capability_fails_closed_on_incomplete_layout(tmp_path: Path) -> None:
    _write_synthetic_sidecar(tmp_path, omit=("mtp.norm.weight",))
    model, _ = _fake_sidecar_model()
    backend = _detect_on_fake_model(tmp_path, model)
    assert not backend.supports_mtp_forward
    assert backend._mtp_sidecar is None


def test_sidecar_capability_fails_closed_on_unexpected_extra_weight(tmp_path: Path) -> None:
    _write_synthetic_sidecar(
        tmp_path,
        rename={"mtp.layers.0.self_attn.o_proj.weight": "mtp.layers.0.linear_attn.out_proj.weight"},
    )
    model, _ = _fake_sidecar_model()
    backend = _detect_on_fake_model(tmp_path, model)
    assert not backend.supports_mtp_forward


def test_sidecar_module_quantize_and_restore_roundtrip(tmp_path: Path) -> None:
    _write_synthetic_sidecar(tmp_path)
    model, _ = _fake_sidecar_model()
    backend = _detect_on_fake_model(tmp_path, model)
    assert backend.supports_mtp_forward

    tokens = np.random.default_rng(3).integers(0, 64, size=(1, 10)).astype(np.int32)
    hidden = np.random.default_rng(4).standard_normal((1, 4, _SIDE_HIDDEN)).astype(np.float32)
    reference = np.asarray(backend.forward_mtp(tokens, hidden).draft_logits)

    # HF-style checkpoint naming resolves into the reconstructed block.
    resolved = backend._resolve_module_path("model.language_model.mtp.layers.0.mlp.gate_proj")
    assert resolved == "mtp.layers.0.mlp.gate_proj"

    backend.quantize_module("mtp.layers.0.mlp.gate_proj", 4, 64)
    quantized = np.asarray(backend.forward_mtp(tokens, hidden).draft_logits)
    assert quantized.shape == reference.shape
    backend.restore_module("mtp.layers.0.mlp.gate_proj")
    restored = np.asarray(backend.forward_mtp(tokens, hidden).draft_logits)
    assert np.allclose(restored, reference, atol=1e-4)
    # A second mutation must be allowed after the restore.
    backend.quantize_module("mtp.layers.0.self_attn.q_proj", 4, 64)
    backend.restore_module("mtp.layers.0.self_attn.q_proj")
    assert np.allclose(
        np.asarray(backend.forward_mtp(tokens, hidden).draft_logits), reference, atol=1e-4
    )


# --- Qwen3_5MoE MTP sidecar layout (real-checkpoint convention) ---
#
# The cert-track MoE pack (AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP, validated on
# hardware 2026-09-25) stores the MTP layer's routed experts as stacked 3-D
# tensors: ``mlp.experts.gate_up_proj`` (experts, 2 * intermediate, hidden)
# concatenates gate||up, and ``mlp.experts.down_proj`` is
# (experts, hidden, intermediate). The decoder layer class exposes SwitchGLU
# projections, so the stacked tensor is halved at reconstruction
# (``_sidecar_leaf_source``). Router ``mlp.gate.weight`` and
# ``mlp.shared_expert*_gate.weight`` are plain 2-D weights and are not norm
# shifted; all 1-D norm weights keep the +1.0 storage shift.

_MOE_HIDDEN = 128
_MOE_EXPERTS = 4
_MOE_INTER = 8


def _write_synthetic_moe_sidecar(
    model_dir: Path,
    *,
    hidden: int = _MOE_HIDDEN,
    experts: int = _MOE_EXPERTS,
    inter: int = _MOE_INTER,
    gate_value: float = 0.11,
    up_value: float = 0.77,
) -> dict[str, np.ndarray]:
    """Synthetic mtp.safetensors with the real Qwen3_5MoE key layout.

    The gate_up tensor uses two distinct constant halves so tests can verify
    the gate||up split order.
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(9)
    arrays: dict[str, np.ndarray] = {}
    dense_names = [
        "mtp.fc.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    ]
    for name in dense_names:
        if name == "mtp.fc.weight":
            arrays[name] = (rng.standard_normal((hidden, 2 * hidden)) * 0.02).astype(np.float16)
        elif name.endswith(("q_proj.weight",)):
            arrays[name] = (rng.standard_normal((2 * hidden, hidden)) * 0.02).astype(np.float16)
        elif name.endswith(("k_proj.weight", "v_proj.weight")):
            arrays[name] = (rng.standard_normal((hidden, hidden)) * 0.02).astype(np.float16)
        elif name.endswith("o_proj.weight"):
            arrays[name] = (rng.standard_normal((hidden, 2 * hidden)) * 0.02).astype(np.float16)
        elif name.endswith("q_norm.weight"):
            arrays[name] = (0.5 + rng.standard_normal((2 * hidden,)) * 0.01).astype(np.float16)
        elif name.endswith("k_norm.weight"):
            arrays[name] = (0.5 + rng.standard_normal((hidden,)) * 0.01).astype(np.float16)
        elif name == "mtp.layers.0.mlp.gate.weight":
            arrays[name] = (rng.standard_normal((experts, hidden)) * 0.02).astype(np.float16)
        elif name.endswith(("shared_expert.gate_proj.weight", "shared_expert.up_proj.weight")):
            arrays[name] = (rng.standard_normal((inter, hidden)) * 0.02).astype(np.float16)
        elif name.endswith("shared_expert.down_proj.weight"):
            arrays[name] = (rng.standard_normal((hidden, inter)) * 0.02).astype(np.float16)
        elif name.endswith("shared_expert_gate.weight"):
            arrays[name] = (rng.standard_normal((1, hidden)) * 0.02).astype(np.float16)
        else:  # 1-D RMSNorm weights, stored in the shifted (gamma - 1) convention
            arrays[name] = (0.5 + rng.standard_normal((hidden,)) * 0.01).astype(np.float16)
    gate_half = np.full((experts, inter, hidden), gate_value, dtype=np.float16)
    up_half = np.full((experts, inter, hidden), up_value, dtype=np.float16)
    arrays["mtp.layers.0.mlp.experts.gate_up_proj"] = np.concatenate([gate_half, up_half], axis=1)
    arrays["mtp.layers.0.mlp.experts.down_proj"] = (
        rng.standard_normal((experts, hidden, inter)) * 0.02
    ).astype(np.float16)
    save_file(arrays, str(model_dir / "mtp.safetensors"))
    return arrays


def _fake_moe_sidecar_model(
    vocab: int = 64, hidden: int = _MOE_HIDDEN, experts: int = _MOE_EXPERTS, inter: int = _MOE_INTER
) -> Any:
    """Tiny Qwen3_5MoE-shaped nn.Module tree (needs mlx)."""
    mlx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class _Args:
        def __init__(self) -> None:
            self.hidden_size = hidden
            self.rms_norm_eps = 1e-6
            self.tie_word_embeddings = False
            self.num_experts = experts
            self.moe_intermediate_size = inter
            self.shared_expert_intermediate_size = inter

    class _Attn(nn.Module):
        def __init__(self, args: _Args) -> None:
            dim = args.hidden_size
            self.q_proj = nn.Linear(dim, 2 * dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(2 * dim, dim, bias=False)
            self.q_norm = nn.RMSNorm(2 * dim, eps=args.rms_norm_eps)
            self.k_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)

        def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
            del mask, cache
            queries = mlx.tanh(self.q_norm(self.q_proj(x)))
            gate = mlx.concatenate([self.k_norm(self.k_proj(x)), self.v_proj(x)], axis=-1)
            return self.o_proj(queries * mlx.sigmoid(gate))

    class _SwitchLinear(nn.Module):
        """Shape-compatible stand-in for mlx SwitchLinear (3-D stacked weight)."""

        def __init__(self, in_dim: int, out_dim: int, num_experts: int) -> None:
            self.weight = mlx.zeros((num_experts, out_dim, in_dim))

        def __call__(self, x: Any, indices: Any) -> Any:
            del indices
            # Dense weighted-expert computation: (B, P, E, out). Accepts
            # token input (B, P, in) or per-expert input (B, P, E, in).
            outs = []
            for i in range(self.weight.shape[0]):
                per = x[..., i, :] if x.ndim == 4 else x
                outs.append(per @ self.weight[i].swapaxes(-1, -2))
            return mlx.stack(outs, axis=-2)

    class _MoeMlp(nn.Module):
        def __init__(self, args: _Args) -> None:
            dim = args.hidden_size
            self.gate = nn.Linear(dim, args.num_experts, bias=False)
            self.switch_mlp = _SwitchGlu(args)
            self.shared_expert = _SharedExpert(args)
            self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        def __call__(self, x: Any) -> Any:
            weights = mlx.softmax(self.gate(x), axis=-1)  # (B, P, E)
            gate = self.switch_mlp.gate_proj(x, None)  # (B, P, E, I)
            up = self.switch_mlp.up_proj(x, None)
            activated = nn.silu(gate) * up
            routed = self.switch_mlp.down_proj(activated, None)  # (B, P, E, H)
            y = (routed * weights[..., None]).sum(axis=-2)
            shared = self.shared_expert(x)
            return y + mlx.sigmoid(self.shared_expert_gate(x)) * shared

    class _SwitchGlu(nn.Module):
        def __init__(self, args: _Args) -> None:
            dim = args.hidden_size
            self.gate_proj = _SwitchLinear(dim, args.moe_intermediate_size, args.num_experts)
            self.up_proj = _SwitchLinear(dim, args.moe_intermediate_size, args.num_experts)
            self.down_proj = _SwitchLinear(args.moe_intermediate_size, dim, args.num_experts)

    class _SharedExpert(nn.Module):
        def __init__(self, args: _Args) -> None:
            dim = args.hidden_size
            self.gate_proj = nn.Linear(dim, args.shared_expert_intermediate_size, bias=False)
            self.up_proj = nn.Linear(dim, args.shared_expert_intermediate_size, bias=False)
            self.down_proj = nn.Linear(args.shared_expert_intermediate_size, dim, bias=False)

        def __call__(self, x: Any) -> Any:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class _Layer(nn.Module):
        def __init__(self, args: _Args, layer_idx: int) -> None:
            del layer_idx
            self.self_attn = _Attn(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mlp = _MoeMlp(args)

        def __call__(self, x: Any, mask: Any = None, cache: Any = None) -> Any:
            hidden = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
            return hidden + self.mlp(self.post_attention_layernorm(hidden))

    class _Backbone(nn.Module):
        def __init__(self, args: _Args) -> None:
            self.embed_tokens = nn.Embedding(vocab, args.hidden_size)
            self.layers = [_Layer(args, 0), _Layer(args, 1)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    class _TextModel(nn.Module):
        def __init__(self, args: _Args) -> None:
            self.args = args
            self.model = _Backbone(args)
            self.lm_head = nn.Linear(args.hidden_size, vocab, bias=False)

    class _Model(nn.Module):
        def __init__(self) -> None:
            self.language_model = _TextModel(_Args())

    return _Model()


def test_sidecar_capability_reconstructs_moe_layout(tmp_path: Path) -> None:
    mlx = pytest.importorskip("mlx.core")
    stored = _write_synthetic_moe_sidecar(tmp_path)
    backend = _detect_on_fake_model(tmp_path, _fake_moe_sidecar_model())

    assert backend.supports_mtp_forward
    layer = backend._mtp_sidecar["layers"][0]
    # gate||up split order: gate half first (locked on the real MoE pack).
    gate_proj = np.asarray(layer.mlp.switch_mlp.gate_proj.weight.astype(mlx.float32))
    up_proj = np.asarray(layer.mlp.switch_mlp.up_proj.weight.astype(mlx.float32))
    assert np.allclose(gate_proj, 0.11, atol=1e-3)
    assert np.allclose(up_proj, 0.77, atol=1e-3)
    down_proj = np.asarray(layer.mlp.switch_mlp.down_proj.weight.astype(mlx.float32))
    expected_down = stored["mtp.layers.0.mlp.experts.down_proj"].astype(np.float32)
    assert np.allclose(down_proj, expected_down, atol=1e-3)
    # 2-D router and shared-expert-gate weights are not norm shifted.
    router = np.asarray(layer.mlp.gate.weight.astype(mlx.float32))
    assert np.allclose(router, stored["mtp.layers.0.mlp.gate.weight"].astype(np.float32), atol=1e-3)
    # 1-D norm weights keep the +1.0 storage shift.
    norm_w = np.asarray(backend._mtp_sidecar["pre_fc_norm_embedding"].weight.astype(mlx.float32))
    assert np.allclose(
        norm_w,
        stored["mtp.pre_fc_norm_embedding.weight"].astype(np.float32) + 1.0,
        atol=1e-3,
    )

    rng = np.random.default_rng(5)
    tokens = rng.integers(0, 64, size=(1, 12)).astype(np.int32)
    hidden = rng.standard_normal((1, 5, _MOE_HIDDEN)).astype(np.float32)
    result = backend.forward_mtp(tokens, hidden)
    assert result.draft_logits.shape == (1, 5, 64)
    assert np.isfinite(result.draft_logits).all()


def test_sidecar_capability_fails_closed_on_odd_gate_up(tmp_path: Path) -> None:
    stored = _write_synthetic_moe_sidecar(tmp_path)
    # Corrupt the stacked projection to an odd intermediate axis.
    stored["mtp.layers.0.mlp.experts.gate_up_proj"] = stored[
        "mtp.layers.0.mlp.experts.gate_up_proj"
    ][:, :15, :]
    from safetensors.numpy import save_file

    save_file(stored, str(tmp_path / "mtp.safetensors"))
    backend = _detect_on_fake_model(tmp_path, _fake_moe_sidecar_model())
    assert not backend.supports_mtp_forward
    assert backend._mtp_sidecar is None
