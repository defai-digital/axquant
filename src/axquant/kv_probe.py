"""Measured per-layer KV-cache sensitivity probing (AXQ-024).

For each text layer and candidate bit-width, a forward pass runs with only that
layer's KV cache quantized (every other layer at BF16) over the same verified
tokenized calibration cache the weight probe uses. Logits are compared against
the all-BF16 baseline with output KL and token disagreement at fixed metric
positions. Like the weight probe, the MLX backend is a lazy optional dependency
and results are development evidence until release KV gating exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import structlog

from axquant.errors import BackendUnavailableError, ProbeError
from axquant.probe import (
    _calibration_dataset_id,
    _load_calibration_inputs,
    compute_kl_divergence,
    compute_token_disagreement,
)
from axquant.schema import (
    CalibrationEvidence,
    CandidateMeasurement,
    EvidenceKind,
    Inventory,
    KvLayerSensitivity,
    KvSensitivityReport,
    MetricVector,
    ProfileName,
    QuantMethod,
)
from axquant.serde import stable_sha256

_LOG = structlog.get_logger()

_KV_PROBE_BACKEND_VERSION = "kv-probe-v1"


class KvProbeBackend(Protocol):
    """Protocol for per-layer KV-cache probe backends."""

    def load_model(self, model_dir: Path) -> None:
        """Load the source model into memory."""
        ...

    def forward_logits(
        self,
        input_ids: Any,
        *,
        layer_bits: dict[int, int] | None,
        group_size: int,
    ) -> Any:
        """Run a forward pass with the given per-layer KV quantization.

        ``layer_bits`` maps text-layer indices to KV bit-widths; unlisted layers
        (and ``None``) use BF16 KV. Returns logits as a numpy-compatible array
        of shape ``(1, positions, vocab)``.
        """
        ...


class MlxKvProbeBackend:
    """MLX-based KV probe backend with lazy imports."""

    backend_id = "mlx-kv"

    def __init__(self) -> None:
        self._model: Any = None
        self._mlx: Any = None
        self._cache_module: Any = None
        self._load: Any = None

    def _ensure_mlx(self) -> None:
        if self._mlx is not None:
            return
        try:
            import importlib

            self._mlx = importlib.import_module("mlx.core")
            self._cache_module = importlib.import_module("mlx_lm.models.cache")
            self._load = importlib.import_module("mlx_lm.utils").load
        except ImportError as exc:
            raise BackendUnavailableError(
                f"MLX KV probe backend requires mlx and mlx-lm: {exc}"
            ) from exc

    def load_model(self, model_dir: Path) -> None:
        self._ensure_mlx()
        self._model, _ = self._load(str(model_dir))

    def forward_logits(
        self,
        input_ids: Any,
        *,
        layer_bits: dict[int, int] | None,
        group_size: int,
    ) -> Any:
        if self._model is None:
            raise ProbeError("KV probe backend has no loaded model")
        try:
            import numpy as np
        except ImportError:
            raise BackendUnavailableError("KV probing requires numpy") from None
        mx = self._mlx
        tokens = mx.array(np.asarray(input_ids, dtype=np.int32))
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.shape[1] < 2:
            raise ProbeError("KV probes require at least two tokens")
        layers = getattr(self._model, "layers", None) or self._model.model.layers
        caches = []
        for index in range(len(layers)):
            bits = (layer_bits or {}).get(index)
            if bits is None or bits >= 16:
                caches.append(self._cache_module.KVCache())
            else:
                caches.append(self._cache_module.QuantizedKVCache(group_size=group_size, bits=bits))
        logits = self._model(tokens, cache=caches)
        mx.eval(logits)
        return np.asarray(logits.astype(mx.float32))


def _kv_candidate_metrics(
    baseline_logits: list[Any],
    candidate_logits: list[Any],
    *,
    metric_positions: int,
) -> MetricVector:
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("KV probing requires numpy") from None
    kl_values: list[float] = []
    disagreements: list[float] = []
    for reference, candidate in zip(baseline_logits, candidate_logits, strict=True):
        reference = np.asarray(reference, dtype=np.float32)
        candidate = np.asarray(candidate, dtype=np.float32)
        positions = min(metric_positions, reference.shape[-2])
        reference_tail = reference[..., -positions:, :]
        candidate_tail = candidate[..., -positions:, :]
        kl_values.append(compute_kl_divergence(reference_tail, candidate_tail))
        disagreements.append(
            compute_token_disagreement(
                reference_tail.argmax(axis=-1),
                candidate_tail.argmax(axis=-1),
            )
        )
    return MetricVector(
        output_kl=float(sum(kl_values) / len(kl_values)),
        token_disagreement=float(sum(disagreements) / len(disagreements)),
    )


def measure_kv_sensitivity(
    inventory: Inventory,
    *,
    model_dir: str | Path,
    calibration_cache: str | Path,
    profile: ProfileName,
    candidate_bits: tuple[int, ...] = (4, 6, 8),
    group_size: int = 64,
    token_budget: int = 2048,
    metric_positions: int = 32,
    backend: KvProbeBackend | None = None,
) -> KvSensitivityReport:
    """Measure per-layer KV-cache sensitivity over a verified calibration cache."""
    if inventory.quantized_source:
        raise ProbeError("measured KV sensitivity requires an unquantized BF16 source inventory")
    if inventory.model.revision is None:
        raise ProbeError("measured KV sensitivity requires a revision-pinned source model")
    layer_count = inventory.architecture_profile.text_layer_count
    if layer_count is None:
        raise ProbeError("measured KV sensitivity requires a known text layer count")
    quantized_bits = tuple(sorted({bits for bits in candidate_bits if bits < 16}))
    if not quantized_bits:
        raise ProbeError("KV probing requires at least one quantized candidate bit-width")
    if backend is None:
        backend = MlxKvProbeBackend()

    cache_path = Path(calibration_cache).expanduser().resolve()
    cache_manifest, batches, measured_tokens = _load_calibration_inputs(
        cache_path,
        token_budget=token_budget,
    )
    if cache_manifest.model.model_id != inventory.model.model_id:
        raise ProbeError("calibration cache model does not match the probe model")
    if cache_manifest.model.revision != inventory.model.revision:
        raise ProbeError("calibration cache revision does not match the probe revision")
    if cache_manifest.profile != profile:
        raise ProbeError("calibration cache profile does not match the probe profile")

    backend.load_model(Path(model_dir).expanduser().resolve())
    baseline = [
        backend.forward_logits(batch, layer_bits=None, group_size=group_size) for batch in batches
    ]

    entries: list[KvLayerSensitivity] = []
    for layer_index in range(layer_count):
        candidates: list[CandidateMeasurement] = []
        for bits in quantized_bits:
            candidate_logits = [
                backend.forward_logits(
                    batch,
                    layer_bits={layer_index: bits},
                    group_size=group_size,
                )
                for batch in batches
            ]
            metrics = _kv_candidate_metrics(
                baseline,
                candidate_logits,
                metric_positions=metric_positions,
            )
            candidates.append(
                CandidateMeasurement(
                    bits=bits,
                    method=QuantMethod.AFFINE,
                    group_size=group_size,
                    metrics=metrics,
                    measured_tokens=measured_tokens,
                )
            )
        candidates.append(
            CandidateMeasurement(
                bits=16,
                method=QuantMethod.BF16,
                metrics=MetricVector(),
                measured_tokens=measured_tokens,
                note="BF16 KV baseline",
            )
        )
        entries.append(KvLayerSensitivity(layer_index=layer_index, candidates=candidates))
        _LOG.info("kv_layer_probed", layer=layer_index, candidates=len(candidates))

    calibration = CalibrationEvidence(
        dataset_id=_calibration_dataset_id(cache_path, cache_manifest),
        dataset_sha256=cache_manifest.dataset_sha256,
        samples=cache_manifest.samples,
        domains=cache_manifest.domains,
        sequence_length=cache_manifest.sequence_length,
        backend=getattr(backend, "backend_id", type(backend).__name__),
        reference=str(cache_path),
        metadata={
            "cache_key_sha256": cache_manifest.cache_key_sha256,
            "token_budget": token_budget,
            "metric_positions": metric_positions,
            "kv_probe_backend_version": _KV_PROBE_BACKEND_VERSION,
        },
    )
    return KvSensitivityReport(
        model=inventory.model,
        architecture_profile=inventory.architecture_profile,
        profile=profile,
        evidence_kind=EvidenceKind.MEASURED_DEVELOPMENT,
        inventory_sha256=stable_sha256(inventory.model_dump(mode="json", exclude={"created_at"})),
        probe_backend=getattr(backend, "backend_id", type(backend).__name__),
        group_size=group_size,
        text_layer_count=layer_count,
        entries=entries,
        calibration=calibration,
    )
