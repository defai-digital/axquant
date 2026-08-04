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
from pydantic import ValidationError

from axquant.errors import BackendUnavailableError, ProbeError
from axquant.probe import (
    _calibration_dataset_id,
    _compute_logit_kl,
    _load_calibration_inputs,
    compute_token_disagreement,
)
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    AX_ENGINE_EXECUTABLE_BITS,
    AX_ENGINE_EXECUTABLE_GROUP_SIZES,
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

_KV_PROBE_BACKEND_VERSION = "kv-probe-v2-fake-quant"


class KvProbeBackend(Protocol):
    """Protocol for per-layer KV-cache probe backends."""

    def load_model(self, model_dir: Path) -> None:
        """Load the source model into memory."""
        ...

    def quantizable_layers(self) -> set[int]:
        """Layer indices whose KV cache supports quantization."""
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
        (and ``None``) use the model's default cache. Returns logits as a
        numpy-compatible array of shape ``(1, positions, vocab)``.
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

    def _prompt_caches(self) -> list[Any]:
        return list(self._cache_module.make_prompt_cache(self._model))

    def quantizable_layers(self) -> set[int]:
        """Layer indices whose default cache is a standard KV cache.

        Hybrid architectures (e.g. Qwen 3.6 gated-delta linear attention) use
        recurrent state caches on most layers; KV quantization applies only to
        the standard-attention layers.
        """
        if self._model is None:
            raise ProbeError("KV probe backend has no loaded model")
        kv_cache_type = self._cache_module.KVCache
        return {
            index
            for index, cache in enumerate(self._prompt_caches())
            if type(cache) is kv_cache_type
        }

    def _fake_quant_cache(self, bits: int, group_size: int) -> Any:
        """A KVCache that round-trips K/V through quantize→dequantize.

        Hybrid attention implementations do not all execute the packed
        `QuantizedKVCache` path, so sensitivity probing measures the numerical
        effect of KV quantization directly while keeping the standard
        attention kernels. The runtime kernel path (AX Engine) differs either
        way; the probe's subject is precision loss, not kernel throughput.
        """
        mx = self._mlx
        base = self._cache_module.KVCache

        def fake_quant(array: Any) -> Any:
            shape = array.shape
            dtype = array.dtype
            if shape[-1] % group_size != 0:
                raise ProbeError(
                    f"KV head dimension {shape[-1]} is not divisible by group size {group_size}"
                )
            flat = array.reshape(-1, shape[-1])
            packed, scales, biases = mx.quantize(flat, group_size=group_size, bits=bits)
            restored = mx.dequantize(
                packed,
                scales,
                biases,
                group_size=group_size,
                bits=bits,
            )
            return restored.reshape(shape).astype(dtype)

        class FakeQuantKVCache(base):  # type: ignore[misc, valid-type]
            def update_and_fetch(self, keys: Any, values: Any) -> Any:
                return super().update_and_fetch(fake_quant(keys), fake_quant(values))

        return FakeQuantKVCache()

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
        caches = self._prompt_caches()
        quantizable = self.quantizable_layers()
        for index, bits in (layer_bits or {}).items():
            if bits >= 16:
                continue
            if index not in quantizable:
                raise ProbeError(
                    f"layer {index} does not use a standard KV cache and cannot be quantized"
                )
            caches[index] = self._fake_quant_cache(bits, group_size)
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
    if type(metric_positions) is not int or metric_positions <= 0:
        raise ProbeError("KV metric_positions must be a positive integer")
    if not baseline_logits or not candidate_logits:
        raise ProbeError("KV metric computation requires non-empty baseline and candidate logits")
    if len(baseline_logits) != len(candidate_logits):
        raise ProbeError("KV baseline and candidate logit batches are not aligned")
    kl_values: list[float] = []
    disagreements: list[float] = []
    for reference, candidate in zip(baseline_logits, candidate_logits, strict=True):
        try:
            reference = np.asarray(reference, dtype=np.float32)
            candidate = np.asarray(candidate, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProbeError(f"KV backend returned non-numeric logits: {exc}") from exc
        if reference.ndim < 2 or candidate.ndim < 2:
            raise ProbeError("KV backend logits must have position and vocabulary dimensions")
        if reference.shape != candidate.shape:
            raise ProbeError(
                f"KV logit shape mismatch: reference {reference.shape}, candidate {candidate.shape}"
            )
        if reference.shape[-2] <= 0 or reference.shape[-1] <= 0:
            raise ProbeError("KV backend returned empty position or vocabulary dimensions")
        if not bool(np.all(np.isfinite(reference))) or not bool(np.all(np.isfinite(candidate))):
            raise ProbeError("KV backend returned non-finite logits")
        positions = min(metric_positions, reference.shape[-2])
        reference_tail = reference[..., -positions:, :]
        candidate_tail = candidate[..., -positions:, :]
        output_kl = _compute_logit_kl(reference_tail, candidate_tail)
        disagreement = compute_token_disagreement(
            reference_tail.argmax(axis=-1),
            candidate_tail.argmax(axis=-1),
        )
        if not bool(np.isfinite(output_kl)) or output_kl < -1e-9:
            raise ProbeError("KV output KL is non-finite or negative")
        if not bool(np.isfinite(disagreement)):
            raise ProbeError("KV token disagreement is non-finite")
        kl_values.append(max(0.0, output_kl))
        disagreements.append(disagreement)
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
    try:
        inventory = Inventory.model_validate(inventory.model_dump(mode="python"))
    except ValidationError as exc:
        raise ProbeError(f"invalid inventory for KV probing: {exc}") from exc
    if inventory.quantized_source:
        raise ProbeError("measured KV sensitivity requires an unquantized BF16 source inventory")
    if not is_immutable_revision(inventory.model.revision):
        raise ProbeError("measured KV sensitivity requires a revision-pinned source model")
    layer_count = inventory.architecture_profile.text_layer_count
    if layer_count is None:
        raise ProbeError("measured KV sensitivity requires a known text layer count")
    if not candidate_bits or any(type(bits) is not int for bits in candidate_bits):
        raise ProbeError("KV candidate bit-widths must be non-empty integers")
    unsupported_bits = set(candidate_bits) - AX_ENGINE_EXECUTABLE_BITS
    if unsupported_bits:
        raise ProbeError(
            f"AX Engine KV probing does not support bit-widths {sorted(unsupported_bits)}"
        )
    quantized_bits = tuple(sorted({bits for bits in candidate_bits if bits < 16}))
    if not quantized_bits:
        raise ProbeError("KV probing requires at least one quantized candidate bit-width")
    if type(group_size) is not int or group_size not in AX_ENGINE_EXECUTABLE_GROUP_SIZES:
        raise ProbeError(f"AX Engine KV probing does not support group size {group_size!r}")
    if type(token_budget) is not int or token_budget <= 0:
        raise ProbeError("KV token_budget must be a positive integer")
    if type(metric_positions) is not int or metric_positions <= 0:
        raise ProbeError("KV metric_positions must be a positive integer")
    if backend is None:
        backend = MlxKvProbeBackend()

    resolved_model_dir = Path(model_dir).expanduser().resolve()
    if not resolved_model_dir.is_dir():
        raise ProbeError(f"KV probe model directory does not exist: {resolved_model_dir}")
    if inventory.model.local_path is None:
        raise ProbeError("measured KV sensitivity requires a local-path-bound inventory")
    if Path(inventory.model.local_path).expanduser().resolve() != resolved_model_dir:
        raise ProbeError("KV probe model directory does not match the inventory source")

    cache_path = Path(calibration_cache).expanduser().resolve()
    cache_manifest, batches, measured_tokens = _load_calibration_inputs(
        cache_path,
        token_budget=token_budget,
    )
    if (
        cache_manifest.model.model_id != inventory.model.model_id
        or cache_manifest.model.format != inventory.model.format
    ):
        raise ProbeError("calibration cache model does not match the probe model")
    if cache_manifest.model.revision != inventory.model.revision:
        raise ProbeError("calibration cache revision does not match the probe revision")
    if cache_manifest.profile != profile:
        raise ProbeError("calibration cache profile does not match the probe profile")
    if not cache_manifest.calibration_evaluation_separation_attested:
        raise ProbeError(
            "measured KV sensitivity requires calibration/evaluation separation attestation"
        )
    calibration_dataset_id = _calibration_dataset_id(cache_path, cache_manifest)

    backend.load_model(resolved_model_dir)
    try:
        quantizable = set(backend.quantizable_layers())
    except TypeError as exc:
        raise ProbeError("KV backend returned an invalid quantizable-layer set") from exc
    invalid_layers = [
        layer
        for layer in quantizable
        if type(layer) is not int or layer < 0 or layer >= layer_count
    ]
    if invalid_layers:
        raise ProbeError(f"KV backend returned out-of-range layer indices: {invalid_layers!r}")
    if not quantizable:
        raise ProbeError("KV backend found no quantizable standard-attention cache layers")
    baseline = [
        backend.forward_logits(batch, layer_bits=None, group_size=group_size) for batch in batches
    ]

    entries: list[KvLayerSensitivity] = []
    for layer_index in range(layer_count):
        candidates: list[CandidateMeasurement] = []
        for bits in quantized_bits:
            if layer_index not in quantizable:
                candidates.append(
                    CandidateMeasurement(
                        bits=bits,
                        method=QuantMethod.AFFINE,
                        group_size=group_size,
                        metrics=MetricVector(),
                        supported=False,
                        measured_tokens=0,
                        note="layer uses a non-KV recurrent cache; KV quantization not applicable",
                    )
                )
                continue
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
        _LOG.info(
            "kv_layer_probed",
            layer=layer_index,
            quantizable=layer_index in quantizable,
            candidates=len(candidates),
        )

    calibration = CalibrationEvidence(
        dataset_id=calibration_dataset_id,
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
