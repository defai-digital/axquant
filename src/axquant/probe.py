"""Measured forward-probe sensitivity backend.

Implements per-tensor and module-group sensitivity probing using MLX
forward passes.  MLX is a lazy optional dependency imported only when
the probe backend is actually invoked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import structlog

from axquant.activation_cache import (
    is_cache_complete,
    load_cache_manifest,
    verify_cache_integrity,
)
from axquant.dwq import apply_mlx_dwq_clip
from axquant.errors import BackendUnavailableError, PlanningError, ProbeError
from axquant.module_paths import mlx_module_aliases
from axquant.schema import (
    CalibrationEvidence,
    CalibrationManifest,
    CandidateMeasurement,
    EvidenceKind,
    Inventory,
    MetricVector,
    ProbeConfig,
    ProbeProgress,
    ProfileName,
    QuantMethod,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
    TokenizedCacheManifest,
)
from axquant.serde import load_model, stable_sha256, write_data

log = structlog.get_logger()

_MIN_RELEASE_CALIBRATION_SAMPLES = 128
_MIN_RELEASE_CALIBRATION_TOKENS = 8192
_REQUIRED_AGENT_CODING_DOMAINS = {
    "coding",
    "json",
    "tool",
    "multilingual",
    "long-context",
}
_PROBE_BACKEND_VERSION = "axquant-mlx-isolated-probe-v4"
_PROBE_MIN_BITS = {
    TensorRole.EMBEDDING: 8,
    TensorRole.NORM: 16,
    # AXQ-026: probe down to the lowest floor any governed plan may use, so an
    # 8-bit LM-head choice is backed by a measurement instead of being
    # unmeasurable by construction. The planner default floor stays BF16.
    TensorRole.LM_HEAD: 8,
    TensorRole.ROUTER: 8,
    TensorRole.VISION: 16,
}


@dataclass
class ForwardResult:
    """Result of a single forward pass through the model."""

    logits: Any = None
    hidden_states: Any = None
    loss: float | None = None
    token_count: int = 0
    peak_memory_bytes: int | None = None
    latency_seconds: float = 0.0


class ProbeBackend(Protocol):
    """Protocol for model probe backends."""

    def load_model(self, model_dir: Path) -> None:
        """Load the source model into memory."""
        ...

    def quantize_module(
        self,
        module_path: str,
        bits: int,
        group_size: int,
        method: QuantMethod = QuantMethod.AFFINE,
    ) -> None:
        """Quantize a single module in-place."""
        ...

    def restore_module(self, module_path: str) -> None:
        """Restore a module to its original unquantized state."""
        ...

    def forward(self, input_ids: Any) -> ForwardResult:
        """Run a forward pass and capture outputs."""
        ...


def _candidate_bits_for_tensor(tensor: TensorSpec, config: ProbeConfig) -> tuple[int, ...]:
    """Apply role floors without turning every protected recommendation into BF16-only."""
    if not tensor.quantizable:
        return (16,)
    if tensor.role.is_mtp and Path(tensor.file).name.lower() in {
        "mtp.safetensors",
        "mtp_head.safetensors",
    }:
        return (16,)
    minimum_bits = _PROBE_MIN_BITS.get(tensor.role, 8 if tensor.role.is_mtp else 2)
    candidates = tuple(bits for bits in config.candidate_bits if bits >= minimum_bits)
    return candidates or (16,)


class MlxProbeBackend:
    """MLX-based probe backend with lazy imports."""

    def __init__(self) -> None:
        self._model: Any = None
        self._original_modules: dict[str, tuple[Any, str, Any]] = {}
        self._mlx: Any = None
        self._mlx_lm: Any = None
        self.metric_positions_per_sample = 32

    def _ensure_mlx(self) -> None:
        if self._mlx is not None:
            return
        try:
            import importlib

            self._mlx = importlib.import_module("mlx.core")
            self._mlx_lm = importlib.import_module("mlx_lm")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"MLX probe backend requires mlx and mlx-lm: {exc}"
            ) from exc

    def load_model(self, model_dir: Path) -> None:
        self._ensure_mlx()
        loaded = self._mlx_lm.load(str(model_dir), lazy=False)
        self._model = loaded[0]
        self._mlx.eval(self._model.parameters())
        log.info("probe_model_loaded", model_dir=str(model_dir))

    def quantize_module(
        self,
        module_path: str,
        bits: int,
        group_size: int,
        method: QuantMethod = QuantMethod.AFFINE,
    ) -> None:
        self._ensure_mlx()
        if self._model is None:
            raise ProbeError("model not loaded")
        if self._original_modules:
            raise ProbeError("a probe mutation is already active")
        resolved_path = self._resolve_module_path(module_path)
        parent, child_name, module = self._get_parent_and_module(resolved_path)
        to_quantized = getattr(module, "to_quantized", None)
        if not callable(to_quantized):
            raise ProbeError(f"module does not support affine quantization: {resolved_path}")
        if method not in {QuantMethod.AFFINE, QuantMethod.DWQ}:
            raise ProbeError(f"probe backend does not support method {method.value}")
        original_weight = getattr(module, "weight", None)
        mutation_installed = False
        try:
            if method == QuantMethod.DWQ:
                apply_mlx_dwq_clip(module)
            quantized_module = to_quantized(
                group_size=group_size,
                bits=bits,
                mode="affine",
            )
            self._set_child(parent, child_name, quantized_module)
            mutation_installed = True
            self._mlx.eval(quantized_module.parameters())
        except (PlanningError, TypeError, ValueError, RuntimeError) as exc:
            if mutation_installed:
                self._set_child(parent, child_name, module)
            raise ProbeError(
                f"cannot quantize {resolved_path} with {method.value} at {bits}-bit: {exc}"
            ) from exc
        finally:
            if method == QuantMethod.DWQ and original_weight is not None:
                module.weight = original_weight
        self._original_modules[module_path] = (parent, child_name, module)
        log.debug(
            "probe_quantize_module",
            module=module_path,
            bits=bits,
            group_size=group_size,
            method=method.value,
        )

    def restore_module(self, module_path: str) -> None:
        original = self._original_modules.pop(module_path, None)
        if original is not None:
            parent, child_name, module = original
            self._set_child(parent, child_name, module)

    def forward(self, input_ids: Any) -> ForwardResult:
        self._ensure_mlx()
        if self._model is None:
            raise ProbeError("model not loaded")
        try:
            import numpy as np
        except ImportError:
            raise BackendUnavailableError("MLX probing requires numpy") from None
        tokens = self._mlx.array(np.asarray(input_ids, dtype=np.int32))
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.shape[1] < 2:
            raise ProbeError("forward probes require at least two tokens")
        metric_positions = min(self.metric_positions_per_sample, int(tokens.shape[1]) - 1)
        self._mlx.reset_peak_memory()
        start = time.monotonic()
        hidden_states: Any = None
        language_model = getattr(self._model, "language_model", None)
        text_backbone = getattr(language_model, "model", None)
        if language_model is not None and callable(text_backbone):
            hidden_states = text_backbone(tokens)
            metric_hidden = hidden_states[:, -(metric_positions + 1) : -1, :]
            if bool(getattr(getattr(language_model, "args", None), "tie_word_embeddings", False)):
                metric_logits = text_backbone.embed_tokens.as_linear(metric_hidden)
            else:
                metric_logits = language_model.lm_head(metric_hidden)
        else:
            logits = self._model(tokens)
            metric_logits = logits[:, -(metric_positions + 1) : -1, :]
            metric_hidden = None
        metric_logits = metric_logits.astype(self._mlx.float32)
        if metric_hidden is not None:
            metric_hidden = metric_hidden.astype(self._mlx.float32)
        if metric_hidden is None:
            self._mlx.eval(metric_logits)
        else:
            self._mlx.eval(metric_logits, metric_hidden)
        elapsed = time.monotonic() - start
        logits_array = np.asarray(metric_logits)
        hidden_array = np.asarray(metric_hidden) if metric_hidden is not None else None
        target_ids = np.asarray(tokens[:, -metric_positions:])
        return ForwardResult(
            logits=logits_array,
            hidden_states=hidden_array,
            loss=_causal_cross_entropy(logits_array, target_ids),
            token_count=int(tokens.size),
            peak_memory_bytes=int(self._mlx.get_peak_memory()),
            latency_seconds=elapsed,
        )

    def _resolve_module_path(self, module_path: str) -> str:
        names = [str(name) for name, _ in self._model.named_modules() if name]
        aliases = mlx_module_aliases(module_path)
        exact_matches = [alias for alias in aliases if alias in names]
        if len(exact_matches) == 1:
            return exact_matches[0]
        suffix_matches = [
            name
            for name in names
            if any(name.endswith(f".{alias}") or alias.endswith(f".{name}") for alias in aliases)
        ]
        if len(suffix_matches) != 1:
            raise ProbeError(
                f"cannot uniquely resolve module path {module_path!r}; "
                f"found {len(suffix_matches)} matches"
            )
        return str(suffix_matches[0])

    def _get_parent_and_module(self, module_path: str) -> tuple[Any, str, Any]:
        parts = module_path.split(".")
        current = self._model
        for part in parts[:-1]:
            current = self._get_child(current, part)
        child_name = parts[-1]
        return current, child_name, self._get_child(current, child_name)

    @staticmethod
    def _get_child(parent: Any, child_name: str) -> Any:
        if hasattr(parent, child_name):
            return getattr(parent, child_name)
        try:
            return parent[int(child_name)]
        except (TypeError, ValueError, IndexError, KeyError):
            try:
                return parent[child_name]
            except (TypeError, IndexError, KeyError) as exc:
                raise ProbeError(f"cannot resolve child module {child_name!r}") from exc

    @staticmethod
    def _set_child(parent: Any, child_name: str, module: Any) -> None:
        if hasattr(parent, child_name):
            setattr(parent, child_name, module)
            return
        try:
            parent[int(child_name)] = module
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise ProbeError(f"cannot replace child module {child_name!r}") from exc


@dataclass
class ProbeState:
    """Tracks probe progress for resume support."""

    completed_tensors: dict[str, list[CandidateMeasurement]] = field(default_factory=dict)
    total_tensors: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def is_tensor_complete(self, tensor_name: str) -> bool:
        return tensor_name in self.completed_tensors

    def record_tensor(self, tensor_name: str, candidates: list[CandidateMeasurement]) -> None:
        self.completed_tensors[tensor_name] = candidates


def compute_kl_divergence(reference: Any, candidate: Any) -> float:
    """Compute KL(P || Q) between reference and candidate distributions.

    Uses numerically stable log probabilities.
    """
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("metric computation requires numpy") from None

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)

    # Normalize to valid probability distributions
    ref = np.clip(ref, 1e-10, None)
    cand = np.clip(cand, 1e-10, None)
    ref = ref / ref.sum(axis=-1, keepdims=True)
    cand = cand / cand.sum(axis=-1, keepdims=True)

    kl = np.sum(ref * np.log(ref / cand), axis=-1)
    return float(np.mean(kl))


def compute_hidden_state_error(reference: Any, candidate: Any) -> float:
    """Compute mean squared error between reference and candidate hidden states."""
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("metric computation requires numpy") from None

    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    return float(np.mean((ref - cand) ** 2))


def compute_cosine_distance(reference: Any, candidate: Any) -> float:
    """Compute 1 - cosine_similarity between reference and candidate."""
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("metric computation requires numpy") from None

    ref = np.asarray(reference, dtype=np.float64).flatten()
    cand = np.asarray(candidate, dtype=np.float64).flatten()

    norm_ref = np.linalg.norm(ref)
    norm_cand = np.linalg.norm(cand)
    if norm_ref == 0 or norm_cand == 0:
        return 1.0
    similarity = float(np.dot(ref, cand) / (norm_ref * norm_cand))
    return 1.0 - similarity


def compute_token_disagreement(reference_tokens: Any, candidate_tokens: Any) -> float:
    """Compute fraction of positions where argmax tokens differ."""
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("metric computation requires numpy") from None

    ref = np.asarray(reference_tokens)
    cand = np.asarray(candidate_tokens)
    if ref.shape != cand.shape:
        raise ProbeError(f"token shape mismatch: reference {ref.shape} vs candidate {cand.shape}")
    if ref.size == 0:
        return 0.0
    return float(np.mean(ref != cand))


def _causal_cross_entropy(logits: Any, target_ids: Any) -> float:
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("metric computation requires numpy") from None
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64)
    if values.ndim != 3 or targets.shape != values.shape[:2]:
        raise ProbeError(f"loss shape mismatch: logits {values.shape}, target IDs {targets.shape}")
    maxima = np.max(values, axis=-1, keepdims=True)
    log_normalizer = maxima + np.log(np.sum(np.exp(values - maxima), axis=-1, keepdims=True))
    log_probabilities = values - log_normalizer
    selected = np.take_along_axis(log_probabilities, targets[..., None], axis=-1)
    return float(-np.mean(selected))


def _compute_logit_kl(reference: Any, candidate: Any) -> float:
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("metric computation requires numpy") from None
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ProbeError(f"logit shape mismatch: reference {ref.shape}, candidate {cand.shape}")

    def log_softmax(values: Any) -> Any:
        maxima = np.max(values, axis=-1, keepdims=True)
        return values - maxima - np.log(np.sum(np.exp(values - maxima), axis=-1, keepdims=True))

    ref_log = log_softmax(ref)
    cand_log = log_softmax(cand)
    return float(np.mean(np.sum(np.exp(ref_log) * (ref_log - cand_log), axis=-1)))


def _load_calibration_inputs(
    cache_dir: Path,
    *,
    token_budget: int,
    replay_batch_size: int = 1,
) -> tuple[TokenizedCacheManifest, list[Any], int]:
    manifest = load_cache_manifest(cache_dir)
    if manifest is None:
        raise ProbeError(f"calibration cache manifest is missing or invalid: {cache_dir}")
    if not manifest.complete or not is_cache_complete(cache_dir):
        raise ProbeError(f"calibration cache is incomplete: {cache_dir}")
    issues = verify_cache_integrity(cache_dir, manifest)
    if issues:
        raise ProbeError(f"calibration cache failed verification: {issues}")
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("measured probing requires numpy") from None
    sample_inputs: list[Any] = []
    measured_tokens = 0
    for shard_index in range(manifest.shard_count):
        shard = cache_dir / "tokenized" / f"shard-{shard_index:04d}.npz"
        with np.load(shard, allow_pickle=False) as data:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            for row in range(len(input_ids)):
                length = int(attention_mask[row].sum())
                remaining = token_budget - measured_tokens
                length = min(length, remaining)
                if length < 2:
                    continue
                sample_inputs.append(np.asarray(input_ids[row, :length], dtype=np.int32))
                measured_tokens += length
                if measured_tokens >= token_budget:
                    break
            if measured_tokens >= token_budget:
                break
    if not sample_inputs:
        raise ProbeError("calibration cache contains no sequences with at least two tokens")
    packed_inputs: list[Any] = []
    current: list[int] = []
    for sample in sample_inputs:
        offset = 0
        while offset < len(sample):
            remaining = manifest.sequence_length - len(current)
            take = min(remaining, len(sample) - offset)
            current.extend(int(token) for token in sample[offset : offset + take])
            offset += take
            if len(current) == manifest.sequence_length:
                packed_inputs.append(np.asarray(current, dtype=np.int32))
                current = []
    if current:
        packed_inputs.append(np.asarray(current, dtype=np.int32))
    replay_batches: list[Any] = []
    index = 0
    while index < len(packed_inputs):
        sequence_length = len(packed_inputs[index])
        end = index + 1
        while (
            end < len(packed_inputs)
            and end - index < replay_batch_size
            and len(packed_inputs[end]) == sequence_length
        ):
            end += 1
        group = packed_inputs[index:end]
        replay_batches.append(np.stack(group) if len(group) > 1 else group[0])
        index = end
    return manifest, replay_batches, measured_tokens


def _calibration_dataset_id(cache_dir: Path, cache_manifest: TokenizedCacheManifest) -> str:
    """Resolve the source dataset identity recorded by a verified cache.

    Older development caches do not bind a calibration manifest and retain the
    cache path as their only provenance.  Release caches do bind one: use its
    dataset identifier after checking the canonical manifest hash and the
    fields shared with the tokenized cache.
    """
    expected_sha256 = cache_manifest.calibration_manifest_sha256
    if expected_sha256 in {None, "", "unknown"}:
        return str(cache_dir)

    source = cache_dir / "calibration_manifest.json"
    if not source.is_file():
        raise ProbeError("calibration cache is missing its bound calibration manifest")
    manifest = load_model(source, CalibrationManifest)
    actual_sha256 = stable_sha256(manifest.model_dump(mode="json", exclude={"created_at"}))
    if actual_sha256 != expected_sha256:
        raise ProbeError("calibration cache manifest checksum does not match its cache binding")
    same_model = (
        manifest.model.model_id == cache_manifest.model.model_id
        and manifest.model.revision == cache_manifest.model.revision
        and manifest.model.format == cache_manifest.model.format
    )
    if (
        not same_model
        or manifest.profile != cache_manifest.profile
        or manifest.dataset_sha256 != cache_manifest.dataset_sha256
        or manifest.samples != cache_manifest.samples
        or set(manifest.domains) != set(cache_manifest.domains)
        or manifest.sequence_length != cache_manifest.sequence_length
    ):
        raise ProbeError("calibration cache manifest does not match the tokenized cache")
    return manifest.dataset_id


def _measure_candidate(
    backend: ProbeBackend,
    inputs: list[Any],
    references: list[ForwardResult],
    *,
    require_hidden_states: bool,
    long_context_min_tokens: int,
) -> MetricVector:
    import numpy as np

    output_kl: list[float] = []
    hidden_error: list[float] = []
    cosine_distance: list[float] = []
    token_disagreement: list[float] = []
    task_loss_delta: list[float] = []
    long_context_loss: list[float] = []
    peak_memory: list[float] = []
    latency: list[float] = []
    for input_ids, reference in zip(inputs, references, strict=True):
        candidate = backend.forward(input_ids)
        if reference.logits is None or candidate.logits is None:
            raise ProbeError("probe backend did not return logits")
        output_kl.append(_compute_logit_kl(reference.logits, candidate.logits))
        token_disagreement.append(
            compute_token_disagreement(
                np.argmax(reference.logits, axis=-1),
                np.argmax(candidate.logits, axis=-1),
            )
        )
        if reference.hidden_states is not None and candidate.hidden_states is not None:
            hidden_error.append(
                compute_hidden_state_error(reference.hidden_states, candidate.hidden_states)
            )
            cosine_distance.append(
                compute_cosine_distance(reference.hidden_states, candidate.hidden_states)
            )
        elif require_hidden_states:
            raise ProbeError("probe backend did not return required hidden states")
        if reference.loss is not None and candidate.loss is not None:
            loss_delta = max(0.0, candidate.loss - reference.loss)
            task_loss_delta.append(loss_delta)
            if int(input_ids.shape[-1]) >= long_context_min_tokens:
                long_context_loss.append(loss_delta)
        peak_memory.append(float(candidate.peak_memory_bytes or 0))
        latency.append(candidate.latency_seconds)

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    reference_peak = max(
        (float(reference.peak_memory_bytes or 0) for reference in references),
        default=0.0,
    )
    reference_latency = sum(reference.latency_seconds for reference in references)
    candidate_peak = max(peak_memory, default=0.0)
    candidate_latency = sum(latency)
    return MetricVector(
        output_kl=mean(output_kl),
        hidden_state_error=mean(hidden_error),
        cosine_distance=mean(cosine_distance),
        token_disagreement=mean(token_disagreement),
        task_loss_delta=mean(task_loss_delta),
        mtp_acceptance_loss=0.0,
        long_context_loss=mean(long_context_loss),
        peak_memory_cost=candidate_peak / reference_peak if reference_peak > 0 else 0.0,
        prefill_latency_cost=(
            candidate_latency / reference_latency if reference_latency > 0 else 0.0
        ),
        decode_latency_cost=0.0,
    )


def _reference_metrics(references: list[ForwardResult]) -> MetricVector:
    has_memory_measurement = any((reference.peak_memory_bytes or 0) > 0 for reference in references)
    has_latency_measurement = any(reference.latency_seconds > 0 for reference in references)
    return MetricVector(
        peak_memory_cost=1.0 if has_memory_measurement else 0.0,
        prefill_latency_cost=1.0 if has_latency_measurement else 0.0,
    )


def _module_group_for_tensor(tensor_name: str) -> str | None:
    """Determine the module group (transformer block) for a tensor.

    Returns the group identifier (e.g. 'model.layers.0.self_attn') or None
    if the tensor doesn't belong to a recognizable group.
    """
    parts = tensor_name.split(".")
    # Look for pattern: model.layers.N.{self_attn|mlp}
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                if i + 2 < len(parts):
                    return ".".join(parts[: i + 3])
            except ValueError:
                pass
    return None


def _validated_base_entries(
    inventory: Inventory,
    config: ProbeConfig,
    inventory_sha256: str,
    base_report: SensitivityReport | None,
) -> dict[str, TensorSensitivity]:
    inventory_names = {tensor.name for tensor in inventory.tensors}
    unknown_targets = sorted(set(config.target_tensors) - inventory_names)
    if unknown_targets:
        raise ProbeError(
            f"probe target tensors are absent from the inventory: {unknown_targets[:10]}"
        )
    if base_report is None:
        if config.target_tensors:
            raise ProbeError("targeted probing requires a base sensitivity report")
        return {}
    if base_report.model != inventory.model or base_report.profile != config.profile:
        raise ProbeError("base sensitivity report does not match the probe inventory/profile")
    # AXQ-017: the support tier is current registry policy, not recorded
    # measurement evidence — a base report probed before a tier promotion is
    # still the same measured contract, so the tier is excluded like notes.
    current_architecture = inventory.architecture_profile.model_dump(
        exclude={"notes", "support_tier"}
    )
    base_architecture = base_report.architecture_profile.model_dump(
        exclude={"notes", "support_tier"}
    )
    if current_architecture != base_architecture:
        raise ProbeError("base sensitivity architecture contract differs from the inventory")
    if base_report.inventory_sha256 != inventory_sha256:
        historical_inventory = inventory.model_copy(
            update={
                "architecture_profile": base_report.architecture_profile,
                "warnings": list(base_report.architecture_profile.notes),
            }
        )
        historical_dump = historical_inventory.model_dump(mode="json", exclude={"created_at"})
        if stable_sha256(historical_dump) != base_report.inventory_sha256:
            # Reports recorded before AXQ-017 serialized no support tier, so
            # their inventory hash covers a dump without that key. Reproduce
            # that exact historical byte contract before failing closed.
            historical_dump["architecture_profile"].pop("support_tier", None)
            if stable_sha256(historical_dump) != base_report.inventory_sha256:
                raise ProbeError("base sensitivity inventory hash cannot be reproduced")
    if base_report.evidence_kind == EvidenceKind.ARCHITECTURE_PRIOR:
        raise ProbeError("method refinement requires measured base sensitivity")
    if base_report.calibration is None:
        raise ProbeError("base sensitivity report has no calibration provenance")
    entries = {entry.tensor.name: entry for entry in base_report.entries}
    if len(entries) != len(base_report.entries) or set(entries) != inventory_names:
        raise ProbeError("base sensitivity report does not exactly cover the inventory")
    for tensor in inventory.tensors:
        if entries[tensor.name].tensor != tensor:
            raise ProbeError(f"base sensitivity tensor metadata differs for {tensor.name}")
    return entries


def probe_tensor_sensitivity(
    inventory: Inventory,
    *,
    config: ProbeConfig,
    backend: ProbeBackend | None = None,
    state: ProbeState | None = None,
    state_path: str | Path | None = None,
    base_report: SensitivityReport | None = None,
) -> SensitivityReport:
    """Probe per-tensor sensitivity using forward passes.

    For each eligible tensor and candidate configuration:
    1. Restore source state
    2. Quantize only the target module
    3. Replay fixed calibration samples
    4. Capture metrics at declared points
    5. Restore completely
    6. Persist result with provenance

    Supports deterministic replay, early termination, and in-process resume.
    """
    if inventory.quantized_source:
        raise ProbeError("measured sensitivity requires an unquantized BF16 source inventory")
    if config.model.revision is None:
        raise ProbeError("measured sensitivity requires a revision-pinned source model")
    if config.module_group_probing:
        raise ProbeError(
            "module-group probing is not available in the tensor-isolation backend; "
            "disable it rather than relabelling representative tensor results"
        )
    if backend is None:
        backend = MlxProbeBackend()
    if isinstance(backend, MlxProbeBackend):
        backend.metric_positions_per_sample = config.metric_positions_per_sample

    inventory_sha256 = stable_sha256(inventory.model_dump(mode="json", exclude={"created_at"}))
    base_entries = _validated_base_entries(
        inventory,
        config,
        inventory_sha256,
        base_report,
    )
    base_sha256 = stable_sha256(base_report) if base_report is not None else None
    config_sha256 = stable_sha256(
        {
            "config": config.model_dump(mode="json"),
            "probe_backend_version": _PROBE_BACKEND_VERSION,
            "base_sensitivity_sha256": base_sha256,
        }
    )
    progress_path = Path(state_path).expanduser().resolve() if state_path is not None else None
    if state is None and progress_path is not None and progress_path.is_file():
        progress = load_model(progress_path, ProbeProgress)
        if progress.inventory_sha256 != inventory_sha256:
            raise ProbeError("probe progress inventory does not match the current inventory")
        if progress.config_sha256 != config_sha256:
            raise ProbeError("probe progress configuration does not match the current probe")
        state = ProbeState(completed_tensors=progress.completed_tensors)
    if state is None:
        state = ProbeState()
    target_tensors = set(config.target_tensors) or {tensor.name for tensor in inventory.tensors}
    state.total_tensors = len(target_tensors)

    cache_path = Path(config.calibration_cache).expanduser().resolve()
    cache_manifest, calibration_inputs, measured_tokens = _load_calibration_inputs(
        cache_path,
        token_budget=config.token_budget_per_candidate,
        replay_batch_size=config.replay_batch_size,
    )
    if cache_manifest.model.model_id != config.model.model_id:
        raise ProbeError("calibration cache model does not match the probe model")
    if cache_manifest.model.revision != config.model.revision:
        raise ProbeError("calibration cache revision does not match the probe revision")
    if cache_manifest.profile != config.profile:
        raise ProbeError("calibration cache profile does not match the probe profile")
    if not cache_manifest.calibration_evaluation_separation_attested:
        raise ProbeError(
            "measured release evidence requires calibration/evaluation separation attestation"
        )
    calibration_dataset_id = _calibration_dataset_id(cache_path, cache_manifest)
    calibration_random_seed: int | None = None
    if cache_manifest.calibration_manifest_sha256 not in {None, "", "unknown"}:
        calibration_random_seed = load_model(
            cache_path / "calibration_manifest.json",
            CalibrationManifest,
        ).random_seed
    if base_report is not None:
        assert base_report.calibration is not None
        base_calibration = base_report.calibration
        compatible_dataset_ids = {calibration_dataset_id, str(cache_path)}
        if (
            base_calibration.dataset_id not in compatible_dataset_ids
            or base_calibration.dataset_sha256 != cache_manifest.dataset_sha256
            or base_calibration.samples != cache_manifest.samples
            or set(base_calibration.domains) != set(cache_manifest.domains)
            or base_calibration.sequence_length != cache_manifest.sequence_length
        ):
            raise ProbeError("base sensitivity uses different calibration evidence")
        required_protocol = {
            "token_budget_per_candidate": config.token_budget_per_candidate,
            "replay_batch_size": config.replay_batch_size,
            "metric_positions_per_sample": config.metric_positions_per_sample,
            "long_context_min_tokens": config.long_context_min_tokens,
            "warmup_replays": config.warmup_replays,
            "capture_points": ",".join(config.capture_points),
        }
        mismatched_protocol = sorted(
            name
            for name, expected in required_protocol.items()
            if base_calibration.metadata.get(name) != expected
        )
        if mismatched_protocol:
            raise ProbeError(
                f"base sensitivity uses a different probe protocol: {mismatched_protocol}"
            )

    probe_required = not target_tensors.issubset(state.completed_tensors)
    references: list[ForwardResult] = []
    reference_metrics = MetricVector()
    if probe_required:
        model_path = config.model.local_path or config.model.model_id
        backend.load_model(Path(model_path).expanduser().resolve())
        for _ in range(config.warmup_replays):
            backend.forward(calibration_inputs[0])
        references = [backend.forward(input_ids) for input_ids in calibration_inputs]
        if any(reference.logits is None for reference in references):
            raise ProbeError("probe backend did not return reference logits")
        if "hidden" in config.capture_points and any(
            reference.hidden_states is None for reference in references
        ):
            raise ProbeError("probe backend did not return requested reference hidden states")
        reference_metrics = _reference_metrics(references)
    else:
        log.info(
            "probe_resume_complete",
            completed=len(state.completed_tensors),
            total=state.total_tensors,
        )

    entries: list[TensorSensitivity] = []

    for tensor in inventory.tensors:
        base_entry = base_entries.get(tensor.name)
        if tensor.name not in target_tensors:
            assert base_entry is not None
            entries.append(base_entry)
            continue
        # Check resume state
        if state.is_tensor_complete(tensor.name):
            entries.append(
                TensorSensitivity(tensor=tensor, candidates=state.completed_tensors[tensor.name])
            )
            continue

        tensor_candidate_bits = _candidate_bits_for_tensor(tensor, config)
        if tensor_candidate_bits == (16,):
            if base_entry is not None:
                candidates = list(base_entry.candidates)
            else:
                preservation_reason = (
                    "non-quantizable tensor preserved"
                    if not tensor.quantizable
                    else "role policy permits only reference precision"
                )
                candidates = [
                    CandidateMeasurement(
                        bits=16,
                        method=QuantMethod.BF16,
                        group_size=None,
                        metrics=MetricVector(),
                        evidence_scope="preserved",
                        measured_tokens=measured_tokens,
                        note=preservation_reason,
                    )
                ]
            state.record_tensor(tensor.name, candidates)
            entries.append(TensorSensitivity(tensor=tensor, candidates=candidates))
            if progress_path is not None:
                write_data(
                    progress_path,
                    ProbeProgress(
                        inventory_sha256=inventory_sha256,
                        config_sha256=config_sha256,
                        completed_tensors=state.completed_tensors,
                        total_tensors=state.total_tensors,
                    ),
                )
            log.info(
                "probe_tensor_completed",
                tensor=tensor.name,
                completed=len(state.completed_tensors),
                total=state.total_tensors,
                candidates=len(candidates),
            )
            continue

        probe_candidates = list(base_entry.candidates) if base_entry is not None else []
        existing_keys = {(candidate.bits, candidate.method) for candidate in probe_candidates}
        cheaper_loss: dict[QuantMethod, float] = {}
        for candidate in probe_candidates:
            if candidate.bits < 16 and candidate.supported:
                previous = cheaper_loss.get(candidate.method)
                cheaper_loss[candidate.method] = (
                    candidate.metrics.output_kl
                    if previous is None
                    else min(previous, candidate.metrics.output_kl)
                )

        for bits in tensor_candidate_bits:
            if bits == 16:
                if (16, QuantMethod.BF16) not in existing_keys:
                    probe_candidates.append(
                        CandidateMeasurement(
                            bits=16,
                            method=QuantMethod.BF16,
                            group_size=None,
                            metrics=reference_metrics,
                            measured_tokens=measured_tokens,
                            note="reference precision",
                        )
                    )
                continue

            for method in config.candidate_methods:
                if (bits, method) in existing_keys:
                    continue
                try:
                    backend.quantize_module(
                        tensor.module_path,
                        bits,
                        config.group_size,
                        method,
                    )
                    for _ in range(config.warmup_replays):
                        backend.forward(calibration_inputs[0])
                    metrics = _measure_candidate(
                        backend,
                        calibration_inputs,
                        references,
                        require_hidden_states="hidden" in config.capture_points,
                        long_context_min_tokens=config.long_context_min_tokens,
                    )
                    packing_control = next(
                        (
                            candidate
                            for candidate in probe_candidates
                            if candidate.bits == bits
                            and candidate.method == QuantMethod.AFFINE
                            and candidate.group_size == config.group_size
                        ),
                        None,
                    )
                    if method == QuantMethod.DWQ and packing_control is not None:
                        metrics = metrics.model_copy(
                            update={
                                "peak_memory_cost": packing_control.metrics.peak_memory_cost,
                                "prefill_latency_cost": (
                                    packing_control.metrics.prefill_latency_cost
                                ),
                                "decode_latency_cost": (
                                    packing_control.metrics.decode_latency_cost
                                ),
                            }
                        )
                    previous_loss = cheaper_loss.get(method)
                    dominated = (
                        previous_loss is not None
                        and metrics.output_kl > previous_loss * config.early_termination_factor
                    )
                    note_parts: list[str] = []
                    if method == QuantMethod.DWQ:
                        note_parts.append(
                            "sampled 0.1/99.9-percentile clipping followed by affine packing"
                        )
                        if packing_control is not None:
                            note_parts.append(
                                "hardware costs normalized to the identical affine packing control"
                            )
                    if dominated:
                        note_parts.append(
                            "dominated by cheaper "
                            f"{method.value} candidate at "
                            f"{config.early_termination_factor}x bound"
                        )
                    probe_candidates.append(
                        CandidateMeasurement(
                            bits=bits,
                            method=method,
                            group_size=config.group_size,
                            metrics=metrics,
                            supported=not dominated,
                            measured_tokens=measured_tokens,
                            note="; ".join(note_parts) or None,
                        )
                    )
                    cheaper_loss[method] = (
                        metrics.output_kl
                        if previous_loss is None
                        else min(previous_loss, metrics.output_kl)
                    )
                except ProbeError as exc:
                    probe_candidates.append(
                        CandidateMeasurement(
                            bits=bits,
                            method=method,
                            group_size=config.group_size,
                            metrics=MetricVector(),
                            supported=False,
                            measured_tokens=0,
                            note=f"probe failed: {exc}",
                        )
                    )
                finally:
                    backend.restore_module(tensor.module_path)

        probe_candidates.sort(key=lambda candidate: (candidate.bits, candidate.method.value))
        state.record_tensor(tensor.name, probe_candidates)
        entries.append(TensorSensitivity(tensor=tensor, candidates=probe_candidates))
        if progress_path is not None:
            write_data(
                progress_path,
                ProbeProgress(
                    inventory_sha256=inventory_sha256,
                    config_sha256=config_sha256,
                    completed_tensors=state.completed_tensors,
                    total_tensors=state.total_tensors,
                ),
            )
        log.info(
            "probe_tensor_completed",
            tensor=tensor.name,
            completed=len(state.completed_tensors),
            total=state.total_tensors,
            candidates=len(probe_candidates),
        )

    # Build calibration evidence
    calibration_metadata: dict[str, str | int | float | bool] = {
        "cache_key_sha256": cache_manifest.cache_key_sha256,
        "sample_order_sha256": cache_manifest.sample_order_sha256 or "unknown",
        "tokenizer_sha256": cache_manifest.tokenizer_sha256 or "unknown",
        "calibration_manifest_sha256": (cache_manifest.calibration_manifest_sha256 or "unknown"),
        "domain_provenance": cache_manifest.domain_provenance,
        "token_budget_per_candidate": config.token_budget_per_candidate,
        "measured_tokens_per_candidate": measured_tokens,
        "packed_replay_sequences": sum(
            int(input_ids.shape[0]) if input_ids.ndim == 2 else 1
            for input_ids in calibration_inputs
        ),
        "replay_batches": len(calibration_inputs),
        "replay_batch_size": config.replay_batch_size,
        "packing": "ordered token concatenation bounded by cache sequence length",
        "metric_positions_per_sample": config.metric_positions_per_sample,
        "long_context_min_tokens": config.long_context_min_tokens,
        "warmup_replays": config.warmup_replays,
        "capture_points": ",".join(config.capture_points),
        "module_group_probing": False,
        "candidate_methods": ",".join(method.value for method in config.candidate_methods),
        "target_tensor_count": len(target_tensors),
        "normalization": (
            "quality means over fixed token positions; memory and prefill latency "
            "are candidate/reference ratios after warmup; method refinements with identical "
            "bit/group/packing reuse the base affine hardware-cost control"
        ),
    }
    if calibration_random_seed is not None:
        calibration_metadata["calibration_random_seed"] = calibration_random_seed
    if base_report is not None:
        assert base_sha256 is not None
        assert base_report.calibration is not None
        calibration_metadata.update(
            {
                "base_sensitivity_sha256": base_sha256,
                "base_inventory_sha256": base_report.inventory_sha256,
                "base_probe_backend": base_report.calibration.backend,
                "refinement_probe_backend": _PROBE_BACKEND_VERSION,
            }
        )
    calibration = CalibrationEvidence(
        dataset_id=calibration_dataset_id,
        dataset_sha256=cache_manifest.dataset_sha256,
        samples=cache_manifest.samples,
        domains=cache_manifest.domains,
        sequence_length=cache_manifest.sequence_length,
        backend=_PROBE_BACKEND_VERSION,
        reference=(
            "measured-forward-probe-refinement"
            if base_report is not None
            else "measured-forward-probe"
        ),
        metadata=calibration_metadata,
    )

    normalized_domains = {
        domain.strip().casefold().replace("_", "-") for domain in cache_manifest.domains
    }
    required_domains = (
        _REQUIRED_AGENT_CODING_DOMAINS if config.profile == ProfileName.AGENT_CODING else set()
    )
    release_evidence = (
        cache_manifest.samples >= _MIN_RELEASE_CALIBRATION_SAMPLES
        and measured_tokens >= _MIN_RELEASE_CALIBRATION_TOKENS
        and cache_manifest.domain_provenance == "sample-records"
        and any(
            int(input_ids.shape[-1]) >= config.long_context_min_tokens
            for input_ids in calibration_inputs
        )
        and required_domains.issubset(normalized_domains)
        and (base_report is None or base_report.evidence_kind == EvidenceKind.MEASURED)
    )
    evidence_kind = EvidenceKind.MEASURED if release_evidence else EvidenceKind.MEASURED_DEVELOPMENT
    warnings = [
        "MTP acceptance and decode latency are validated by the candidate benchmark stage, "
        "not by isolated tensor forward probes."
    ]
    if not release_evidence:
        warnings.append(
            "Measured development evidence does not meet the release calibration sample, "
            "token-budget, and workload-domain minimums."
        )
    report = SensitivityReport(
        model=inventory.model,
        architecture_profile=inventory.architecture_profile,
        profile=config.profile,
        evidence_kind=evidence_kind,
        inventory_sha256=inventory_sha256,
        entries=entries,
        calibration=calibration,
        warnings=warnings,
    )
    if progress_path is not None:
        write_data(
            progress_path,
            ProbeProgress(
                inventory_sha256=inventory_sha256,
                config_sha256=config_sha256,
                completed_tensors=state.completed_tensors,
                total_tensors=state.total_tensors,
                complete=True,
            ),
        )
    return report
