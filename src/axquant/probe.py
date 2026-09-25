"""Measured forward-probe sensitivity backend.

Implements per-tensor and module-group sensitivity probing using MLX
forward passes.  MLX is a lazy optional dependency imported only when
the probe backend is actually invoked.

MH6 (AXQ-046): backends may optionally declare an MTP teacher-forced
forward (``supports_mtp_forward`` / ``forward_mtp``) so MTP-role
candidates record a measured, strictly-positive acceptance proxy loss
instead of the honest 0.0 unmeasured marker. See MlxProbeBackend for the
supported-layout limits of the MLX implementation.
"""

from __future__ import annotations

import importlib
import json
import re
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import structlog

from axquant.activation_cache import (
    is_cache_complete,
    load_cache_manifest,
    verify_cache_integrity,
)
from axquant.awq import apply_mlx_awq_scale
from axquant.calibration import calibration_manifest_sha256
from axquant.capture_binding import LoadedActivationCapture, activation_capture_metadata
from axquant.dwq import apply_mlx_dwq_clip
from axquant.errors import BackendUnavailableError, PlanningError, ProbeError
from axquant.gptq import apply_mlx_gptq_refine
from axquant.module_paths import mlx_module_aliases
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    PROTECTED_MIN_BITS,
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
from axquant.schema.loading import (
    load_probe_progress,
)
from axquant.schema.sensitivity import candidate_key
from axquant.serde import load_model, stable_sha256, write_data

log = structlog.get_logger()

_MXFP4_CANDIDATE_BITS = 4
_MXFP4_CANDIDATE_GROUP_SIZE = 32

# MH6 (AXQ-046): Qwen3.5/Qwen3Next-native MTP checkpoints store every MTP
# RMSNorm weight in the shifted (gamma - 1) convention (the same storage form
# MLX-LM's qwen3_5 sanitize compensates when it sees mtp.* keys at load). All
# 1-D tensors in the validated MTP sidecar layout are norm weights, so the
# 1.0 is added back unconditionally there.
_MTP_NORM_SHIFT = 1.0

_MIN_RELEASE_CALIBRATION_SAMPLES = 128
_MIN_RELEASE_CALIBRATION_TOKENS = 8192
_REQUIRED_AGENT_CODING_DOMAINS = {
    "coding",
    "json",
    "tool",
    "multilingual",
    "long-context",
}
_PROBE_BACKEND_VERSION = "axquant-mlx-isolated-probe-v6"
# MH6 (AXQ-046): the MH1 gate treats any non-zero mtp_acceptance_loss as
# measured, so a measured proxy is floored at epsilon to stay strictly
# positive even when draft tokens agree perfectly. 1e-6 is far below the
# smallest observable disagreement (1/metric_positions per sample) and far
# below any profile weight scale, so the floor never inverts a real
# candidate comparison.
MTP_ACCEPTANCE_LOSS_EPSILON = 1e-6
# Methods that refine the float weight before identical affine packing. They
# reuse the matching AFFINE candidate as the hardware-cost control.
_REFINEMENT_METHODS = frozenset(
    {QuantMethod.DWQ, QuantMethod.AWQ, QuantMethod.GPTQ, QuantMethod.GPTQ_ACT}
)
# Methods whose refinement is driven by captured calibration activations.
_ACTIVATION_DRIVEN_METHODS = frozenset({QuantMethod.AWQ, QuantMethod.GPTQ, QuantMethod.GPTQ_ACT})
_REFINEMENT_NOTES = {
    QuantMethod.DWQ: "sampled 0.1/99.9-percentile clipping followed by affine packing",
    QuantMethod.AWQ: "activation-aware channel scaling followed by affine packing",
    QuantMethod.GPTQ: "hessian error-compensated rounding followed by affine packing",
    QuantMethod.GPTQ_ACT: (
        "group-preserving act-order hessian error-compensated rounding followed by affine packing"
    ),
}
# Base role floors match PROTECTED_MIN_BITS. AXQ-026: probe LM-head down to 8
# so an 8-bit LM-head plan choice is measurable; planner default floor stays 16.
_PROBE_MIN_BITS = {
    **PROTECTED_MIN_BITS,
    TensorRole.LM_HEAD: 8,
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
    # MH6: draft-token logits from a teacher-forced MTP forward at the same
    # metric positions as logits/hidden_states; None when the backend cannot
    # run an MTP forward or hidden states were not captured.
    mtp_draft_logits: Any = None


@dataclass
class MtpForwardResult:
    """Draft-token logits from a teacher-forced MTP forward (MH6)."""

    draft_logits: Any


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


class MtpForwardBackend(Protocol):
    """Optional MH6 capability: teacher-forced MTP forward.

    Backends declare ``supports_mtp_forward = True`` (set only when the
    loaded model actually exposes usable MTP modules) and implement
    ``forward_mtp`` to draft-token logits for the current in-memory module
    state. Backends without this capability keep producing the honest 0.0
    unmeasured marker in MetricVector.mtp_acceptance_loss.
    """

    supports_mtp_forward: bool

    def forward_mtp(self, input_ids: Any, hidden_states: Any) -> MtpForwardResult:
        """Draft-token logits of the MTP block teacher-forced with trunk hidden states."""
        ...


def _mtp_forward_capability(backend: ProbeBackend) -> MtpForwardBackend | None:
    """Return the backend's MTP forward capability when fully declared.

    A capability flag without a callable forward is treated as unsupported:
    the probe must fail closed to the unmeasured zero marker, never guess.
    """
    if not bool(getattr(backend, "supports_mtp_forward", False)):
        return None
    forward_mtp = getattr(backend, "forward_mtp", None)
    if not callable(forward_mtp):
        return None
    return cast(MtpForwardBackend, backend)


def _candidate_bits_for_tensor(tensor: TensorSpec, config: ProbeConfig) -> tuple[int, ...]:
    """Apply role floors without turning every protected recommendation into BF16-only."""
    if not tensor.quantizable:
        return (16,)
    if tensor.role.is_mtp and Path(tensor.file).name.lower() in EXTERNAL_MTP_SIDECAR_FILENAMES:
        return (16,)
    minimum_bits = _PROBE_MIN_BITS.get(tensor.role, 8 if tensor.role.is_mtp else 2)
    candidates = tuple(bits for bits in config.candidate_bits if bits >= minimum_bits)
    return candidates or (16,)


class _MtpRopeOffsetCache:
    """Minimal KV-cache duck-type for a standalone MTP block pass (MH6).

    Teacher-forced MTP drafts run on the trailing metric window whose true
    absolute positions start at ``offset``; full attention layers in MLX-LM
    derive rope phases from ``cache.offset``. This shim supplies the offset
    without prefix keys: ``update_and_fetch`` returns only the new keys and
    SDPA ignores the object otherwise.
    """

    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        return keys, values

    def make_mask(self, *args: Any, **kwargs: Any) -> str:
        return "causal"


@dataclass(frozen=True)
class _MtpTensorRef:
    """On-disk location of one integrated-MTP tensor."""

    path: Path
    name: str
    shape: tuple[int, ...]
    dtype: str
    data_offsets: tuple[int, int]


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a safetensors header without reading tensor data."""
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ProbeError(f"truncated safetensors file: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        try:
            header = json.loads(handle.read(header_len))
        except ValueError as exc:
            raise ProbeError(f"invalid safetensors header in {path}: {exc}") from exc
    return {name: meta for name, meta in header.items() if name != "__metadata__"}


def _canonical_mtp_entries(model_dir: Path) -> dict[str, _MtpTensorRef]:
    """Collect integrated-MTP weights on disk, canonicalized past the ``mtp.`` segment.

    MLX-LM sanitizers drop every ``mtp.`` tensor at load, so the loaded model
    tree never shows them; the weights remain on disk either in a dedicated
    ``mtp.safetensors`` sidecar or as ``mtp.`` keys inside the indexed shards
    (HF-native export). The canonical layout is what remains after the
    ``mtp.`` prefix, e.g. ``fc.weight``, ``layers.0.self_attn.q_proj.weight``.
    """
    try:
        files = sorted(model_dir.glob("*.safetensors"))
    except OSError:
        return {}
    if not files:
        return {}
    indexed_files: set[str] = set()
    for index_path in sorted(model_dir.glob("*.index.json")):
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            indexed_files |= {str(name) for name in weight_map.values()}
        except (OSError, ValueError):
            continue  # a broken index must not hide a dedicated sidecar file
    entries: dict[str, _MtpTensorRef] = {}
    for path in [p for p in files if p.name not in indexed_files] + [
        p for p in files if p.name in indexed_files
    ]:
        try:
            header = _safetensors_header(path)
        except (OSError, ProbeError) as exc:
            log.info("mtp_sidecar_header_unreadable", file=str(path), error=str(exc))
            continue
        for name, meta in header.items():
            parts = name.split(".")
            if "mtp" not in parts:
                continue
            canonical = ".".join(parts[parts.index("mtp") + 1 :])
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            if not canonical or not shape or not isinstance(offsets, list) or len(offsets) != 2:
                continue
            entries.setdefault(
                canonical,
                _MtpTensorRef(
                    path=path,
                    name=name,
                    shape=tuple(int(dim) for dim in shape),
                    dtype=str(meta.get("dtype", "")),
                    data_offsets=(int(offsets[0]), int(offsets[1])),
                ),
            )
    return entries


def _read_safetensors_tensor(ref: _MtpTensorRef) -> Any:
    """Load one tensor without pulling its whole shard into memory.

    Applies the shifted-norm storage convention (``_MTP_NORM_SHIFT`` on 1-D
    weights). Unsupported dtypes fail closed instead of guessing a cast.
    """
    mlx = importlib.import_module("mlx.core")
    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError("MLX probing requires numpy") from None
    with ref.path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ProbeError(f"truncated safetensors file: {ref.path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header = json.loads(handle.read(header_len))
        meta = header.get(ref.name)
        if meta is None:
            raise ProbeError(f"tensor {ref.name} vanished from {ref.path}")
        start, end = (int(value) for value in meta["data_offsets"])
        handle.seek(8 + header_len + start)
        nbytes = end - start
        dtype = str(meta.get("dtype", ""))
        if dtype == "BF16":
            raw = np.fromfile(handle, dtype="<u2", count=nbytes // 2)
            array = mlx.array(raw).reshape(ref.shape).view(mlx.bfloat16)
        elif dtype == "F16":
            raw = np.fromfile(handle, dtype="<f2", count=nbytes // 2)
            array = mlx.array(raw).reshape(ref.shape)
        elif dtype == "F32":
            raw = np.fromfile(handle, dtype="<f4", count=nbytes // 4)
            array = mlx.array(raw).reshape(ref.shape)
        else:
            raise ProbeError(f"unsupported MTP weight dtype {dtype} for {ref.name}")
    if len(ref.shape) == 1:
        shifted = array.astype(mlx.float32) + _MTP_NORM_SHIFT
        array = shifted.astype(array.dtype)
    return array


def _module_param_paths(root_name: str, module: Any) -> set[str]:
    """Dotted paths of every submodule under ``root_name`` that carries a weight."""
    paths: set[str] = set()
    named = getattr(module, "named_modules", None)
    if not callable(named):
        return paths
    for name, sub in named():
        path = f"{root_name}.{name}" if name else root_name
        weight = getattr(sub, "weight", None)
        if weight is not None and not callable(weight):
            paths.add(path)
    return paths


# Qwen3_5MoE MTP sidecar stacks the routed experts as 3-D tensors:
# ``mlp.experts.gate_up_proj`` is (experts, 2 * intermediate, hidden) with the
# gate and up projections concatenated along the intermediate axis, and
# ``mlp.experts.down_proj`` is (experts, hidden, intermediate). The MLX-LM
# decoder layer class exposes them as SwitchGLU projections with separate
# weights, so the stacked tensor is halved at load.
_MOE_GATE_UP_KEY = "mlp.experts.gate_up_proj"
_MOE_DOWN_KEY = "mlp.experts.down_proj"
_MOE_SWITCH_LEAVES = {
    "mlp.switch_mlp.gate_proj.weight": _MOE_GATE_UP_KEY,
    "mlp.switch_mlp.up_proj.weight": _MOE_GATE_UP_KEY,
    "mlp.switch_mlp.down_proj.weight": _MOE_DOWN_KEY,
}


def _sidecar_leaf_source(
    leaf: str, mapping: Mapping[str, _MtpTensorRef]
) -> tuple[str, _MtpTensorRef, int | None] | None:
    """Resolve a decoder-layer leaf to its sidecar source tensor.

    Returns ``(source_key, ref, gate_up_half)`` where ``gate_up_half`` is 0
    (gate) or 1 (up) for the concatenated Qwen3_5MoE projection, else None.
    Direct names win; unknown leaves return None so the caller fails closed.
    """
    ref = mapping.get(leaf)
    if ref is not None:
        return leaf, ref, None
    source_key = _MOE_SWITCH_LEAVES.get(leaf)
    if source_key is not None and source_key in mapping:
        if source_key == _MOE_GATE_UP_KEY:
            half = 0 if leaf.endswith("gate_proj.weight") else 1
            return source_key, mapping[source_key], half
        return source_key, mapping[source_key], None
    return None


class MlxProbeBackend:
    """MLX-based probe backend with lazy imports.

    MH6 (AXQ-046): optionally supports a teacher-forced MTP forward used to
    measure MTP acceptance proxy losses. Public MLX-LM has no stable
    cross-architecture MTP entry point, so capability is duck-typed after
    load. Two layouts are recognized, both fail-closed:

    1. The loaded model exposes MTP-block modules directly (a named module
       with an ``mtp`` path segment) plus a resolvable token embedding.
    2. The checkpoint integrates MTP weights that the MLX-LM sanitizer
       stripped at load (Qwen3.5 / Qwen3-Next native MTP, in a dedicated
       ``mtp.safetensors`` or as ``mtp.`` keys inside the indexed shards).
       The block is then rebuilt from the model's own full-attention decoder
       layer class; see ``_build_sidecar_mtp_capability`` for the exact
       structural requirements.

    ``forward_mtp`` additionally fails closed: any runtime uncertainty
    (missing modules, block signature mismatch, shape mismatch) raises
    ProbeError and the probe records the honest 0.0 unmeasured marker.
    Checkpoints whose MTP block cannot be reconstructed keep the unmeasured
    marker instead of producing guessed acceptance evidence.
    """

    def __init__(self, *, calibration_activations: Mapping[str, Any] | None = None) -> None:
        self._model: Any = None
        self._model_dir: Path | None = None
        self._original_modules: dict[str, tuple[Any, str, Any]] = {}
        self._mlx: Any = None
        self._mlx_lm: Any = None
        self._calibration_activations = dict(calibration_activations or {})
        self.metric_positions_per_sample = 32
        self.supports_mtp_forward = False
        self._mtp_blocks: list[Any] = []
        self._embed_tokens_module: Any = None
        self._mtp_sidecar: dict[str, Any] | None = None
        self._mtp_sidecar_paths: set[str] = set()

    def _ensure_mlx(self) -> None:
        if self._mlx is not None:
            return
        try:
            self._mlx = importlib.import_module("mlx.core")
            self._mlx_lm = importlib.import_module("mlx_lm")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"MLX probe backend requires mlx and mlx-lm: {exc}"
            ) from exc

    def load_model(self, model_dir: Path) -> None:
        self._ensure_mlx()
        self._model_dir = Path(model_dir).expanduser().resolve()
        loaded = self._mlx_lm.load(str(self._model_dir), lazy=False)
        self._model = loaded[0]
        self._mlx.eval(self._model.parameters())
        self._detect_mtp_forward_capability()
        log.info(
            "probe_model_loaded",
            model_dir=str(model_dir),
            supports_mtp_forward=self.supports_mtp_forward,
        )

    def _detect_mtp_forward_capability(self) -> None:
        """Duck-type the pieces a teacher-forced MTP forward needs (MH6).

        The flag stays False unless every structural prerequisite resolves;
        runtime failures inside forward_mtp are additionally converted to
        ProbeError so the probe falls back to the unmeasured zero marker.
        """
        self.supports_mtp_forward = False
        self._mtp_blocks = []
        self._embed_tokens_module = None
        self._mtp_sidecar = None
        self._mtp_sidecar_paths = set()
        if self._model is None:
            return
        named = {str(name): module for name, module in self._model.named_modules() if name}
        mtp_names = [name for name in named if any(part == "mtp" for part in name.split("."))]
        embedding = self._resolve_embedding_module(named)
        if embedding is None:
            return
        # Keep only the outermost MTP containers; nested projections must not
        # be invoked as standalone blocks.
        blocks = [
            named[name]
            for name in sorted(mtp_names)
            if not any(other != name and other.startswith(f"{name}.") for other in mtp_names)
        ]
        if blocks and all(callable(block) for block in blocks):
            self._mtp_blocks = blocks
            self._embed_tokens_module = embedding
            self.supports_mtp_forward = True
            return
        # MLX-LM sanitizers strip integrated mtp.* weights at load; rebuild
        # the block from the checkpoint directory when possible.
        self._build_sidecar_mtp_capability(embedding)

    def _resolve_embedding_module(self, named: Mapping[str, Any]) -> Any | None:
        candidates = set()
        for probe in ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight"):
            for alias in mlx_module_aliases(probe):
                candidates.add(alias[: -len(".weight")] if alias.endswith(".weight") else alias)
        for name in sorted(candidates, key=len, reverse=True):
            module = named.get(name)
            if module is not None and callable(module):
                return module
        return None

    def _build_sidecar_mtp_capability(self, embedding: Any) -> bool:
        """Rebuild an integrated MTP block that the MLX-LM sanitizer stripped (MH6).

        Qwen3.5 / Qwen3-Next / Qwen3_5MoE native MTP checkpoints keep the draft
        block in ``mtp.*`` tensors that ``mlx_lm.load`` drops, so the loaded
        model exposes no MTP modules. When the checkpoint directory still
        holds them (a dedicated ``mtp.safetensors`` or ``mtp.`` keys inside
        the indexed shards), the block is reconstructed with the model's own
        full-attention decoder layer class:

        ``fc(cat([pre_fc_norm_embedding(embed(t+1)),
        pre_fc_norm_hidden(hidden_t)]))`` -> MTP decoder layer(s) -> ``norm``
        -> draft head. Norm weights are stored shifted (gamma - 1) and get
        ``_MTP_NORM_SHIFT`` added back. MoE MTP layers store the routed
        experts as stacked 3-D tensors (``mlp.experts.gate_up_proj`` =
        gate||up); they are split into the SwitchGLU projections the decoder
        layer class exposes (``_sidecar_leaf_source``).

        Every structural deviation leaves the capability off (fail closed).
        """
        self._ensure_mlx()
        if self._model_dir is None or self._model is None:
            return False
        try:
            entries = _canonical_mtp_entries(self._model_dir)
        except OSError as exc:
            log.info("mtp_sidecar_scan_failed", error=str(exc))
            return False
        if not entries:
            return False
        language_model = getattr(self._model, "language_model", None)
        backbone = getattr(language_model, "model", None) if language_model is not None else None
        layers = getattr(backbone, "layers", None) if backbone is not None else None
        if not layers:
            return False
        trunk_index = next(
            (i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")), None
        )
        if trunk_index is None:
            return False
        args = getattr(language_model, "args", None) or getattr(self._model, "args", None)
        eps = float(getattr(args, "rms_norm_eps", None) or 1e-6)
        fc_ref = entries.get("fc.weight")
        scalar_refs = [
            entries.get(name)
            for name in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight")
        ]
        if fc_ref is None or any(ref is None for ref in scalar_refs):
            log.info("mtp_sidecar_layout_incomplete", keys=sorted(entries)[:8])
            return False
        norm_emb_ref, norm_hidden_ref, final_norm_ref = scalar_refs
        if norm_emb_ref is None or norm_hidden_ref is None or final_norm_ref is None:
            return False
        if (
            len(fc_ref.shape) != 2
            or fc_ref.shape[1] % 2 != 0
            or fc_ref.shape[0] != (fc_ref.shape[1] // 2)
        ):
            log.info("mtp_sidecar_fc_shape_unexpected", shape=fc_ref.shape)
            return False
        if any(ref is not None and len(ref.shape) != 1 for ref in scalar_refs):
            log.info("mtp_sidecar_norm_shape_unexpected")
            return False
        hidden = fc_ref.shape[0]
        layer_indices = sorted(
            {
                int(match.group(1))
                for key in entries
                if (match := re.match(r"layers\.(\d+)\.", key)) is not None
            }
        )
        if not layer_indices or layer_indices != list(range(len(layer_indices))):
            log.info("mtp_sidecar_layers_not_contiguous", layers=layer_indices)
            return False
        nn = importlib.import_module("mlx.nn")
        try:
            mtp_layers = []
            for index in layer_indices:
                prefix = f"layers.{index}."
                mapping = {
                    key[len(prefix) :]: ref
                    for key, ref in entries.items()
                    if key.startswith(prefix)
                }
                mtp_layers.append(
                    self._instantiate_sidecar_layer(
                        type(layers[trunk_index]), args, trunk_index, mapping
                    )
                )
            fc = nn.Linear(2 * hidden, hidden, bias=False)
            fc.weight = _read_safetensors_tensor(fc_ref)
            norm_emb = nn.RMSNorm(hidden, eps=eps)
            norm_emb.weight = _read_safetensors_tensor(norm_emb_ref)
            norm_hidden = nn.RMSNorm(hidden, eps=eps)
            norm_hidden.weight = _read_safetensors_tensor(norm_hidden_ref)
            final_norm = nn.RMSNorm(hidden, eps=eps)
            final_norm.weight = _read_safetensors_tensor(final_norm_ref)
            for module in [fc, norm_emb, norm_hidden, final_norm, *mtp_layers]:
                self._mlx.eval(module.parameters())
        except (ProbeError, TypeError, ValueError, RuntimeError, OSError) as exc:
            log.info("mtp_sidecar_reconstruction_failed", error=str(exc))
            self._mtp_sidecar = None
            self._mtp_sidecar_paths = set()
            return False
        # Single tree shared by forward_mtp and quantize/restore so module
        # replacement and restore mutate exactly what the forward reads.
        self._mtp_sidecar = {
            "fc": fc,
            "pre_fc_norm_embedding": norm_emb,
            "pre_fc_norm_hidden": norm_hidden,
            "norm": final_norm,
            "layers": {
                index: layer for index, layer in zip(layer_indices, mtp_layers, strict=True)
            },
        }
        self._embed_tokens_module = embedding
        self._mtp_sidecar_paths = set()
        for name, module in self._mtp_sidecar.items():
            if name == "layers":
                for index, layer in module.items():
                    self._mtp_sidecar_paths |= _module_param_paths(f"layers.{index}", layer)
            else:
                self._mtp_sidecar_paths |= _module_param_paths(name, module)
        log.info(
            "mtp_sidecar_reconstructed",
            layers=len(mtp_layers),
            tensors=len(entries),
            quantizable_paths=sorted(self._mtp_sidecar_paths),
        )
        self.supports_mtp_forward = True
        return True

    def _instantiate_sidecar_layer(
        self,
        layer_class: Any,
        args: Any,
        trunk_index: int,
        mapping: Mapping[str, _MtpTensorRef],
    ) -> Any:
        """Construct an MTP decoder layer and overwrite every parameter from disk.

        Fail closed unless the mapping covers exactly the layer's parameter
        leaves with matching shapes: any leftover destination keeps random
        init weights, and any unconsumed sidecar weight signals a layout the
        forward cannot trust.
        """
        try:
            layer = layer_class(args, trunk_index)
        except (TypeError, ValueError) as exc:
            raise ProbeError(f"cannot construct MTP layer from trunk class: {exc}") from exc
        tree = layer.parameters()
        consumed: set[str] = set()

        def assign(node: dict[str, Any], prefix: str) -> None:
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    assign(value, path)
                    continue
                source = _sidecar_leaf_source(path, mapping)
                if source is None:
                    raise ProbeError(f"MTP sidecar has no weight for layer leaf {path!r}")
                source_key, ref, gate_up_half = source
                array = _read_safetensors_tensor(ref)
                if gate_up_half is not None:
                    if len(ref.shape) != 3 or ref.shape[1] % 2 != 0:
                        raise ProbeError(
                            f"MTP sidecar gate_up projection must be 3-D with an even "
                            f"intermediate axis, got {ref.shape} for {source_key!r}"
                        )
                    width = ref.shape[1] // 2
                    array = array[:, gate_up_half * width : (gate_up_half + 1) * width, :]
                if tuple(array.shape) != tuple(value.shape):
                    raise ProbeError(
                        f"MTP sidecar shape mismatch for {path!r}: "
                        f"{tuple(array.shape)} from {source_key!r} "
                        f"vs module {tuple(value.shape)}"
                    )
                node[key] = array
                consumed.add(source_key)

        assign(tree, "")
        extra = sorted(set(mapping) - consumed)
        if extra:
            raise ProbeError(f"MTP sidecar weights have no module destination: {extra[:5]}")
        layer.update(tree)
        return layer

    def forward_mtp(self, input_ids: Any, hidden_states: Any) -> MtpForwardResult:
        """Teacher-force the loaded model's MTP blocks and return draft logits.

        DeepSeek/Qwen-Next MTP convention: the draft block at position t sees
        the embedding of token t+1 fused with the trunk hidden state at t.
        The returned logits sit at the same metric positions as the trunk
        hidden states, so reference and candidate drafts compare position by
        position. Fail closed: any structural uncertainty raises ProbeError.
        """
        self._ensure_mlx()
        if self._model is None:
            raise ProbeError("model not loaded")
        if not self.supports_mtp_forward:
            raise ProbeError("loaded model exposes no usable MTP modules")
        try:
            import numpy as np
        except ImportError:
            raise BackendUnavailableError("MLX probing requires numpy") from None
        tokens = self._mlx.array(np.asarray(input_ids, dtype=np.int32))
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        hidden = self._mlx.array(np.asarray(hidden_states, dtype=np.float32))
        positions = int(hidden.shape[1]) if hidden.ndim == 3 else 0
        if positions < 1 or tokens.ndim != 2 or hidden.shape[0] != tokens.shape[0]:
            raise ProbeError(
                "MTP forward shape mismatch: "
                f"tokens {tuple(tokens.shape)}, hidden {tuple(hidden.shape)}"
            )
        try:
            embedded = self._embed_tokens_module(tokens)
            if isinstance(embedded, (tuple, list)):
                embedded = embedded[0]
            if int(embedded.shape[1]) < positions:
                raise ProbeError(
                    "MTP forward shape mismatch: embedding has fewer positions "
                    f"({int(embedded.shape[1])}) than trunk hidden states ({positions})"
                )
            if self._mtp_sidecar is not None:
                block_out = self._forward_sidecar_mtp_fusion(embedded, hidden, tokens)
            else:
                # Trunk hidden states cover tokens[:-1] at the metric positions;
                # the draft input at those positions pairs them with the trailing
                # token embeddings (token t+1 fused with hidden state t).
                mtp_input = self._mlx.concatenate([embedded[:, -positions:], hidden], axis=-1)
                block_out = mtp_input
                for block in self._mtp_blocks:
                    block_out = block(block_out)
                    if isinstance(block_out, (tuple, list)):
                        block_out = block_out[0]
            language_model = getattr(self._model, "language_model", None)
            tied = bool(
                getattr(getattr(language_model, "args", None), "tie_word_embeddings", False)
            )
            if tied and hasattr(self._embed_tokens_module, "as_linear"):
                draft_logits = self._embed_tokens_module.as_linear(block_out)
            else:
                head_root = language_model if language_model is not None else self._model
                lm_head = getattr(head_root, "lm_head", None)
                if not callable(lm_head):
                    raise ProbeError("cannot resolve a draft output head for MTP forward")
                draft_logits = lm_head(block_out)
        except ProbeError:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ProbeError(f"MTP forward failed: {exc}") from exc
        draft_logits = draft_logits.astype(self._mlx.float32)
        self._mlx.eval(draft_logits)
        return MtpForwardResult(draft_logits=np.asarray(draft_logits))

    def _forward_sidecar_mtp_fusion(self, embedded: Any, hidden: Any, tokens: Any) -> Any:
        """Run the reconstructed Qwen3.5/Qwen3Next MTP block (MH6).

        ``embedded`` covers the full token sequence; its trailing ``positions``
        embeddings are token t+1 relative to the trunk hidden states, so the
        fusion needs the full sequence behind the window (``positions + 1``
        embeddings). Attention runs with a rope-offset shim so the trailing
        window keeps its true absolute positions.
        """
        sidecar = self._mtp_sidecar
        if sidecar is None:
            raise ProbeError("MTP sidecar block not reconstructed")
        positions = int(hidden.shape[1])
        if int(embedded.shape[1]) < positions + 1:
            raise ProbeError(
                "MTP forward shape mismatch: sidecar fusion needs the full "
                f"trailing token window ({positions + 1} embeddings), "
                f"got {int(embedded.shape[1])}"
            )
        hidden = hidden.astype(embedded.dtype)
        fused = self._mlx.concatenate(
            [
                sidecar["pre_fc_norm_embedding"](embedded[:, -positions:]),
                sidecar["pre_fc_norm_hidden"](hidden),
            ],
            axis=-1,
        )
        block_out = sidecar["fc"](fused)
        cache = _MtpRopeOffsetCache(int(tokens.shape[1]) - positions)
        mask = "causal" if positions > 1 else None
        for layer in sidecar["layers"].values():
            block_out = layer(block_out, mask=mask, cache=cache)
        return sidecar["norm"](block_out)

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
        if method not in {QuantMethod.AFFINE, QuantMethod.MXFP4, *_REFINEMENT_METHODS}:
            raise ProbeError(f"probe backend does not support method {method.value}")
        original_weight = getattr(module, "weight", None)
        mutation_installed = False
        try:
            quantized_mode = "affine"
            if method == QuantMethod.MXFP4:
                # ADR-0015 / AXQ-046 MH5: native MLX MXFP4 block-float packing.
                if bits != _MXFP4_CANDIDATE_BITS or group_size != _MXFP4_CANDIDATE_GROUP_SIZE:
                    raise ProbeError("mxfp4 probe candidates require 4-bit with group size 32")
                quantized_mode = "mxfp4"
            if method == QuantMethod.DWQ:
                apply_mlx_dwq_clip(module)
            elif method in _ACTIVATION_DRIVEN_METHODS:
                activations = self._calibration_for(module_path, resolved_path, method)
                if method == QuantMethod.AWQ:
                    apply_mlx_awq_scale(
                        module,
                        activations=activations,
                        bits=bits,
                        group_size=group_size,
                    )
                else:
                    apply_mlx_gptq_refine(
                        module,
                        activations=activations,
                        bits=bits,
                        group_size=group_size,
                        act_order=method == QuantMethod.GPTQ_ACT,
                    )
            quantized_module = to_quantized(
                group_size=group_size,
                bits=bits,
                mode=quantized_mode,
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
            if method in _REFINEMENT_METHODS and original_weight is not None:
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
        if len(suffix_matches) == 1:
            return str(suffix_matches[0])
        sidecar_match = self._resolve_sidecar_module_path(aliases)
        if sidecar_match is not None:
            return sidecar_match
        raise ProbeError(
            f"cannot uniquely resolve module path {module_path!r}; "
            f"found {len(suffix_matches)} matches"
        )

    def _resolve_sidecar_module_path(self, aliases: tuple[str, ...]) -> str | None:
        """Match an inventory module path against the reconstructed MTP block.

        Integrated-MTP tensors are named ``...mtp.<canonical>`` in the source
        checkpoint while the reconstructed block lives outside the loaded
        model tree, so named-module resolution can never see it.
        """
        if not self._mtp_sidecar_paths:
            return None
        candidates = set()
        for alias in aliases:
            parts = alias.split(".")
            if "mtp" not in parts:
                continue
            remainder = ".".join(parts[parts.index("mtp") + 1 :])
            if remainder in self._mtp_sidecar_paths:
                candidates.add(f"mtp.{remainder}")
        if len(candidates) == 1:
            return candidates.pop()
        return None

    def _is_sidecar_module_path(self, module_path: str) -> bool:
        return self._mtp_sidecar is not None and (
            module_path == "mtp" or module_path.startswith("mtp.")
        )

    def _calibration_for(self, module_path: str, resolved_path: str, method: QuantMethod) -> Any:
        """Resolve captured calibration activations for an inventory module path."""
        keys = [module_path, resolved_path]
        for key in (module_path, resolved_path):
            if key.endswith(".weight"):
                keys.append(key[: -len(".weight")])
        for alias in mlx_module_aliases(module_path):
            keys.append(alias)
            if alias.endswith(".weight"):
                keys.append(alias[: -len(".weight")])
        for key in keys:
            if key in self._calibration_activations:
                return self._calibration_activations[key]
        raise ProbeError(
            f"{method.value} probing requires calibration activations for module "
            f"{module_path}; run capture-activations to produce them"
        )

    def _get_parent_and_module(self, module_path: str) -> tuple[Any, str, Any]:
        parts = module_path.split(".")
        if self._is_sidecar_module_path(module_path):
            # Reconstructed MTP block: walk the sidecar tree instead of the
            # loaded model tree (dict parents resolve via key access).
            parts = parts[1:]
            current: Any = self._mtp_sidecar
        else:
            current = self._model
        for part in parts[:-1]:
            current = self._get_child(current, part)
        child_name = parts[-1]
        return current, child_name, self._get_child(current, child_name)

    @staticmethod
    def _get_child(parent: Any, child_name: str) -> Any:
        if isinstance(parent, Mapping) and child_name in parent:
            return parent[child_name]
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
        if isinstance(parent, dict) and child_name in parent:
            parent[child_name] = module
            return
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
    return max(0.0, 1.0 - similarity)


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
    if len(current) >= 2:
        packed_inputs.append(np.asarray(current, dtype=np.int32))
    elif current:
        # A single-token remainder cannot be probed (forward passes need at
        # least two tokens); drop it instead of aborting the whole run.
        measured_tokens -= len(current)
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
    actual_sha256 = calibration_manifest_sha256(manifest)
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
    mtp_forward: MtpForwardBackend | None = None,
    mtp_reference_drafts: list[Any] | None = None,
) -> MetricVector:
    import numpy as np

    # Replay batches carry unequal sequence counts (the packed tail always
    # forms a short batch), so every per-batch mean is weighted by its
    # sequence count; an unweighted mean over batches would over-weight
    # positions in small batches and make the metric depend on
    # replay_batch_size.
    output_kl: list[tuple[float, float]] = []
    hidden_error: list[tuple[float, float]] = []
    cosine_distance: list[tuple[float, float]] = []
    token_disagreement: list[tuple[float, float]] = []
    task_loss_delta: list[tuple[float, float]] = []
    long_context_loss: list[tuple[float, float]] = []
    mtp_disagreement: list[tuple[float, float]] = []
    peak_memory: list[float] = []
    latency: list[float] = []
    mtp_measurement_failed = False
    for batch_index, (input_ids, reference) in enumerate(zip(inputs, references, strict=True)):
        candidate = backend.forward(input_ids)
        rows = float(input_ids.shape[0]) if getattr(input_ids, "ndim", 1) == 2 else 1.0
        if reference.logits is None or candidate.logits is None:
            raise ProbeError("probe backend did not return logits")
        output_kl.append((_compute_logit_kl(reference.logits, candidate.logits), rows))
        token_disagreement.append(
            (
                compute_token_disagreement(
                    np.argmax(reference.logits, axis=-1),
                    np.argmax(candidate.logits, axis=-1),
                ),
                rows,
            )
        )
        if reference.hidden_states is not None and candidate.hidden_states is not None:
            hidden_error.append(
                (
                    compute_hidden_state_error(reference.hidden_states, candidate.hidden_states),
                    rows,
                )
            )
            cosine_distance.append(
                (
                    compute_cosine_distance(reference.hidden_states, candidate.hidden_states),
                    rows,
                )
            )
        elif require_hidden_states:
            raise ProbeError("probe backend did not return required hidden states")
        if reference.loss is not None and candidate.loss is not None:
            loss_delta = max(0.0, candidate.loss - reference.loss)
            task_loss_delta.append((loss_delta, rows))
            if int(input_ids.shape[-1]) >= long_context_min_tokens:
                long_context_loss.append((loss_delta, rows))
        if mtp_forward is not None and mtp_reference_drafts is not None:
            # MH6: teacher-force the (candidate-state) MTP block with the
            # reference trunk hidden states and compare draft-token agreement.
            # Any forward uncertainty fails closed to the unmeasured marker
            # for this whole candidate.
            try:
                if reference.hidden_states is None:
                    raise ProbeError("probe backend did not return trunk hidden states")
                reference_draft = mtp_reference_drafts[batch_index]
                candidate_draft = mtp_forward.forward_mtp(
                    input_ids, reference.hidden_states
                ).draft_logits
                mtp_disagreement.append(
                    (
                        compute_token_disagreement(
                            np.argmax(reference_draft, axis=-1),
                            np.argmax(candidate_draft, axis=-1),
                        ),
                        rows,
                    )
                )
            except (ProbeError, TypeError, ValueError, RuntimeError) as exc:
                mtp_measurement_failed = True
                log.warning("mtp_forward_candidate_failed", error=str(exc))
        peak_memory.append(float(candidate.peak_memory_bytes or 0))
        latency.append(candidate.latency_seconds)

    def mean(values: list[tuple[float, float]]) -> float:
        total_weight = sum(weight for _value, weight in values)
        if total_weight <= 0:
            return 0.0
        return float(sum(value * weight for value, weight in values) / total_weight)

    reference_peak = max(
        (float(reference.peak_memory_bytes or 0) for reference in references),
        default=0.0,
    )
    reference_latency = sum(reference.latency_seconds for reference in references)
    candidate_peak = max(peak_memory, default=0.0)
    candidate_latency = sum(latency)
    mtp_acceptance_loss = 0.0
    if mtp_forward is not None and mtp_reference_drafts is not None and not mtp_measurement_failed:
        # Epsilon floor: a perfect-acceptance candidate must stay strictly
        # positive so the MH1 gate reads it as measured, not as the zero
        # unmeasured marker.
        mtp_acceptance_loss = max(MTP_ACCEPTANCE_LOSS_EPSILON, mean(mtp_disagreement))
    return MetricVector(
        output_kl=mean(output_kl),
        hidden_state_error=mean(hidden_error),
        cosine_distance=mean(cosine_distance),
        token_disagreement=mean(token_disagreement),
        task_loss_delta=mean(task_loss_delta),
        mtp_acceptance_loss=mtp_acceptance_loss,
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
    calibration_activations: Mapping[str, Any] | None = None,
    allow_legacy_4bit: bool = False,
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
    AXQ-047: a 4-bit candidate grid yields MXFP4 candidates only unless
    ``allow_legacy_4bit`` is set (affine 4-bit is a retired product line).
    """
    if inventory.quantized_source:
        raise ProbeError("measured sensitivity requires an unquantized BF16 source inventory")
    if not is_immutable_revision(config.model.revision):
        raise ProbeError("measured sensitivity requires a revision-pinned source model")
    if config.model.model_id != inventory.model.model_id:
        raise ProbeError("probe model does not match the inventory model")
    if config.model.revision != inventory.model.revision:
        raise ProbeError("probe revision does not match the inventory revision")
    if config.model.format != inventory.model.format:
        raise ProbeError("probe model format does not match the inventory format")
    if config.model.local_path is not None and inventory.model.local_path is not None:
        probe_model_dir = Path(config.model.local_path).expanduser().resolve()
        inventory_model_dir = Path(inventory.model.local_path).expanduser().resolve()
        if probe_model_dir != inventory_model_dir:
            raise ProbeError("probe model directory does not match the inventory source")
    if config.module_group_probing:
        raise ProbeError(
            "module-group probing is not available in the tensor-isolation backend; "
            "disable it rather than relabelling representative tensor results"
        )
    activation_driven = bool(_ACTIVATION_DRIVEN_METHODS & set(config.candidate_methods))
    bound_capture: LoadedActivationCapture | None = None
    if activation_driven:
        if calibration_activations is None:
            raise ProbeError(
                "AWQ/GPTQ measured probing requires captured calibration activations; "
                "run capture-activations and pass the artifact via --calibration-activations"
            )
        if not isinstance(calibration_activations, LoadedActivationCapture):
            raise ProbeError(
                "AWQ/GPTQ measured probing requires a checksum-bound capture loaded with "
                "load_capture_activations; an unbound activation mapping is not evidence"
            )
        bound_capture = calibration_activations
    if backend is None:
        backend = MlxProbeBackend(calibration_activations=calibration_activations)
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
            "activation_capture_manifest_sha256": (
                bound_capture.manifest_sha256 if bound_capture is not None else None
            ),
        }
    )
    progress_path = Path(state_path).expanduser().resolve() if state_path is not None else None
    if state is None and progress_path is not None and progress_path.is_file():
        progress = load_probe_progress(progress_path)
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
    if bound_capture is not None:
        capture_manifest = bound_capture.manifest
        tokenized_cache_sha256 = stable_sha256(cache_manifest)
        if (
            capture_manifest.model != config.model.model_id
            or capture_manifest.revision != config.model.revision
        ):
            raise ProbeError("activation capture source model does not match the probe model")
        if capture_manifest.tokenized_cache_manifest_sha256 != tokenized_cache_sha256:
            raise ProbeError("activation capture does not bind the probe calibration cache")
        if capture_manifest.cache_key_sha256 != cache_manifest.cache_key_sha256:
            raise ProbeError("activation capture cache key does not match the probe cache")
        if capture_manifest.calibration_dataset_id != calibration_dataset_id:
            raise ProbeError("activation capture dataset does not match the probe calibration")
    calibration_random_seed: int | None = None
    if cache_manifest.calibration_manifest_sha256 not in {None, "", "unknown"}:
        calibration_random_seed = load_model(
            cache_path / "calibration_manifest.json",
            CalibrationManifest,
        ).random_seed
    if base_report is not None:
        if base_report.calibration is None:
            raise ProbeError("base sensitivity report has no calibration provenance")
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
    mtp_target_tensors = {
        tensor.name
        for tensor in inventory.tensors
        if tensor.role.is_mtp and tensor.name in target_tensors
    }
    mtp_forward: MtpForwardBackend | None = None
    references: list[ForwardResult] = []
    reference_metrics = MetricVector()
    mtp_reference_drafts: list[Any] | None = None
    if probe_required:
        model_path = config.model.local_path or config.model.model_id
        backend.load_model(Path(model_path).expanduser().resolve())
        # MTP forward capability is only known once load_model inspected the
        # loaded module tree, so resolve it here: a fresh backend reports False
        # until then, which would silently skip every MTP acceptance
        # measurement and misreport the provenance as capability-less.
        if mtp_target_tensors:
            mtp_forward = _mtp_forward_capability(backend)
        for _ in range(config.warmup_replays):
            backend.forward(calibration_inputs[0])
        references = [backend.forward(input_ids) for input_ids in calibration_inputs]
        if any(reference.logits is None for reference in references):
            raise ProbeError("probe backend did not return reference logits")
        if "hidden" in config.capture_points and any(
            reference.hidden_states is None for reference in references
        ):
            raise ProbeError(
                "probe backend did not return requested reference hidden states; "
                "plain dense backbones expose logits only — retry with "
                "--capture-points output"
            )
        reference_metrics = _reference_metrics(references)
        if mtp_forward is not None:
            # MH6: capture reference draft logits from the untouched model so
            # each MTP-role candidate can be scored by draft-token agreement.
            # Any uncertainty (missing hidden states, forward failure) keeps
            # the honest zero unmeasured marker for every MTP candidate.
            try:
                drafts = [
                    mtp_forward.forward_mtp(input_ids, reference.hidden_states).draft_logits
                    for input_ids, reference in zip(calibration_inputs, references, strict=True)
                    if reference.hidden_states is not None
                ]
            except (ProbeError, TypeError, ValueError, RuntimeError) as exc:
                log.warning("mtp_reference_forward_failed", error=str(exc))
            else:
                if len(drafts) == len(references):
                    mtp_reference_drafts = drafts
                else:
                    log.warning(
                        "mtp_reference_forward_missing_hidden_states",
                        batches=len(references) - len(drafts),
                    )
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
            if base_entry is None:
                raise ProbeError(f"base sensitivity is missing tensor {tensor.name}")
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
                preserved_metrics = MetricVector()
                if tensor.role.is_mtp and mtp_reference_drafts is not None:
                    # MH6: reference-precision MTP keeps the measured marker —
                    # reference draft agreement against itself floors at epsilon.
                    preserved_metrics = preserved_metrics.model_copy(
                        update={"mtp_acceptance_loss": MTP_ACCEPTANCE_LOSS_EPSILON}
                    )
                    preservation_reason += (
                        "; MTP acceptance loss measured via teacher-forced MTP forward"
                    )
                elif tensor.role.is_mtp:
                    preservation_reason += (
                        "; MTP acceptance loss unmeasured (backend has no MTP forward capability)"
                    )
                candidates = [
                    CandidateMeasurement(
                        bits=16,
                        method=QuantMethod.BF16,
                        group_size=None,
                        metrics=preserved_metrics,
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
        existing_keys = {
            candidate_key(candidate.bits, candidate.method, candidate.group_size)
            for candidate in probe_candidates
        }
        # Early-termination compares a candidate only against strictly
        # lower-bit (genuinely cheaper) measurements on the same
        # (method, group_size) track. Tracking a single minimum across all
        # bit-widths would let higher-bit seeds from a base report falsely
        # dominate a healthy low-bit candidate.
        track_losses: dict[tuple[QuantMethod, int], dict[int, float]] = {}
        for candidate in probe_candidates:
            if candidate.bits < 16 and candidate.supported and candidate.group_size is not None:
                per_bits = track_losses.setdefault((candidate.method, candidate.group_size), {})
                previous = per_bits.get(candidate.bits)
                per_bits[candidate.bits] = (
                    candidate.metrics.output_kl
                    if previous is None
                    else min(previous, candidate.metrics.output_kl)
                )

        effective_group_sizes = config.effective_group_sizes()
        for bits in tensor_candidate_bits:
            if bits == 16:
                bf16_key = candidate_key(16, QuantMethod.BF16, None)
                if bf16_key not in existing_keys:
                    bf16_metrics = reference_metrics
                    bf16_note = "reference precision"
                    if tensor.role.is_mtp and mtp_reference_drafts is not None:
                        # MH6: reference draft agreement floors at epsilon so
                        # every MTP-scoped candidate stays strictly positive.
                        bf16_metrics = bf16_metrics.model_copy(
                            update={"mtp_acceptance_loss": MTP_ACCEPTANCE_LOSS_EPSILON}
                        )
                        bf16_note += "; MTP acceptance loss measured via teacher-forced MTP forward"
                    elif tensor.role.is_mtp:
                        bf16_note += (
                            "; MTP acceptance loss unmeasured "
                            "(backend has no MTP forward capability)"
                        )
                    probe_candidates.append(
                        CandidateMeasurement(
                            bits=16,
                            method=QuantMethod.BF16,
                            group_size=None,
                            metrics=bf16_metrics,
                            measured_tokens=measured_tokens,
                            note=bf16_note,
                        )
                    )
                    existing_keys.add(bf16_key)
                continue

            bits_group_sizes = effective_group_sizes
            if (
                bits == 4
                and not allow_legacy_4bit
                and QuantMethod.MXFP4 in config.candidate_methods
                and _MXFP4_CANDIDATE_GROUP_SIZE not in effective_group_sizes
            ):
                # The enforced MXFP4 rung carries its fixed group size 32 even
                # when the requested grid names coarser groups only (AXQ-047).
                bits_group_sizes = (*effective_group_sizes, _MXFP4_CANDIDATE_GROUP_SIZE)
            for group_size in bits_group_sizes:
                for method in config.candidate_methods:
                    if bits == 4 and not allow_legacy_4bit and method != QuantMethod.MXFP4:
                        # AXQ-047 (ADR 0016): the affine 4-bit product line is
                        # retired; measured probing at 4-bit runs MXFP4 only
                        # unless the explicit legacy opt-in is set.
                        continue
                    if method == QuantMethod.MXFP4 and (
                        bits != _MXFP4_CANDIDATE_BITS or group_size != _MXFP4_CANDIDATE_GROUP_SIZE
                    ):
                        # MXFP4 exists only at 4-bit / group 32; the candidate
                        # grid naturally yields one mxfp4 candidate per 4-bit
                        # gs32 tensor instead of measuring impossible configs.
                        continue
                    cand_key = candidate_key(bits, method, group_size)
                    if cand_key in existing_keys:
                        continue
                    try:
                        backend.quantize_module(
                            tensor.module_path,
                            bits,
                            group_size,
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
                            mtp_forward=mtp_forward if tensor.role.is_mtp else None,
                            mtp_reference_drafts=(
                                mtp_reference_drafts if tensor.role.is_mtp else None
                            ),
                        )
                        packing_control = next(
                            (
                                candidate
                                for candidate in probe_candidates
                                if candidate.bits == bits
                                and candidate.method == QuantMethod.AFFINE
                                and candidate.group_size == group_size
                                and candidate.supported
                            ),
                            None,
                        )
                        if method in _REFINEMENT_METHODS and packing_control is not None:
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
                        per_bits = track_losses.setdefault((method, group_size), {})
                        cheaper = [
                            loss for track_bits, loss in per_bits.items() if track_bits < bits
                        ]
                        previous_loss = min(cheaper) if cheaper else None
                        dominated = (
                            previous_loss is not None
                            and metrics.output_kl > previous_loss * config.early_termination_factor
                        )
                        note_parts: list[str] = []
                        refinement_note = _REFINEMENT_NOTES.get(method)
                        if refinement_note is not None:
                            note_parts.append(refinement_note)
                            if packing_control is not None:
                                note_parts.append(
                                    "hardware costs normalized to the identical "
                                    "affine packing control"
                                )
                        if dominated:
                            note_parts.append(
                                "dominated by cheaper "
                                f"{method.value} candidate at "
                                f"{config.early_termination_factor}x bound"
                            )
                        if tensor.role.is_mtp:
                            # MH6: per-candidate provenance of the MTP
                            # acceptance loss source.
                            note_parts.append(
                                "MTP acceptance loss measured via teacher-forced MTP forward"
                                if metrics.mtp_acceptance_loss > 0
                                else (
                                    "MTP acceptance loss unmeasured "
                                    "(backend has no MTP forward capability)"
                                )
                            )
                        probe_candidates.append(
                            CandidateMeasurement(
                                bits=bits,
                                method=method,
                                group_size=group_size,
                                metrics=metrics,
                                supported=not dominated,
                                measured_tokens=measured_tokens,
                                note="; ".join(note_parts) or None,
                            )
                        )
                        existing_keys.add(cand_key)
                        recorded = per_bits.get(bits)
                        per_bits[bits] = (
                            metrics.output_kl
                            if recorded is None
                            else min(recorded, metrics.output_kl)
                        )
                    except ProbeError as exc:
                        probe_candidates.append(
                            CandidateMeasurement(
                                bits=bits,
                                method=method,
                                group_size=group_size,
                                metrics=MetricVector(),
                                supported=False,
                                measured_tokens=0,
                                note=f"probe failed: {exc}",
                            )
                        )
                        existing_keys.add(cand_key)
                    finally:
                        backend.restore_module(tensor.module_path)

        probe_candidates.sort(
            key=lambda candidate: (
                candidate.bits,
                candidate.group_size or 0,
                candidate.method.value,
            )
        )
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
    if bound_capture is not None:
        calibration_metadata.update(activation_capture_metadata(bound_capture))
    if mtp_target_tensors:
        # MH6: report-level provenance of the MTP acceptance loss source,
        # derived from the recorded candidates so resumed runs report what
        # is actually on record.
        mtp_measured = any(
            candidate.metrics.mtp_acceptance_loss > 0
            for entry in entries
            if entry.tensor.role.is_mtp
            for candidate in entry.candidates
        )
        calibration_metadata["mtp_acceptance_provenance"] = (
            "mtp-forward" if mtp_measured else "unmeasured"
        )
    if calibration_random_seed is not None:
        calibration_metadata["calibration_random_seed"] = calibration_random_seed
    if base_report is not None:
        if base_sha256 is None or base_report.calibration is None:
            raise ProbeError("base sensitivity provenance is incomplete")
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
    mtp_acceptance_measured = "mtp_acceptance_provenance" in calibration_metadata and (
        calibration_metadata["mtp_acceptance_provenance"] == "mtp-forward"
    )
    if mtp_acceptance_measured:
        warnings = [
            "MTP acceptance loss was measured by teacher-forced MTP forward probes; "
            "decode latency is validated by the candidate benchmark stage, not by "
            "isolated tensor forward probes."
        ]
    else:
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
