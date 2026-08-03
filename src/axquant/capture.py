"""Calibration activation capture for weight-refinement backends.

AWQ/GPTQ-style refinement needs per-module INPUT activations collected from a
forward replay of a verified tokenized calibration cache.  MLX has no forward
hooks, so capture swaps each target ``nn.Linear`` with a recording wrapper
installed on its parent module (the same child-replacement mechanics as the
probe backend), replays the calibration batches, and writes one
checksum-bound npz per module plus an ``ActivationCaptureManifest``.

MLX is a lazy optional dependency imported only when capture runs; loading a
capture artifact back (``load_capture_activations``) does not require MLX.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from axquant.activation_cache import _write_npz_atomic, load_cache_manifest
from axquant.errors import BackendUnavailableError, CaptureError
from axquant.module_paths import mlx_module_aliases
from axquant.probe import _calibration_dataset_id, _load_calibration_inputs
from axquant.schema import ActivationCaptureEntry, ActivationCaptureManifest
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

if TYPE_CHECKING:
    import numpy as np

log = structlog.get_logger()

CAPTURE_MANIFEST_NAME = "activation_capture_manifest.json"
CAPTURE_ACTIVATIONS_DIR = "activations"


def _ensure_mlx() -> tuple[Any, Any, Any]:
    try:
        import importlib

        mlx = importlib.import_module("mlx.core")
        nn = importlib.import_module("mlx.nn")
        mlx_lm = importlib.import_module("mlx_lm")
    except ImportError as exc:
        raise BackendUnavailableError(f"activation capture requires mlx and mlx-lm: {exc}") from exc
    return mlx, nn, mlx_lm


def _sanitize_filename(module_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", module_path)


def _get_child(parent: Any, child_name: str) -> Any:
    if hasattr(parent, child_name):
        return getattr(parent, child_name)
    try:
        return parent[int(child_name)]
    except (TypeError, ValueError, IndexError, KeyError):
        try:
            return parent[child_name]
        except (TypeError, IndexError, KeyError) as exc:
            raise CaptureError(f"cannot resolve child module {child_name!r}") from exc


def _set_child(parent: Any, child_name: str, module: Any) -> None:
    if hasattr(parent, child_name):
        setattr(parent, child_name, module)
        return
    try:
        parent[int(child_name)] = module
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise CaptureError(f"cannot replace child module {child_name!r}") from exc


def _parent_and_module(model: Any, module_path: str) -> tuple[Any, str, Any]:
    parts = module_path.split(".")
    current = model
    for part in parts[:-1]:
        current = _get_child(current, part)
    child_name = parts[-1]
    return current, child_name, _get_child(current, child_name)


class _RowRecorder:
    """Deterministic fixed-stride row subsampler bounded to ``max_rows``.

    Rows are kept at global stream positions ``0, stride, 2*stride, ...`` so
    the retained sample spreads across the whole calibration replay instead of
    front-loading the first batch.
    """

    def __init__(self, *, stride: int, max_rows: int) -> None:
        if stride < 1:
            raise CaptureError("capture row stride must be positive")
        self.stride = stride
        self.max_rows = max_rows
        self.seen = 0
        self.kept = 0
        self.chunks: list[Any] = []

    def record(self, rows: np.ndarray) -> None:
        import numpy as np

        count = len(rows)
        if count == 0 or self.kept >= self.max_rows:
            self.seen += count
            return
        positions = np.arange(self.seen, self.seen + count)
        keep = positions[(positions % self.stride) == 0]
        keep = keep[: self.max_rows - self.kept] - self.seen
        self.seen += count
        if keep.size:
            self.chunks.append(np.ascontiguousarray(rows[keep]))
            self.kept += int(keep.size)


def _discover_targets(
    model: Any,
    nn: Any,
    target_modules: tuple[str, ...] | list[str] | None,
) -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        discovered[str(name)] = module
    if not target_modules:
        if not discovered:
            raise CaptureError("model exposes no capturable nn.Linear modules")
        return discovered
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for target in sorted(set(target_modules)):
        aliases = mlx_module_aliases(target)
        exact = [name for name in discovered if name in aliases]
        candidates = exact or [
            name
            for name in discovered
            if any(name.endswith(f".{alias}") or alias.endswith(f".{name}") for alias in aliases)
        ]
        if len(candidates) != 1:
            unresolved.append(target)
            continue
        resolved[candidates[0]] = discovered[candidates[0]]
    if unresolved:
        raise CaptureError(
            f"target modules did not resolve to a unique capturable Linear: {unresolved[:10]}"
        )
    return resolved


def capture_calibration_activations(
    *,
    model_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    target_modules: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 2048,
    token_budget: int | None = None,
) -> ActivationCaptureManifest:
    """Capture per-module Linear input activations over a verified cache.

    Replays the tokenized calibration cache through the source model with
    recording wrappers on every eligible ``nn.Linear`` (embeddings, LM head,
    and fused MoE SwitchLinears are excluded), subsamples each module's input
    rows with a fixed stride so positions spread across the replay, and writes
    one fp16 npz per module plus the checksum-bound manifest.  ``token_budget``
    defaults to the full cache.
    """
    import numpy as np

    mlx, nn, mlx_lm = _ensure_mlx()
    if max_rows < 1:
        raise CaptureError("max_rows must be positive")
    model_path = Path(model_dir).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve()
    capture_dir = Path(output_dir).expanduser().resolve()

    manifest_hint = load_cache_manifest(cache_path)
    if manifest_hint is None:
        raise CaptureError(f"calibration cache manifest is missing or invalid: {cache_path}")
    budget = token_budget if token_budget is not None else max(manifest_hint.total_tokens, 2)
    cache_manifest, replay_batches, measured_tokens = _load_calibration_inputs(
        cache_path,
        token_budget=budget,
        replay_batch_size=1,
    )
    tokenized_manifest_sha256 = stable_sha256(
        cache_manifest.model_dump(mode="json", exclude={"created_at"})
    )
    dataset_id = _calibration_dataset_id(cache_path, cache_manifest)

    loaded = mlx_lm.load(str(model_path), lazy=False)
    model = loaded[0]
    mlx.eval(model.parameters())
    log.info("capture_model_loaded", model_dir=str(model_path))

    targets = _discover_targets(model, nn, target_modules)
    total_rows = sum(int(np.asarray(batch).size) for batch in replay_batches)
    stride = max(1, math.ceil(total_rows / max_rows))
    recorders = {name: _RowRecorder(stride=stride, max_rows=max_rows) for name in targets}

    class _RecordingLinear:
        """Plain callable wrapper; the parent module only ever calls it.

        A plain object (rather than an ``nn.Module`` subclass) keeps the
        wrapper out of parameter traversal while ``_set_child``/``__call__``
        mechanics behave exactly like the original child module.
        """

        def __init__(self, inner: Any, recorder: _RowRecorder) -> None:
            self.inner = inner
            self._recorder = recorder

        def __call__(self, x: Any) -> Any:
            rows = np.asarray(x.reshape(-1, x.shape[-1]), dtype=np.float16)
            self._recorder.record(rows)
            return self.inner(x)

    originals: dict[str, tuple[Any, str, Any]] = {}
    for name in targets:
        parent, child_name, original = _parent_and_module(model, name)
        _set_child(parent, child_name, _RecordingLinear(original, recorders[name]))
        originals[name] = (parent, child_name, original)
    try:
        for batch in replay_batches:
            tokens = mlx.array(np.asarray(batch, dtype=np.int32))
            if tokens.ndim == 1:
                tokens = tokens[None, :]
            language_model = getattr(model, "language_model", None)
            text_backbone = getattr(language_model, "model", None)
            if language_model is not None and callable(text_backbone):
                output = text_backbone(tokens)
            else:
                output = model(tokens)
            mlx.eval(output)
    finally:
        for parent, child_name, original in originals.values():
            _set_child(parent, child_name, original)

    activations_dir = capture_dir / CAPTURE_ACTIVATIONS_DIR
    activations_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ActivationCaptureEntry] = []
    for index, name in enumerate(sorted(targets)):
        recorder = recorders[name]
        in_features = int(targets[name].weight.shape[1])
        if recorder.chunks:
            x_rows = np.concatenate(recorder.chunks, axis=0).astype(np.float16, copy=False)
        else:
            x_rows = np.zeros((0, in_features), dtype=np.float16)
        filename = f"{index:04d}-{_sanitize_filename(name)}.npz"
        _write_npz_atomic(activations_dir / filename, x_rows=x_rows)
        entries.append(
            ActivationCaptureEntry(
                module_path=name,
                rows=int(x_rows.shape[0]),
                in_features=in_features,
                file=filename,
                sha256=file_sha256(activations_dir / filename),
            )
        )

    manifest = ActivationCaptureManifest(
        model=cache_manifest.model.model_id,
        revision=cache_manifest.model.revision,
        tokenized_cache_manifest_sha256=tokenized_manifest_sha256,
        cache_key_sha256=cache_manifest.cache_key_sha256,
        calibration_dataset_id=dataset_id,
        max_rows=max_rows,
        entries=tuple(entries),
    )
    write_data(capture_dir / CAPTURE_MANIFEST_NAME, manifest)
    log.info(
        "activation_capture_completed",
        output=str(capture_dir / CAPTURE_MANIFEST_NAME),
        modules=len(entries),
        rows=sum(entry.rows for entry in entries),
        max_rows=max_rows,
        stride=stride,
        measured_tokens=measured_tokens,
    )
    return manifest


def load_capture_activations(
    capture_dir: str | Path,
    *,
    model: str,
    revision: str | None = None,
) -> dict[str, np.ndarray]:
    """Load a capture artifact, failing closed on any identity or checksum drift."""
    import numpy as np

    root = Path(capture_dir).expanduser().resolve()
    manifest = load_model(root / CAPTURE_MANIFEST_NAME, ActivationCaptureManifest)
    if manifest.model != model:
        raise CaptureError(
            f"activation capture model {manifest.model!r} does not match requested {model!r}"
        )
    if revision is not None and manifest.revision != revision:
        raise CaptureError(
            f"activation capture revision {manifest.revision!r} does not match {revision!r}"
        )
    result: dict[str, np.ndarray] = {}
    for entry in manifest.entries:
        if Path(entry.file).name != entry.file:
            raise CaptureError(f"unsafe activation file name in manifest: {entry.file!r}")
        path = root / CAPTURE_ACTIVATIONS_DIR / entry.file
        if not path.is_file():
            raise CaptureError(f"activation capture file is missing: {path}")
        if file_sha256(path) != entry.sha256:
            raise CaptureError(f"activation capture file checksum mismatch: {entry.file}")
        with np.load(path, allow_pickle=False) as data:
            if "x_rows" not in data.files:
                raise CaptureError(f"activation capture file lacks x_rows: {entry.file}")
            rows = np.asarray(data["x_rows"], dtype=np.float16)
        if rows.shape != (entry.rows, entry.in_features):
            raise CaptureError(
                f"activation capture shape mismatch for {entry.module_path}: "
                f"{rows.shape} != {(entry.rows, entry.in_features)}"
            )
        result[entry.module_path] = rows
    return result
