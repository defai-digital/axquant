"""Calibration activation capture for weight-refinement backends.

AWQ/GPTQ-style refinement needs per-module INPUT activations collected from a
forward replay of a verified tokenized calibration cache.  MLX has no forward
hooks, so capture swaps each target ``nn.Linear`` with a recording wrapper
installed on its parent module (the same child-replacement mechanics as the
probe backend), replays the calibration batches, and writes checksum-bound
npz activations plus an ``ActivationCaptureManifest``.

The replay is segmented: after each segment the newly captured rows are
flushed to per-segment chunk files under ``activations/.partial/`` and a
``capture_progress.json`` checkpoint is rewritten, so an interrupted capture
resumes from the last completed segment instead of losing everything.  On
full completion the chunks are concatenated into the final per-module (or
sharded) npz files, the partial state is removed, and a ``completion.json``
marker is written; ``load_capture_activations`` refuses partial captures.

MLX is a lazy optional dependency imported only when capture runs; loading a
capture artifact back (``load_capture_activations``) does not require MLX.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import ValidationError

from axquant.activation_cache import _write_npz_atomic, load_cache_manifest
from axquant.capture_binding import LoadedActivationCapture
from axquant.errors import ArtifactError, BackendUnavailableError, CaptureError
from axquant.module_paths import mlx_module_aliases
from axquant.probe import _calibration_dataset_id, _load_calibration_inputs
from axquant.schema import (
    ActivationCaptureEntry,
    ActivationCaptureManifest,
    CaptureProgress,
    CaptureProgressModule,
    SourceConversionProvenance,
    TokenizedCacheManifest,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

if TYPE_CHECKING:
    import numpy as np

log = structlog.get_logger()

CAPTURE_MANIFEST_NAME = "activation_capture_manifest.json"
CAPTURE_ACTIVATIONS_DIR = "activations"
CAPTURE_PROGRESS_NAME = "capture_progress.json"
CAPTURE_COMPLETION_MARKER = "completion.json"
CAPTURE_COMPLETION_SCHEMA = "axquant.activation-capture-completion.v1"
_PARTIAL_DIRNAME = ".partial"
_LEGACY_ARRAY_KEY = "x_rows"
_SEGMENT_CHUNK = re.compile(r"^(?P<index>\d{4})-.+\.seg-(?P<segment>\d{4})\.npz$")


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


def _shard_array_key(module_path: str) -> str:
    return f"rows::{module_path}"


def _segment_chunk_path(
    partial_dir: Path,
    module_index: int,
    module_path: str,
    segment_index: int,
) -> Path:
    filename = f"{module_index:04d}-{_sanitize_filename(module_path)}.seg-{segment_index:04d}.npz"
    return partial_dir / filename


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


def _activation_rows_f16(mlx: Any, x: Any) -> Any:
    """Materialize a module input as an fp16 numpy row matrix.

    The cast happens inside MLX first: numpy's PEP 3118 buffer import rejects
    bfloat16 mlx arrays ("Item size 2 ... does not match ... item size 1"),
    which is the native activation dtype of BF16 checkpoints.
    """
    import numpy as np

    rows = x.astype(mlx.float16).reshape(-1, x.shape[-1])
    mlx.eval(rows)
    return np.asarray(rows)


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


def _replay_segment(model: Any, mlx: Any, batches: list[Any]) -> None:
    """Replay one segment of calibration batches through the wrapped model."""
    import numpy as np

    for batch in batches:
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


def _flush_segment(
    partial_dir: Path,
    sorted_names: list[str],
    recorders: dict[str, _RowRecorder],
    segment_index: int,
) -> None:
    """Flush each recorder's newly accumulated rows to a per-segment chunk file."""
    import numpy as np

    for index, name in enumerate(sorted_names):
        recorder = recorders[name]
        destination = _segment_chunk_path(partial_dir, index, name, segment_index)
        if not recorder.chunks:
            # A crash can leave an uncommitted chunk for this segment before
            # the progress checkpoint advances.  Replaying a segment that
            # now retains no rows must remove that stale file.
            destination.unlink(missing_ok=True)
            continue
        rows = np.concatenate(recorder.chunks, axis=0).astype(np.float16, copy=False)
        recorder.chunks = []
        _write_npz_atomic(destination, x_rows=rows)


def _write_completion_marker(
    capture_dir: Path,
    *,
    cache_key_sha256: str,
    manifest_sha256: str,
    modules: int,
    rows: int,
) -> None:
    """Write the capture completion marker (mirrors the calibration cache convention)."""
    marker_data = {
        "schema_version": CAPTURE_COMPLETION_SCHEMA,
        "complete": True,
        "cache_key_sha256": cache_key_sha256,
        "manifest_sha256": manifest_sha256,
        "modules": modules,
        "rows": rows,
    }
    write_data(capture_dir / CAPTURE_COMPLETION_MARKER, marker_data)


def _verify_completion_marker(
    capture_dir: Path,
    manifest: ActivationCaptureManifest,
) -> None:
    """Verify the marker written last during capture finalization.

    The pre-v1.1.1 development marker only carried ``complete``.  It remains
    readable for migration, while every field present is validated and new
    markers must bind the manifest semantic digest.
    """
    marker_path = capture_dir / CAPTURE_COMPLETION_MARKER
    if not marker_path.is_file():
        raise CaptureError(
            f"activation capture is incomplete (no completion marker): {capture_dir}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"activation capture completion marker is unreadable: {exc}") from exc
    if not isinstance(marker, dict) or marker.get("complete") is not True:
        raise CaptureError("activation capture completion marker is invalid")
    schema_version = marker.get("schema_version")
    if schema_version not in {None, CAPTURE_COMPLETION_SCHEMA}:
        raise CaptureError(f"unsupported activation capture completion schema: {schema_version!r}")
    expected_manifest_sha256 = stable_sha256(manifest)
    marker_manifest_sha256 = marker.get("manifest_sha256")
    if schema_version == CAPTURE_COMPLETION_SCHEMA and not isinstance(marker_manifest_sha256, str):
        raise CaptureError("activation capture completion marker lacks manifest checksum")
    if marker_manifest_sha256 is not None and marker_manifest_sha256 != expected_manifest_sha256:
        raise CaptureError("activation capture completion marker manifest checksum mismatch")
    expected_fields = {
        "cache_key_sha256": manifest.cache_key_sha256,
        "modules": len(manifest.entries),
        "rows": sum(entry.rows for entry in manifest.entries),
    }
    for name, expected in expected_fields.items():
        if name in marker and marker[name] != expected:
            raise CaptureError(f"activation capture completion marker {name} mismatch")


def _load_resume_progress(
    progress_path: Path,
    binding: dict[str, Any],
) -> CaptureProgress | None:
    """Load and bind-check an existing progress checkpoint, or None if absent."""
    if not progress_path.is_file():
        return None
    try:
        progress = load_model(progress_path, CaptureProgress)
    except (ArtifactError, ValidationError, ValueError) as exc:
        raise CaptureError(
            f"capture progress checkpoint is unreadable: {exc}; use a fresh output directory"
        ) from exc
    resumed_binding = progress.model_dump(mode="json", include=set(binding))
    if resumed_binding != binding:
        raise CaptureError(
            "existing capture progress does not match this invocation "
            "(model, revision, cache, max_rows, stride, token budget, segment_batches, "
            "or target modules differ); use a fresh output directory"
        )
    return progress


def _verify_source_provenance(
    model_path: Path,
    cache_manifest: TokenizedCacheManifest,
) -> None:
    """Bind a prepared BF16 checkpoint to the cache's immutable model identity."""
    provenance_path = model_path / "axquant_source.json"
    if not provenance_path.is_file():
        return
    try:
        provenance = load_model(provenance_path, SourceConversionProvenance)
    except (ArtifactError, ValidationError, ValueError) as exc:
        raise CaptureError(f"BF16 source provenance is unreadable: {exc}") from exc
    if (
        provenance.source_model != cache_manifest.model.model_id
        or provenance.source_revision != cache_manifest.model.revision
    ):
        raise CaptureError(
            "BF16 source provenance does not match the tokenized calibration cache model"
        )


def _validate_resume_chunks(
    partial_dir: Path,
    sorted_names: list[str],
    targets: dict[str, Any],
    progress: CaptureProgress,
    *,
    stride: int,
    max_rows: int,
) -> None:
    """Verify committed resume chunks and discard only uncommitted replay output."""
    import numpy as np

    if set(progress.modules) != set(sorted_names):
        raise CaptureError(
            "capture progress module inventory is incomplete; use a fresh output directory"
        )
    if progress.segments_completed and not partial_dir.is_dir():
        raise CaptureError(
            "capture progress has no partial activation directory; use a fresh output directory"
        )

    for path in sorted(partial_dir.glob("*.npz")) if partial_dir.is_dir() else ():
        match = _SEGMENT_CHUNK.fullmatch(path.name)
        if match is None:
            raise CaptureError(
                f"unexpected activation capture resume chunk {path.name!r}; "
                "use a fresh output directory"
            )
        module_index = int(match.group("index"))
        segment_index = int(match.group("segment"))
        if module_index >= len(sorted_names) or path != _segment_chunk_path(
            partial_dir,
            module_index,
            sorted_names[module_index],
            segment_index,
        ):
            raise CaptureError(
                f"activation capture resume chunk does not match its module: {path.name}; "
                "use a fresh output directory"
            )
        if segment_index >= progress.segments_completed:
            # Flush happens before checkpoint.  A chunk at or beyond the
            # checkpoint is uncommitted and will be deterministically replayed.
            path.unlink()

    for module_index, name in enumerate(sorted_names):
        state = progress.modules[name]
        expected_kept = min(max_rows, (state.seen + stride - 1) // stride)
        if state.kept != expected_kept:
            raise CaptureError(
                f"capture progress row accounting is invalid for {name}; "
                "use a fresh output directory"
            )
        in_features = int(targets[name].weight.shape[1])
        committed_rows = 0
        for segment_index in range(progress.segments_completed):
            path = _segment_chunk_path(partial_dir, module_index, name, segment_index)
            if not path.is_file():
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    if _LEGACY_ARRAY_KEY not in data.files:
                        raise CaptureError(f"capture resume chunk lacks rows: {path.name}")
                    rows = np.asarray(data[_LEGACY_ARRAY_KEY])
            except (OSError, ValueError) as exc:
                raise CaptureError(
                    f"capture resume chunk is unreadable: {path.name}: {exc}"
                ) from exc
            if (
                rows.dtype != np.float16
                or rows.ndim != 2
                or rows.shape[1] != in_features
                or rows.shape[0] == 0
            ):
                raise CaptureError(f"capture resume chunk shape is invalid: {path.name}")
            committed_rows += int(rows.shape[0])
        if committed_rows != state.kept:
            raise CaptureError(
                f"capture resume chunks contain {committed_rows} rows for {name}, "
                f"but progress records {state.kept}; use a fresh output directory"
            )


def capture_calibration_activations(
    *,
    model_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    target_modules: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 2048,
    token_budget: int | None = None,
    segment_batches: int = 8,
    modules_per_shard: int = 1,
) -> ActivationCaptureManifest:
    """Capture per-module Linear input activations over a verified cache.

    Replays the tokenized calibration cache through the source model with
    recording wrappers on every eligible ``nn.Linear`` (embeddings, LM head,
    and fused MoE SwitchLinears are excluded), subsamples each module's input
    rows with a fixed stride so positions spread across the replay, and writes
    fp16 npz activations plus the checksum-bound manifest.  ``token_budget``
    defaults to the full cache.

    The replay runs in segments of ``segment_batches`` batches with a
    checkpoint after each; rerunning with identical inputs resumes from the
    last completed segment and produces a byte-identical artifact.  With
    ``modules_per_shard`` above one, modules are grouped into shared
    ``shard-NNNN.npz`` archives keyed ``rows::<module_path>`` instead of one
    npz per module.
    """
    import numpy as np

    if max_rows < 1:
        raise CaptureError("max_rows must be positive")
    if segment_batches < 1:
        raise CaptureError("segment_batches must be positive")
    if modules_per_shard < 1:
        raise CaptureError("modules_per_shard must be positive")
    model_path = Path(model_dir).expanduser().resolve()
    cache_path = Path(cache_dir).expanduser().resolve()
    capture_dir = Path(output_dir).expanduser().resolve()
    if (capture_dir / CAPTURE_COMPLETION_MARKER).exists():
        raise CaptureError(
            f"activation capture output is already complete: {capture_dir}; "
            "use a fresh output directory"
        )

    manifest_hint = load_cache_manifest(cache_path)
    if manifest_hint is None:
        raise CaptureError(f"calibration cache manifest is missing or invalid: {cache_path}")
    _verify_source_provenance(model_path, manifest_hint)
    mlx, nn, mlx_lm = _ensure_mlx()
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
    sorted_names = sorted(targets)
    total_rows = sum(int(np.asarray(batch).size) for batch in replay_batches)
    stride = max(1, math.ceil(total_rows / max_rows))
    segments = [
        replay_batches[start : start + segment_batches]
        for start in range(0, len(replay_batches), segment_batches)
    ]

    activations_dir = capture_dir / CAPTURE_ACTIVATIONS_DIR
    partial_dir = activations_dir / _PARTIAL_DIRNAME
    progress_path = capture_dir / CAPTURE_PROGRESS_NAME
    binding: dict[str, Any] = {
        "model": cache_manifest.model.model_id,
        "revision": cache_manifest.model.revision,
        "cache_key_sha256": cache_manifest.cache_key_sha256,
        "max_rows": max_rows,
        "stride": stride,
        "token_budget": budget,
        "segment_batches": segment_batches,
        "target_modules": sorted_names,
    }

    segments_completed = 0
    resumed_modules: dict[str, CaptureProgressModule] = {}
    progress = _load_resume_progress(progress_path, binding)
    if progress is not None:
        segments_completed = progress.segments_completed
        if segments_completed > len(segments):
            raise CaptureError(
                f"capture progress claims {segments_completed} completed segments but only "
                f"{len(segments)} exist; use a fresh output directory"
            )
        resumed_modules = dict(progress.modules)
        _validate_resume_chunks(
            partial_dir,
            sorted_names,
            targets,
            progress,
            stride=stride,
            max_rows=max_rows,
        )
        log.info(
            "capture_resume_started",
            output=str(capture_dir),
            segments_completed=segments_completed,
        )
    elif partial_dir.is_dir() and any(partial_dir.iterdir()):
        raise CaptureError(
            "activation capture has partial chunks without a progress checkpoint; "
            "use a fresh output directory"
        )

    recorders: dict[str, _RowRecorder] = {}
    for name in sorted_names:
        recorder = _RowRecorder(stride=stride, max_rows=max_rows)
        state = resumed_modules.get(name)
        if state is not None:
            recorder.seen = state.seen
            recorder.kept = state.kept
        recorders[name] = recorder

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
            self._recorder.record(_activation_rows_f16(mlx, x))
            return self.inner(x)

    originals: dict[str, tuple[Any, str, Any]] = {}
    for name in sorted_names:
        parent, child_name, original = _parent_and_module(model, name)
        _set_child(parent, child_name, _RecordingLinear(original, recorders[name]))
        originals[name] = (parent, child_name, original)

    try:
        for segment_index in range(segments_completed, len(segments)):
            _replay_segment(model, mlx, segments[segment_index])
            partial_dir.mkdir(parents=True, exist_ok=True)
            _flush_segment(partial_dir, sorted_names, recorders, segment_index)
            checkpoint = CaptureProgress(
                **binding,
                segments_completed=segment_index + 1,
                modules={
                    name: CaptureProgressModule(
                        seen=recorders[name].seen,
                        kept=recorders[name].kept,
                    )
                    for name in sorted_names
                },
            )
            write_data(progress_path, checkpoint)
            log.info(
                "capture_segment_completed",
                segment=segment_index + 1,
                segments=len(segments),
            )
    finally:
        for parent, child_name, original in originals.values():
            _set_child(parent, child_name, original)

    activations_dir.mkdir(parents=True, exist_ok=True)
    module_rows: dict[str, np.ndarray] = {}
    module_in_features: dict[str, int] = {}
    for index, name in enumerate(sorted_names):
        in_features = int(targets[name].weight.shape[1])
        module_in_features[name] = in_features
        arrays: list[np.ndarray] = []
        for segment_index in range(len(segments)):
            segment_file = _segment_chunk_path(partial_dir, index, name, segment_index)
            if not segment_file.is_file():
                continue
            try:
                with np.load(segment_file, allow_pickle=False) as data:
                    if _LEGACY_ARRAY_KEY not in data.files:
                        raise CaptureError(
                            f"activation capture chunk lacks rows: {segment_file.name}"
                        )
                    rows = np.asarray(data[_LEGACY_ARRAY_KEY])
            except (OSError, ValueError) as exc:
                raise CaptureError(
                    f"activation capture chunk is unreadable: {segment_file.name}: {exc}"
                ) from exc
            if (
                rows.dtype != np.float16
                or rows.ndim != 2
                or rows.shape[1] != in_features
                or rows.shape[0] == 0
            ):
                raise CaptureError(
                    f"activation capture chunk shape is invalid: {segment_file.name}"
                )
            arrays.append(rows)
        if arrays:
            module_rows[name] = np.concatenate(arrays, axis=0)
        else:
            module_rows[name] = np.zeros((0, in_features), dtype=np.float16)
        if len(module_rows[name]) != recorders[name].kept:
            raise CaptureError(
                f"activation capture chunks contain {len(module_rows[name])} rows for {name}, "
                f"but replay recorded {recorders[name].kept}"
            )

    entries: list[ActivationCaptureEntry] = []
    if modules_per_shard == 1:
        for index, name in enumerate(sorted_names):
            x_rows = module_rows[name]
            filename = f"{index:04d}-{_sanitize_filename(name)}.npz"
            _write_npz_atomic(activations_dir / filename, compressed=True, x_rows=x_rows)
            entries.append(
                ActivationCaptureEntry(
                    module_path=name,
                    rows=int(x_rows.shape[0]),
                    in_features=module_in_features[name],
                    file=filename,
                    sha256=file_sha256(activations_dir / filename),
                )
            )
    else:
        groups = [
            sorted_names[start : start + modules_per_shard]
            for start in range(0, len(sorted_names), modules_per_shard)
        ]
        for shard_index, group in enumerate(groups):
            filename = f"shard-{shard_index:04d}.npz"
            _write_npz_atomic(
                activations_dir / filename,
                compressed=True,
                **{_shard_array_key(name): module_rows[name] for name in group},
            )
            shard_sha256 = file_sha256(activations_dir / filename)
            for name in group:
                x_rows = module_rows[name]
                entries.append(
                    ActivationCaptureEntry(
                        module_path=name,
                        rows=int(x_rows.shape[0]),
                        in_features=module_in_features[name],
                        file=filename,
                        sha256=shard_sha256,
                        array_key=_shard_array_key(name),
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
    shutil.rmtree(partial_dir, ignore_errors=True)
    progress_path.unlink(missing_ok=True)
    _write_completion_marker(
        capture_dir,
        cache_key_sha256=cache_manifest.cache_key_sha256,
        manifest_sha256=stable_sha256(manifest),
        modules=len(entries),
        rows=sum(entry.rows for entry in entries),
    )
    log.info(
        "activation_capture_completed",
        output=str(capture_dir / CAPTURE_MANIFEST_NAME),
        modules=len(entries),
        rows=sum(entry.rows for entry in entries),
        max_rows=max_rows,
        stride=stride,
        measured_tokens=measured_tokens,
        segments=len(segments),
        modules_per_shard=modules_per_shard,
    )
    return manifest


def load_capture_activations(
    capture_dir: str | Path,
    *,
    model: str,
    revision: str | None = None,
) -> LoadedActivationCapture:
    """Load a capture artifact, failing closed on any identity or checksum drift."""
    import numpy as np

    root = Path(capture_dir).expanduser().resolve()
    if not (root / CAPTURE_COMPLETION_MARKER).is_file():
        raise CaptureError(f"activation capture is incomplete (no completion marker): {root}")
    manifest = load_model(root / CAPTURE_MANIFEST_NAME, ActivationCaptureManifest)
    _verify_completion_marker(root, manifest)
    if manifest.model != model:
        raise CaptureError(
            f"activation capture model {manifest.model!r} does not match requested {model!r}"
        )
    if revision is not None and manifest.revision != revision:
        raise CaptureError(
            f"activation capture revision {manifest.revision!r} does not match {revision!r}"
        )
    verified_files: dict[str, tuple[Path, str]] = {}
    result: dict[str, np.ndarray] = {}
    for entry in manifest.entries:
        if Path(entry.file).name != entry.file:
            raise CaptureError(f"unsafe activation file name in manifest: {entry.file!r}")
        path = root / CAPTURE_ACTIVATIONS_DIR / entry.file
        if not path.is_file():
            raise CaptureError(f"activation capture file is missing: {path}")
        if entry.file not in verified_files:
            actual_sha256 = file_sha256(path)
            if actual_sha256 != entry.sha256:
                raise CaptureError(f"activation capture file checksum mismatch: {entry.file}")
            verified_files[entry.file] = (path, actual_sha256)
        elif verified_files[entry.file][1] != entry.sha256:
            raise CaptureError(
                f"activation capture manifest has inconsistent checksums: {entry.file}"
            )
        array_key = entry.array_key or _LEGACY_ARRAY_KEY
        if entry.array_key is not None and entry.array_key != _shard_array_key(entry.module_path):
            raise CaptureError(
                f"activation capture array key mismatch for {entry.module_path}: "
                f"{entry.array_key!r}"
            )
        with np.load(path, allow_pickle=False) as data:
            if array_key not in data.files:
                raise CaptureError(f"activation capture file lacks {array_key}: {entry.file}")
            rows = np.asarray(data[array_key], dtype=np.float16)
        if rows.shape != (entry.rows, entry.in_features):
            raise CaptureError(
                f"activation capture shape mismatch for {entry.module_path}: "
                f"{rows.shape} != {(entry.rows, entry.in_features)}"
            )
        rows.setflags(write=False)
        result[entry.module_path] = rows
    return LoadedActivationCapture(
        manifest=manifest,
        manifest_sha256=stable_sha256(manifest),
        activations=result,
        source_dir=root,
    )
