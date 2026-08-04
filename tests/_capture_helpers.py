"""Test-only helpers for materializing trusted activation-capture wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from axquant.capture import (
    CAPTURE_ACTIVATIONS_DIR,
    CAPTURE_COMPLETION_MARKER,
    CAPTURE_COMPLETION_SCHEMA,
    CAPTURE_MANIFEST_NAME,
    load_capture_activations,
)
from axquant.capture_binding import LoadedActivationCapture
from axquant.schema import ActivationCaptureEntry, ActivationCaptureManifest
from axquant.serde import file_sha256, stable_sha256, write_data


def load_test_activation_capture(
    capture_dir: Path,
    *,
    manifest: ActivationCaptureManifest,
    activations: Mapping[str, np.ndarray],
) -> LoadedActivationCapture:
    """Write a complete checksum-bound fixture and load it through the real trust boundary."""

    activation_dir = capture_dir / CAPTURE_ACTIVATIONS_DIR
    activation_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ActivationCaptureEntry] = []
    for index, (module_path, values) in enumerate(sorted(activations.items())):
        rows = np.asarray(values, dtype=np.float16)
        if rows.ndim != 2:
            raise ValueError(f"test activation rows must be rank 2: {module_path}")
        filename = f"{index:04d}.npz"
        path = activation_dir / filename
        np.savez_compressed(path, x_rows=rows)
        entries.append(
            ActivationCaptureEntry(
                module_path=module_path,
                rows=int(rows.shape[0]),
                in_features=int(rows.shape[1]),
                file=filename,
                sha256=file_sha256(path),
            )
        )
    max_rows = max([manifest.max_rows, *(entry.rows for entry in entries)])
    bound_manifest = ActivationCaptureManifest.model_validate(
        {
            **manifest.model_dump(mode="json"),
            "entries": [entry.model_dump(mode="json") for entry in entries],
            "max_rows": max_rows,
        }
    )
    write_data(capture_dir / CAPTURE_MANIFEST_NAME, bound_manifest)
    write_data(
        capture_dir / CAPTURE_COMPLETION_MARKER,
        {
            "schema_version": CAPTURE_COMPLETION_SCHEMA,
            "complete": True,
            "cache_key_sha256": bound_manifest.cache_key_sha256,
            "manifest_sha256": stable_sha256(bound_manifest),
            "modules": len(bound_manifest.entries),
            "rows": sum(entry.rows for entry in bound_manifest.entries),
        },
    )
    return load_capture_activations(
        capture_dir,
        model=bound_manifest.model,
        revision=bound_manifest.revision,
    )
