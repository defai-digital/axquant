"""In-memory identity for a verified activation-capture artifact.

The arrays used by AWQ/GPTQ are intentionally exposed through the ordinary
``Mapping`` interface so existing refinement backends stay simple.  The
wrapper also carries the checksum-bound manifest identity, which lets probe,
plan, conversion, and release gates prove that they used the same capture.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from axquant.schema import ActivationCaptureManifest
from axquant.serde import stable_sha256

CAPTURE_MANIFEST_SHA256_KEY = "activation_capture_manifest_sha256"
CAPTURE_CACHE_MANIFEST_SHA256_KEY = "activation_capture_tokenized_cache_manifest_sha256"
CAPTURE_CACHE_KEY_SHA256_KEY = "activation_capture_cache_key_sha256"
CAPTURE_DATASET_ID_KEY = "activation_capture_dataset_id"
CAPTURE_METADATA_KEYS = (
    CAPTURE_MANIFEST_SHA256_KEY,
    CAPTURE_CACHE_MANIFEST_SHA256_KEY,
    CAPTURE_CACHE_KEY_SHA256_KEY,
    CAPTURE_DATASET_ID_KEY,
)


@dataclass(frozen=True)
class LoadedActivationCapture(Mapping[str, Any]):
    """Verified capture arrays plus their semantic artifact identity.

    ``load_capture_activations`` is the normal constructor.  Mapping
    compatibility keeps low-level consumers and existing Python callers able
    to iterate and index activations without knowing about provenance fields.
    """

    manifest: ActivationCaptureManifest
    manifest_sha256: str
    activations: Mapping[str, Any]
    source_dir: Path

    def __post_init__(self) -> None:
        if self.manifest_sha256 != stable_sha256(self.manifest):
            raise ValueError("activation capture digest does not match its manifest")
        object.__setattr__(self, "activations", MappingProxyType(dict(self.activations)))

    def __getitem__(self, key: str) -> Any:
        return self.activations[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.activations)

    def __len__(self) -> int:
        return len(self.activations)


def activation_capture_metadata(capture: LoadedActivationCapture) -> dict[str, str]:
    """Return the complete provenance binding stored in calibration evidence."""
    manifest = capture.manifest
    return {
        CAPTURE_MANIFEST_SHA256_KEY: capture.manifest_sha256,
        CAPTURE_CACHE_MANIFEST_SHA256_KEY: manifest.tokenized_cache_manifest_sha256,
        CAPTURE_CACHE_KEY_SHA256_KEY: manifest.cache_key_sha256,
        CAPTURE_DATASET_ID_KEY: manifest.calibration_dataset_id,
    }


def activation_capture_evidence_issues(
    manifest: ActivationCaptureManifest,
    metadata: Mapping[str, str | int | float | bool],
    *,
    model_id: str,
    revision: str | None,
    dataset_id: str,
) -> list[str]:
    """Validate a manifest against the capture binding in calibration evidence."""
    issues: list[str] = []
    expected = {
        CAPTURE_MANIFEST_SHA256_KEY: stable_sha256(manifest),
        CAPTURE_CACHE_MANIFEST_SHA256_KEY: manifest.tokenized_cache_manifest_sha256,
        CAPTURE_CACHE_KEY_SHA256_KEY: manifest.cache_key_sha256,
        CAPTURE_DATASET_ID_KEY: manifest.calibration_dataset_id,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            issues.append(f"calibration evidence {key} does not match the activation capture")
    if manifest.model != model_id or manifest.revision != revision:
        issues.append("activation capture source model does not match the measured plan")
    if manifest.calibration_dataset_id != dataset_id:
        issues.append("activation capture dataset does not match calibration evidence")
    return issues
