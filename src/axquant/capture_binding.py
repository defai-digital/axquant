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

_TRUSTED_CAPTURE_CONSTRUCTION = object()

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


@dataclass(frozen=True, init=False)
class LoadedActivationCapture(Mapping[str, Any]):
    """Verified capture arrays plus their semantic artifact identity.

    Instances are capabilities proving that ``load_capture_activations``
    validated an on-disk artifact. Direct construction is intentionally
    rejected so an arbitrary mapping cannot satisfy the probe/conversion
    ``isinstance`` gate. Mapping compatibility keeps low-level consumers able
    to iterate and index activations without knowing about provenance fields.
    """

    _manifest: ActivationCaptureManifest
    manifest_sha256: str
    _activations: Mapping[str, Any]
    source_dir: Path

    def __init__(
        self,
        *,
        manifest: ActivationCaptureManifest,
        manifest_sha256: str,
        activations: Mapping[str, Any],
        source_dir: Path,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _TRUSTED_CAPTURE_CONSTRUCTION:
            raise TypeError(
                "LoadedActivationCapture cannot be constructed directly; "
                "use load_capture_activations"
            )
        trusted_manifest = manifest.model_copy(deep=True)
        if manifest_sha256 != stable_sha256(trusted_manifest):
            raise ValueError("activation capture digest does not match its manifest")
        object.__setattr__(self, "_manifest", trusted_manifest)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "_activations", MappingProxyType(dict(activations)))
        object.__setattr__(self, "source_dir", source_dir)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del kwargs
        raise TypeError("LoadedActivationCapture cannot be subclassed")

    @property
    def manifest(self) -> ActivationCaptureManifest:
        """Return a defensive copy of the checksum-verified manifest."""

        return self._manifest.model_copy(deep=True)

    @property
    def activations(self) -> Mapping[str, Any]:
        """Return the immutable activation lookup."""

        return self._activations

    def __getitem__(self, key: str) -> Any:
        return self._activations[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._activations)

    def __len__(self) -> int:
        return len(self._activations)


def _loaded_activation_capture_from_validated_disk(
    *,
    manifest: ActivationCaptureManifest,
    manifest_sha256: str,
    activations: Mapping[str, Any],
    source_dir: Path,
) -> LoadedActivationCapture:
    """Create the trusted wrapper after ``capture.py`` validates every file."""

    return LoadedActivationCapture(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        activations=activations,
        source_dir=source_dir,
        _construction_token=_TRUSTED_CAPTURE_CONSTRUCTION,
    )


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
