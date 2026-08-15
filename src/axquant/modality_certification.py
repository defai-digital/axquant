"""Capability-gated vision/audio certification policy (AXQuant 1.8.0).

Best practice:

* If a modality is **not supported** by the pack (no architecture declaration and
  no protected sidecar weights), certification **disables** it:
  ``status=not-applicable``, ``supported=false``. No smoke or quality suite runs.
* If a modality **is supported**, certification must either:
  1. run a bound **smoke** (load + generation/transcription) → ``smoke-certified``;
  2. run a bound **quality** suite with thresholds → ``quality-certified``; or
  3. explicitly leave the modality **present but uncertified**
     (``present-not-certified``) when weights are only protected/preserved and no
     multimodal quality claim is authorized.

Public language must not treat ``Vision present=True`` on a model card as a
quality pass. Text dual-suite Tier 1 never implies vision-tower or audio quality
(see Certification Specification §8).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from axquant.schema.public_certification import (
    ModalityClaimStatus,
    PublicModalitiesBlock,
    PublicModalityClaim,
)

ModalityName = Literal["vision", "audio"]

_SMOKE_EVIDENCE = {
    "vision": "runtime-smoke-mlx-vlm",
    "audio": "runtime-smoke-mlx-audio",
}
_QUALITY_EVIDENCE = {
    "vision": "multimodal-quality-vision",
    "audio": "multimodal-quality-audio",
}


def derive_modality_claim(
    *,
    supported: bool,
    smoke_passed: bool | None = None,
    quality_passed: bool | None = None,
    reason: str | None = None,
    runtime: str | None = None,
    modality: ModalityName = "vision",
) -> PublicModalityClaim:
    """Derive one modality claim from support flags and optional evidence.

    Precedence when supported: quality > smoke > present-not-certified.
    Unsupported always maps to not-applicable regardless of evidence flags.
    """

    if not supported:
        return PublicModalityClaim(
            status="not-applicable",
            supported=False,
            reason=reason or f"{modality} not supported on this pack",
            runtime=None,
            evidence_kind=None,
        )
    if quality_passed is True:
        return PublicModalityClaim(
            status="quality-certified",
            supported=True,
            reason=reason or f"{modality} quality suite passed thresholds",
            runtime=runtime,
            evidence_kind=_QUALITY_EVIDENCE[modality],
        )
    if smoke_passed is True:
        return PublicModalityClaim(
            status="smoke-certified",
            supported=True,
            reason=reason or f"{modality} runtime smoke passed",
            runtime=runtime or ("mlx-vlm" if modality == "vision" else "mlx-audio"),
            evidence_kind=_SMOKE_EVIDENCE[modality],
        )
    return PublicModalityClaim(
        status="present-not-certified",
        supported=True,
        reason=reason
        or (
            f"{modality} weights present or protected; no multimodal "
            f"{'quality or smoke' if smoke_passed is None else 'quality'} claim"
        ),
        runtime=runtime,
        evidence_kind=None,
    )


def build_modalities_block(
    *,
    vision_supported: bool,
    audio_supported: bool,
    vision_smoke_passed: bool | None = None,
    audio_smoke_passed: bool | None = None,
    vision_quality_passed: bool | None = None,
    audio_quality_passed: bool | None = None,
    vision_reason: str | None = None,
    audio_reason: str | None = None,
    vision_runtime: str | None = None,
    audio_runtime: str | None = None,
) -> PublicModalitiesBlock:
    """Build the Tier 1 ``modalities`` envelope from capability + evidence."""

    return PublicModalitiesBlock(
        policy="capability-gated-v1",
        vision=derive_modality_claim(
            supported=vision_supported,
            smoke_passed=vision_smoke_passed,
            quality_passed=vision_quality_passed,
            reason=vision_reason,
            runtime=vision_runtime,
            modality="vision",
        ),
        audio=derive_modality_claim(
            supported=audio_supported,
            smoke_passed=audio_smoke_passed,
            quality_passed=audio_quality_passed,
            reason=audio_reason,
            runtime=audio_runtime,
            modality="audio",
        ),
    )


def claim_allows_public_quality(status: ModalityClaimStatus) -> bool:
    """Whether public language may claim multimodal quality for this status."""

    return status == "quality-certified"


def claim_allows_public_smoke(status: ModalityClaimStatus) -> bool:
    """Whether public language may claim multimodal runtime smoke."""

    return status in {"smoke-certified", "quality-certified"}


def validate_modality_evidence_consistency(
    block: PublicModalitiesBlock,
    *,
    vision_supported: bool,
    audio_supported: bool,
) -> list[str]:
    """Return human-readable issues; empty means consistent."""

    issues: list[str] = []
    if block.vision.supported != vision_supported:
        issues.append(
            "vision.supported "
            f"({block.vision.supported}) disagrees with capability flag "
            f"({vision_supported})"
        )
    if block.audio.supported != audio_supported:
        issues.append(
            "audio.supported "
            f"({block.audio.supported}) disagrees with capability flag "
            f"({audio_supported})"
        )
    if vision_supported and block.vision.status == "not-applicable":
        issues.append("vision is supported but claim status is not-applicable")
    if audio_supported and block.audio.status == "not-applicable":
        issues.append("audio is supported but claim status is not-applicable")
    if not vision_supported and block.vision.status != "not-applicable":
        issues.append("vision is unsupported but claim status is not not-applicable")
    if not audio_supported and block.audio.status != "not-applicable":
        issues.append("audio is unsupported but claim status is not not-applicable")
    return issues


def modalities_to_public_dict(block: PublicModalitiesBlock) -> dict[str, Any]:
    """Stable JSON-ready dict for certificate writers."""

    return block.model_dump(mode="json")


def summarize_modalities_for_markdown(block: PublicModalitiesBlock | None) -> str:
    """One-line human summary for cert markdown / cards."""

    if block is None:
        return "Modalities: legacy record (capability-gated block not stated)."
    return (
        f"Vision: `{block.vision.status}`"
        f"{'' if block.vision.supported else ' (disabled)'}; "
        f"Audio: `{block.audio.status}`"
        f"{'' if block.audio.supported else ' (disabled)'}."
    )


_VISION_WEIGHT_NAMES = frozenset(
    {
        "vision.safetensors",
        "vision_tower.safetensors",
    }
)
_AUDIO_WEIGHT_NAMES = frozenset(
    {
        "audio.safetensors",
        "audio_tower.safetensors",
    }
)
_AUDIO_CONFIG_KEYS = ("audio_config", "audio_encoder_config")


@dataclass(frozen=True, slots=True)
class ArtifactModalityInspect:
    """Per-pack inspect of whether vision/audio are actually shipped."""

    vision_declared: bool
    audio_declared: bool
    vision_weight_files: tuple[str, ...]
    audio_weight_files: tuple[str, ...]
    vision_key_prefixes: tuple[str, ...] = ()
    source: str = "local-dir"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def vision_supported(self) -> bool:
        return self.vision_declared or bool(self.vision_weight_files)

    @property
    def audio_supported(self) -> bool:
        return self.audio_declared or bool(self.audio_weight_files)


def _basename(entry: str) -> str:
    return entry.replace("\\", "/").rsplit("/", 1)[-1]


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _declared_from_config(config: dict[str, Any]) -> tuple[bool, bool]:
    vision_declared = isinstance(config.get("vision_config"), dict)
    audio_declared = any(isinstance(config.get(key), dict) for key in _AUDIO_CONFIG_KEYS)
    return vision_declared, audio_declared


def _weight_files_from_names(names: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    basenames = [_basename(name) for name in names]
    vision = tuple(sorted({name for name in basenames if name in _VISION_WEIGHT_NAMES}))
    audio = tuple(sorted({name for name in basenames if name in _AUDIO_WEIGHT_NAMES}))
    if "axquant_vision_sidecar_manifest.json" in basenames and not vision:
        vision = ("axquant_vision_sidecar_manifest.json",)
    if "axquant_audio_sidecar_manifest.json" in basenames and not audio:
        audio = ("axquant_audio_sidecar_manifest.json",)
    return vision, audio


def _sidecar_key_prefixes(path: Path, *, limit: int = 24) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    try:
        from safetensors import SafetensorError, safe_open
    except ImportError:
        return ()
    try:
        with safe_open(str(path), framework="np") as handle:
            keys = list(handle.keys())
    except (OSError, ValueError, RuntimeError, SafetensorError):
        return ()
    prefixes: set[str] = set()
    for key in keys[:limit]:
        parts = key.split(".")
        prefixes.add(".".join(parts[:2]) if len(parts) >= 2 else key)
    return tuple(sorted(prefixes))


def inspect_hub_listing(
    *,
    filenames: Iterable[str],
    config: dict[str, Any] | None = None,
    source: str = "hub-listing",
) -> ArtifactModalityInspect:
    """Inspect a published pack from its file list and optional ``config.json``."""

    names = tuple(filenames)
    vision_declared = audio_declared = False
    notes: list[str] = []
    if config is not None:
        vision_declared, audio_declared = _declared_from_config(config)
    elif "config.json" not in {_basename(name) for name in names}:
        notes.append("config.json not provided; support inferred from filenames only")
    vision_files, audio_files = _weight_files_from_names(names)
    return ArtifactModalityInspect(
        vision_declared=vision_declared,
        audio_declared=audio_declared,
        vision_weight_files=vision_files,
        audio_weight_files=audio_files,
        source=source,
        notes=tuple(notes),
    )


def inspect_artifact_modalities(model_dir: str | Path) -> ArtifactModalityInspect:
    """Inspect a local artifact directory for vision/audio capability.

    A modality is supported when the pack declares a tower config **or**
    ships sidecar weights / a sidecar manifest. A lone ``audio_token_id``
    (or similar placeholder token) is not support.
    """

    directory = Path(model_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {directory}")
    names = [path.name for path in directory.iterdir()]
    config: dict[str, Any] | None = None
    config_path = directory / "config.json"
    if config_path.is_file():
        config = _load_json_object(config_path)
    inspect = inspect_hub_listing(
        filenames=names,
        config=config,
        source=str(directory),
    )
    prefixes: list[str] = []
    for name in inspect.vision_weight_files:
        if name.endswith(".safetensors"):
            prefixes.extend(_sidecar_key_prefixes(directory / name))
    return ArtifactModalityInspect(
        vision_declared=inspect.vision_declared,
        audio_declared=inspect.audio_declared,
        vision_weight_files=inspect.vision_weight_files,
        audio_weight_files=inspect.audio_weight_files,
        vision_key_prefixes=tuple(sorted(set(prefixes))),
        source=inspect.source,
        notes=inspect.notes,
    )


def format_modalities_card_section(block: PublicModalitiesBlock) -> str:
    """Markdown section for Hub cards / cert notes (capability-gated)."""

    def line(name: str, claim: PublicModalityClaim) -> str:
        reason = claim.reason or ""
        return f"| {name} | `{claim.status}` | `{str(claim.supported).lower()}` | {reason} |"

    return (
        "## Modalities (capability-gated)\n\n"
        "Text checkpoint Tier 1 does **not** imply vision or audio quality. "
        "`Vision present=true` on a pack is not a quality pass.\n\n"
        "| Modality | Claim | Supported | Reason |\n"
        "| --- | --- | --- | --- |\n"
        f"{line('Vision', block.vision)}\n"
        f"{line('Audio', block.audio)}\n"
    )
