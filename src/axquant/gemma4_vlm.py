"""Gemma 4 protected-multimodal layout helpers for public MLX-VLM runtimes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from safetensors import safe_open

from axquant.errors import ArtifactError

GEMMA4_MLX_VLM_VISION_LAYOUT = "mlx-vlm-gemma4-v1"
_MODEL_PREFIX = "model."
_VISION_PREFIXES = (
    "vision_tower.",
    "vision_embedder.",
    "embed_vision.",
    "embed_audio.",
)
_SUPPORTED_MODEL_TYPES = frozenset({"gemma4", "gemma4_unified"})


def normalize_gemma4_vision_tensor_names(names: Iterable[str]) -> tuple[str, ...]:
    """Return Gemma 4 vision names in the public MLX-VLM module namespace."""

    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        output_name = name
        for prefix in _VISION_PREFIXES:
            if name.startswith(f"{_MODEL_PREFIX}{prefix}"):
                output_name = name.removeprefix(_MODEL_PREFIX)
                break
            if name.startswith(prefix):
                break
        else:
            raise ArtifactError(f"unsupported Gemma 4 protected vision tensor name: {name}")
        if output_name in seen:
            raise ArtifactError(
                f"Gemma 4 vision tensor name collision after normalization: {output_name}"
            )
        seen.add(output_name)
        normalized.append(output_name)
    if not normalized:
        raise ArtifactError("Gemma 4 protected vision sidecar contains no tensors")
    return tuple(normalized)


def _load_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Gemma 4 Safetensors index is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError("Gemma 4 Safetensors index must contain a JSON object")
    return value


def validate_gemma4_mlx_vlm_vision_layout(directory: str | Path) -> tuple[str, ...]:
    """Fail closed unless an AXQ Gemma target is loadable by public MLX-VLM."""

    root = Path(directory).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ArtifactError("Gemma 4 oMLX/MLX-VLM compatibility requires config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Gemma 4 config.json is unreadable: {exc}") from exc
    if not isinstance(config, dict):
        raise ArtifactError("Gemma 4 config.json must contain a JSON object")
    model_type = config.get("model_type")
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ArtifactError(
            "Gemma 4 oMLX/MLX-VLM compatibility requires model_type gemma4 or gemma4_unified"
        )

    sidecar = root / "vision.safetensors"
    if not sidecar.is_file():
        raise ArtifactError("Gemma 4 oMLX/MLX-VLM compatibility requires vision.safetensors")
    try:
        with safe_open(sidecar, framework="numpy") as handle:
            names = tuple(handle.keys())
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise ArtifactError(f"Gemma 4 vision.safetensors is unreadable: {exc}") from exc
    normalized = normalize_gemma4_vision_tensor_names(names)
    if normalized != names:
        raise ArtifactError(
            "Gemma 4 vision.safetensors uses source-prefixed tensor names; rebuild the "
            "Tier 1 target with the MLX-VLM vision layout"
        )
    if metadata.get("format") != "mlx":
        raise ArtifactError("Gemma 4 vision.safetensors must declare Safetensors format=mlx")
    if metadata.get("axquant_layout") != GEMMA4_MLX_VLM_VISION_LAYOUT:
        raise ArtifactError(
            "Gemma 4 vision.safetensors is missing the AXQuant MLX-VLM layout marker"
        )
    if model_type == "gemma4_unified":
        required_prefixes = ["vision_embedder.", "embed_vision."]
        audio_config = config.get("audio_config")
        if isinstance(audio_config, dict) and audio_config:
            required_prefixes.append("embed_audio.")
        missing_prefixes = [
            prefix
            for prefix in required_prefixes
            if not any(name.startswith(prefix) for name in names)
        ]
        if missing_prefixes:
            raise ArtifactError(
                f"Gemma 4 unified sidecar is missing MLX-VLM multimodal modules: {missing_prefixes}"
            )

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ArtifactError(
            "Gemma 4 oMLX/MLX-VLM compatibility requires model.safetensors.index.json"
        )
    index = _load_index(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError("Gemma 4 Safetensors index has no non-empty weight_map")
    indexed_vision_names = {
        name for name, shard in weight_map.items() if shard == "vision.safetensors"
    }
    if indexed_vision_names != set(names):
        raise ArtifactError(
            "Gemma 4 vision.safetensors keys do not exactly match its Safetensors index entries"
        )
    return names
