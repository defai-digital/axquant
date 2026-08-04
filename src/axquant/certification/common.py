from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from axquant.errors import ArtifactError
from axquant.schema import (
    ArchitectureFingerprint,
    Inventory,
    SourceCheckpointFile,
    SourceCheckpointManifest,
)
from axquant.serde import file_sha256, stable_sha256

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "axquant_source.json",
    "model.safetensors.index.json",
)


def resolved(base: Path, value: str) -> Path:
    root = base.expanduser().resolve()
    path = Path(value)
    if path.is_absolute() or value.startswith("~"):
        raise ArtifactError(f"certification evidence path must be relative: {value}")
    resolved_path = (root / path).resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(f"certification evidence path escapes its root: {value}") from exc
    return resolved_path


def required_file(base: Path, value: str, label: str) -> Path:
    path = resolved(base, value)
    if not path.is_file():
        raise ArtifactError(f"{label} does not exist: {path}")
    return path


def required_directory(base: Path, value: str, label: str) -> Path:
    path = resolved(base, value)
    if not path.is_dir():
        raise ArtifactError(f"{label} does not exist: {path}")
    return path


def bound_file(base: Path, value: str, expected_sha256: str, label: str) -> Path:
    path = required_file(base, value, label)
    if file_sha256(path) != expected_sha256:
        raise ArtifactError(f"{label} checksum does not match its index: {path}")
    return path


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} must contain a JSON object: {path}")
    return value


def tokenizer_sha256(model_dir: str | Path) -> str:
    directory = Path(model_dir).expanduser().resolve()
    bindings = {
        name: file_sha256(directory / name)
        for name in _TOKENIZER_FILES
        if (directory / name).is_file()
    }
    if not bindings:
        raise ArtifactError(f"checkpoint has no supported tokenizer files: {directory}")
    return stable_sha256(bindings)


def architecture_fingerprint(
    model_dir: str | Path,
    *,
    inventory: Inventory,
) -> ArchitectureFingerprint:
    directory = Path(model_dir).expanduser().resolve()
    config_path = directory / "config.json"
    config = _read_json_object(config_path, "model config")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen3NextForCausalLM" not in architectures:
        raise ArtifactError("Qwen3-Next config does not declare Qwen3NextForCausalLM")

    def positive_int(key: str) -> int:
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ArtifactError(f"Qwen3-Next config field {key!r} must be a positive integer")
        return value

    return ArchitectureFingerprint(
        model_type=config.get("model_type"),
        architecture="Qwen3NextForCausalLM",
        text_layer_count=positive_int("num_hidden_layers"),
        hidden_size=positive_int("hidden_size"),
        full_attention_interval=positive_int("full_attention_interval"),
        expert_count=positive_int("num_experts"),
        experts_per_token=positive_int("num_experts_per_tok"),
        expert_intermediate_size=positive_int("moe_intermediate_size"),
        mtp_declared=inventory.architecture_profile.mtp_declared,
        vision_present=inventory.architecture_profile.vision_present,
        config_sha256=file_sha256(config_path),
        tokenizer_sha256=tokenizer_sha256(directory),
    )


def source_checkpoint_files(model_dir: str | Path, inventory: Inventory) -> list[Path]:
    directory = Path(model_dir).expanduser().resolve()
    inventory_weight_names = set(inventory.source_files)
    if len(inventory_weight_names) != len(inventory.source_files):
        raise ArtifactError("source inventory contains duplicate weight-file paths")
    actual_weight_names = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.safetensors")
        if path.is_file()
    }
    if actual_weight_names != inventory_weight_names:
        missing = sorted(inventory_weight_names - actual_weight_names)
        untracked = sorted(actual_weight_names - inventory_weight_names)
        raise ArtifactError(
            "source inventory Safetensors membership differs from the checkpoint: "
            f"missing={missing}, untracked={untracked}"
        )
    relative_names = set(inventory_weight_names)
    relative_names.update(name for name in _METADATA_FILES if (directory / name).is_file())
    relative_names.update(name for name in _TOKENIZER_FILES if (directory / name).is_file())
    files: list[Path] = []
    for name in sorted(relative_names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ArtifactError(f"unsafe source checkpoint member: {name}")
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ArtifactError(f"source checkpoint member escapes its root: {name}") from exc
        if not path.is_file():
            raise ArtifactError(f"source checkpoint member is missing: {name}")
        files.append(path)
    return files


def build_source_checkpoint_manifest(
    model_dir: str | Path,
    *,
    inventory: Inventory,
) -> SourceCheckpointManifest:
    directory = Path(model_dir).expanduser().resolve()
    if (
        inventory.model.local_path is None
        or Path(inventory.model.local_path).resolve() != directory
    ):
        raise ArtifactError("inventory local path does not match the source checkpoint")
    fingerprint = architecture_fingerprint(directory, inventory=inventory)
    records = [
        SourceCheckpointFile(
            path=path.relative_to(directory).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in source_checkpoint_files(directory, inventory)
    ]
    return SourceCheckpointManifest(
        source_model=inventory.model,
        config_sha256=fingerprint.config_sha256,
        tokenizer_sha256=fingerprint.tokenizer_sha256,
        files=records,
    )


def source_manifest_issues(
    model_dir: str | Path,
    manifest: SourceCheckpointManifest,
    *,
    inventory: Inventory,
) -> list[str]:
    directory = Path(model_dir).expanduser().resolve()
    issues: list[str] = []
    for record in manifest.files:
        path = directory / record.path
        if not path.is_file():
            issues.append(f"source checkpoint member is missing: {record.path}")
        elif path.stat().st_size != record.size_bytes:
            issues.append(f"source checkpoint member size changed: {record.path}")
        elif file_sha256(path) != record.sha256:
            issues.append(f"source checkpoint member checksum changed: {record.path}")
    try:
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in source_checkpoint_files(directory, inventory)
        }
        recorded_files = {record.path for record in manifest.files}
        if actual_files != recorded_files:
            issues.append("source checkpoint manifest membership changed")
    except (ArtifactError, ValueError) as exc:
        issues.append(str(exc))
    return issues


def finite_median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute a median from no values")
    return float(median(values))
