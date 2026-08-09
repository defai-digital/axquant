"""Compose byte-identical AXQ Gemma targets with assistant-MTP drafters.

Implements the ST2 composite pack layout from
``.internal/engineering/sibling-tier2-expansion-technical-specification.md``:

- copy/hardlink Tier 1 AXQ target files without rewriting tensors;
- attach ``assistant/`` + ``ax_gemma4_assistant_mtp.json``;
- emit ``ax_composite_pack_manifest.json`` with digests for claim binding.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from axquant.errors import ArtifactError
from axquant.serde import file_sha256

COMPOSITE_MANIFEST_NAME = "ax_composite_pack_manifest.json"
ASSISTANT_CONTRACT_NAME = "ax_gemma4_assistant_mtp.json"
COMPOSITE_SCHEMA_VERSION = "axquant.composite-pack-manifest.v1"
ASSISTANT_CONTRACT_SCHEMA = "ax.gemma4_assistant_mtp.v1"

# Engine known-pair leaves (gemma4_assistant_mtp.rs).
KNOWN_TARGET_LEAVES: frozenset[str] = frozenset(
    {
        "gemma-4-e2b-it",
        "gemma-4-e4b-it",
        "gemma-4-12b-it",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    }
)

# Files that must not be copied from the target root into the composite (assistant owns them).
_SKIP_TARGET_NAMES: frozenset[str] = frozenset(
    {
        ASSISTANT_CONTRACT_NAME,
        COMPOSITE_MANIFEST_NAME,
        ".git",
        ".gitattributes",
        ".cache",
    }
)


@dataclass(frozen=True, slots=True)
class Gemma4AssistantComposeRequest:
    """Inputs for composite pack assembly."""

    target_dir: Path
    assistant_dir: Path
    output_dir: Path
    target_model_id: str
    assistant_model_id: str
    base_pack_id: str | None = None
    base_tier1_certificate: str | None = None
    assistant_source_id: str | None = None
    max_depth: int = 1
    prefer_hardlink: bool = True
    axquant_version: str | None = None


@dataclass(frozen=True, slots=True)
class Gemma4AssistantComposeResult:
    """Paths and digests produced by composition."""

    output_dir: Path
    contract_path: Path
    manifest_path: Path
    contract_sha256: str
    base_weight_digests: dict[str, str]
    assistant_weight_digests: dict[str, str]


def _model_id_leaf(model_id: str) -> str:
    leaf = model_id.rsplit("/", 1)[-1].strip().lower()
    return leaf


def validate_known_gemma4_assistant_pair(target_model_id: str, assistant_model_id: str) -> None:
    """Fail closed unless target/assistant leaves form an engine-known pair."""
    target = _model_id_leaf(target_model_id)
    assistant = _model_id_leaf(assistant_model_id)
    if target not in KNOWN_TARGET_LEAVES:
        raise ArtifactError(
            f"target_model_id leaf {target!r} is not a known Gemma4 assistant pair target"
        )
    prefix = assistant.removesuffix("-assistant")
    if not assistant.endswith("-assistant") or prefix != target:
        raise ArtifactError(
            f"assistant_model_id leaf {assistant!r} must be {target}-assistant "
            f"(engine exact-pair contract)"
        )


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path)
    return files


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _digest_tree(root: Path, *, prefix: str = "") -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in _iter_files(root):
        rel = _relative_posix(path, root)
        key = f"{prefix}{rel}" if prefix else rel
        digests[key] = file_sha256(path)
    return digests


def _link_or_copy(src: Path, dst: Path, *, prefer_hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _copy_target_tree(target_dir: Path, output_dir: Path, *, prefer_hardlink: bool) -> None:
    for path in _iter_files(target_dir):
        rel = path.relative_to(target_dir)
        if rel.parts and rel.parts[0] in _SKIP_TARGET_NAMES:
            continue
        if rel.name in _SKIP_TARGET_NAMES:
            continue
        # Never copy an existing assistant/ from target into composite (source of truth is request).
        if rel.parts and rel.parts[0] == "assistant":
            continue
        _link_or_copy(path, output_dir / rel, prefer_hardlink=prefer_hardlink)


def _copy_assistant_tree(
    assistant_dir: Path, output_assistant: Path, *, prefer_hardlink: bool
) -> None:
    if not assistant_dir.is_dir():
        raise ArtifactError(f"assistant directory is missing: {assistant_dir}")
    config_path = assistant_dir / "config.json"
    if not config_path.is_file():
        raise ArtifactError(f"assistant config.json is missing under {assistant_dir}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"assistant config.json is unreadable: {exc}") from exc
    model_type = str(config.get("model_type", "")).strip().lower()
    if model_type != "gemma4_assistant":
        raise ArtifactError(
            f"assistant model_type must be 'gemma4_assistant', got {model_type!r}"
        )
    for path in _iter_files(assistant_dir):
        rel = path.relative_to(assistant_dir)
        _link_or_copy(path, output_assistant / rel, prefer_hardlink=prefer_hardlink)


def _write_contract(
    path: Path,
    *,
    target_model_id: str,
    assistant_model_id: str,
    max_depth: int,
) -> str:
    if max_depth < 1:
        raise ArtifactError("max_depth must be >= 1")
    payload = {
        "schema_version": ASSISTANT_CONTRACT_SCHEMA,
        "backend": "gemma4_assistant",
        "target_model_id": _model_id_leaf(target_model_id),
        "assistant_model_id": _model_id_leaf(assistant_model_id),
        "assistant_path": "assistant",
        "max_depth": int(max_depth),
        "pairing": "exact",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_sha256(path)


def compose_gemma4_assistant_mtp(request: Gemma4AssistantComposeRequest) -> Gemma4AssistantComposeResult:
    """Build a composite AXQ+assistant pack under ``request.output_dir``."""
    target_dir = request.target_dir.expanduser().resolve()
    assistant_dir = request.assistant_dir.expanduser().resolve()
    output_dir = request.output_dir.expanduser().resolve()

    if not target_dir.is_dir():
        raise ArtifactError(f"target directory is missing: {target_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ArtifactError(f"output directory must be empty or new: {output_dir}")

    validate_known_gemma4_assistant_pair(request.target_model_id, request.assistant_model_id)

    # Digest base before copy so we can prove identity after composition.
    base_weight_digests = {
        key: digest
        for key, digest in _digest_tree(target_dir).items()
        if not key.startswith("assistant/")
        and Path(key).name not in _SKIP_TARGET_NAMES
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_target_tree(target_dir, output_dir, prefer_hardlink=request.prefer_hardlink)
    _copy_assistant_tree(
        assistant_dir,
        output_dir / "assistant",
        prefer_hardlink=request.prefer_hardlink,
    )

    # Verify base digests in the composite match the source target.
    composed_base = {
        key: digest
        for key, digest in _digest_tree(output_dir).items()
        if not key.startswith("assistant/")
        and Path(key).name
        not in {ASSISTANT_CONTRACT_NAME, COMPOSITE_MANIFEST_NAME, *_SKIP_TARGET_NAMES}
    }
    # Only compare keys that existed on the target (ignore new contract/manifest).
    for key, digest in base_weight_digests.items():
        composed = composed_base.get(key)
        if composed is None:
            raise ArtifactError(f"composed pack is missing base file {key!r}")
        if composed != digest:
            raise ArtifactError(
                f"composed base file {key!r} digest mismatch "
                f"(target={digest} composite={composed})"
            )

    contract_path = output_dir / ASSISTANT_CONTRACT_NAME
    contract_sha256 = _write_contract(
        contract_path,
        target_model_id=request.target_model_id,
        assistant_model_id=request.assistant_model_id,
        max_depth=request.max_depth,
    )
    assistant_weight_digests = _digest_tree(output_dir / "assistant", prefix="assistant/")

    manifest: dict[str, Any] = {
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "product_class": "gemma4-axq-assistant-mtp",
        "base_pack_id": request.base_pack_id,
        "base_tier1_certificate": request.base_tier1_certificate,
        "base_weight_digests": dict(sorted(base_weight_digests.items())),
        "assistant_source_id": request.assistant_source_id,
        "assistant_weight_digests": dict(sorted(assistant_weight_digests.items())),
        "contract_file": ASSISTANT_CONTRACT_NAME,
        "contract_sha256": contract_sha256,
        "target_model_id": _model_id_leaf(request.target_model_id),
        "assistant_model_id": _model_id_leaf(request.assistant_model_id),
        "composed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool_versions": {
            "axquant": request.axquant_version,
        },
        "prefer_hardlink": request.prefer_hardlink,
    }
    manifest_path = output_dir / COMPOSITE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return Gemma4AssistantComposeResult(
        output_dir=output_dir,
        contract_path=contract_path,
        manifest_path=manifest_path,
        contract_sha256=contract_sha256,
        base_weight_digests=base_weight_digests,
        assistant_weight_digests=assistant_weight_digests,
    )


def load_composite_manifest(path: Path) -> Mapping[str, Any]:
    """Load and lightly validate a composite pack manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"composite manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("composite manifest must be a JSON object")
    if payload.get("schema_version") != COMPOSITE_SCHEMA_VERSION:
        raise ArtifactError(
            f"unsupported composite manifest schema_version: {payload.get('schema_version')!r}"
        )
    return payload
