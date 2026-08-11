"""Immutable schema contract snapshots for freeze-class artifact models.

Registered ``schema_version`` owners emit a canonical JSON Schema under
``schemas/``. CI fails when a model drifts from its snapshot without a new
version string (AXQ-042 / Codex CG1-CG3).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import pkgutil
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, get_args, get_origin

from pydantic import BaseModel

from axquant.schema._base import StrictModel
from axquant.schema.public_certification import (
    CHECKPOINT_SCHEMA_VERSION,
    MTP_SCHEMA_VERSION,
)
from axquant.schema.registry import (
    CompatibilityClass,
    SchemaRegistryEntry,
)
from axquant.schema.registry import (
    schema_registry as public_cert_registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMAS_DIR = _REPO_ROOT / "schemas"
_DEFAULT_CATALOG = _REPO_ROOT / "docs" / "schema-catalog.md"
_MANIFEST_NAME = "manifest.json"

# Drop documentation-only keys so Pydantic docstring/title churn is not a freeze break.
_NOISE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "title",
        "description",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)


def _version_from_model(model: type[StrictModel]) -> str | None:
    field = model.model_fields.get("schema_version")
    if field is None:
        return None
    annotation = field.annotation
    if get_origin(annotation) is Literal:
        args = [arg for arg in get_args(annotation) if isinstance(arg, str)]
        if len(args) == 1:
            return args[0]
    default = field.default
    if isinstance(default, str) and default:
        return default
    return None


def discover_versioned_models() -> dict[str, type[StrictModel]]:
    """Map ``schema_version`` → owning StrictModel for top-level schema modules."""

    import axquant.schema as schema_pkg

    owned: dict[str, type[StrictModel]] = {}
    modules = [schema_pkg]
    modules.extend(
        importlib.import_module(f"axquant.schema.{module.name}")
        for module in pkgutil.iter_modules(schema_pkg.__path__)
        if not module.name.startswith("_")
    )
    for module in modules:
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is StrictModel or not issubclass(obj, StrictModel):
                continue
            if obj.__module__ != module.__name__:
                continue
            version = _version_from_model(obj)
            if version is None:
                continue
            if version in owned and owned[version] is not obj:
                raise ValueError(
                    f"duplicate schema_version {version!r}: "
                    f"{owned[version].__module__}.{owned[version].__name__} vs "
                    f"{obj.__module__}.{obj.__name__}"
                )
            owned[version] = obj
    return owned


def _compatibility_class(version: str) -> CompatibilityClass:
    if version in {CHECKPOINT_SCHEMA_VERSION, MTP_SCHEMA_VERSION}:
        return "public-certification"
    if version.startswith("axquant.public-"):
        return "public-certification"
    if any(
        token in version
        for token in (
            "release-audit",
            "release-validation",
            "release-exception",
            "artifact-lifecycle",
            "public-claim",
            "flagship-release",
        )
    ):
        return "release"
    if any(
        token in version
        for token in (
            "capture-progress",
            "coding-evaluation-state",
            "probe-capacity",
            "plan-request",
        )
    ):
        return "operational"
    return "evidence"


def build_schema_registry() -> tuple[SchemaRegistryEntry, ...]:
    """Full freeze registry: public-cert entries first, then discovered models."""

    owned = discover_versioned_models()
    public = {entry.schema_version: entry for entry in public_cert_registry()}
    entries: list[SchemaRegistryEntry] = []
    for version in sorted(owned):
        model = owned[version]
        if version in public:
            base = public[version]
            entries.append(
                SchemaRegistryEntry(
                    schema_version=version,
                    model=model,
                    compatibility_class=base.compatibility_class,
                    freeze_policy=base.freeze_policy,
                    description=base.description,
                )
            )
            continue
        entries.append(
            SchemaRegistryEntry(
                schema_version=version,
                model=model,
                compatibility_class=_compatibility_class(version),
                freeze_policy="immutable-envelope",
                description=f"{model.__module__}.{model.__name__}",
            )
        )
    # Ensure public-cert versions are always present even if discovery fails.
    for version, entry in public.items():
        if version not in owned:
            entries.insert(0, entry)
    entries.sort(
        key=lambda item: (
            0 if item.compatibility_class == "public-certification" else 1,
            item.schema_version,
        )
    )
    return tuple(entries)


def _clean_schema(node: Any) -> Any:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _NOISE_KEYS:
                continue
            cleaned[key] = _clean_schema(value)
        return cleaned
    if isinstance(node, list):
        return [_clean_schema(item) for item in node]
    return node


def canonical_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema(mode="validation")
    cleaned = _clean_schema(raw)
    if not isinstance(cleaned, dict):
        raise TypeError("JSON Schema root must be an object")
    return cleaned


def schema_snapshot_text(model: type[BaseModel]) -> str:
    payload = json.dumps(
        canonical_json_schema(model),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    return payload + "\n"


def schema_filename(schema_version: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", schema_version)
    return f"{safe}.schema.json"


def schemas_dir(root: Path | None = None) -> Path:
    return (root or _REPO_ROOT) / "schemas"


def catalog_path(root: Path | None = None) -> Path:
    return (root or _REPO_ROOT) / "docs" / "schema-catalog.md"


@dataclass(frozen=True, slots=True)
class SchemaManifestEntry:
    schema_version: str
    filename: str
    model: str
    compatibility_class: str
    freeze_policy: str
    sha256: str


def build_manifest(entries: tuple[SchemaRegistryEntry, ...]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for entry in entries:
        body = schema_snapshot_text(entry.model)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        items.append(
            {
                "schema_version": entry.schema_version,
                "filename": schema_filename(entry.schema_version),
                "model": f"{entry.model.__module__}.{entry.model.__name__}",
                "compatibility_class": entry.compatibility_class,
                "freeze_policy": entry.freeze_policy,
                "sha256": digest,
            }
        )
    items.sort(key=lambda item: item["schema_version"])
    return {
        "schema_version": "axquant.schema-contract-manifest.v1",
        "count": len(items),
        "entries": items,
    }


def render_schema_catalog(entries: tuple[SchemaRegistryEntry, ...] | None = None) -> str:
    registry = entries if entries is not None else build_schema_registry()
    lines = [
        "<!-- Generated by axquant.schema_contracts — do not edit by hand. -->",
        "",
        "# Schema contract catalog",
        "",
        "Machine-owned freeze catalog for versioned artifact models. Source of truth is",
        "the Pydantic model registered for each `schema_version`; JSON Schema snapshots",
        "under [`schemas/`](../schemas/) lock the serialized contract.",
        "",
        "Regenerate:",
        "",
        "```bash",
        "python scripts/render_schema_contracts.py --write",
        "```",
        "",
        "Policy: [schema governance](schema-governance.md).",
        "",
        "| schema_version | class | freeze | model | snapshot |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in sorted(registry, key=lambda item: item.schema_version):
        model_name = f"`{entry.model.__module__}.{entry.model.__name__}`"
        filename = schema_filename(entry.schema_version)
        snap = f"[`{filename}`](../schemas/{filename})"
        lines.append(
            f"| `{entry.schema_version}` | `{entry.compatibility_class}` | "
            f"`{entry.freeze_policy}` | {model_name} | {snap} |"
        )
    lines.append("")
    return "\n".join(lines)


def expected_schema_files(
    *,
    root: Path | None = None,
    entries: tuple[SchemaRegistryEntry, ...] | None = None,
) -> dict[Path, str]:
    repo = root or _REPO_ROOT
    registry = entries if entries is not None else build_schema_registry()
    out_dir = schemas_dir(repo)
    files: dict[Path, str] = {}
    for entry in registry:
        path = out_dir / schema_filename(entry.schema_version)
        files[path] = schema_snapshot_text(entry.model)
    files[out_dir / _MANIFEST_NAME] = (
        json.dumps(build_manifest(registry), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    files[catalog_path(repo)] = render_schema_catalog(registry)
    return files


def write_schema_contracts(*, root: Path | None = None) -> list[Path]:
    written: list[Path] = []
    for path, content in expected_schema_files(root=root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
        if previous != content:
            path.write_text(content, encoding="utf-8")
            written.append(path)
    return written


def check_schema_contracts(*, root: Path | None = None) -> list[str]:
    """Return drift messages; empty means snapshots match the live models."""

    messages: list[str] = []
    repo = root or _REPO_ROOT
    expected = expected_schema_files(root=repo)
    out_dir = schemas_dir(repo)
    if not out_dir.is_dir():
        return [f"missing schemas directory: {out_dir}"]

    expected_names = {path.name for path in expected if path.parent == out_dir}
    for path in sorted(out_dir.glob("*.schema.json")):
        if path.name not in expected_names:
            messages.append(f"{path.relative_to(repo)}: orphan schema snapshot (not in registry)")
    for path, content in expected.items():
        rel = path.relative_to(repo)
        if not path.is_file():
            messages.append(f"missing generated schema contract: {rel}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != content:
            messages.append(
                f"{rel}: schema contract out of date "
                "(run: python scripts/render_schema_contracts.py --write)"
            )
    return messages


def check_base_ref_immutability(
    *,
    root: Path | None = None,
    base_ref: str | None = None,
) -> list[str]:
    """Fail if an existing schema_version snapshot changed vs git base ref.

    When ``base_ref`` is None, uses ``origin/main`` if available, else ``main``,
    else skips with no messages (local first freeze).
    """

    repo = root or _REPO_ROOT
    ref = base_ref
    if ref is None:
        for candidate in ("origin/main", "main"):
            probe = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", candidate],
                check=False,
                capture_output=True,
            )
            if probe.returncode == 0:
                ref = candidate
                break
    if ref is None:
        return []

    show = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:schemas/manifest.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if show.returncode != 0:
        # First introduction of schemas/ on this branch — nothing to compare.
        return []

    try:
        base_manifest = json.loads(show.stdout)
    except json.JSONDecodeError as exc:
        return [f"base ref {ref}: schemas/manifest.json is not valid JSON: {exc}"]

    base_entries = {
        item["schema_version"]: item
        for item in base_manifest.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("schema_version"), str)
    }
    current_path = schemas_dir(repo) / _MANIFEST_NAME
    if not current_path.is_file():
        return ["schemas/manifest.json missing in working tree"]
    current_manifest = json.loads(current_path.read_text(encoding="utf-8"))
    current_entries = {
        item["schema_version"]: item
        for item in current_manifest.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("schema_version"), str)
    }

    messages: list[str] = []
    for version, base_item in sorted(base_entries.items()):
        if version not in current_entries:
            messages.append(
                f"schema_version {version!r} removed from freeze registry "
                f"(forbidden without explicit retirement; base={ref})"
            )
            continue
        cur = current_entries[version]
        if cur.get("sha256") != base_item.get("sha256"):
            messages.append(
                f"schema_version {version!r} snapshot sha256 changed under the same "
                f"version string (base={ref}). Bump schema_version instead of editing "
                f"{base_item.get('filename')}."
            )
        if cur.get("filename") != base_item.get("filename"):
            messages.append(
                f"schema_version {version!r} snapshot filename renamed under the same "
                f"version (base={ref})."
            )
    return messages
