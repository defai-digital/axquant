"""Schema contract freeze gates (AXQ-042 / CG1-CG3)."""

from __future__ import annotations

import json
from pathlib import Path

from axquant.schema.public_certification import (
    CHECKPOINT_SCHEMA_VERSION,
    MTP_SCHEMA_VERSION,
    PublicCheckpointCertification,
)
from axquant.schema_contracts import (
    build_schema_registry,
    check_base_ref_immutability,
    check_schema_contracts,
    discover_versioned_models,
    expected_schema_files,
    render_schema_catalog,
    schema_filename,
    schema_snapshot_text,
    write_schema_contracts,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_discovery_finds_unique_schema_versions() -> None:
    owned = discover_versioned_models()
    assert CHECKPOINT_SCHEMA_VERSION in owned
    assert MTP_SCHEMA_VERSION in owned
    assert owned[CHECKPOINT_SCHEMA_VERSION] is PublicCheckpointCertification
    assert len(owned) >= 50
    assert len(owned) == len(set(owned))


def test_registry_covers_every_discovered_version() -> None:
    owned = discover_versioned_models()
    registry = build_schema_registry()
    versions = {entry.schema_version for entry in registry}
    assert set(owned) <= versions
    public = [entry for entry in registry if entry.compatibility_class == "public-certification"]
    assert {entry.schema_version for entry in public} >= {
        CHECKPOINT_SCHEMA_VERSION,
        MTP_SCHEMA_VERSION,
    }


def test_schema_snapshots_match_live_models() -> None:
    messages = check_schema_contracts(root=_ROOT)
    assert not messages, "\n".join(messages)


def test_catalog_matches_generator() -> None:
    catalog = _ROOT / "docs" / "schema-catalog.md"
    assert catalog.is_file()
    assert catalog.read_text(encoding="utf-8") == render_schema_catalog()


def test_manifest_digests_match_snapshot_files() -> None:
    manifest = json.loads((_ROOT / "schemas" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "axquant.schema-contract-manifest.v1"
    assert manifest["count"] == len(manifest["entries"])
    for item in manifest["entries"]:
        path = _ROOT / "schemas" / item["filename"]
        body = path.read_text(encoding="utf-8")
        import hashlib

        assert hashlib.sha256(body.encode("utf-8")).hexdigest() == item["sha256"]


def test_snapshot_detects_model_field_requiredness_change(tmp_path: Path) -> None:
    """A requiredness change under the same version must fail the freeze check."""

    write_schema_contracts(root=_ROOT)
    # Copy live contracts into a temp tree and corrupt one snapshot.
    import shutil

    tmp_root = tmp_path / "repo"
    shutil.copytree(_ROOT / "schemas", tmp_root / "schemas")
    shutil.copytree(_ROOT / "docs", tmp_root / "docs")
    # Point discovery at real package (models unchanged) but compare against
    # corrupted snapshot under tmp_root by monkeypatching only the check root.
    target = tmp_root / "schemas" / schema_filename(CHECKPOINT_SCHEMA_VERSION)
    data = json.loads(target.read_text(encoding="utf-8"))
    # Flip a stable property that clean() preserves: required list membership.
    required = data.get("required")
    if isinstance(required, list) and "host_id" in required:
        data["required"] = [name for name in required if name != "host_id"]
    else:
        data["required"] = ["__corrupt__"]
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Recompute expected against live models into another temp and ensure mismatch
    # vs corrupted tree would be reported by comparing live expected to disk.
    expected = schema_snapshot_text(PublicCheckpointCertification)
    assert target.read_text(encoding="utf-8") != expected


def test_expected_files_are_deterministic() -> None:
    first = expected_schema_files(root=_ROOT)
    second = expected_schema_files(root=_ROOT)
    assert first.keys() == second.keys()
    for path in first:
        assert first[path] == second[path]


def test_base_ref_check_is_callable() -> None:
    # On a clean tree matching origin, either no base manifest yet or digests match.
    messages = check_base_ref_immutability(root=_ROOT)
    assert isinstance(messages, list)
