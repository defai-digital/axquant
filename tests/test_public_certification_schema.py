"""Frozen public certification schema contracts (AXQ-042)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from axquant.public_cert_index import check_documents, load_public_cert_rows
from axquant.schema.public_certification import (
    CHECKPOINT_SCHEMA_VERSION,
    MTP_SCHEMA_VERSION,
    PublicCheckpointCertification,
    PublicMtpAccelerationCertification,
    load_public_checkpoint_certification,
    load_public_mtp_acceleration_certification,
)
from axquant.schema.registry import public_certification_schema_versions, schema_registry

_ROOT = Path(__file__).resolve().parents[1]
_CERT_DIR = _ROOT / "docs" / "certifications"


def test_all_public_certificate_json_loads_via_frozen_models() -> None:
    tier1 = sorted(_CERT_DIR.glob("*-tier1.json"))
    tier2 = sorted(_CERT_DIR.glob("*-tier2.json"))
    assert tier1, "expected checkpoint Tier 1 certificates"
    for path in tier1:
        cert = load_public_checkpoint_certification(path)
        assert cert.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert cert.public_index.display_name
        assert cert.public_index.edition_label
    for path in tier2:
        cert = load_public_mtp_acceleration_certification(path)
        assert cert.schema_version == MTP_SCHEMA_VERSION
        assert cert.certification_tier == "mtp-acceleration"


def test_unknown_top_level_field_is_rejected() -> None:
    path = _CERT_DIR / "gemma4-12b-axq4-tier1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["unexpected_envelope_field"] = True
    with pytest.raises(ValidationError):
        PublicCheckpointCertification.model_validate(data)


def test_invalid_status_is_rejected() -> None:
    path = _CERT_DIR / "gemma4-12b-axq4-tier1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "maybe"
    with pytest.raises(ValidationError):
        PublicCheckpointCertification.model_validate(data)


def test_missing_public_index_is_rejected() -> None:
    path = _CERT_DIR / "gemma4-12b-axq4-tier1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["public_index"]
    with pytest.raises(ValidationError):
        PublicCheckpointCertification.model_validate(data)


def test_missing_timestamp_is_rejected() -> None:
    path = _CERT_DIR / "gpt-oss-20b-axq4-tier1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("certified_at", None)
    data.pop("evaluated_at", None)
    with pytest.raises(ValidationError, match="certified_at or evaluated_at"):
        PublicCheckpointCertification.model_validate(data)


def test_registry_lists_both_public_certification_versions() -> None:
    versions = public_certification_schema_versions()
    assert CHECKPOINT_SCHEMA_VERSION in versions
    assert MTP_SCHEMA_VERSION in versions
    entries = schema_registry()
    assert all(entry.compatibility_class == "public-certification" for entry in entries)
    assert all(entry.freeze_policy == "immutable-envelope" for entry in entries)
    assert {entry.model for entry in entries} == {
        PublicCheckpointCertification,
        PublicMtpAccelerationCertification,
    }


def test_index_loader_uses_schema_and_docs_stay_aligned() -> None:
    rows = load_public_cert_rows(listed_only=False)
    assert len(rows) >= 17
    listed = [row for row in rows if row.listed]
    assert listed
    # Schema-validated load must still drive documentation SSOT.
    assert not check_documents(root=_ROOT)


def test_index_rejects_tier2_certificate_for_a_different_artifact(tmp_path: Path) -> None:
    stem = "qwen36-27b-axq6"
    tier1_source = _CERT_DIR / f"{stem}-tier1.json"
    tier2_source = _CERT_DIR / f"{stem}-tier2.json"
    (tmp_path / f"{stem}-tier1.json").write_bytes(tier1_source.read_bytes())
    tier2 = json.loads(tier2_source.read_text(encoding="utf-8"))
    tier2["artifact"]["hub_repo_id"] = "AutomatosX/different-checkpoint"
    (tmp_path / f"{stem}-tier2.json").write_text(
        json.dumps(tier2, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / f"{stem}-tier1.md").write_text("# tier 1\n", encoding="utf-8")
    (tmp_path / f"{stem}-tier2.md").write_text("# tier 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Tier 2 artifact does not match Tier 1"):
        load_public_cert_rows(tmp_path, listed_only=False)
