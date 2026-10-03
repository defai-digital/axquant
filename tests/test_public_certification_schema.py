"""Frozen public certification schema contracts (AXQ-042)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _cert_fixtures import tier1_payload, write_cert_pair
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


def test_public_catalog_is_empty_pending_recertification() -> None:
    assert sorted(_CERT_DIR.glob("*-tier1.json")) == []
    assert sorted(_CERT_DIR.glob("*-tier2.json")) == []
    assert load_public_cert_rows(listed_only=False) == []
    assert not check_documents(root=_ROOT)


def test_synthetic_pair_loads_via_frozen_models(tmp_path: Path) -> None:
    tier1_path, tier2_path = write_cert_pair(tmp_path, "demo-axq6")
    cert = load_public_checkpoint_certification(tier1_path)
    assert cert.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert cert.public_index.display_name
    assert cert.public_index.edition_label
    mtp = load_public_mtp_acceleration_certification(tier2_path)
    assert mtp.schema_version == MTP_SCHEMA_VERSION
    assert mtp.certification_tier == "mtp-acceleration"


def test_unknown_top_level_field_is_rejected() -> None:
    data = tier1_payload()
    data["unexpected_envelope_field"] = True
    with pytest.raises(ValidationError):
        PublicCheckpointCertification.model_validate(data)


def test_invalid_status_is_rejected() -> None:
    data = tier1_payload()
    data["status"] = "maybe"
    with pytest.raises(ValidationError):
        PublicCheckpointCertification.model_validate(data)


def test_missing_public_index_is_rejected() -> None:
    data = tier1_payload()
    del data["public_index"]
    with pytest.raises(ValidationError):
        PublicCheckpointCertification.model_validate(data)


def test_missing_timestamp_is_rejected() -> None:
    data = tier1_payload()
    data.pop("certified_at", None)
    data.pop("evaluated_at", None)
    with pytest.raises(ValidationError, match="certified_at or evaluated_at"):
        PublicCheckpointCertification.model_validate(data)


def test_registry_lists_both_public_certification_versions() -> None:
    versions = public_certification_schema_versions()
    assert CHECKPOINT_SCHEMA_VERSION in versions
    assert MTP_SCHEMA_VERSION in versions
    entries = schema_registry()
    public_entries = [
        entry for entry in entries if entry.compatibility_class == "public-certification"
    ]
    assert all(entry.freeze_policy == "immutable-envelope" for entry in public_entries)
    assert {entry.model for entry in public_entries} == {
        PublicCheckpointCertification,
        PublicMtpAccelerationCertification,
    }


def test_index_loader_uses_schema_and_docs_stay_aligned(tmp_path: Path) -> None:
    write_cert_pair(tmp_path, "demo-axq6")
    rows = load_public_cert_rows(tmp_path, listed_only=False)
    assert len(rows) == 1
    assert rows[0].tier1_status == "certified"
    assert rows[0].tier2_status == "certified"
    # Schema-validated load must still drive documentation SSOT.
    assert not check_documents(root=_ROOT)


def test_index_rejects_tier2_certificate_for_a_different_artifact(tmp_path: Path) -> None:
    stem = "demo-axq6"
    _, tier2_path = write_cert_pair(tmp_path, stem)
    tier2 = json.loads(tier2_path.read_text(encoding="utf-8"))
    tier2["artifact"]["hub_repo_id"] = "AutomatosX/different-checkpoint"
    tier2_path.write_text(json.dumps(tier2, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Tier 2 artifact does not match Tier 1"):
        load_public_cert_rows(tmp_path, listed_only=False)
