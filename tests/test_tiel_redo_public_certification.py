"""2026-09-23 Tiel / Cyber-Tiel redo Tier 1 records bind measured factory evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axquant.schema.public_certification import load_public_checkpoint_certification

CERTS = Path(__file__).resolve().parents[1] / "docs" / "certifications"

CASES = {
    "": {
        "status": "certified",
        "agent_coding_retention": 1.0,
        "general_retention": 1.0,
        "timestamp_field": "certified_at",
    },
    "cyber-": {
        "status": "not_certified",
        "agent_coding_retention": 0.9615384615384616,
        "general_retention": 1.0,
        "timestamp_field": "evaluated_at",
    },
}


def _load(prefix: str):
    return load_public_checkpoint_certification(
        CERTS / f"{prefix}tiel-coder-35b-axq-mxfp4-mtp-redo-tier1.json"
    )


@pytest.mark.parametrize("prefix", ["", "cyber-"])
def test_redo_record_binds_evidence_and_factory_verdict(prefix: str) -> None:
    expected = CASES[prefix]
    cert = _load(prefix)
    assert cert.status == expected["status"]
    assert cert.host_id == "df-macstudio-m2"
    assert cert.artifact.product_class == "MXFP4"
    assert cert.artifact.hub_repo_id.endswith("-MXFP4-MTP")
    assert cert.public_index.edition_label == f"main@{cert.artifact.hub_commit[:8]}"
    stamp = getattr(cert, expected["timestamp_field"])
    assert stamp is not None
    other_field = (
        "evaluated_at" if expected["timestamp_field"] == "certified_at" else "certified_at"
    )
    assert getattr(cert, other_field) is None

    assert cert.quality["agent-coding"]["retention"] == expected["agent_coding_retention"]
    assert cert.quality["general"]["retention"] == expected["general_retention"]
    assert cert.quality["agent-coding"]["reference_kind"] == "quantized-source-same-pin"
    assert cert.quality["agent-coding"]["samples"] == 15
    assert cert.thresholds["minimum_quality_retention"] == 0.98
    if cert.status == "not_certified":
        minimum = cert.thresholds["minimum_quality_retention"]
        assert cert.quality["agent-coding"]["retention"] < minimum

    assert cert.mtp_acceleration.status == "not-certified"
    assert cert.mtp_acceleration.ax_engine_version == "7.5.4"
    assert cert.runtime is not None
    assert cert.runtime["ax_engine"]["status"] == "pass"
    assert cert.runtime["mlx_lm"]["status"] == "pass"
    assert cert.runtime_default is not None
    assert cert.runtime_default["promoted"] is False

    # Every evidence hash the certificate records resolves on disk.
    assert cert.evidence_hashes
    for relative, expected_sha in cert.evidence_hashes.items():
        assert hashlib.sha256((CERTS / relative).read_bytes()).hexdigest() == expected_sha

    # The offline verification run on the factory passed every check.
    model_dir = "cyber-tiel" if prefix else "tiel"
    verify = json.loads(
        (CERTS / f"evidence/tiel-redo-20260923/{model_dir}/verify-cert.json").read_text()
    )
    assert verify["passed"] is True
    assert verify["issues"] == []
    assert len(verify["checks"]) == 20
    assert all(check["passed"] for check in verify["checks"])
    assert (
        verify["certificate_sha256"]
        == hashlib.sha256(
            (CERTS / f"{prefix}tiel-coder-35b-axq-mxfp4-mtp-redo-tier1.json").read_bytes()
        ).hexdigest()
    )
    assert verify["manifest_sha256"] == cert.artifact.candidate_manifest_sha256
    assert verify["product_class"] == "MXFP4"


@pytest.mark.parametrize("prefix", ["", "cyber-"])
def test_redo_artifact_integrity_matches_published_revision(prefix: str) -> None:
    cert = _load(prefix)
    model_dir = "cyber-tiel" if prefix else "tiel"
    integrity_path = CERTS / f"evidence/tiel-redo-20260923/{model_dir}/size/artifact-integrity.json"
    integrity = json.loads(integrity_path.read_text())
    assert integrity["all_match"] is True
    assert len(integrity["files"]) == 20
    assert integrity["manifest_sha256"] == cert.artifact.candidate_manifest_sha256
    assert all(
        file["manifest_sha256_match"] and file["manifest_size_match"] for file in integrity["files"]
    )
    # The metadata correction revision republished plan/manifest/README only.
    publication = json.loads(
        (CERTS / "evidence/tiel-redo-20260923/classfix-publication.json").read_text()
    )
    entry = publication[model_dir]
    assert entry["revision"] == cert.artifact.hub_commit
    assert all(file["match"] for file in entry["files"])
    assert {file["file"] for file in entry["files"]} == {
        "README.md",
        "axquant_manifest.json",
        "axquant_plan.json",
    }
