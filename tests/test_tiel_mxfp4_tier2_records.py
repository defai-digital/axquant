"""2026-09-24 Tiel / Cyber-Tiel MXFP4 MTP Tier 2 records bind measured evidence.

Both scoped MTP acceleration (Tier 2) records are ``not_certified``: greedy
MTP-vs-direct exactness fails on both packs on the factory host, and the
``df-macbookpro-m5`` comparison evidence (same engine binary) shows the same
inexactness, establishing it as a pack property rather than a host defect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axquant.schema.public_certification import (
    load_public_checkpoint_certification,
    load_public_mtp_acceleration_certification,
)

CERTS = Path(__file__).resolve().parents[1] / "docs" / "certifications"
EVIDENCE = CERTS / "evidence" / "tiel-mxfp4-tier2-20260924"

AGENT_CODING_DATASET_SHA256 = "92e9803670152665b81814461460ee8cd33fe3dbd841c8b9eb10a913ffdfcfef"
GENERAL_DATASET_SHA256 = "f02e68877bbae8c100d7359b9bb286a9687e388c8912c0f8cb730d615b768925"

CASES = {
    "": {
        "exactness": {"agent-coding": False, "general-long": False},
        "weighted": {"agent-coding": 0.6677, "general-long": 1.0145},
        "divergent": {"agent-coding": 1, "general-long": 2},
        "failed_trials": {"agent-coding": 1, "general-long": 0},
    },
    "cyber-": {
        "exactness": {"agent-coding": True, "general-long": False},
        "weighted": {"agent-coding": 1.2088, "general-long": 1.0889},
        "divergent": {"agent-coding": 0, "general-long": 2},
        "failed_trials": {"agent-coding": 0, "general-long": 0},
    },
}


def _load(prefix: str):
    return load_public_mtp_acceleration_certification(
        CERTS / f"{prefix}tiel-coder-35b-axq-mxfp4-mtp-redo-tier2.json"
    )


@pytest.mark.parametrize("prefix", ["", "cyber-"])
def test_tier2_record_binds_factory_evidence_and_verdict(prefix: str) -> None:
    expected = CASES[prefix]
    cert = _load(prefix)
    assert cert.status == "not_certified"
    assert cert.host_id == "df-macstudio-m2"
    assert cert.certification_tier == "mtp-acceleration"
    assert cert.artifact.product_class == "MXFP4"
    assert cert.artifact.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert cert.thresholds["exactness_required"] == 1.0
    assert cert.thresholds["token_weighted_decode_speedup_min"] == 1.2
    assert cert.thresholds["prompt_median_speedup_min"] == 1.1

    mtp = cert.mtp_acceleration
    assert mtp["status"] == "not-certified"
    assert mtp["ax_engine_version"] == "7.5.4"
    assert (
        mtp["engine_binary_sha256"]
        == "eeb404133ea7cec4fc1da7b0dee7aebcd4d8006b3b2f0eda96c2112908cacac9"
    )

    profiles = mtp["profiles"]
    for workload in ("agent-coding", "general-long"):
        assert len(profiles[workload]["dataset_sha256"]) == 64
        assert len(profiles[workload]["comparison_sha256"]) == 64
    assert profiles["agent-coding"]["dataset_sha256"] == AGENT_CODING_DATASET_SHA256
    assert profiles["general-long"]["dataset_sha256"] == GENERAL_DATASET_SHA256
    for workload in ("agent-coding", "general-long"):
        profile = profiles[workload]
        assert profile["exactness_pass"] is expected["exactness"][workload]
        assert profile["divergent_trial_count"] == expected["divergent"][workload]
        assert profile["failed_trial_count"] == expected["failed_trials"][workload]
        assert profile["token_weighted_decode_speedup"] == pytest.approx(
            expected["weighted"][workload], abs=5e-4
        )
        assert profile["measured_trial_count"] == 2
        # Every recorded comparison hash resolves on disk.
        comparison = (
            EVIDENCE
            / "factory"
            / ("cyber-tiel" if prefix else "tiel")
            / f"{workload}-mtp_ab_comparison.json"
        )
        assert hashlib.sha256(comparison.read_bytes()).hexdigest() == (profile["comparison_sha256"])


@pytest.mark.parametrize("prefix", ["", "cyber-"])
def test_tier2_artifact_matches_checkpoint_tier1(prefix: str) -> None:
    cert = _load(prefix)
    tier1 = load_public_checkpoint_certification(
        CERTS / f"{prefix}tiel-coder-35b-axq-mxfp4-mtp-redo-tier1.json"
    )
    assert cert.artifact.hub_repo_id == tier1.artifact.hub_repo_id
    assert cert.artifact.hub_commit == tier1.artifact.hub_commit
    assert cert.artifact.candidate_manifest_sha256 == tier1.artifact.candidate_manifest_sha256
    assert cert.related_certificates is not None
    assert cert.related_certificates["checkpoint_tier1"] == (
        f"{prefix}tiel-coder-35b-axq-mxfp4-mtp-redo-tier1.json"
    )
    # The Tier 1 record keeps its not-certified MTP block; no pointer upgrade.
    assert tier1.mtp_acceleration.status == "not-certified"

    # The evidence package directory exists with the summary and both suites.
    package = CERTS / cert.evidence_package
    assert (package / "TIER2_TECHNICAL_SUMMARY.json").is_file()
    assert (package / "agent-coding-mtp_ab_comparison.json").is_file()
    assert (package / "general-long-mtp_ab_comparison.json").is_file()


def test_m5_comparison_evidence_present_and_labeled_development() -> None:
    comparison = json.loads(
        (EVIDENCE / "comparison-df-macbookpro-m5" / "host-comparison.json").read_text()
    )
    assert comparison["schema_version"] == "axquant.tiel-tier2-host-comparison.v1"
    assert "same packs" in comparison["summary"]
    assert "engine_dev_build_finding" in comparison

    for pack in ("tiel", "cyber-tiel"):
        summary = json.loads(
            (
                EVIDENCE
                / "comparison-df-macbookpro-m5"
                / "engine-754"
                / pack
                / "TIER2_TECHNICAL_SUMMARY.json"
            ).read_text()
        )
        assert summary["host_id"] == "df-macbookpro-m5"
        assert summary["hostname"] == "df-macbookpro-m5"
        # MTP genuinely engaged on the comparison host (unlike the dev-build round).
        assert summary["technical_tier2_pass"] is False

    # Dev-build round: MTP never became available (engine-side gate finding).
    raw = json.loads(
        (
            EVIDENCE
            / "comparison-df-macbookpro-m5"
            / "engine-dev-build"
            / "cyber-tiel-agent-coding-mtp-on-rawlog.json"
        ).read_text()
    )
    mtp = raw["trials"][1]["backend_report"]["performance"]["mtp"]
    assert mtp["requested"] is True
    assert mtp["available"] is False
    assert mtp["active"] is False
