"""Holo3-35B-A3B public checkpoint records bind measured gates honestly."""

from __future__ import annotations

from pathlib import Path

from axquant.schema.public_certification import (
    load_public_checkpoint_certification,
    load_public_mtp_acceleration_certification,
)

_ROOT = Path(__file__).resolve().parents[1]
_CERT_DIR = _ROOT / "docs" / "certifications"


def test_holo3_6bit_is_certified_with_passing_gates() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "holo3-35b-axq6-tier1.json")
    assert cert.status == "certified"
    assert cert.host_id == "df-macstudio-m2"
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit"
    assert cert.artifact.source_model_id == "Hcompany/Holo3-35B-A3B"
    assert cert.artifact.source_revision == "208d5ae3a03f99d561f32ab5e606f73397a390ea"
    assert cert.mtp_acceleration.status == "not-applicable"
    assert cert.public_index.listed is True

    size = cert.size
    assert float(size["size_ratio_vs_uniform"]) <= float(size["max_size_ratio_applied"])
    assert size["pass"] is True

    quality = cert.quality
    min_ret = float(cert.thresholds["minimum_quality_retention"])
    for suite in ("agent-coding", "general"):
        retention = float(quality[suite]["retention"])
        assert retention >= min_ret, suite
    # Not the official Qwen 3.6 product id
    assert "Qwen3.6" not in cert.artifact.hub_repo_id
    assert cert.artifact.source_model_id.startswith("Hcompany/")


def test_holo3_4bit_is_certified_with_recovery_layout() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "holo3-35b-axq4-tier1.json")
    assert cert.status == "certified"
    assert cert.public_index.listed is True
    assert cert.mtp_acceleration.status == "not-applicable"
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit"
    assert cert.plan.get("recipe") == "examples/holo3-35b-axq4-agent-v0.1.yaml"

    quality = cert.quality
    min_ret = float(cert.thresholds["minimum_quality_retention"])
    assert float(quality["agent-coding"]["retention"]) >= min_ret
    assert float(quality["general"]["retention"]) >= min_ret

    size = cert.size
    assert float(size["size_ratio_vs_uniform"]) <= float(size["max_size_ratio_applied"])
    assert size["pass"] is True


def test_holo3_4bit_mtp_is_certified_with_grafted_sidecar() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "holo3-35b-axq4-mtp-tier1.json")
    assert cert.status == "certified"
    assert cert.public_index.listed is True
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP"
    assert cert.artifact.hub_commit == "c048f577843225ac0545be5674b4d68b9a51dcf0"
    assert cert.mtp_acceleration.status == "not-certified"
    assert "graft" in (cert.mtp_acceleration.reason or "").lower()
    assert cert.plan.get("mtp_donor", "").startswith("Qwen/Qwen3.5-35B-A3B@")
    assert float(cert.quality["agent-coding"]["retention"]) >= float(
        cert.thresholds["minimum_quality_retention"]
    )
    assert float(cert.size["size_ratio_vs_uniform"]) <= float(cert.size["max_size_ratio_applied"])
    assert "Qwen3.6" not in cert.artifact.hub_repo_id


def test_holo3_6bit_mtp_is_certified_with_grafted_sidecar() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "holo3-35b-axq6-mtp-tier1.json")
    assert cert.status == "certified"
    assert cert.public_index.listed is True
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP"
    assert cert.artifact.hub_commit == "f474549461817cafb73909847af43af2431d4a0d"
    assert cert.mtp_acceleration.status == "not-certified"
    assert "not co-trained" in (cert.mtp_acceleration.reason or "").lower()
    assert float(cert.quality["general"]["retention"]) >= float(
        cert.thresholds["minimum_quality_retention"]
    )
    assert float(cert.size["size_ratio_vs_uniform"]) <= float(cert.size["max_size_ratio_applied"])


def test_holo3_mtp_tier2_records_are_not_certified() -> None:
    for name in ("holo3-35b-axq4-mtp-tier2.json", "holo3-35b-axq6-mtp-tier2.json"):
        cert = load_public_mtp_acceleration_certification(_CERT_DIR / name)
        assert cert.status == "not_certified"
        assert cert.certification_tier == "mtp-acceleration"
        assert cert.host_id == "df-macstudio-m2"
        assert "MTP" in cert.artifact.hub_repo_id
        assert cert.mtp_acceleration["status"] == "not_certified"
