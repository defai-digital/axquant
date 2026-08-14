"""Holo3-35B-A3B public checkpoint records bind measured gates honestly."""

from __future__ import annotations

from pathlib import Path

from axquant.schema.public_certification import load_public_checkpoint_certification

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


def test_holo3_mtp_product_certs_are_withdrawn() -> None:
    """-MTP public certificates must not ship after product withdrawal."""
    for name in (
        "holo3-35b-axq4-mtp-tier1.json",
        "holo3-35b-axq6-mtp-tier1.json",
        "holo3-35b-axq4-mtp-tier2.json",
        "holo3-35b-axq6-mtp-tier2.json",
    ):
        assert not (_CERT_DIR / name).exists(), name
    # Public matrices must not list Holo3 -MTP SKUs.
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    matrix = (_ROOT / "docs/releases/certification-matrix.md").read_text(encoding="utf-8")
    assert "AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP" not in readme
    assert "AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP" not in readme
    assert "Holo3-35B-A3B AXQ 4-bit-MTP" not in matrix
    assert "Holo3-35B-A3B AXQ 6-bit-MTP" not in matrix
