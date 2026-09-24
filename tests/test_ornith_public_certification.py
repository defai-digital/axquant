"""Ornith-1.0-35B records stay honest after the 2026-09-19 repo withdrawal."""

# The Hub repositories were deleted in the AutomatosX catalog cleanup;
# the records remain as unlisted historical evidence of the factory
# measurement and must not render in the public index.

from __future__ import annotations

from pathlib import Path

from axquant.schema.public_certification import load_public_checkpoint_certification

_ROOT = Path(__file__).resolve().parents[1]
_CERT_DIR = _ROOT / "docs" / "certifications"


def test_ornith_6bit_is_certified_with_passing_gates() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "ornith-35b-axq6-tier1.json")
    assert cert.status == "certified"
    assert cert.host_id == "df-macstudio-m2"
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit"
    assert cert.artifact.source_model_id == "deepreinforce-ai/Ornith-1.0-35B"
    assert cert.artifact.source_revision == "5df2ed3f675c7beaa490328cc70bb573b65fb660"
    assert cert.mtp_acceleration.status == "not-applicable"
    assert cert.public_index.listed is False

    size = cert.size
    assert float(size["size_ratio_vs_uniform"]) <= float(size["max_size_ratio_applied"])
    assert size["pass"] is True

    quality = cert.quality
    min_ret = float(cert.thresholds["minimum_quality_retention"])
    for suite in ("agent-coding", "general"):
        retention = float(quality[suite]["retention"])
        assert retention >= min_ret, suite
    assert "Qwen3.6" not in cert.artifact.hub_repo_id
    assert cert.artifact.source_model_id.startswith("deepreinforce-ai/")


def test_ornith_4bit_is_certified_architecture_prior() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "ornith-35b-axq4-tier1.json")
    assert cert.status == "certified"
    assert cert.public_index.listed is False
    assert cert.mtp_acceleration.status == "not-applicable"
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit"
    assert cert.plan.get("recipe") is None

    quality = cert.quality
    min_ret = float(cert.thresholds["minimum_quality_retention"])
    assert float(quality["agent-coding"]["retention"]) >= min_ret
    assert float(quality["general"]["retention"]) >= min_ret

    size = cert.size
    assert float(size["size_ratio_vs_uniform"]) <= float(size["max_size_ratio_applied"])
    assert size["pass"] is True
