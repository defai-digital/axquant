"""Holo-3.1-35B-A3B public records bind measured gates honestly."""

from __future__ import annotations

from pathlib import Path

from axquant.schema.public_certification import load_public_checkpoint_certification

_CERT_DIR = Path(__file__).resolve().parents[1] / "docs" / "certifications"


def test_holo31_mxfp4_is_certified() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "holo31-35b-axq-mxfp4-tier1.json")
    assert cert.status == "certified"
    assert cert.host_id == "df-macstudio-m2"
    assert cert.artifact.hub_repo_id == "AutomatosX/AX-Holo-3.1-35B-A3B-MLX-AXQ-MXFP4"
    assert cert.artifact.source_model_id == "Hcompany/Holo-3.1-35B-A3B"
    assert cert.artifact.source_revision == "2bdb92851a8cd9d72cdd891fdf38cfcc7fefae2c"
    assert cert.artifact.hub_commit == "23aa374ff6a70f740b5992c80e6d3e2405d8a324"
    assert cert.public_index.listed is True
    assert cert.mtp_acceleration.status == "not-applicable"
    min_ret = float(cert.thresholds["minimum_quality_retention"])
    for suite in ("agent-coding", "general"):
        assert float(cert.quality[suite]["retention"]) >= min_ret, suite


def test_holo31_6_and_8bit_stay_unlisted() -> None:
    for name, repo, commit in (
        (
            "holo31-35b-axq6-tier1.json",
            "AutomatosX/AX-Holo-3.1-35B-A3B-MLX-AXQ-6bit",
            "344d66ed45af48e1de1a85a7590292a338e599b5",
        ),
        (
            "holo31-35b-axq8-tier1.json",
            "AutomatosX/AX-Holo-3.1-35B-A3B-MLX-AXQ-8bit",
            "4b1284784455e915adc08a9b67d414d781067bdf",
        ),
    ):
        cert = load_public_checkpoint_certification(_CERT_DIR / name)
        assert cert.status == "not_certified", name
        assert cert.public_index.listed is False, name
        assert cert.artifact.hub_repo_id == repo
        assert cert.artifact.hub_commit == commit
        assert float(cert.quality["general"]["retention"]) < 0.98
