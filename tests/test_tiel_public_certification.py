"""Tiel certification records bind runtime evidence without promoting its scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axquant.schema.public_certification import load_public_checkpoint_certification

CERTS = Path(__file__).resolve().parents[1] / "docs" / "certifications"


@pytest.mark.parametrize("prefix", ["", "cyber-"])
def test_tiel_runtime_evidence_is_bound_and_not_quality_or_speed_certified(prefix: str) -> None:
    cert = load_public_checkpoint_certification(
        CERTS / f"{prefix}tiel-coder-35b-axq-mxfp4-mtp-tier1.json"
    )
    assert cert.status == "not_certified"
    assert cert.certified_at is None
    assert cert.mtp_acceleration.status == "not-certified"
    assert cert.quality["status"] == "not_assessed"
    assert cert.size["pass"] is None
    assert cert.runtime_default is not None
    assert cert.runtime_default["promoted"] is False
    assert cert.runtime is not None
    runtime = cert.runtime["ax_engine"]
    for gate in ("mtp_s", "mtp_p", "mtp_d"):
        assert runtime[gate] == "not_assessed"
    assert cert.evidence_hashes
    for relative, expected_sha in cert.evidence_hashes.items():
        assert hashlib.sha256((CERTS / relative).read_bytes()).hexdigest() == expected_sha

    integrity_path = next(p for p in cert.evidence_hashes if p.endswith("artifact-integrity.json"))
    integrity = json.loads((CERTS / integrity_path).read_text())
    model = next(m for m in integrity["models"] if m["repo_id"] == cert.artifact.hub_repo_id)
    assert model["revision"] == cert.artifact.hub_commit
    assert model["manifest_sha256"] == cert.artifact.candidate_manifest_sha256
    assert model["passed"] and len(model["files"]) == 20
    assert all(file["passed"] for file in model["files"])
    manifest_path = next(p for p in cert.evidence_hashes if p.endswith("-axquant-manifest.json"))
    manifest_bytes = (CERTS / manifest_path).read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == model["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    observed = {f["path"]: (f["sha256"], f["size_bytes"]) for f in model["files"]}
    expected = {f["path"]: (f["sha256"], f["size_bytes"]) for f in manifest["files"]}
    assert observed == expected
    assert cert.artifact.source_revision == manifest["source_model"]["revision"]
    assert cert.artifact.source_model_id == manifest["source_model"]["model_id"]
    assert integrity["binaries"] == cert.toolchain["binary_sha256"]
    assert cert.mtp_acceleration.engine_binary_sha256 == integrity["binaries"]["ax-engine-server"]

    for field in ("bench_evidence", "server_evidence"):
        assert runtime[field] in cert.evidence_hashes
        response = json.loads((CERTS / runtime[field]).read_text())
        assert response["status"] == "finished"
        assert response["prompt_tokens"] == list(range(1, 17))
        assert len(response["output_tokens"]) == 32
        assert response["runtime"]["host"]["detected_soc"] == "Apple M5 Max"
        counters = response["route"]["crossover_decisions"]
        for counter in (
            "ax_mtp_requested",
            "ax_mtp_available",
            "ax_mlx_mtp_model_policy_active",
            "ax_mtp_draft_tokens",
            "ax_mtp_verify_tokens",
            "ax_mtp_accepted_tokens",
        ):
            assert counters[counter] > 0, counter
        assert counters["ax_mtp_direct_fallback_steps"] == 0
