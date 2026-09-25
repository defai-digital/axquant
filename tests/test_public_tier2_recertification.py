"""Tier 2 recertification record (axquant.public-tier2-recert.v1) and verification rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from axquant.certification.verify import verify_tier2_recertification
from axquant.public_cert_index import load_public_cert_rows, load_tier2_recertifications
from axquant.schema import (
    PublicCertArtifact,
    PublicMtpAccelerationCertification,
    PublicTier2Recertification,
    load_public_tier2_recertification,
)
from axquant.serde import file_sha256

_ROOT = Path(__file__).resolve().parents[1]
_CERT_DIR = _ROOT / "docs" / "certifications"


def _original_cert() -> PublicMtpAccelerationCertification:
    return PublicMtpAccelerationCertification(
        status="certified",
        host_id="df-macstudio-m2",
        certified_at="2026-08-14T12:00:00+00:00",
        artifact=PublicCertArtifact(
            hub_repo_id="owner/AX-Tiny-4B-MLX-AXQ-4bit-MTP",
            hub_commit="c" * 40,
            product_class="4bit",
            candidate_manifest_sha256="d" * 64,
        ),
        thresholds={
            "exactness_required": 1.0,
            "prompt_median_speedup_min": 1.1,
            "token_weighted_decode_speedup_min": 1.2,
        },
        mtp_acceleration={
            "status": "certified",
            "scope": "decode-heavy-authorizing-profiles",
            "ax_engine_version": "7.1.0",
            "profiles": {
                "agent-coding": {"gate_pass": True},
                "general-long": {"gate_pass": True},
            },
        },
        toolchain={"ax_engine": "7.1.0", "axquant": "1.9.0"},
    )


def _recert_payload() -> dict[str, object]:
    return {
        "schema_version": "axquant.public-tier2-recert.v1",
        "original_tier2_sha256": "e" * 64,
        "hub_commit": "c" * 40,
        "candidate_manifest_sha256": "d" * 64,
        "host_id": "df-macstudio-m2",
        "artifact": {
            "hub_repo_id": "owner/AX-Tiny-4B-MLX-AXQ-4bit-MTP",
            "hub_commit": "c" * 40,
            "product_class": "4bit",
            "candidate_manifest_sha256": "d" * 64,
        },
        "ax_engine_version": "7.5.4",
        "exactness_pass": True,
        "mtp_acceleration": {
            "profiles": {
                "agent-coding": {
                    "exactness_pass": True,
                    "prompt_median_speedup": 1.25,
                    "token_weighted_decode_speedup": 1.30,
                },
                "general-long": {
                    "exactness_pass": True,
                    "prompt_median_speedup": 1.15,
                    "token_weighted_decode_speedup": 1.22,
                },
            }
        },
        "toolchain": {"ax_engine": "7.5.4", "axquant": "1.9.0"},
        "created_at": "2026-09-24T12:00:00+00:00",
    }


def _verify(
    recert: PublicTier2Recertification,
    original: PublicMtpAccelerationCertification | None = None,
    original_sha256: str = "e" * 64,
) -> list[str]:
    return verify_tier2_recertification(
        recertification=recert,
        original_cert=original if original is not None else _original_cert(),
        original_sha256=original_sha256,
    )


def test_schema_round_trip_and_unknown_field_rejected() -> None:
    payload = _recert_payload()
    recert = PublicTier2Recertification.model_validate(payload)
    loaded = PublicTier2Recertification.model_validate_json(recert.model_dump_json())
    assert loaded.schema_version == "axquant.public-tier2-recert.v1"
    assert loaded.ax_engine_version == "7.5.4"

    payload["unexpected_envelope_field"] = True
    with pytest.raises(ValidationError):
        PublicTier2Recertification.model_validate(payload)


def test_schema_rejects_missing_profiles() -> None:
    payload = _recert_payload()
    payload["mtp_acceleration"] = {}
    with pytest.raises(ValidationError, match="profiles"):
        PublicTier2Recertification.model_validate(payload)


def test_schema_rejects_bad_original_hash() -> None:
    payload = _recert_payload()
    payload["original_tier2_sha256"] = "not-a-sha"
    with pytest.raises(ValidationError):
        PublicTier2Recertification.model_validate(payload)


def test_valid_recertification_passes() -> None:
    recert = PublicTier2Recertification.model_validate(_recert_payload())
    assert _verify(recert) == []


def test_non_certified_original_is_rejected() -> None:
    recert = PublicTier2Recertification.model_validate(_recert_payload())
    original = _original_cert().model_copy(update={"status": "not_certified"})
    issues = _verify(recert, original)
    assert any("not certified" in issue for issue in issues)


def test_original_hash_mismatch_is_rejected() -> None:
    recert = PublicTier2Recertification.model_validate(_recert_payload())
    issues = _verify(recert, original_sha256="f" * 64)
    assert any("byte hash" in issue for issue in issues)


@pytest.mark.parametrize(
    "field",
    ["hub_commit", "candidate_manifest_sha256"],
)
def test_identity_field_mismatch_is_rejected(field: str) -> None:
    payload = _recert_payload()
    payload[field] = "0" * len(payload[field])  # type: ignore[arg-type]
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any(field in issue for issue in issues)


def test_artifact_mismatch_is_rejected() -> None:
    payload = _recert_payload()
    artifact = dict(payload["artifact"])  # type: ignore[arg-type]
    artifact["product_class"] = "6bit"
    payload["artifact"] = artifact
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("artifact identity" in issue for issue in issues)


def test_same_engine_version_is_rejected() -> None:
    payload = _recert_payload()
    payload["ax_engine_version"] = "7.1.0"
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("ax_engine_version" in issue for issue in issues)


def test_failed_exactness_is_rejected() -> None:
    payload = _recert_payload()
    payload["exactness_pass"] = False
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("exactness_pass is false" in issue for issue in issues)


def test_profile_exactness_failure_is_rejected() -> None:
    payload = _recert_payload()
    profiles = dict(payload["mtp_acceleration"]["profiles"])  # type: ignore[index]
    agent = dict(profiles["agent-coding"])  # type: ignore[index]
    agent["exactness_pass"] = False
    profiles["agent-coding"] = agent
    payload["mtp_acceleration"]["profiles"] = profiles  # type: ignore[index]
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("profile agent-coding exactness_pass" in issue for issue in issues)


def test_speedup_below_original_threshold_is_rejected() -> None:
    payload = _recert_payload()
    profiles = dict(payload["mtp_acceleration"]["profiles"])  # type: ignore[index]
    general = dict(profiles["general-long"])  # type: ignore[index]
    general["token_weighted_decode_speedup"] = 1.19
    profiles["general-long"] = general
    payload["mtp_acceleration"]["profiles"] = profiles  # type: ignore[index]
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("token_weighted_decode_speedup" in issue for issue in issues)


def test_missing_authorizing_profile_is_rejected() -> None:
    payload = _recert_payload()
    profiles = dict(payload["mtp_acceleration"]["profiles"])  # type: ignore[index]
    del profiles["general-long"]
    payload["mtp_acceleration"]["profiles"] = profiles  # type: ignore[index]
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("general-long" in issue for issue in issues)


def test_unauthorized_host_is_rejected() -> None:
    payload = _recert_payload()
    payload["host_id"] = "df-macbookpro-m3"
    recert = PublicTier2Recertification.model_validate(payload)
    issues = _verify(recert)
    assert any("host_id" in issue for issue in issues)


def test_documented_large_memory_exception_host_is_accepted() -> None:
    payload = _recert_payload()
    payload["host_id"] = "tn-macstudio-m3"
    recert = PublicTier2Recertification.model_validate(payload)
    assert _verify(recert) == []


def _write_recert_file(directory: Path, record_id: str, payload: dict[str, object]) -> Path:
    path = directory / f"{record_id}-tier2-recert.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _index_fixture(tmp_path: Path) -> tuple[str, dict[str, object]]:
    stem = "qwen36-27b-axq6"
    for suffix in ("tier1.json", "tier1.md", "tier2.json", "tier2.md"):
        source = _CERT_DIR / f"{stem}-{suffix}"
        (tmp_path / source.name).write_bytes(source.read_bytes())
    tier2_path = tmp_path / f"{stem}-tier2.json"
    original = json.loads(tier2_path.read_text(encoding="utf-8"))
    payload = _recert_payload()
    payload["original_tier2_sha256"] = file_sha256(tier2_path)
    payload["hub_commit"] = original["artifact"]["hub_commit"]
    payload["candidate_manifest_sha256"] = original["artifact"]["candidate_manifest_sha256"]
    payload["artifact"] = original["artifact"]
    payload["ax_engine_version"] = "7.5.4"
    payload["exactness_pass"] = True
    payload["mtp_acceleration"] = {"profiles": {}}
    for name, profile in original["mtp_acceleration"]["profiles"].items():
        assert profile["exactness_pass"] is True
        payload["mtp_acceleration"]["profiles"][name] = {  # type: ignore[index]
            "exactness_pass": True,
            "prompt_median_speedup": profile["prompt_median_speedup"] + 0.01,
            "token_weighted_decode_speedup": profile["token_weighted_decode_speedup"] + 0.01,
        }
    return stem, payload


def test_index_lists_verified_recertification(tmp_path: Path) -> None:
    stem, payload = _index_fixture(tmp_path)
    recert_path = _write_recert_file(tmp_path, stem, payload)

    grouped = load_tier2_recertifications(tmp_path)
    assert len(grouped[stem]) == 1
    assert grouped[stem][0].ax_engine_version == "7.5.4"

    rows = load_public_cert_rows(tmp_path, listed_only=False)
    row = next(item for item in rows if item.record_id == stem)
    assert len(row.tier2_recerts) == 1
    assert row.tier2_recerts[0].path == recert_path
    assert row.tier2_recerts[0].original_tier2_sha256 == payload["original_tier2_sha256"]


def test_index_rejects_tampered_recertification(tmp_path: Path) -> None:
    stem, payload = _index_fixture(tmp_path)
    original = json.loads((tmp_path / f"{stem}-tier2.json").read_text(encoding="utf-8"))
    payload["ax_engine_version"] = original["mtp_acceleration"]["ax_engine_version"]
    _write_recert_file(tmp_path, stem, payload)

    with pytest.raises(ValueError, match="recertification verification failed"):
        load_tier2_recertifications(tmp_path)
    with pytest.raises(ValueError, match="recertification verification failed"):
        load_public_cert_rows(tmp_path, listed_only=False)


def test_index_rejects_recertification_without_original(tmp_path: Path) -> None:
    stem, payload = _index_fixture(tmp_path)
    (tmp_path / f"{stem}-tier2.json").unlink()
    _write_recert_file(tmp_path, stem, payload)

    with pytest.raises(ValueError, match="missing original Tier 2 certificate"):
        load_tier2_recertifications(tmp_path)


def test_loader_function_round_trips(tmp_path: Path) -> None:
    stem, payload = _index_fixture(tmp_path)
    recert_path = _write_recert_file(tmp_path, stem, payload)
    loaded = load_public_tier2_recertification(recert_path)
    assert loaded.schema_version == "axquant.public-tier2-recert.v1"
    assert loaded.original_tier2_sha256 == payload["original_tier2_sha256"]
