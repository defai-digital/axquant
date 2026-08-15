"""Capability-gated multimodal certification policy (1.8.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from axquant.modality_certification import (
    build_modalities_block,
    claim_allows_public_quality,
    claim_allows_public_smoke,
    derive_modality_claim,
    format_modalities_card_section,
    inspect_artifact_modalities,
    inspect_hub_listing,
    summarize_modalities_for_markdown,
    validate_modality_evidence_consistency,
)
from axquant.schema.public_certification import (
    PublicModalityClaim,
    load_public_checkpoint_certification,
)

_ROOT = Path(__file__).resolve().parents[1]
_CERT_DIR = _ROOT / "docs" / "certifications"


def test_unsupported_disables_modality() -> None:
    claim = derive_modality_claim(supported=False, modality="vision")
    assert claim.status == "not-applicable"
    assert claim.supported is False
    assert not claim_allows_public_smoke(claim.status)
    assert not claim_allows_public_quality(claim.status)


def test_supported_without_evidence_is_present_not_certified() -> None:
    claim = derive_modality_claim(supported=True, modality="vision")
    assert claim.status == "present-not-certified"
    assert claim.supported is True
    assert not claim_allows_public_quality(claim.status)


def test_smoke_outranks_present() -> None:
    claim = derive_modality_claim(
        supported=True,
        smoke_passed=True,
        modality="vision",
        runtime="mlx-vlm",
    )
    assert claim.status == "smoke-certified"
    assert claim.evidence_kind == "runtime-smoke-mlx-vlm"
    assert claim_allows_public_smoke(claim.status)
    assert not claim_allows_public_quality(claim.status)


def test_quality_outranks_smoke() -> None:
    claim = derive_modality_claim(
        supported=True,
        smoke_passed=True,
        quality_passed=True,
        modality="audio",
    )
    assert claim.status == "quality-certified"
    assert claim.evidence_kind == "multimodal-quality-audio"
    assert claim_allows_public_quality(claim.status)


def test_unsupported_ignores_smoke_flags() -> None:
    claim = derive_modality_claim(
        supported=False,
        smoke_passed=True,
        quality_passed=True,
        modality="vision",
    )
    assert claim.status == "not-applicable"


def test_build_modalities_block_and_consistency() -> None:
    block = build_modalities_block(
        vision_supported=True,
        audio_supported=False,
        vision_smoke_passed=True,
    )
    assert block.policy == "capability-gated-v1"
    assert block.vision.status == "smoke-certified"
    assert block.audio.status == "not-applicable"
    assert not validate_modality_evidence_consistency(
        block,
        vision_supported=True,
        audio_supported=False,
    )
    assert validate_modality_evidence_consistency(
        block,
        vision_supported=False,
        audio_supported=False,
    )


def test_claim_rejects_incoherent_support_status() -> None:
    with pytest.raises(ValidationError):
        PublicModalityClaim(status="not-applicable", supported=True)
    with pytest.raises(ValidationError):
        PublicModalityClaim(status="present-not-certified", supported=False)


def test_smoke_requires_evidence_kind() -> None:
    with pytest.raises(ValidationError):
        PublicModalityClaim(status="smoke-certified", supported=True)


def test_all_in_tree_tier1_certs_have_valid_modalities() -> None:
    paths = sorted(_CERT_DIR.glob("*-tier1.json"))
    assert paths
    for path in paths:
        cert = load_public_checkpoint_certification(path)
        assert cert.modalities is not None, path.name
        assert cert.modalities.policy == "capability-gated-v1"
        # Smoke/quality must carry evidence_kind (enforced by model).
        for claim in (cert.modalities.vision, cert.modalities.audio):
            if claim.status in {"smoke-certified", "quality-certified"}:
                assert claim.evidence_kind


def test_qwen3_vl_is_vision_smoke_audio_na() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "qwen3-vl-30b-axq4-tier1.json")
    assert cert.modalities is not None
    assert cert.modalities.vision.status == "smoke-certified"
    assert cert.modalities.audio.status == "not-applicable"


def test_gemma4_hub_vision_sidecar_is_not_disabled() -> None:
    cert = load_public_checkpoint_certification(_CERT_DIR / "gemma4-12b-axq4-tier1.json")
    assert cert.modalities is not None
    assert cert.modalities.vision.supported is True
    assert cert.modalities.vision.status != "not-applicable"
    assert cert.modalities.audio.status == "not-applicable"
    text = format_modalities_card_section(cert.modalities)
    assert "capability-gated" in text
    assert "not a quality pass" in text


def test_certified_non_deepseek_support_matches_status() -> None:
    for path in sorted(_CERT_DIR.glob("*-tier1.json")):
        cert = load_public_checkpoint_certification(path)
        if cert.status != "certified" or cert.modalities is None:
            continue
        for claim in (cert.modalities.vision, cert.modalities.audio):
            if claim.status == "not-applicable":
                assert claim.supported is False, path.name
            else:
                assert claim.supported is True, path.name


def test_inspect_treats_sidecar_and_config_as_supported(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "demo", "vision_config": {"depth": 1}, "audio_token_id": 7}),
        encoding="utf-8",
    )
    (tmp_path / "vision.safetensors").write_bytes(b"not-a-real-sidecar")
    inspect = inspect_artifact_modalities(tmp_path)
    assert inspect.vision_supported
    assert inspect.vision_declared
    assert inspect.vision_weight_files == ("vision.safetensors",)
    assert not inspect.audio_supported


def test_inspect_hub_listing_audio_config_without_sidecar() -> None:
    inspect = inspect_hub_listing(
        filenames=("config.json", "model-00001-of-00002.safetensors"),
        config={"audio_config": {"hidden_size": 8}},
    )
    assert inspect.audio_supported
    assert not inspect.vision_supported


def test_format_modalities_card_rejects_quality_wording() -> None:
    block = build_modalities_block(
        vision_supported=True,
        audio_supported=False,
        vision_smoke_passed=False,
    )
    text = format_modalities_card_section(block)
    assert "capability-gated" in text
    assert "not a quality pass" in text
    assert "`present-not-certified`" in text
    assert "`not-applicable`" in text


def test_legacy_cert_without_modalities_still_loads(tmp_path: Path) -> None:
    source = _CERT_DIR / "gpt-oss-20b-axq4-tier1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data.pop("modalities", None)
    path = tmp_path / "legacy-tier1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cert = load_public_checkpoint_certification(path)
    assert cert.modalities is None
    assert "legacy" in summarize_modalities_for_markdown(None).lower()
