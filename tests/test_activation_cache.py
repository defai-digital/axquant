"""Tests for the tokenized calibration and activation cache (v0.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from axquant.activation_cache import (
    compute_cache_key,
    is_cache_complete,
    load_cache_manifest,
    tokenize_calibration,
    verify_cache_integrity,
)
from axquant.errors import CacheError
from axquant.schema import ModelIdentity, ProfileName, SoftwareVersions, TokenizedCacheManifest
from axquant.serde import stable_sha256, write_data


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    special_tokens_map: ClassVar[dict[str, str]] = {
        "eos_token": "</s>",
        "pad_token": "<pad>",
    }

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "def": 1, "</s>": 2}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> list[int]:
        assert add_special_tokens
        assert truncation
        return ([len(word) % 97 + 3 for word in text.split()] + [self.eos_token_id])[:max_length]


@pytest.fixture
def calibration_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "calibration.jsonl"
    lines = [
        json.dumps({"text": "def sort_list(items): return sorted(items)"}),
        json.dumps({"text": "Fix the bug in this function"}),
        json.dumps({"text": "Generate a JSON response"}),
    ]
    dataset.write_text("\n".join(lines), encoding="utf-8")
    return dataset


@pytest.fixture
def model_identity() -> ModelIdentity:
    return ModelIdentity(model_id="test-model", revision="rev123")


class TestCacheKeyDeterminism:
    def test_same_inputs_same_key(self, model_identity: ModelIdentity) -> None:
        key1 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=0,
        )
        key2 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=0,
        )
        assert key1 == key2

    def test_different_seed_different_key(self, model_identity: ModelIdentity) -> None:
        key1 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=0,
        )
        key2 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=1,
        )
        assert key1 != key2

    def test_different_dataset_different_key(self, model_identity: ModelIdentity) -> None:
        key1 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=0,
        )
        key2 = compute_cache_key(
            model=model_identity,
            dataset_sha256="def456",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=0,
        )
        assert key1 != key2

    def test_different_profile_different_key(self, model_identity: ModelIdentity) -> None:
        key1 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.AGENT_CODING,
            sequence_length=2048,
            random_seed=0,
        )
        key2 = compute_cache_key(
            model=model_identity,
            dataset_sha256="abc123",
            profile=ProfileName.GENERAL,
            sequence_length=2048,
            random_seed=0,
        )
        assert key1 != key2


@pytest.mark.parametrize(("field", "value"), [("shard_count", 0), ("total_tokens", 0)])
def test_tokenized_cache_manifest_requires_nonempty_payload(
    model_identity: ModelIdentity,
    field: str,
    value: int,
) -> None:
    payload = {
        "cache_key_sha256": "a" * 64,
        "model": model_identity,
        "dataset_sha256": "b" * 64,
        "profile": ProfileName.AGENT_CODING,
        "sequence_length": 8,
        "samples": 1,
        "shard_count": 1,
        "total_tokens": 1,
        "software_versions": SoftwareVersions(
            axquant="0.1.0",
            python="3.13",
            safetensors="0.5",
            pydantic="2.0",
        ),
    }
    payload[field] = value

    with pytest.raises(ValueError):
        TokenizedCacheManifest.model_validate(payload)


class TestTokenizeCalibration:
    def test_writes_real_token_ids_and_verified_checksums(
        self,
        model_identity: ModelIdentity,
        calibration_dataset: Path,
        tmp_path: Path,
    ) -> None:
        import numpy as np

        output_dir = tmp_path / "cache"
        manifest = tokenize_calibration(
            model=model_identity,
            dataset_path=calibration_dataset,
            output_dir=output_dir,
            profile=ProfileName.AGENT_CODING,
            sequence_length=32,
            random_seed=7,
            tokenizer=_FakeTokenizer(),
        )

        assert manifest.complete
        assert manifest.total_tokens > manifest.samples
        assert manifest.shard_sha256
        assert is_cache_complete(output_dir, manifest)
        assert verify_cache_integrity(output_dir, manifest) == []
        completion = json.loads((output_dir / "completion.json").read_text(encoding="utf-8"))
        assert completion["schema_version"] == "axquant.tokenized-cache-completion.v1"
        assert completion["manifest_sha256"] == stable_sha256(manifest)
        shard = output_dir / "tokenized" / "shard-0000.npz"
        with np.load(shard, allow_pickle=False) as data:
            assert data["input_ids"].shape[0] == 3
            assert data["attention_mask"].sum() == manifest.total_tokens
            assert np.any(data["input_ids"] != 0)

    def test_checksum_tampering_is_detected(
        self,
        model_identity: ModelIdentity,
        calibration_dataset: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "cache"
        manifest = tokenize_calibration(
            model=model_identity,
            dataset_path=calibration_dataset,
            output_dir=output_dir,
            profile=ProfileName.AGENT_CODING,
            sequence_length=32,
            random_seed=0,
            tokenizer=_FakeTokenizer(),
        )
        shard = output_dir / "tokenized" / "shard-0000.npz"
        shard.write_bytes(shard.read_bytes() + b"tampered")

        assert "checksum mismatch: shard-0000.npz" in verify_cache_integrity(output_dir, manifest)

    def test_missing_shard_checksum_binding_is_rejected(
        self,
        model_identity: ModelIdentity,
        calibration_dataset: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "cache"
        manifest = tokenize_calibration(
            model=model_identity,
            dataset_path=calibration_dataset,
            output_dir=output_dir,
            profile=ProfileName.AGENT_CODING,
            sequence_length=32,
            random_seed=0,
            tokenizer=_FakeTokenizer(),
        )
        unbound = manifest.model_copy(update={"shard_sha256": {}})

        assert "missing checksum binding: shard-0000.npz" in verify_cache_integrity(
            output_dir,
            unbound,
        )

    def test_sample_domains_are_observed_release_provenance(
        self,
        model_identity: ModelIdentity,
        tmp_path: Path,
    ) -> None:
        dataset = tmp_path / "domains.jsonl"
        dataset.write_text(
            "\n".join(
                [
                    json.dumps({"text": "repair code", "domain": "coding"}),
                    json.dumps({"text": "emit an object", "domain": "json"}),
                ]
            ),
            encoding="utf-8",
        )
        manifest = tokenize_calibration(
            model=model_identity,
            dataset_path=dataset,
            output_dir=tmp_path / "cache",
            profile=ProfileName.AGENT_CODING,
            sequence_length=32,
            random_seed=0,
            tokenizer=_FakeTokenizer(),
            domains=["coding", "json"],
        )

        assert manifest.domains == ["coding", "json"]
        assert manifest.domain_provenance == "sample-records"

    def test_declared_domain_without_sample_is_rejected(
        self,
        model_identity: ModelIdentity,
        tmp_path: Path,
    ) -> None:
        dataset = tmp_path / "domains.jsonl"
        dataset.write_text(
            json.dumps({"text": "repair code", "domain": "coding"}),
            encoding="utf-8",
        )

        with pytest.raises(CacheError, match="no matching samples"):
            tokenize_calibration(
                model=model_identity,
                dataset_path=dataset,
                output_dir=tmp_path / "cache",
                profile=ProfileName.AGENT_CODING,
                sequence_length=32,
                random_seed=0,
                tokenizer=_FakeTokenizer(),
                domains=["coding", "json"],
            )

    def test_missing_dataset(self, model_identity: ModelIdentity, tmp_path: Path) -> None:
        with pytest.raises(CacheError, match="does not exist"):
            tokenize_calibration(
                model=model_identity,
                dataset_path=tmp_path / "nonexistent.jsonl",
                output_dir=tmp_path / "cache",
                profile=ProfileName.AGENT_CODING,
                sequence_length=512,
                random_seed=0,
            )

    def test_empty_dataset(self, model_identity: ModelIdentity, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(CacheError, match="no samples"):
            tokenize_calibration(
                model=model_identity,
                dataset_path=empty,
                output_dir=tmp_path / "cache",
                profile=ProfileName.AGENT_CODING,
                sequence_length=512,
                random_seed=0,
            )

    def test_changed_inputs_rejected(
        self,
        model_identity: ModelIdentity,
        calibration_dataset: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "cache"
        # First call may fail due to missing transformers, but we test the
        # cache rejection logic by pre-creating a manifest with different key
        output_dir.mkdir(parents=True)
        manifest_data = {
            "schema_version": "axquant.tokenized-cache.v1",
            "cache_key_sha256": "different_key",
            "model": {"model_id": "test-model", "revision": "rev123"},
            "dataset_sha256": "old_sha",
            "profile": "agent-coding",
            "sequence_length": 512,
            "samples": 3,
            "shard_count": 1,
            "total_tokens": 512,
            "software_versions": {
                "axquant": "0.1.0",
                "python": "3.13",
                "safetensors": "0.5",
                "pydantic": "2.0",
            },
            "complete": True,
        }
        (output_dir / "tokenized_cache_manifest.json").write_text(
            json.dumps(manifest_data), encoding="utf-8"
        )
        (output_dir / "completion.json").write_text("{}", encoding="utf-8")

        with pytest.raises(CacheError, match="different inputs"):
            tokenize_calibration(
                model=model_identity,
                dataset_path=calibration_dataset,
                output_dir=output_dir,
                profile=ProfileName.AGENT_CODING,
                sequence_length=512,
                random_seed=0,
            )

    def test_completed_unbound_cache_is_not_reused_for_bound_calibration(
        self,
        model_identity: ModelIdentity,
        calibration_dataset: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "cache"
        first = tokenize_calibration(
            model=model_identity,
            dataset_path=calibration_dataset,
            output_dir=output_dir,
            profile=ProfileName.AGENT_CODING,
            sequence_length=32,
            random_seed=0,
            tokenizer=_FakeTokenizer(),
        )
        assert first.calibration_manifest_sha256 is None

        with pytest.raises(CacheError, match="different calibration manifest binding"):
            tokenize_calibration(
                model=model_identity,
                dataset_path=calibration_dataset,
                output_dir=output_dir,
                profile=ProfileName.AGENT_CODING,
                sequence_length=32,
                random_seed=0,
                tokenizer=_FakeTokenizer(),
                calibration_manifest_sha256="a" * 64,
            )


class TestCacheIntegrity:
    def test_missing_tokenized_dir(self, tmp_path: Path) -> None:
        manifest = TokenizedCacheManifest(
            cache_key_sha256="test",
            model=ModelIdentity(model_id="m"),
            dataset_sha256="sha",
            profile=ProfileName.GENERAL,
            sequence_length=512,
            samples=10,
            shard_count=2,
            total_tokens=5120,
            software_versions=SoftwareVersions(
                axquant="0.1.0",
                python="3.13",
                safetensors="0.5",
                pydantic="2.0",
            ),
        )
        issues = verify_cache_integrity(tmp_path, manifest)
        assert "tokenized directory missing" in issues

    def test_missing_shards(self, tmp_path: Path) -> None:
        tokenized = tmp_path / "tokenized"
        tokenized.mkdir()
        manifest = TokenizedCacheManifest(
            cache_key_sha256="test",
            model=ModelIdentity(model_id="m"),
            dataset_sha256="sha",
            profile=ProfileName.GENERAL,
            sequence_length=512,
            samples=10,
            shard_count=2,
            total_tokens=5120,
            shard_sha256={
                "shard-0000.npz": "a" * 64,
                "shard-0001.npz": "b" * 64,
            },
            software_versions=SoftwareVersions(
                axquant="0.1.0",
                python="3.13",
                safetensors="0.5",
                pydantic="2.0",
            ),
        )
        issues = verify_cache_integrity(tmp_path, manifest)
        assert len(issues) == 2
        assert "missing shard: shard-0000.npz" in issues
        assert "missing shard: shard-0001.npz" in issues


class TestCompletionMarker:
    def test_not_complete_initially(self, tmp_path: Path) -> None:
        assert not is_cache_complete(tmp_path)

    def test_marker_presence_without_manifest_binding_is_not_complete(self, tmp_path: Path) -> None:
        (tmp_path / "completion.json").write_text("{}", encoding="utf-8")
        assert not is_cache_complete(tmp_path)

    def test_legacy_marker_must_match_manifest_fields(self, tmp_path: Path) -> None:
        manifest = TokenizedCacheManifest(
            cache_key_sha256="cache-key",
            model=ModelIdentity(model_id="m"),
            dataset_sha256="dataset",
            profile=ProfileName.GENERAL,
            sequence_length=8,
            samples=1,
            shard_count=1,
            total_tokens=8,
            shard_sha256={"shard-0000.npz": "a" * 64},
            software_versions=SoftwareVersions(
                axquant="0.1.0",
                python="3.13",
                safetensors="0.5",
                pydantic="2.0",
            ),
            complete=True,
        )
        write_data(tmp_path / "tokenized_cache_manifest.json", manifest)
        write_data(
            tmp_path / "completion.json",
            {
                "complete": True,
                "cache_key_sha256": manifest.cache_key_sha256,
                "shard_count": manifest.shard_count,
                "total_tokens": manifest.total_tokens,
            },
        )

        assert is_cache_complete(tmp_path, manifest)

        write_data(
            tmp_path / "completion.json",
            {
                "complete": True,
                "cache_key_sha256": "different",
                "shard_count": manifest.shard_count,
                "total_tokens": manifest.total_tokens,
            },
        )
        assert not is_cache_complete(tmp_path, manifest)

    def test_current_marker_rejects_manifest_digest_drift(self, tmp_path: Path) -> None:
        manifest = TokenizedCacheManifest(
            cache_key_sha256="cache-key",
            model=ModelIdentity(model_id="m"),
            dataset_sha256="dataset",
            profile=ProfileName.GENERAL,
            sequence_length=8,
            samples=1,
            shard_count=1,
            total_tokens=8,
            shard_sha256={"shard-0000.npz": "a" * 64},
            software_versions=SoftwareVersions(
                axquant="0.1.0",
                python="3.13",
                safetensors="0.5",
                pydantic="2.0",
            ),
            complete=True,
        )
        write_data(tmp_path / "tokenized_cache_manifest.json", manifest)
        write_data(
            tmp_path / "completion.json",
            {
                "schema_version": "axquant.tokenized-cache-completion.v1",
                "complete": True,
                "cache_key_sha256": manifest.cache_key_sha256,
                "manifest_sha256": stable_sha256(manifest),
                "shard_count": manifest.shard_count,
                "total_tokens": manifest.total_tokens,
            },
        )

        changed = manifest.model_copy(update={"domains": ["different"]})
        write_data(tmp_path / "tokenized_cache_manifest.json", changed)
        assert not is_cache_complete(tmp_path, changed)


class TestLoadManifest:
    def test_no_manifest(self, tmp_path: Path) -> None:
        assert load_cache_manifest(tmp_path) is None

    def test_invalid_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "tokenized_cache_manifest.json").write_text("not json", encoding="utf-8")
        assert load_cache_manifest(tmp_path) is None
