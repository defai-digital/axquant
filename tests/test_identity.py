from __future__ import annotations

import pytest

from axquant.errors import ArtifactError
from axquant.identity import (
    checkpoint_key_from_source_manifest,
    model_identity_key,
    same_model_identity,
    semantic_artifact_manifest_sha256,
    semantic_model_identity,
)
from axquant.schema import (
    ArtifactFile,
    CandidateKey,
    CheckpointKey,
    ModelIdentity,
    SourceCheckpointFile,
    SourceCheckpointManifest,
)
from axquant.serde import stable_sha256

_REVISION = "a" * 40
_SHA = "b" * 64


def _source_manifest(*, local_path: str) -> SourceCheckpointManifest:
    return SourceCheckpointManifest(
        source_model=ModelIdentity(
            model_id="Qwen/Qwen3.6-27B",
            revision=_REVISION,
            architecture="Qwen3_5ForConditionalGeneration",
            local_path=local_path,
        ),
        config_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        files=[
            SourceCheckpointFile(
                path="config.json",
                size_bytes=1,
                sha256="c" * 64,
            ),
            SourceCheckpointFile(
                path="model.safetensors.index.json",
                size_bytes=2,
                sha256="e" * 64,
            ),
            SourceCheckpointFile(
                path="model-00001-of-00001.safetensors",
                size_bytes=3,
                sha256="f" * 64,
            ),
        ],
    )


def test_model_identity_is_path_neutral() -> None:
    left = ModelIdentity(
        model_id="org/model",
        revision=_REVISION,
        architecture="Fixture",
        local_path="/Volumes/one/model",
    )
    right = left.model_copy(update={"local_path": "/Volumes/two/model"})

    assert same_model_identity(left, right)
    assert model_identity_key(left) == model_identity_key(right)
    assert semantic_model_identity(left).local_path is None


def test_model_identity_still_detects_semantic_drift() -> None:
    model = ModelIdentity(model_id="org/model", revision=_REVISION)
    assert not same_model_identity(
        model,
        model.model_copy(update={"revision": "c" * 40}),
    )


def test_checkpoint_key_is_stable_across_hosts() -> None:
    left = checkpoint_key_from_source_manifest(_source_manifest(local_path="/host-a/model"))
    right = checkpoint_key_from_source_manifest(_source_manifest(local_path="/host-b/model"))

    assert left == right
    assert left.model.local_path is None


def test_checkpoint_key_requires_safetensors_index() -> None:
    manifest = _source_manifest(local_path="/model")
    manifest.files = [record for record in manifest.files if not record.path.endswith("index.json")]
    with pytest.raises(ArtifactError, match="omits the Safetensors index"):
        checkpoint_key_from_source_manifest(manifest)


def test_checkpoint_key_changes_when_member_bytes_change_at_same_path() -> None:
    first = checkpoint_key_from_source_manifest(_source_manifest(local_path="/model"))
    changed = _source_manifest(local_path="/model")
    changed.files[-1].sha256 = "1" * 64
    second = checkpoint_key_from_source_manifest(changed)

    assert first != second
    assert first.checkpoint_members_sha256 != second.checkpoint_members_sha256


@pytest.mark.parametrize(
    "field",
    [
        "certification_policy_sha256",
        "calibration_sha256",
        "activation_capture_sha256",
        "sensitivity_sha256",
        "plan_sha256",
        "artifact_manifest_sha256",
        "checkpoint_members_sha256",
    ],
)
def test_candidate_key_changes_for_every_semantic_input(field: str) -> None:
    source = CheckpointKey(
        model=ModelIdentity(model_id="org/model", revision=_REVISION),
        config_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        weight_index_sha256="3" * 64,
        checkpoint_members_sha256="4" * 64,
    )
    candidate = CandidateKey(
        source=source,
        certification_policy_sha256="5" * 64,
        calibration_sha256="6" * 64,
        activation_capture_sha256="7" * 64,
        sensitivity_sha256="8" * 64,
        plan_sha256="9" * 64,
        artifact_manifest_sha256="a" * 64,
        checkpoint_members_sha256="b" * 64,
    )
    changed = candidate.model_copy(update={field: "f" * 64})

    assert stable_sha256(candidate) != stable_sha256(changed)


def test_artifact_semantic_digest_ignores_non_checkpoint_records() -> None:
    # The manifest constructor is intentionally avoided here; this test isolates
    # the record-selection invariant without duplicating the large artifact fixture.
    class Manifest:
        def __init__(self) -> None:
            self.source_model = ModelIdentity(
                model_id="org/model",
                revision=_REVISION,
                local_path="/host/model",
            )
            self.files = [
                ArtifactFile(path="model.safetensors", size_bytes=10, sha256=_SHA),
                ArtifactFile(path="README.md", size_bytes=20, sha256="c" * 64),
            ]

        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "source_model": self.source_model.model_dump(mode="json"),
                "plan_sha256": "d" * 64,
                "files": [record.model_dump(mode="json") for record in self.files],
            }

    manifest = Manifest()
    first = semantic_artifact_manifest_sha256(manifest, plan_sha256="e" * 64)  # type: ignore[arg-type]
    manifest.source_model.local_path = "/another-host/model"
    manifest.files[1] = ArtifactFile(path="README.md", size_bytes=21, sha256="f" * 64)
    second = semantic_artifact_manifest_sha256(manifest, plan_sha256="e" * 64)  # type: ignore[arg-type]
    assert first == second
