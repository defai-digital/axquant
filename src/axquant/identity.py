from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from axquant.errors import ArtifactError
from axquant.schema.artifacts import ArtifactFile, ArtifactManifest
from axquant.schema.certification import SourceCheckpointManifest
from axquant.schema.flagship import CandidateKey, CheckpointKey
from axquant.schema.inventory import ModelIdentity
from axquant.schema.planning import QuantizationPlan
from axquant.serde import stable_sha256

_INDEX_PATH = "model.safetensors.index.json"
_CHECKPOINT_METADATA = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    _INDEX_PATH,
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def semantic_model_identity(identity: ModelIdentity) -> ModelIdentity:
    """Return the model identity used at cross-host evidence boundaries."""

    return identity.model_copy(update={"local_path": None})


def same_model_identity(left: ModelIdentity, right: ModelIdentity) -> bool:
    return semantic_model_identity(left) == semantic_model_identity(right)


def model_identity_key(
    identity: ModelIdentity,
) -> tuple[str, str | None, str, str | None]:
    normalized = semantic_model_identity(identity)
    return (
        normalized.model_id,
        normalized.revision,
        normalized.format,
        normalized.architecture,
    )


def _checkpoint_file(record: ArtifactFile) -> bool:
    return record.path.endswith(".safetensors") or record.path in _CHECKPOINT_METADATA


def artifact_checkpoint_records(manifest: ArtifactManifest) -> list[ArtifactFile]:
    records = sorted(
        (record for record in manifest.files if _checkpoint_file(record)),
        key=lambda record: record.path,
    )
    if not records or not any(record.path.endswith(".safetensors") for record in records):
        raise ArtifactError("artifact manifest contains no checkpoint Safetensors members")
    return records


def checkpoint_key_from_source_manifest(manifest: SourceCheckpointManifest) -> CheckpointKey:
    records = sorted(manifest.files, key=lambda record: record.path)
    index = next((record for record in records if record.path == _INDEX_PATH), None)
    if index is None:
        raise ArtifactError("flagship source checkpoint manifest omits the Safetensors index")
    return CheckpointKey(
        model=semantic_model_identity(manifest.source_model),
        config_sha256=manifest.config_sha256,
        tokenizer_sha256=manifest.tokenizer_sha256,
        weight_index_sha256=index.sha256,
        checkpoint_members_sha256=stable_sha256(
            [record.model_dump(mode="json") for record in records]
        ),
    )


def semantic_plan_sha256(plan: QuantizationPlan) -> str:
    payload = plan.model_dump(mode="json")
    source = dict(payload["source_model"])
    source["local_path"] = None
    payload["source_model"] = source
    return stable_sha256(payload)


def semantic_artifact_manifest_sha256(
    manifest: ArtifactManifest,
    *,
    plan_sha256: str,
) -> str:
    payload = manifest.model_dump(mode="json")
    source = dict(payload["source_model"])
    source["local_path"] = None
    payload["source_model"] = source
    payload["plan_sha256"] = plan_sha256
    payload["files"] = [
        record.model_dump(mode="json") for record in artifact_checkpoint_records(manifest)
    ]
    return stable_sha256(payload)


def candidate_key_from_artifacts(
    *,
    source_manifest: SourceCheckpointManifest,
    certification_policy_sha256: str,
    calibration_sha256: str,
    activation_capture_sha256: str,
    sensitivity_sha256: str,
    plan: QuantizationPlan,
    artifact_manifest: ArtifactManifest,
) -> CandidateKey:
    if not same_model_identity(source_manifest.source_model, plan.source_model):
        raise ArtifactError("source checkpoint manifest and plan identify different models")
    if not same_model_identity(plan.source_model, artifact_manifest.source_model):
        raise ArtifactError("plan and artifact manifest identify different source models")
    plan_digest = semantic_plan_sha256(plan)
    records = artifact_checkpoint_records(artifact_manifest)
    return CandidateKey(
        source=checkpoint_key_from_source_manifest(source_manifest),
        certification_policy_sha256=certification_policy_sha256,
        calibration_sha256=calibration_sha256,
        activation_capture_sha256=activation_capture_sha256,
        sensitivity_sha256=sensitivity_sha256,
        plan_sha256=plan_digest,
        artifact_manifest_sha256=semantic_artifact_manifest_sha256(
            artifact_manifest,
            plan_sha256=plan_digest,
        ),
        checkpoint_members_sha256=stable_sha256(
            [record.model_dump(mode="json") for record in records]
        ),
    )


def semantic_payload(value: BaseModel | dict[str, Any] | list[Any]) -> Any:
    """Normalize nested ModelIdentity-shaped objects for path-neutral comparison."""

    payload: Any = value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            normalized = {key: normalize(child) for key, child in item.items()}
            if {"model_id", "format", "local_path"}.issubset(normalized):
                normalized["local_path"] = None
            return normalized
        if isinstance(item, list):
            return [normalize(child) for child in item]
        return item

    return normalize(payload)
