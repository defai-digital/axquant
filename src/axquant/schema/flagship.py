from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from axquant.revisions import is_immutable_revision
from axquant.schema._base import StrictModel
from axquant.schema.inventory import ModelIdentity

_SHA256 = r"^[0-9a-f]{64}$"


class CheckpointKey(StrictModel):
    """Path-neutral identity for one immutable source checkpoint."""

    schema_version: Literal["axquant.checkpoint-key.v1"] = "axquant.checkpoint-key.v1"
    model: ModelIdentity
    config_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    weight_index_sha256: str = Field(pattern=_SHA256)
    checkpoint_members_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def immutable_and_path_neutral(self) -> CheckpointKey:
        if not is_immutable_revision(self.model.revision):
            raise ValueError("checkpoint key requires a full immutable source revision")
        if self.model.local_path is not None:
            raise ValueError("checkpoint key must not contain a local path")
        return self


class CandidateKey(StrictModel):
    """Semantic identity for one converted release candidate."""

    schema_version: Literal["axquant.candidate-key.v1"] = "axquant.candidate-key.v1"
    source: CheckpointKey
    certification_policy_sha256: str = Field(pattern=_SHA256)
    calibration_sha256: str = Field(pattern=_SHA256)
    activation_capture_sha256: str = Field(pattern=_SHA256)
    sensitivity_sha256: str = Field(pattern=_SHA256)
    plan_sha256: str = Field(pattern=_SHA256)
    artifact_manifest_sha256: str = Field(pattern=_SHA256)
    checkpoint_members_sha256: str = Field(pattern=_SHA256)


class ActivationCaptureSentinel(StrictModel):
    """Explicit proof that the frozen plan does not require activation capture."""

    schema_version: Literal["axquant.activation-capture-sentinel.v1"] = (
        "axquant.activation-capture-sentinel.v1"
    )
    kind: Literal["not-required"] = "not-required"
    policy_reason: Literal["plan-uses-no-awq-or-gptq"] = "plan-uses-no-awq-or-gptq"
    plan_sha256: str = Field(pattern=_SHA256)
