"""DFlash2 target-pack models (AXQ-067 DF-1 / DF-1b).

AX Engine runs DFlash2 block drafting as its primary greedy-text route for a
small set of admitted targets. It selects the drafter by the pack's recorded
``source_model.model_id``, and the drafter reuses the target's token embedding
and LM head. AXQuant mirrors the admitted list as a dated input and records,
per pack, whether it is a DFlash2 target and what the drafter will consume.
None of these records is runtime, speed, or quality evidence.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from itertools import pairwise
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now

_SHA256 = r"^[0-9a-f]{64}$"
_REVISION = r"^[0-9a-f]{40}$"
_HUB_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
_ENGINE_COMMIT = r"^[0-9a-f]{7,40}$"
_FORMAT_LANE = r"^[a-z0-9_]+$"
# A shard is a plain file name inside the pack, never a path.
_SHARD_FILE = r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.safetensors$"


def _check_tap_layers(ids: list[int], layers: int) -> None:
    if any(value < 0 for value in ids) or any(b <= a for a, b in pairwise(ids)):
        raise ValueError("tap_layer_ids must be non-negative and strictly increasing")
    if any(layer >= layers for layer in ids):
        raise ValueError("tap_layer_ids must be below the layer count")


class Dflash2ArchitectureSignature(StrictModel):
    """Config fields that identify checkpoints shaped like an admitted target.

    A fine-tune or a sibling release with the same shapes matches the signature;
    only the source identity decides whether the pack is a DFlash2 target.
    """

    model_types: list[str] = Field(min_length=1)
    hidden_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    # None for dense targets; MoE targets record the routed expert count.
    num_experts: int | None = Field(default=None, gt=0)


class Dflash2AdmittedTarget(StrictModel):
    source_model_id: str = Field(pattern=_HUB_ID)
    signature: Dflash2ArchitectureSignature
    drafter_repo: str = Field(pattern=_HUB_ID)
    drafter_revision: str = Field(pattern=_REVISION)
    drafter_config_sha256: str = Field(pattern=_SHA256)
    drafter_weights_sha256: str = Field(pattern=_SHA256)
    tap_layer_ids: list[int] = Field(min_length=1)
    # Format lanes AX Engine's format policy admits for this target. None means
    # AX Engine applies no format policy to it (a runtime fact, not a gap).
    admitted_format_lanes: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _tap_layers_increase(self) -> Dflash2AdmittedTarget:
        _check_tap_layers(self.tap_layer_ids, self.signature.num_hidden_layers)
        lanes = self.admitted_format_lanes or []
        if len(lanes) != len(set(lanes)) or any(not re.fullmatch(_FORMAT_LANE, x) for x in lanes):
            raise ValueError("admitted_format_lanes must be unique lowercase lane names")
        return self


class Dflash2AdmittedTargets(StrictModel):
    """Dated mirror of AX Engine ``ADMITTED_DRAFTERS``."""

    schema_version: Literal["axquant.dflash2_admitted_targets.v1"] = (
        "axquant.dflash2_admitted_targets.v1"
    )
    observed_at: date
    ax_engine_commit: str = Field(pattern=_ENGINE_COMMIT)
    targets: list[Dflash2AdmittedTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_targets(self) -> Dflash2AdmittedTargets:
        ids = [target.source_model_id.lower() for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("admitted targets must have unique source_model_id values")
        for index, first in enumerate(self.targets):
            for second in self.targets[index + 1 :]:
                a, b = first.signature, second.signature
                if set(a.model_types) & set(b.model_types) and (
                    a.hidden_size,
                    a.num_hidden_layers,
                    a.vocab_size,
                    a.num_experts,
                ) == (b.hidden_size, b.num_hidden_layers, b.vocab_size, b.num_experts):
                    raise ValueError("admitted targets must have distinguishable signatures")
        return self


class Dflash2MirrorReference(StrictModel):
    observed_at: date
    ax_engine_commit: str = Field(pattern=_ENGINE_COMMIT)


class Dflash2DrafterPin(StrictModel):
    """The drafter AX Engine is expected to select. No drafter bytes are read."""

    repo: str = Field(pattern=_HUB_ID)
    revision: str = Field(pattern=_REVISION)
    config_sha256: str = Field(pattern=_SHA256)
    weights_sha256: str = Field(pattern=_SHA256)


class Dflash2SharedTensorPart(StrictModel):
    """One stored tensor of a drafter-shared module (weight, scales or biases)."""

    name: str = Field(min_length=1)
    file: str = Field(pattern=_SHARD_FILE)
    dtype: str = Field(min_length=1)
    shape: list[int]
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _non_negative_shape(self) -> Dflash2SharedTensorPart:
        if any(dim < 0 for dim in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        return self


class Dflash2SharedTensor(StrictModel):
    """A target module the drafter reuses: the token embedding or the LM head."""

    role: Literal["token_embedding", "lm_head"]
    module: str = Field(min_length=1)
    # None when the module is stored unquantized.
    mode: str | None = None
    bits: int | None = Field(default=None, gt=0)
    group_size: int | None = Field(default=None, gt=0)
    tied_to_embedding: bool = False
    parts: list[Dflash2SharedTensorPart] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tied_head_has_no_parts(self) -> Dflash2SharedTensor:
        if self.tied_to_embedding and (self.role != "lm_head" or self.parts):
            raise ValueError("only a tied lm_head may be recorded, and without parts")
        if not self.tied_to_embedding and not self.parts:
            raise ValueError("an untied shared tensor must record its stored parts")
        quantization = (self.mode, self.bits, self.group_size)
        if self.tied_to_embedding and quantization != (None, None, None):
            raise ValueError("a tied lm_head inherits the embedding quantization")
        if (self.mode is None) != (self.bits is None):
            raise ValueError("mode and bits are recorded together")
        names = [part.name for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("shared tensor parts must be unique")
        allowed = {f"{self.module}.{suffix}" for suffix in ("weight", "scales", "biases")}
        if any(name not in allowed for name in names):
            raise ValueError("shared tensor parts must belong to the recorded module")
        return self


class Dflash2TargetDeclaration(StrictModel):
    """Per-pack DFlash2 declaration written at convert (``axquant_dflash2_target.json``).

    ``target``: built from the exact admitted source; AX Engine will select the
    pinned drafter. ``non_target``: an admitted-signature checkpoint with a
    different Hub identity (for example a fine-tune); AX Engine selects no
    drafter. ``declared_non_target``: the operator explicitly declined, with a
    reason. Development metadata only; never speed or quality evidence.
    """

    schema_version: Literal["axquant.dflash2_target.v1"] = "axquant.dflash2_target.v1"
    status: Literal["target", "non_target", "declared_non_target"]
    # None only for a declared non-target converted without a Hub identity.
    source_model_id: str | None = Field(default=None, pattern=_HUB_ID)
    source_revision: str | None = Field(default=None, pattern=_REVISION)
    admitted_source_model_id: str = Field(pattern=_HUB_ID)
    mirror: Dflash2MirrorReference
    drafter: Dflash2DrafterPin | None = None
    tap_layer_ids: list[int] = Field(min_length=1)
    target_num_hidden_layers: int = Field(gt=0)
    format_lane: str = Field(pattern=_FORMAT_LANE)
    shared_tensors: list[Dflash2SharedTensor] = Field(default_factory=list)
    reason: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _status_fields(self) -> Dflash2TargetDeclaration:
        if self.status == "target":
            if self.drafter is None or self.source_revision is None:
                raise ValueError("a DFlash2 target must record the drafter pin and source revision")
            if (self.source_model_id or "").lower() != self.admitted_source_model_id.lower():
                raise ValueError("a DFlash2 target must record the admitted source model ID")
            roles = [tensor.role for tensor in self.shared_tensors]
            if sorted(roles) != ["lm_head", "token_embedding"]:
                raise ValueError("a DFlash2 target must record the embedding and LM head once")
        else:
            if self.drafter is not None:
                raise ValueError("a non-target must not record a drafter pin")
            if (self.source_model_id or "").lower() == self.admitted_source_model_id.lower():
                raise ValueError("the admitted source model ID cannot be a non-target")
            if self.status == "non_target" and self.source_model_id is None:
                raise ValueError("a non-target must record its own Hub identity")
        roles = [tensor.role for tensor in self.shared_tensors]
        if len(roles) != len(set(roles)):
            raise ValueError("shared tensor roles must be unique")
        if self.source_model_id is None and self.source_revision is not None:
            raise ValueError("a source revision requires a source model ID")
        if self.status == "declared_non_target":
            if not (self.reason or "").strip():
                raise ValueError("a declared non-target requires a reason")
        elif self.reason is not None:
            raise ValueError("only a declared non-target records a reason")
        _check_tap_layers(self.tap_layer_ids, self.target_num_hidden_layers)
        return self


# A composite member lives under ``dflash2/`` and is a plain relative path.
_COMPANION_FILE = r"^dflash2/[A-Za-z0-9][A-Za-z0-9_.-]*$"


class Dflash2CompanionFile(StrictModel):
    path: str = Field(pattern=_COMPANION_FILE)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256)


class Dflash2Companion(StrictModel):
    """Composite target + drafter pack record (AXQ-067 DF-5, ``axquant_dflash2_companion.json``).

    The target files are byte-identical to the bound target pack and the
    drafter files byte-identical to the admitted upstream revision; nothing is
    requantized. Development metadata only: the target's certificates, if any,
    do not transfer, and no route, speed or quality claim follows.
    """

    schema_version: Literal["axquant.dflash2_companion.v1"] = "axquant.dflash2_companion.v1"
    target_artifact_manifest_sha256: str = Field(pattern=_SHA256)
    target_declaration_sha256: str = Field(pattern=_SHA256)
    source_model_id: str = Field(pattern=_HUB_ID)
    source_revision: str = Field(pattern=_REVISION)
    drafter: Dflash2DrafterPin
    drafter_license: str = Field(min_length=1)
    files: list[Dflash2CompanionFile] = Field(min_length=2)
    composer_version: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _drafter_files(self) -> Dflash2Companion:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("companion files must be unique")
        by_path = {item.path: item.sha256 for item in self.files}
        if by_path.get("dflash2/config.json") != self.drafter.config_sha256:
            raise ValueError("dflash2/config.json must match the drafter config pin")
        if by_path.get("dflash2/model.safetensors") != self.drafter.weights_sha256:
            raise ValueError("dflash2/model.safetensors must match the drafter weights pin")
        return self
