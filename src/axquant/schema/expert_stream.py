from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel

ExpertStreamProjection = Literal["gate_up", "gate", "up", "down"]
ExpertStreamResidentRole = Literal[
    "embedding",
    "attention",
    "router",
    "shared_expert",
    "mlp",
    "norm",
    "lm_head",
    "mtp",
    "vision",
    "audio",
    "other",
]


def _streamed_roles_default() -> list[Literal["expert"]]:
    return ["expert"]


class ExpertStreamTensor(StrictModel):
    name: str = Field(min_length=1)
    file: str = Field(min_length=1)
    layer: int = Field(ge=0)
    proj: ExpertStreamProjection
    expert_axis: Literal[0] = 0
    num_experts: int = Field(ge=1)
    bits: int = Field(ge=2, le=16)
    group_size: int = Field(ge=1)

    @field_validator("file")
    @classmethod
    def repo_relative_file(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix():
            raise ValueError("expert stream tensor file must be a normalized repo-relative path")
        return value


class ExpertStreamManifest(StrictModel):
    schema_version: Literal["axquant.expert-stream.v1"] = "axquant.expert-stream.v1"
    generated_by: Literal["axquant"] = "axquant"
    required: bool
    mode: Literal["layer-stack"] = "layer-stack"
    num_experts: int = Field(ge=1)
    experts_per_tok: int = Field(ge=1)
    estimated_resident_bytes: int = Field(ge=0)
    estimated_full_resident_bytes: int = Field(ge=1)
    estimated_max_layer_expert_bytes: int = Field(ge=1)
    resident_roles: list[ExpertStreamResidentRole] = Field(min_length=1)
    streamed_roles: list[Literal["expert"]] = Field(default_factory=_streamed_roles_default)
    tensors: list[ExpertStreamTensor] = Field(min_length=1)

    @model_validator(mode="after")
    def internally_consistent(self) -> ExpertStreamManifest:
        if self.experts_per_tok > self.num_experts:
            raise ValueError("experts_per_tok cannot exceed num_experts")
        if self.estimated_resident_bytes > self.estimated_full_resident_bytes:
            raise ValueError("resident bytes cannot exceed full-resident bytes")
        if len(self.resident_roles) != len(set(self.resident_roles)):
            raise ValueError("resident roles must be unique")
        if self.streamed_roles != ["expert"]:
            raise ValueError("v1 streams only the expert role")
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("expert stream tensor names must be unique")
        if any(tensor.num_experts != self.num_experts for tensor in self.tensors):
            raise ValueError("expert stream tensor expert counts must match the manifest")
        return self
