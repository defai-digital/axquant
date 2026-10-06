from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RuntimeExportTarget = Literal["omlx", "mtplx"]


class RuntimeExportVerdict(StrictModel):
    profile: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*-v[1-9][0-9]*$")] | None
    status: Literal["unsupported", "requires-export", "static-compatible"]
    blockers: list[str]
    required_actions: list[str]
    runtime_verified: Literal[False] = False
    mtp_verified: Literal[False] = False
    launch_environment: dict[str, str] = Field(default_factory=dict)
    model_settings: dict[str, bool | int | str] = Field(default_factory=dict)
    modality_constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_status(self) -> RuntimeExportVerdict:
        expected = (
            "unsupported"
            if self.blockers
            else "requires-export"
            if self.required_actions
            else "static-compatible"
        )
        if self.status != expected:
            raise ValueError("runtime export status disagrees with blockers and required actions")
        if self.profile is None and not self.blockers:
            raise ValueError("unknown runtime profiles must fail closed")
        return self


class RuntimeCompatibilityReport(StrictModel):
    schema_version: Literal["axquant.runtime-compatibility.v1"] = "axquant.runtime-compatibility.v1"
    model_type: str
    binding: dict[str, Digest]
    targets: dict[RuntimeExportTarget, RuntimeExportVerdict]
    quality_certified: Literal[False] = False

    @model_validator(mode="after")
    def complete_targets(self) -> RuntimeCompatibilityReport:
        if set(self.targets) != {"omlx", "mtplx"}:
            raise ValueError("runtime report must include oMLX and MTPLX")
        return self


class RuntimeExportManifest(StrictModel):
    schema_version: Literal["axquant.runtime-export.v1"] = "axquant.runtime-export.v1"
    target: RuntimeExportTarget
    source_binding: dict[str, Digest]
    source_manifest_sha256: Digest | None
    source_trunk_norm_layout: Literal["raw_hf_delta", "mlx_multiplier"] | None = None
    files: dict[str, Digest]
    runtime_verified: Literal[False] = False
    quality_certified: Literal[False] = False
    launch_environment: dict[str, str] = Field(default_factory=dict)
    model_settings: dict[str, bool | int | str] = Field(default_factory=dict)
    modality_constraints: list[str] = Field(default_factory=list)
