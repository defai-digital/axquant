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


PublicationSmokeCheckId = Literal[
    "ax-engine:mlx-lm-agreement",
    "ax-engine:mlx-vlm-agreement",
    "mlx-lm:text",
    "mlx-vlm:text",
    "mlx-vlm:image",
    "omlx:mtp-off",
    "omlx:mtp-on",
]


class PublicationSmokeCheck(StrictModel):
    check: PublicationSmokeCheckId
    status: Literal["pass", "fail"]
    runtime_version: str | None = None
    expected: str
    output: str = ""
    detail: str = ""
    seconds: float | None = None
    # Agreement checks: per-prompt worst log-probability deficit (nat) of the
    # candidate's tokens under the reference runtime the check id names (MLX-LM
    # or MLX-VLM), prompts within tolerance, and the digest
    # of the prompt set, budget and tolerance.
    agreed: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=1)
    logprob_deficits: list[float] | None = None
    prompt_set_sha256: Digest | None = None


class PublicationSmokeReceipt(StrictModel):
    """Real load-and-generate smoke runs bound to the runtime payload digest.

    Evidence of loading and short greedy generation only. It does not certify
    quality, speed, or MTP acceptance.
    """

    schema_version: Literal["axquant.publication-smoke.v3"] = "axquant.publication-smoke.v3"
    payload_sha256: Digest
    required_checks: list[PublicationSmokeCheckId]
    checks: list[PublicationSmokeCheck]
    # Chip description, never a hostname: the receipt ships with the pack.
    hardware: str
    created_at: str
    quality_certified: Literal[False] = False
    mtp_acceptance_measured: Literal[False] = False

    @model_validator(mode="after")
    def unique_checks(self) -> PublicationSmokeReceipt:
        names = [check.check for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("publication smoke receipt repeats a check")
        if len(self.required_checks) != len(set(self.required_checks)):
            raise ValueError("publication smoke receipt repeats a required check")
        return self
