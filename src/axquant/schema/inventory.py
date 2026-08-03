from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.enums import (
    ArchitectureSupportLevel,
    OptimizationScope,
    QuantMethod,
    SupportTier,
    TensorRole,
)


class ModelIdentity(StrictModel):
    model_id: str
    revision: str | None = None
    format: Literal["mlx"] = "mlx"
    architecture: str | None = None
    local_path: str | None = None


class SourceConversionProvenance(StrictModel):
    """Immutable Hub identity emitted by the BF16 source-preparation helper."""

    schema_version: Literal["axquant.source-conversion.v1"] = "axquant.source-conversion.v1"
    source_model: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dtype: Literal["bfloat16"] = "bfloat16"
    key_remap_applied: bool


class ArchitectureProfile(StrictModel):
    adapter_id: str = "generic"
    product_family: str = "unknown"
    config_model_type: str | None = None
    support_level: ArchitectureSupportLevel = ArchitectureSupportLevel.INVENTORY_ONLY
    support_tier: SupportTier = SupportTier.INSPECT_ONLY
    optimization_scope: OptimizationScope = OptimizationScope.INVENTORY_ONLY
    dense: bool | None = None
    text_layer_count: int | None = Field(default=None, ge=1)
    mtp_declared: bool = False
    vision_present: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def tier_requires_supported_level(self) -> ArchitectureProfile:
        if (
            self.support_tier is not SupportTier.INSPECT_ONLY
            and self.support_level is not ArchitectureSupportLevel.SUPPORTED
        ):
            raise ValueError("a convertible or certified tier requires a supported architecture")
        return self


class TensorSpec(StrictModel):
    name: str
    module_path: str
    shape: tuple[int, ...]
    dtype: str
    parameters: int = Field(ge=0)
    physical_elements: int = Field(default=0, ge=0)
    storage_bytes: int = Field(default=0, ge=0)
    role: TensorRole
    quantizable: bool
    file: str
    current_precision: str
    current_bits: int | None = Field(default=None, ge=2, le=16)
    current_group_size: int | None = Field(default=None, ge=1)
    current_method: QuantMethod | None = None
    quantization_metadata: bool = False
    protected_recommendation: bool = False
    protection_reason: str | None = None
    tied_to: str | None = None

    @field_validator("shape")
    @classmethod
    def valid_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension < 0 for dimension in value):
            raise ValueError("tensor shape must contain non-negative dimensions")
        return value


class Inventory(StrictModel):
    schema_version: Literal["axquant.inventory.v1"] = "axquant.inventory.v1"
    model: ModelIdentity
    tensors: list[TensorSpec]
    total_parameters: int = Field(ge=0)
    quantizable_parameters: int = Field(ge=0)
    weight_bytes: int = Field(default=0, ge=0)
    mtp_weight_bytes: int = Field(default=0, ge=0)
    precision_parameters: dict[str, int] = Field(default_factory=dict)
    mtp_present: bool
    quantized_source: bool
    source_files: list[str]
    architecture_profile: ArchitectureProfile = Field(default_factory=ArchitectureProfile)
    tied_weight_groups: list[list[str]] = Field(default_factory=list)
    config_sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)
