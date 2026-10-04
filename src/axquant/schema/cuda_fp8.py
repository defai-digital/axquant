"""Independent FP8 W8A8 CUDA development contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.cuda import NVFP4_ROLES, CudaFileDigest, CudaQuantizationPlan
from axquant.schema.enums import TensorRole

_SHA256 = r"^[0-9a-f]{64}$"


class CudaFp8TensorAllocation(StrictModel):
    tensor_name: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)
    role: TensorRole
    method: Literal["fp8", "preserve"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def eligible_fp8(self) -> CudaFp8TensorAllocation:
        if any(size < 0 for size in self.shape):
            raise ValueError("tensor shapes cannot contain negative dimensions")
        if self.method == "fp8" and (
            self.role not in NVFP4_ROLES
            or not self.tensor_name.endswith(".weight")
            or self.dtype not in {"BF16", "F16", "F32"}
            or len(self.shape) != 2
            or min(self.shape) < 1
            or self.shape[1] % 16
        ):
            raise ValueError("FP8 allocations require eligible floating text matrices")
        return self


class CudaFp8QuantizationPlan(StrictModel):
    schema_version: Literal["axquant.cuda-fp8-plan.v1"] = "axquant.cuda-fp8-plan.v1"
    format: Literal["fp8_e4m3"] = "fp8_e4m3"
    algorithm: Literal["rtn"] = "rtn"
    weight_strategy: Literal["channel"] = "channel"
    activation_precision: Literal[8] = 8
    activation_strategy: Literal["dynamic-token"] = "dynamic-token"
    runtime_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    evidence_kind: Literal["unmeasured"] = "unmeasured"
    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    model_type: str = Field(min_length=1)
    config_sha256: str = Field(pattern=_SHA256)
    source_files: list[CudaFileDigest] = Field(min_length=1)
    allocations: list[CudaFp8TensorAllocation] = Field(min_length=1)
    keep_patterns: list[str] = Field(default_factory=list)
    estimated_weight_bytes: int = Field(gt=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("model_id")
    @classmethod
    def portable_identity(cls, value: str) -> str:
        return CudaQuantizationPlan.portable_identity(value)

    @model_validator(mode="after")
    def complete_layout(self) -> CudaFp8QuantizationPlan:
        files = [item.path for item in self.source_files]
        names = [item.tensor_name for item in self.allocations]
        if len(files) != len(set(files)) or len(names) != len(set(names)):
            raise ValueError("FP8 plans cannot contain duplicate files or tensors")
        if any(item.source_file not in files for item in self.allocations):
            raise ValueError("FP8 allocation references an unbound source file")
        if not any(item.method == "fp8" for item in self.allocations):
            raise ValueError("FP8 plan must select at least one tensor")
        return self


class CudaFp8PackManifest(StrictModel):
    schema_version: Literal["axquant.cuda-fp8-pack.v1"] = "axquant.cuda-fp8-pack.v1"
    format: Literal["fp8_e4m3"] = "fp8_e4m3"
    algorithm: Literal["rtn"] = "rtn"
    weight_strategy: Literal["channel"] = "channel"
    activation_precision: Literal[8] = 8
    activation_strategy: Literal["dynamic-token"] = "dynamic-token"
    runtime_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    status: Literal["development"] = "development"
    runtime_verified: Literal[False] = False
    quality_certified: Literal[False] = False
    backend: Literal["torch-cpu", "torch-cuda"]
    plan_sha256: str = Field(pattern=_SHA256)
    quantized_tensors: list[str] = Field(min_length=1)
    files: list[CudaFileDigest] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def unique_members(self) -> CudaFp8PackManifest:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)) or len(self.quantized_tensors) != len(
            set(self.quantized_tensors)
        ):
            raise ValueError("FP8 manifests cannot contain duplicate files or tensors")
        return self
