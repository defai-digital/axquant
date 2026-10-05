"""Contracts for the native CUDA mixed NVFP4/FP8 six-bit development lane (AXQ-058).

A CUDA six-bit lane is a budget class: NVIDIA has no six-bit datatype and the
only six-bit float standard was retired (AXQ-051). The class is realized by
mixing 4-bit NVFP4 matrices with 8-bit FP8 matrices while every protected
tensor keeps its source precision.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.cuda import _expert_unit, _fused_scale_group
from axquant.schema._base import StrictModel, utc_now
from axquant.schema.cuda import NVFP4_ROLES, CudaFileDigest, CudaQuantizationPlan
from axquant.schema.enums import TensorRole

_SHA256 = r"^[0-9a-f]{64}$"


def allocation_unit(name: str) -> str | None:
    """Runtime allocation unit covering ``name``, or ``None`` for a lone matrix.

    Expert tables and fused projections must keep one uniform method so the
    fused runtime module never combines packed and source-precision inputs.
    """
    return _expert_unit(name) or _fused_scale_group(name)


class CudaMixTensorAllocation(StrictModel):
    tensor_name: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)
    role: TensorRole
    method: Literal["nvfp4", "fp8", "preserve"]
    scale_group: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def eligible_mix(self) -> CudaMixTensorAllocation:
        if any(size < 0 for size in self.shape):
            raise ValueError("tensor shapes cannot contain negative dimensions")
        if self.method == "preserve":
            if self.scale_group is not None:
                raise ValueError("preserved tensors cannot declare a quantized scale group")
            return self
        if self.scale_group is not None and self.method != "nvfp4":
            raise ValueError("scale groups are only defined for NVFP4 allocations")
        if (
            self.role not in NVFP4_ROLES
            or not self.tensor_name.endswith(".weight")
            or self.dtype not in {"BF16", "F16", "F32"}
            or len(self.shape) != 2
            or min(self.shape) < 1
            or self.shape[1] % 16
        ):
            raise ValueError("mixed allocations require eligible floating text matrices, block 16")
        return self


class CudaMixQuantizationPlan(StrictModel):
    schema_version: Literal["axquant.cuda-mix-plan.v1"] = "axquant.cuda-mix-plan.v1"
    format: Literal["nvfp4+fp8"] = "nvfp4+fp8"
    algorithm: Literal["rtn"] = "rtn"
    activation_precision: Literal[16] = 16
    activation_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    block_size: Literal[16] = 16
    fp8_weight_strategy: Literal["channel"] = "channel"
    fp8_activation_strategy: Literal["dynamic-token"] = "dynamic-token"
    evidence_kind: Literal["unmeasured"] = "unmeasured"
    target_bpw: float = Field(gt=0)
    trunk_bpw: float = Field(gt=0)
    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    model_type: str = Field(min_length=1)
    config_sha256: str = Field(pattern=_SHA256)
    source_files: list[CudaFileDigest] = Field(min_length=1)
    allocations: list[CudaMixTensorAllocation] = Field(min_length=1)
    keep_patterns: list[str] = Field(default_factory=list)
    estimated_weight_bytes: int = Field(gt=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("model_id")
    @classmethod
    def portable_identity(cls, value: str) -> str:
        return CudaQuantizationPlan.portable_identity(value)

    @model_validator(mode="after")
    def complete_layout(self) -> CudaMixQuantizationPlan:
        files = [item.path for item in self.source_files]
        names = [item.tensor_name for item in self.allocations]
        if len(files) != len(set(files)) or len(names) != len(set(names)):
            raise ValueError("CUDA mix plans cannot contain duplicate files or tensors")
        if any(item.source_file not in files for item in self.allocations):
            raise ValueError("CUDA mix allocation references an unbound source file")
        methods = {item.method for item in self.allocations}
        if "nvfp4" not in methods or "fp8" not in methods:
            raise ValueError("CUDA mix plan must select at least one NVFP4 and one FP8 tensor")
        units: dict[str, set[str]] = {}
        for item in self.allocations:
            key = allocation_unit(item.tensor_name)
            if key is not None:
                units.setdefault(key, set()).add(item.method)
        if any(len(seen) > 1 for seen in units.values()):
            raise ValueError("CUDA mix plans must keep fused and expert units uniform")
        return self


class CudaMixPackManifest(StrictModel):
    schema_version: Literal["axquant.cuda-mix-pack.v1"] = "axquant.cuda-mix-pack.v1"
    format: Literal["nvfp4+fp8"] = "nvfp4+fp8"
    algorithm: Literal["rtn"] = "rtn"
    activation_precision: Literal[16] = 16
    activation_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    block_size: Literal[16] = 16
    status: Literal["development"] = "development"
    runtime_verified: Literal[False] = False
    quality_certified: Literal[False] = False
    backend: Literal["numpy-reference", "torch-cuda"]
    plan_sha256: str = Field(pattern=_SHA256)
    quantized_tensors: list[str] = Field(min_length=1)
    files: list[CudaFileDigest] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def unique_members(self) -> CudaMixPackManifest:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)) or len(self.quantized_tensors) != len(
            set(self.quantized_tensors)
        ):
            raise ValueError("CUDA mix manifests cannot contain duplicate files or tensors")
        return self
