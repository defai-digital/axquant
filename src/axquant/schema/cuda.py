"""Independent experimental CUDA contracts; the frozen MLX chain is unchanged."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.enums import TensorRole

_SHA256 = r"^[0-9a-f]{64}$"
NVFP4_ROLES = frozenset({TensorRole.ATTENTION, TensorRole.MLP, TensorRole.EXPERT})


class CudaFileDigest(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
            raise ValueError("CUDA artifact members require canonical safe relative paths")
        if not path.parts:
            raise ValueError("CUDA artifact members require a file path")
        return value


class CudaTensorAllocation(StrictModel):
    tensor_name: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)
    role: TensorRole
    method: Literal["nvfp4", "preserve"]
    scale_group: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def eligible_nvfp4(self) -> CudaTensorAllocation:
        if any(size < 0 for size in self.shape):
            raise ValueError("tensor shapes cannot contain negative dimensions")
        if self.method == "preserve" and self.scale_group is not None:
            raise ValueError("preserved tensors cannot declare an NVFP4 scale group")
        if self.method == "nvfp4" and (
            self.role not in NVFP4_ROLES
            or not self.tensor_name.endswith(".weight")
            or self.dtype not in {"BF16", "F16", "F32"}
            or len(self.shape) != 2
            or min(self.shape) < 1
            or self.shape[1] % 16
        ):
            raise ValueError("NVFP4 allocations require eligible floating text matrices, block 16")
        return self


class CudaQuantizationPlan(StrictModel):
    schema_version: Literal["axquant.cuda-plan.v1"] = "axquant.cuda-plan.v1"
    format: Literal["nvfp4"] = "nvfp4"
    algorithm: Literal["rtn"] = "rtn"
    activation_precision: Literal[16] = 16
    activation_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    block_size: Literal[16] = 16
    evidence_kind: Literal["unmeasured"] = "unmeasured"
    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    model_type: str = Field(min_length=1)
    config_sha256: str = Field(pattern=_SHA256)
    source_files: list[CudaFileDigest] = Field(min_length=1)
    allocations: list[CudaTensorAllocation] = Field(min_length=1)
    keep_patterns: list[str] = Field(default_factory=list)
    estimated_weight_bytes: int = Field(gt=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("model_id")
    @classmethod
    def portable_identity(cls, value: str) -> str:
        if value.startswith(("/", "~", "file:")) or "\\" in value:
            raise ValueError("CUDA model_id must not contain a local filesystem path")
        return value

    @model_validator(mode="after")
    def complete_layout(self) -> CudaQuantizationPlan:
        files = [item.path for item in self.source_files]
        names = [item.tensor_name for item in self.allocations]
        if len(files) != len(set(files)) or len(names) != len(set(names)):
            raise ValueError("CUDA plans cannot contain duplicate files or tensors")
        if any(item.source_file not in files for item in self.allocations):
            raise ValueError("CUDA allocation references an unbound source file")
        if not any(item.method == "nvfp4" for item in self.allocations):
            raise ValueError("CUDA plan must select at least one NVFP4 tensor")
        return self


class CudaPackManifest(StrictModel):
    schema_version: Literal["axquant.cuda-pack.v1"] = "axquant.cuda-pack.v1"
    format: Literal["nvfp4"] = "nvfp4"
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
    def unique_members(self) -> CudaPackManifest:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)) or len(self.quantized_tensors) != len(
            set(self.quantized_tensors)
        ):
            raise ValueError("CUDA manifests cannot contain duplicate files or quantized tensors")
        return self
