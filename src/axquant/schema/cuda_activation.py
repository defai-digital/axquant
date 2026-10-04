"""Source-bound activation calibration and experimental NVFP4 W4A4 contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.cuda import CudaFileDigest, CudaQuantizationPlan

_SHA256 = r"^[0-9a-f]{64}$"


class CudaActivationStat(StrictModel):
    tensor_name: str = Field(min_length=1)
    input_columns: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    absolute_maximum: float = Field(ge=0, allow_inf_nan=False)


class CudaActivationCalibration(StrictModel):
    schema_version: Literal["axquant.cuda-activation.v1"] = "axquant.cuda-activation.v1"
    weight_plan_sha256: str = Field(pattern=_SHA256)
    input_files: list[CudaFileDigest] = Field(min_length=1)
    runtime: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    source_precision: Literal["bfloat16", "float16"]
    status: Literal["complete"] = "complete"
    method: Literal["observed-input-and-expert-replay-absmax"] = (
        "observed-input-and-expert-replay-absmax"
    )
    statistics: list[CudaActivationStat] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def unique_members(self) -> CudaActivationCalibration:
        names = [item.tensor_name for item in self.statistics]
        inputs = [item.path for item in self.input_files]
        if len(names) != len(set(names)) or len(inputs) != len(set(inputs)):
            raise ValueError("CUDA calibration cannot contain duplicate tensors or inputs")
        return self


class CudaW4A4Plan(StrictModel):
    schema_version: Literal["axquant.cuda-w4a4-plan.v1"] = "axquant.cuda-w4a4-plan.v1"
    format: Literal["nvfp4"] = "nvfp4"
    activation_precision: Literal[4] = 4
    activation_headroom: float = Field(default=1.25, ge=1, le=16, allow_inf_nan=False)
    weight_plan: CudaQuantizationPlan
    calibration: CudaActivationCalibration
    created_at: datetime = Field(default_factory=utc_now)


class CudaW4A4PackManifest(StrictModel):
    schema_version: Literal["axquant.cuda-w4a4-pack.v1"] = "axquant.cuda-w4a4-pack.v1"
    format: Literal["nvfp4"] = "nvfp4"
    algorithm: Literal["rtn"] = "rtn"
    activation_precision: Literal[4] = 4
    activation_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    block_size: Literal[16] = 16
    status: Literal["development"] = "development"
    runtime_verified: Literal[False] = False
    quality_certified: Literal[False] = False
    backend: Literal["numpy-reference", "torch-cuda"]
    plan_sha256: str = Field(pattern=_SHA256)
    calibration_sha256: str = Field(pattern=_SHA256)
    quantized_tensors: list[str] = Field(min_length=1)
    files: list[CudaFileDigest] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def unique_members(self) -> CudaW4A4PackManifest:
        paths = [item.path for item in self.files]
        names = self.quantized_tensors
        if len(paths) != len(set(paths)) or len(names) != len(set(names)):
            raise ValueError("CUDA W4A4 manifests cannot contain duplicate files or tensors")
        return self
