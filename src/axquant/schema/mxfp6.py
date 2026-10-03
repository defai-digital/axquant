"""Experimental reference MXFP6 artifacts, separate from native MLX plans."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axquant.schema._base import StrictModel


class Mxfp6File(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
            or path.as_posix() != value
        ):
            raise ValueError("MXFP6 file paths must be canonical relative POSIX paths")
        return value


class Mxfp6TensorRecord(StrictModel):
    name: str = Field(min_length=1)
    source_file: str
    output_file: str
    shape: tuple[int, ...]
    source_dtype: str
    encoding: Literal["mxfp6", "preserved"]
    data_key: str = Field(min_length=1)
    scales_key: str | None = None
    storage_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_encoding(self) -> Mxfp6TensorRecord:
        if any(size < 0 for size in self.shape):
            raise ValueError("negative tensor dimension")
        if self.encoding == "mxfp6":
            if len(self.shape) < 2 or any(size == 0 for size in self.shape) or self.shape[-1] % 32:
                raise ValueError("MXFP6 tensors require nonempty block-aligned matrix shapes")
            if not self.scales_key or self.scales_key == self.data_key:
                raise ValueError("MXFP6 tensors require distinct data and scale keys")
            if self.source_dtype not in {"BF16", "F16", "F32"}:
                raise ValueError("MXFP6 source must be BF16, F16, or F32")
            if self.storage_bytes != math.prod(self.shape) // 32 * 25:
                raise ValueError("MXFP6 storage must use 25 bytes per block of 32")
        elif self.scales_key is not None or self.data_key != self.name:
            raise ValueError("preserved tensors retain their original key without scales")
        return self


class Mxfp6PackManifest(StrictModel):
    schema_version: Literal["axquant.mxfp6-pack.v1"] = "axquant.mxfp6-pack.v1"
    status: Literal["experimental"] = "experimental"
    evidence_kind: Literal["unmeasured_development"] = "unmeasured_development"
    runtime_support: Literal["reference-only"] = "reference-only"
    element_format: Literal["e2m3", "e3m2"]
    block_size: Literal[32] = 32
    scale_format: Literal["e8m0"] = "e8m0"
    packing: Literal["lsb-first-6bit"] = "lsb-first-6bit"
    rounding: Literal["ties-to-even"] = "ties-to-even"
    scale_policy: Literal["floor-amax-exponent"] = "floor-amax-exponent"
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_index_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_files: list[Mxfp6File] = Field(min_length=1)
    output_files: list[Mxfp6File] = Field(min_length=1)
    tensors: list[Mxfp6TensorRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_members(self) -> Mxfp6PackManifest:
        source = {item.path for item in self.source_files}
        output = {item.path for item in self.output_files}
        if len(source) != len(self.source_files) or len(output) != len(self.output_files):
            raise ValueError("duplicate MXFP6 file paths")
        if len({item.name for item in self.tensors}) != len(self.tensors):
            raise ValueError("duplicate MXFP6 logical tensor names")
        keys: set[tuple[str, str]] = set()
        for tensor in self.tensors:
            if tensor.source_file not in source or tensor.output_file not in output:
                raise ValueError("MXFP6 tensor references an unlisted file")
            for key in (tensor.data_key, tensor.scales_key):
                if key is not None:
                    member = (tensor.output_file, key)
                    if member in keys:
                        raise ValueError("duplicate MXFP6 physical tensor key")
                    keys.add(member)
        if not any(item.encoding == "mxfp6" for item in self.tensors):
            raise ValueError("MXFP6 pack must contain at least one MXFP6 tensor")
        if {item.source_file for item in self.tensors} != source or {
            item.output_file for item in self.tensors
        } != output:
            raise ValueError("MXFP6 pack contains unreferenced files")
        return self
