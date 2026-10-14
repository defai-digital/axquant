"""Custom development storage, explicitly separate from language-model packs."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel

DocLayoutFormat = Literal["nvfp4", "mxfp4", "mxfp8"]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DocLayoutRuntimeCompatibility(StrictModel):
    schema_version: Literal["axq-doclayout.runtime-compatibility/1"] = (
        "axq-doclayout.runtime-compatibility/1"
    )
    native_low_precision_verified: Literal[False] = False
    ax_engine: Literal["unsupported"] = "unsupported"
    omlx: Literal["unsupported"] = "unsupported"
    mtplx: Literal["unsupported"] = "unsupported"
    mlx_lm: Literal["unsupported"] = "unsupported"
    mlx_vlm: Literal["unsupported"] = "unsupported"
    source_from_pretrained: Literal["unsupported: accepts only pinned FP32 source"] = (
        "unsupported: accepts only pinned FP32 source"
    )
    reference_execution: Literal["dequantize-to-fp32-pytorch-cpu"] = (
        "dequantize-to-fp32-pytorch-cpu"
    )


class DocLayoutTensor(StrictModel):
    shape: tuple[int, ...]
    dtype: str
    method: Literal["preserved", "nvfp4", "mxfp4", "mxfp8"]
    padded_columns: int | None = Field(default=None, ge=1)
    group_size: int | None = None
    global_scale: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_absolute_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_layout(self) -> DocLayoutTensor:
        if any(size < 0 for size in self.shape):
            raise ValueError("Negative source dimension")
        if self.method == "preserved":
            if any(
                v is not None
                for v in (
                    self.padded_columns,
                    self.group_size,
                    self.global_scale,
                    self.max_absolute_error,
                )
            ):
                raise ValueError("Preserved tensors cannot declare a quantized layout")
            return self
        group = 16 if self.method == "nvfp4" else 32
        columns = math.prod(self.shape[1:])
        if (
            len(self.shape) not in {2, 4}
            or min(self.shape) < 1
            or self.dtype != "float32"
            or self.group_size != group
            or self.padded_columns != ((columns + group - 1) // group) * group
            or self.max_absolute_error is None
        ):
            raise ValueError("Invalid quantized tensor layout")
        if (self.global_scale is not None) != (self.method == "nvfp4"):
            raise ValueError("Only NVFP4 requires an inverse global scale")
        return self


class DocLayoutManifest(StrictModel):
    schema_version: Literal["axq-doclayout.development/1"] = "axq-doclayout.development/1"
    format: DocLayoutFormat
    source_model: Literal["AutomatosX/AXDocLayout-M-v1.0"]
    source_revision: Literal["69ddc0da17a7fc35dfde29b06adbb83a75284ae1"]
    source_weight_sha256: Literal[
        "4d9ee36db616f3f4ef293588e4258b5f6f42d7a84097a5afb520779076045bf2"
    ]
    status: Literal["development"] = "development"
    quality_certified: Literal[False] = False
    native_runtime_verified: Literal[False] = False
    quantized_parameters: int = Field(gt=0)
    total_parameters_and_buffers: int = Field(gt=0)
    max_absolute_weight_error: float = Field(ge=0, allow_inf_nan=False)
    files: dict[str, Digest]
    tensors: dict[str, DocLayoutTensor]

    @model_validator(mode="after")
    def consistent_storage(self) -> DocLayoutManifest:
        from pathlib import PurePosixPath

        if not self.files or "model.safetensors" not in self.files:
            raise ValueError("Missing bound weight file")
        for name in self.files:
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("Unsafe pack filename")
        count = 0
        for tensor in self.tensors.values():
            if tensor.method != "preserved":
                if tensor.method != self.format:
                    raise ValueError("Mixed physical formats are not supported")
                count += math.prod(tensor.shape)
        if count != self.quantized_parameters:
            raise ValueError("Quantized parameter accounting mismatch")
        if (
            sum(math.prod(t.shape) for t in self.tensors.values())
            != self.total_parameters_and_buffers
        ):
            raise ValueError("Total parameter accounting mismatch")
        return self
