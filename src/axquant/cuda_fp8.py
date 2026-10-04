"""Native, source-bound RTN FP8 W8A8 CUDA checkpoint conversion."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from axquant.cuda import _digest, _quantization_config, _torch, plan_cuda_nvfp4
from axquant.errors import ArtifactError, PlanningError
from axquant.fp8 import quantize_fp8_channel
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.schema.cuda_fp8 import (
    CudaFp8PackManifest,
    CudaFp8QuantizationPlan,
    CudaFp8TensorAllocation,
)
from axquant.serde import read_data, stable_sha256, write_data

PLAN_NAME = "axquant_cuda_plan.json"
MANIFEST_NAME = "axquant_cuda_manifest.json"
_INDEX_NAME = "model.safetensors.index.json"


def plan_cuda_fp8(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    keep_patterns: list[str] | None = None,
    allow_unmeasured: bool = False,
) -> CudaFp8QuantizationPlan:
    """Use the existing CUDA source/protection policy with independent FP8 contracts."""
    if not allow_unmeasured:
        raise PlanningError("FP8 RTN is unmeasured; explicitly pass --allow-unmeasured")
    baseline = plan_cuda_nvfp4(
        model_dir,
        model_id=model_id,
        revision=revision,
        keep_patterns=keep_patterns,
        allow_unmeasured=allow_unmeasured,
    )
    allocations = [
        CudaFp8TensorAllocation(
            tensor_name=item.tensor_name,
            source_file=item.source_file,
            shape=item.shape,
            dtype=item.dtype,
            role=item.role,
            method="fp8" if item.method == "nvfp4" else "preserve",
            reason="Unmeasured native E4M3FN RTN with per-channel FP32 scales"
            if item.method == "nvfp4"
            else item.reason,
        )
        for item in baseline.allocations
    ]
    estimated = baseline.estimated_weight_bytes
    for item in baseline.allocations:
        if item.method == "nvfp4":
            count = item.shape[0] * item.shape[1]
            estimated += count + item.shape[0] * 4 - (count * 9 // 16 + 4)
    return CudaFp8QuantizationPlan(
        runtime_dtype=baseline.activation_dtype,
        model_id=baseline.model_id,
        revision=baseline.revision,
        model_type=baseline.model_type,
        config_sha256=baseline.config_sha256,
        source_files=baseline.source_files,
        allocations=allocations,
        keep_patterns=baseline.keep_patterns,
        estimated_weight_bytes=estimated,
    )


def _verify_source(source: Path, plan: CudaFp8QuantizationPlan) -> None:
    for member in plan.source_files:
        if _digest(source, member.path) != member:
            raise ArtifactError(f"CUDA source content changed: {member.path}")


def _fp8_config(ignore: list[str]) -> dict[str, Any]:
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "fp8": {
                "targets": ["Linear"],
                "format": "float-quantized",
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
                "output_activations": None,
            }
        },
        "ignore": ignore,
    }


def convert_cuda_fp8(
    model_dir: str | Path,
    plan: CudaFp8QuantizationPlan,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    rows_per_chunk: int = 256,
    allow_unmeasured: bool = False,
) -> CudaFp8PackManifest:
    """Atomically export E4M3FN weights; activations are quantized by the runtime."""
    if not allow_unmeasured:
        raise PlanningError("FP8 RTN is unmeasured; explicitly pass --allow-unmeasured")
    if rows_per_chunk < 1:
        raise PlanningError("rows_per_chunk must be positive")
    if device not in {"cpu", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise PlanningError("device must be cpu, cuda or cuda:<index>")
    source = Path(model_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ArtifactError("CUDA output directory already exists")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ArtifactError("CUDA source and output directories must not overlap")
    _verify_source(source, plan)
    expected = plan_cuda_fp8(
        source,
        model_id=plan.model_id,
        revision=plan.revision,
        keep_patterns=plan.keep_patterns,
        allow_unmeasured=True,
    )
    if stable_sha256(expected) != stable_sha256(plan):
        raise PlanningError("CUDA plan differs from the bound source or protection policy")
    protection = plan_cuda_nvfp4(
        source,
        model_id=plan.model_id,
        revision=plan.revision,
        keep_patterns=plan.keep_patterns,
        allow_unmeasured=True,
    )
    ignore = _quantization_config(protection)["ignore"]
    torch = _torch()
    if device != "cpu" and not torch.cuda.is_available():
        raise ArtifactError("FP8 CUDA execution requires an available CUDA device")
    from safetensors.torch import load_file, save_file

    allocations = {item.tensor_name: item for item in plan.allocations}
    selected = {name for name, item in allocations.items() if item.method == "fp8"}
    generated = {name.removesuffix(".weight") + ".weight_scale" for name in selected}
    if generated & allocations.keys():
        raise PlanningError("FP8 generated tensor names collide with source tensors")
    source_index = read_data(source / _INDEX_NAME) if (source / _INDEX_NAME).exists() else None
    main_names = (
        set(source_index["weight_map"])
        if source_index is not None
        else {
            item.tensor_name
            for item in plan.allocations
            if Path(item.source_file).name not in EXTERNAL_MTP_SIDECAR_FILENAMES
        }
    )
    visited: set[str] = set()
    weight_map: dict[str, str] = {}
    total_size = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.fp8-", dir=output.parent))
    try:
        for member in plan.source_files:
            destination = staging / member.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not member.path.endswith(".safetensors"):
                if member.path not in {"config.json", _INDEX_NAME}:
                    shutil.copyfile(source / member.path, destination)
                continue
            tensors = load_file(str(source / member.path), device="cpu")
            transformed: dict[str, Any] = {}
            for name, tensor in tensors.items():
                item = allocations.get(name)
                if item is None or item.source_file != member.path or name in visited:
                    raise PlanningError(f"unexpected or duplicate CUDA source tensor: {name}")
                visited.add(name)
                transformed[name] = tensor
                output_names = [name]
                if item.method == "fp8":
                    encoded, scale = quantize_fp8_channel(
                        tensor,
                        device=device,
                        rows_per_chunk=rows_per_chunk,
                    )
                    scale_name = name.removesuffix(".weight") + ".weight_scale"
                    transformed[name] = encoded
                    transformed[scale_name] = scale
                    output_names.append(scale_name)
                if name in main_names:
                    for output_name in output_names:
                        weight_map[output_name] = member.path
                        saved = transformed[output_name]
                        total_size += saved.numel() * saved.element_size()
            if selected.isdisjoint(tensors):
                shutil.copyfile(source / member.path, destination)
            else:
                save_file(transformed, str(destination), metadata={"format": "pt"})
        if visited != set(allocations):
            raise PlanningError("CUDA backend did not visit every planned tensor")
        config = read_data(source / "config.json")
        config["torch_dtype"] = plan.runtime_dtype
        if "dtype" in config:
            config["dtype"] = plan.runtime_dtype
        config["quantization_config"] = _fp8_config(ignore)
        write_data(staging / "config.json", config)
        write_data(
            staging / _INDEX_NAME,
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
        )
        write_data(staging / PLAN_NAME, plan)
        manifest = CudaFp8PackManifest(
            runtime_dtype=plan.runtime_dtype,
            backend="torch-cpu" if device == "cpu" else "torch-cuda",
            plan_sha256=stable_sha256(plan),
            quantized_tensors=sorted(selected),
            files=[
                _digest(staging, path.relative_to(staging).as_posix())
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ],
        )
        write_data(staging / MANIFEST_NAME, manifest)
        _verify_source(source, plan)
        if output.exists():
            raise ArtifactError("CUDA output directory appeared during conversion")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
