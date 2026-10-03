"""Native, unmeasured NVFP4A16 planning and atomic CUDA checkpoint export."""

from __future__ import annotations

import fnmatch
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, PlanningError
from axquant.inspector import inspect_model
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.nvfp4 import _global_scale, quantize_nvfp4_cuda, quantize_nvfp4_reference
from axquant.schema.cuda import (
    NVFP4_ROLES,
    CudaFileDigest,
    CudaPackManifest,
    CudaQuantizationPlan,
    CudaTensorAllocation,
)
from axquant.schema.enums import SupportTier
from axquant.serde import file_sha256, read_data, stable_sha256, write_data

PLAN_NAME = "axquant_cuda_plan.json"
MANIFEST_NAME = "axquant_cuda_manifest.json"
_INDEX_NAME = "model.safetensors.index.json"
_ASSET_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer*.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "*.model",
    "chat_template*.jinja",
    "*processor_config.json",
    "*.py",
)


def _digest(root: Path, relative: str) -> CudaFileDigest:
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ArtifactError(f"CUDA source requires regular files within the checkpoint: {relative}")
    return CudaFileDigest(path=relative, sha256=file_sha256(path), size_bytes=path.stat().st_size)


def _require_unmeasured(allowed: bool) -> None:
    if not allowed:
        raise PlanningError("NVFP4 RTN is unmeasured; explicitly pass --allow-unmeasured")


def _fused_scale_group(name: str) -> str | None:
    for suffixes, group in (
        (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
        (("gate_proj", "up_proj"), "gate_up_proj"),
        (("w1", "w3"), "w13"),
    ):
        for suffix in suffixes:
            ending = f".{suffix}.weight"
            if name.endswith(ending):
                return name.removesuffix(ending) + "." + group
    return None


def plan_cuda_nvfp4(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    keep_patterns: list[str] | None = None,
    allow_unmeasured: bool = False,
) -> CudaQuantizationPlan:
    """Allocate eligible text matrices to NVFP4 and preserve every other tensor."""
    _require_unmeasured(allow_unmeasured)
    source = Path(model_dir).expanduser().resolve()
    if not source.is_dir():
        raise ArtifactError("CUDA planning requires a local Safetensors checkpoint directory")
    if model_id is not None:
        CudaQuantizationPlan.portable_identity(model_id)
    inventory = inspect_model(source, model_id=model_id, revision=revision)
    identity = model_id or inventory.model.model_id
    if identity.startswith(("/", "~", "file:")) or "\\" in identity:
        identity = source.name
    if inventory.architecture_profile.support_tier == SupportTier.INSPECT_ONLY:
        raise PlanningError("CUDA conversion requires a registered convertible architecture")
    patterns = sorted(set(keep_patterns or []))
    allocations: list[CudaTensorAllocation] = []
    for tensor in sorted(inventory.tensors, key=lambda item: item.name):
        eligible = (
            tensor.quantizable
            and not tensor.protected_recommendation
            and tensor.tied_to is None
            and tensor.role in NVFP4_ROLES
            and tensor.name.endswith(".weight")
            and tensor.dtype in {"BF16", "F16", "F32"}
            and len(tensor.shape) == 2
            and min(tensor.shape) > 0
            and tensor.shape[1] % 16 == 0
            and not any(fnmatch.fnmatchcase(tensor.name, pattern) for pattern in patterns)
        )
        if eligible:
            reason = "Unmeasured native NVFP4 RTN, block 16, activations preserved"
        else:
            reason = "Source precision preserved by protection, shape, dtype or keep policy"
        allocations.append(
            CudaTensorAllocation(
                tensor_name=tensor.name,
                source_file=tensor.file,
                shape=tensor.shape,
                dtype=tensor.dtype,
                role=tensor.role,
                method="nvfp4" if eligible else "preserve",
                reason=reason,
            )
        )
    fused_groups: dict[str, list[int]] = {}
    for index, allocation in enumerate(allocations):
        group = _fused_scale_group(allocation.tensor_name)
        if group is not None:
            fused_groups.setdefault(group, []).append(index)
    for group, indices in fused_groups.items():
        if len(indices) < 2:
            continue
        quantized = all(allocations[index].method == "nvfp4" for index in indices)
        for index in indices:
            allocation = allocations[index]
            allocations[index] = allocation.model_copy(
                update={
                    "method": "nvfp4" if quantized else "preserve",
                    "scale_group": group if quantized else None,
                    "reason": allocation.reason
                    if quantized
                    else "Preserve complete fused runtime unit",
                }
            )
    storage = {tensor.name: tensor for tensor in inventory.tensors}
    estimated_bytes = sum(
        storage[item.tensor_name].parameters * 9 // 16 + 4
        if item.method == "nvfp4"
        else storage[item.tensor_name].storage_bytes
        for item in allocations
    )
    members = set(inventory.source_files)
    members.update(
        path.name
        for path in source.iterdir()
        if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in _ASSET_PATTERNS)
    )
    if (source / _INDEX_NAME).exists():
        members.add(_INDEX_NAME)
    config = read_data(source / "config.json")
    original_dtype = config.get("dtype", config.get("torch_dtype", "bfloat16"))
    return CudaQuantizationPlan(
        activation_dtype="float16" if original_dtype == "float16" else "bfloat16",
        model_id=identity,
        revision=inventory.model.revision,
        model_type=inventory.architecture_profile.config_model_type or "unknown",
        config_sha256=inventory.config_sha256,
        source_files=[_digest(source, name) for name in sorted(members)],
        allocations=allocations,
        keep_patterns=patterns,
        estimated_weight_bytes=estimated_bytes,
    )


def _verify_source(source: Path, plan: CudaQuantizationPlan) -> None:
    for member in plan.source_files:
        if _digest(source, member.path) != member:
            raise ArtifactError(f"CUDA source content changed: {member.path}")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ArtifactError("NVFP4 checkpoint serialization requires axquant[cuda]") from exc
    return torch


def _quantization_config(plan: CudaQuantizationPlan) -> dict[str, Any]:
    ignored = sorted(
        {
            item.tensor_name.removesuffix(".weight")
            for item in plan.allocations
            if item.method == "preserve" and item.tensor_name.endswith(".weight")
        }
    )
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "nvfp4": {
                "targets": ["Linear"],
                "format": "nvfp4-pack-quantized",
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "dynamic": False,
                    "scale_dtype": "float8_e4m3fn",
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ignored,
    }


def convert_cuda_nvfp4(
    model_dir: str | Path,
    plan: CudaQuantizationPlan,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    rows_per_chunk: int = 256,
    allow_unmeasured: bool = False,
) -> CudaPackManifest:
    """Export NVFP4A16 bytes atomically; CUDA is required unless CPU is explicit."""
    _require_unmeasured(allow_unmeasured)
    if rows_per_chunk < 1:
        raise PlanningError("rows_per_chunk must be positive")
    if (
        device != "cpu"
        and device != "cuda"
        and not (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise PlanningError("device must be cpu, cuda or cuda:<index>")
    source = Path(model_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ArtifactError("CUDA output directory already exists")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ArtifactError("CUDA source and output directories must not overlap")
    _verify_source(source, plan)
    expected = plan_cuda_nvfp4(
        source,
        model_id=plan.model_id,
        revision=plan.revision,
        keep_patterns=plan.keep_patterns,
        allow_unmeasured=True,
    )
    if stable_sha256(expected) != stable_sha256(plan):
        raise PlanningError("CUDA plan differs from the bound source or protection policy")
    torch = _torch()
    if device != "cpu" and not torch.cuda.is_available():
        raise ArtifactError("NVFP4 CUDA execution requires an available CUDA device")
    from safetensors.torch import load_file, save_file

    allocations = {item.tensor_name: item for item in plan.allocations}
    selected = {name for name, item in allocations.items() if item.method == "nvfp4"}
    group_maxima: dict[str, float] = {}
    grouped_files = sorted({item.source_file for item in plan.allocations if item.scale_group})
    for relative in grouped_files:
        group_tensors = load_file(str(source / relative), device="cpu")
        for name, weight in group_tensors.items():
            group_allocation = allocations[name]
            if group_allocation.scale_group is not None:
                maximum = float(weight.abs().amax().float().item())
                if not math.isfinite(maximum):
                    raise ArtifactError("NVFP4 source weights must be finite")
                group = group_allocation.scale_group
                group_maxima[group] = max(group_maxima.get(group, 0), maximum)
        del group_tensors
    group_scales = {group: _global_scale(maximum) for group, maximum in group_maxima.items()}
    generated = {
        name.removesuffix(".weight") + suffix
        for name in selected
        for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale")
    }
    if generated & allocations.keys():
        raise PlanningError("NVFP4 generated tensor names collide with source tensors")
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
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.nvfp4-", dir=output.parent))
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
                if item.method == "preserve":
                    transformed[name] = tensor
                else:
                    packed = (
                        quantize_nvfp4_reference(
                            tensor.float().numpy(),
                            global_scale=group_scales.get(item.scale_group or ""),
                        )
                        if device == "cpu"
                        else quantize_nvfp4_cuda(
                            tensor,
                            device=device,
                            rows_per_chunk=rows_per_chunk,
                            global_scale=group_scales.get(item.scale_group or ""),
                        )
                    )
                    prefix = name.removesuffix(".weight")
                    transformed[prefix + ".weight_packed"] = torch.from_numpy(packed.packed)
                    transformed[prefix + ".weight_scale"] = torch.from_numpy(
                        packed.scale_bytes
                    ).view(torch.float8_e4m3fn)
                    transformed[prefix + ".weight_global_scale"] = torch.tensor(
                        packed.global_scale, dtype=torch.float32
                    )
                output_names = (
                    [name]
                    if item.method == "preserve"
                    else [
                        name.removesuffix(".weight") + suffix
                        for suffix in (".weight_packed", ".weight_scale", ".weight_global_scale")
                    ]
                )
                if name in main_names:
                    for output_name in output_names:
                        weight_map[output_name] = member.path
                        saved = transformed[output_name]
                        total_size += saved.numel() * saved.element_size()
            if selected.isdisjoint(tensors):
                # Preserve external MTP and other untouched files byte for byte.
                shutil.copyfile(source / member.path, destination)
            else:
                save_file(transformed, str(destination), metadata={"format": "pt"})
        if visited != set(allocations):
            raise PlanningError("CUDA backend did not visit every planned tensor")
        config = read_data(source / "config.json")
        config["torch_dtype"] = plan.activation_dtype
        if "dtype" in config:
            config["dtype"] = plan.activation_dtype
        config["quantization_config"] = _quantization_config(plan)
        write_data(staging / "config.json", config)
        write_data(
            staging / _INDEX_NAME,
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        write_data(staging / PLAN_NAME, plan)
        manifest = CudaPackManifest(
            activation_dtype=plan.activation_dtype,
            backend="numpy-reference" if device == "cpu" else "torch-cuda",
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
