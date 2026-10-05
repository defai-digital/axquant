"""Native, unmeasured mixed NVFP4+FP8 (six-bit budget class) CUDA export.

The plan starts from the NVFP4 protection and source-binding policy and
promotes whole allocation units to FP8, in ascending unit size, until the
language-trunk bits per weight reaches the requested budget. Promotion order is
a documented unmeasured heuristic, not measured sensitivity.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from axquant.cuda import (
    _INDEX_NAME,
    MANIFEST_NAME,
    PLAN_NAME,
    _digest,
    _expert_unit,
    _fused_scale_group,
    _torch,
    _verify_source,
    plan_cuda_nvfp4,
)
from axquant.errors import ArtifactError, PlanningError
from axquant.fp8 import quantize_fp8_channel
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.nvfp4 import _global_scale, quantize_nvfp4_cuda, quantize_nvfp4_reference
from axquant.schema.cuda import NVFP4_ROLES
from axquant.schema.cuda_mix import (
    CudaMixPackManifest,
    CudaMixQuantizationPlan,
    CudaMixTensorAllocation,
    allocation_unit,
)
from axquant.schema.enums import TensorRole
from axquant.serde import read_data, stable_sha256, write_data

_PROMOTED_REASON = "Promoted to FP8 channel RTN to reach the six-bit trunk budget"
_PRESERVE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "F64": 64}
_NVFP4_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_global_scale")


def _require_unmeasured(allowed: bool) -> None:
    if not allowed:
        raise PlanningError("CUDA mixed RTN is unmeasured; explicitly pass --allow-unmeasured")


def _quantized_bytes(method: str, parameters: int, rows: int) -> int:
    if method == "nvfp4":
        return parameters * 9 // 16 + 4
    return parameters + rows * 4


def trunk_bpw(allocations: list[CudaMixTensorAllocation]) -> float:
    """Realized language-trunk bits per weight for the current allocation mix.

    The trunk is the attention, MLP and expert matrices that carry quantized
    weights. Protected tensors and non-trunk roles are excluded, so the value
    describes the selected language matrices rather than the whole checkpoint.
    """
    parameters = 0
    encoded = 0
    for item in allocations:
        if item.method == "preserve" or item.role not in NVFP4_ROLES:
            continue
        rows, columns = item.shape
        parameters += rows * columns
        encoded += _quantized_bytes(item.method, rows * columns, rows)
    if parameters == 0:
        return 0.0
    return 8 * encoded / parameters


def _promotion_unit(name: str) -> str:
    """Return the promotion unit for a tensor name.

    Self-attention projections share one unit because vLLM builds them without
    a module prefix and can only match them by class, so the whole attention
    block must carry a single method.
    """

    if ".self_attn." in name:
        return name.split(".self_attn.")[0] + ".self_attn"
    return allocation_unit(name) or name


def promote_units_to_target(
    allocations: list[CudaMixTensorAllocation], target_bpw: float
) -> list[CudaMixTensorAllocation]:
    """Promote whole NVFP4 allocation units to FP8 until the trunk reaches the target.

    Units are promoted in ascending parameter count with the unit key as the
    deterministic tie-break. Every member of a fused projection, an expert table
    or the attention block moves together, so a unit never mixes packed and
    source-precision inputs.
    """
    updated = list(allocations)
    units: dict[str, list[int]] = {}
    for index, item in enumerate(updated):
        if item.method == "nvfp4":
            key = _promotion_unit(item.tensor_name)
            units.setdefault(key, []).append(index)
    order = sorted(
        units.items(),
        key=lambda entry: (
            sum(updated[index].shape[0] * updated[index].shape[1] for index in entry[1]),
            entry[0],
        ),
    )
    for _key, indices in order:
        if trunk_bpw(updated) >= target_bpw:
            break
        for index in indices:
            updated[index] = updated[index].model_copy(
                update={"method": "fp8", "scale_group": None, "reason": _PROMOTED_REASON}
            )
    return updated


def _estimated_weight_bytes(allocations: list[CudaMixTensorAllocation]) -> int:
    total = 0
    for item in allocations:
        parameters = math.prod(item.shape)
        if item.method == "preserve":
            total += parameters * _PRESERVE_BITS.get(item.dtype, 32) // 8
        else:
            total += _quantized_bytes(item.method, parameters, item.shape[0])
    return total


def plan_cuda_mix(
    model_dir: str | Path,
    *,
    target_bpw: float = 6.0,
    model_id: str | None = None,
    revision: str | None = None,
    keep_patterns: list[str] | None = None,
    allow_unmeasured: bool = False,
) -> CudaMixQuantizationPlan:
    """Allocate eligible trunk matrices to a mixed NVFP4/FP8 six-bit budget class."""
    _require_unmeasured(allow_unmeasured)
    if not math.isfinite(target_bpw) or target_bpw <= 0:
        raise PlanningError("target_bpw must be positive and finite")
    base = plan_cuda_nvfp4(
        model_dir,
        model_id=model_id,
        revision=revision,
        keep_patterns=keep_patterns,
        allow_unmeasured=True,
    )
    allocations = [
        CudaMixTensorAllocation(
            tensor_name=item.tensor_name,
            source_file=item.source_file,
            shape=item.shape,
            dtype=item.dtype,
            role=item.role,
            method="nvfp4" if item.method == "nvfp4" else "preserve",
            scale_group=item.scale_group,
            reason=item.reason,
        )
        for item in base.allocations
    ]
    allocations = promote_units_to_target(allocations, target_bpw)
    methods = {item.method for item in allocations}
    if "nvfp4" not in methods or "fp8" not in methods:
        raise PlanningError("target_bpw cannot be realized as an NVFP4/FP8 mix")
    return CudaMixQuantizationPlan(
        target_bpw=target_bpw,
        trunk_bpw=trunk_bpw(allocations),
        activation_dtype=base.activation_dtype,
        model_id=base.model_id,
        revision=base.revision,
        model_type=base.model_type,
        config_sha256=base.config_sha256,
        source_files=base.source_files,
        allocations=allocations,
        keep_patterns=base.keep_patterns,
        estimated_weight_bytes=_estimated_weight_bytes(allocations),
    )


def _regex_target(name: str) -> str:
    """Anchored regex target matching a checkpoint module under any runtime wrapper.

    vLLM builds runtime module names with a backend prefix such as
    ``language_model.model.`` in front of the checkpoint path, so exact name
    targets never match. An anchored ``.*`` regex matches the checkpoint path
    regardless of that prefix.
    """

    return "re:.*" + re.escape(name) + "$"


def _combined_target(names: set[str]) -> list[str]:
    if not names:
        return []
    alternation = "|".join(sorted(re.escape(name) for name in names))
    return ["re:.*(?:" + alternation + ")$"]


def _mix_quantization_config(plan: CudaMixQuantizationPlan) -> dict[str, Any]:
    nvfp4_names: set[str] = set()
    fp8_names: set[str] = set()
    for item in plan.allocations:
        if item.method == "preserve":
            continue
        names = nvfp4_names if item.method == "nvfp4" else fp8_names
        names.add(item.tensor_name.removesuffix(".weight"))
        fused = _fused_scale_group(item.tensor_name)
        if fused is not None:
            names.add(fused)
        experts = _expert_unit(item.tensor_name)
        if experts is not None:
            names.add(experts)
    preserve_names: set[str] = {
        item.tensor_name.removesuffix(".weight")
        for item in plan.allocations
        if item.method == "preserve" and item.tensor_name.endswith(".weight")
    }
    preserve_names.update(
        unit
        for item in plan.allocations
        if item.method == "preserve" and (unit := _expert_unit(item.tensor_name)) is not None
    )
    preserve_names.update(
        group
        for item in plan.allocations
        if item.method == "preserve" and (group := _fused_scale_group(item.tensor_name)) is not None
    )
    attention_methods = {
        item.method
        for item in plan.allocations
        if ".self_attn." in item.tensor_name and item.method != "preserve"
    }
    if len(attention_methods) != 1:
        raise PlanningError(
            "vLLM builds self-attention projections without a module prefix; "
            "a mixed pack requires exactly one uniform self-attention method"
        )
    # DeepseekV2 builds qkv_proj/o_proj without a prefix, so those modules can
    # only be matched by the "Linear" class target, not by name or regex.
    method_targets = {
        "nvfp4": _combined_target(nvfp4_names),
        "fp8": _combined_target(fp8_names),
    }
    method_targets[next(iter(attention_methods))] = ["Linear"]
    ignored = _combined_target(preserve_names)
    if any(item.tensor_name.startswith("mtp.") for item in plan.allocations):
        # Fused runtime MoE adds virtual projections absent from source tensors.
        protected_mtp = r"(?:model\.)?mtp(?:\..*)?"
        if any(
            item.method != "preserve" and re.fullmatch(protected_mtp, item.tensor_name)
            for item in plan.allocations
        ):
            raise PlanningError("runtime MTP protection overlaps a selected CUDA tensor")
        ignored.append("re:" + protected_mtp)
    if any(item.role in {TensorRole.VISION, TensorRole.AUDIO} for item in plan.allocations):
        # Runtime vision wrappers may rename internal paths (transformer -> encoder).
        protected = r".*(?:vision|visual|sam_model|projector|view_sep|image_newline|audio).*"
        if any(
            item.method != "preserve" and re.fullmatch(protected, item.tensor_name)
            for item in plan.allocations
        ):
            raise PlanningError("runtime protection pattern overlaps a selected CUDA tensor")
        ignored.append("re:" + protected)
    return {
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "config_groups": {
            "nvfp4": {
                "format": "nvfp4-pack-quantized",
                "targets": method_targets["nvfp4"],
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
            },
            "fp8": {
                "format": "float-quantized",
                "targets": method_targets["fp8"],
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
            },
        },
        "ignore": ignored,
    }


def convert_cuda_mix(
    model_dir: str | Path,
    plan: CudaMixQuantizationPlan,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    rows_per_chunk: int = 256,
    allow_unmeasured: bool = False,
) -> CudaMixPackManifest:
    """Export mixed NVFP4/FP8 bytes atomically; CUDA is required unless CPU is explicit."""
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
    expected = plan_cuda_mix(
        source,
        target_bpw=plan.target_bpw,
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
    nvfp4_names = {name for name, item in allocations.items() if item.method == "nvfp4"}
    fp8_names = {name for name, item in allocations.items() if item.method == "fp8"}
    selected = nvfp4_names | fp8_names
    generated = {
        name.removesuffix(".weight") + suffix for name in nvfp4_names for suffix in _NVFP4_SUFFIXES
    }
    generated |= {name.removesuffix(".weight") + ".weight_scale" for name in fp8_names}
    if generated & allocations.keys():
        raise PlanningError("CUDA mix generated tensor names collide with source tensors")
    group_maxima: dict[str, float] = {}
    grouped_files = sorted({item.source_file for item in allocations.values() if item.scale_group})
    for relative in grouped_files:
        group_tensors = load_file(str(source / relative), device="cpu")
        for name, weight in group_tensors.items():
            group_allocation = allocations.get(name)
            if group_allocation is None or group_allocation.scale_group is None:
                continue
            maximum = float(weight.abs().amax().float().item())
            if not math.isfinite(maximum):
                raise ArtifactError("NVFP4 source weights must be finite")
            group = group_allocation.scale_group
            group_maxima[group] = max(group_maxima.get(group, 0), maximum)
        del group_tensors
    group_scales = {group: _global_scale(maximum) for group, maximum in group_maxima.items()}
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
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.mix-", dir=output.parent))
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
                prefix = name.removesuffix(".weight")
                if item.method == "preserve":
                    transformed[name] = tensor
                elif item.method == "nvfp4":
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
                    transformed[prefix + ".weight_packed"] = torch.from_numpy(packed.packed)
                    transformed[prefix + ".weight_scale"] = torch.from_numpy(
                        packed.scale_bytes
                    ).view(torch.float8_e4m3fn)
                    transformed[prefix + ".weight_global_scale"] = torch.tensor(
                        packed.global_scale, dtype=torch.float32
                    )
                else:
                    encoded, scales = quantize_fp8_channel(
                        tensor, device=device, rows_per_chunk=rows_per_chunk
                    )
                    transformed[name] = encoded
                    transformed[prefix + ".weight_scale"] = scales
                if item.method == "preserve":
                    output_names = [name]
                elif item.method == "nvfp4":
                    output_names = [prefix + suffix for suffix in _NVFP4_SUFFIXES]
                else:
                    output_names = [name, prefix + ".weight_scale"]
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
        config["quantization_config"] = _mix_quantization_config(plan)
        write_data(staging / "config.json", config)
        write_data(
            staging / _INDEX_NAME,
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        write_data(staging / PLAN_NAME, plan)
        manifest = CudaMixPackManifest(
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
