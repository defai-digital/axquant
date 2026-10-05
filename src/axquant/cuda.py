"""Native, unmeasured NVFP4A16 planning and atomic CUDA checkpoint export."""

from __future__ import annotations

import fnmatch
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol

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
from axquant.schema.cuda_activation import (
    CudaActivationCalibration,
    CudaW4A4PackManifest,
    CudaW4A4Plan,
)
from axquant.schema.enums import SupportTier, TensorRole
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
    "chat_template*.json",
    "*processor_config.json",
    "*.py",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
)
_POOLING_ASSETS = ("1_Pooling/config.json", "2_Normalize/config.json")


class SourceBoundPlan(Protocol):
    """Any CUDA plan whose source members are bound by exact content digests."""

    source_files: list[CudaFileDigest]


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


def _expert_unit(name: str) -> str | None:
    match = re.fullmatch(
        r"(.+\.experts)\.\d+\.(?:gate_proj|up_proj|down_proj|w[123])\.weight", name
    )
    return match.group(1) if match else None


def _w4a4_scale_group(name: str) -> str:
    unit = _expert_unit(name)
    if unit is not None:
        down = name.endswith((".down_proj.weight", ".w2.weight"))
        return unit + (".w2" if down else ".w13")
    return _fused_scale_group(name) or name.removesuffix(".weight")


def plan_cuda_nvfp4(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    keep_patterns: list[str] | None = None,
    embedding_protection: bool = False,
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
    if embedding_protection:
        config = read_data(source / "config.json")
        layers = config.get("num_hidden_layers")
        if (
            config.get("model_type") != "qwen3"
            or not isinstance(layers, int)
            or isinstance(layers, bool)
            or layers < 5
            or not all(
                (source / name).is_file()
                for name in (
                    "modules.json",
                    "1_Pooling/config.json",
                    "config_sentence_transformers.json",
                )
            )
        ):
            raise PlanningError("Embedding protection requires a Qwen3 embedding checkpoint")
        pooling = read_data(source / "1_Pooling/config.json")
        enabled = [
            key for key, value in pooling.items() if key.startswith("pooling_mode_") and value
        ]
        if enabled != ["pooling_mode_lasttoken"]:
            raise PlanningError("Qwen3 embedding protection requires last-token pooling")
        flat = any(tensor.name.startswith("layers.") for tensor in inventory.tensors)
        prefix = "" if flat else "model."
        patterns = sorted(
            set(patterns)
            | {
                prefix + "layers.*.self_attn.*",
                *(f"{prefix}layers.{index}.mlp.*" for index in (0, 1, layers - 2, layers - 1)),
            }
        )
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
    expert_units: dict[str, list[int]] = {}
    for index, allocation in enumerate(allocations):
        unit = _expert_unit(allocation.tensor_name)
        if unit is not None:
            expert_units.setdefault(unit, []).append(index)
    for indices in expert_units.values():
        if all(allocations[index].method == "nvfp4" for index in indices):
            continue
        for index in indices:
            allocations[index] = allocations[index].model_copy(
                update={
                    "method": "preserve",
                    "scale_group": None,
                    "reason": "Preserve complete fused runtime expert table",
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
    members.update(name for name in _POOLING_ASSETS if (source / name).exists())
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


def _verify_source(source: Path, plan: SourceBoundPlan) -> None:
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
    expert_units = {
        unit
        for item in plan.allocations
        if item.method == "preserve" and (unit := _expert_unit(item.tensor_name)) is not None
    }
    ignored.extend(sorted(expert_units))
    fused_units = {
        group
        for item in plan.allocations
        if item.method == "preserve" and (group := _fused_scale_group(item.tensor_name)) is not None
    }
    ignored.extend(sorted(fused_units))
    if any(item.tensor_name.startswith("mtp.") for item in plan.allocations):
        # Fused runtime MoE adds virtual projections absent from source tensors.
        protected_mtp = r"(?:model\.)?mtp(?:\..*)?"
        if any(
            item.method == "nvfp4" and re.fullmatch(protected_mtp, item.tensor_name)
            for item in plan.allocations
        ):
            raise PlanningError("runtime MTP protection overlaps a selected NVFP4 tensor")
        ignored.append("re:" + protected_mtp)
    if any(item.tensor_name.startswith("layers.") for item in plan.allocations):
        # Base embedding checkpoints omit the wrapper prefix added by vLLM.
        ignored = sorted(set(ignored) | {"model." + name for name in ignored})
    if any(item.role in {TensorRole.VISION, TensorRole.AUDIO} for item in plan.allocations):
        # Runtime vision wrappers may rename internal paths (transformer -> encoder).
        protected = r".*(?:vision|visual|sam_model|projector|view_sep|image_newline|audio).*"
        if any(
            item.method == "nvfp4" and re.fullmatch(protected, item.tensor_name)
            for item in plan.allocations
        ):
            raise PlanningError("runtime protection pattern overlaps a selected NVFP4 tensor")
        ignored.append("re:" + protected)
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
    result = _convert_cuda_nvfp4(
        model_dir,
        plan,
        output_dir,
        device=device,
        rows_per_chunk=rows_per_chunk,
        allow_unmeasured=allow_unmeasured,
    )
    assert isinstance(result, CudaPackManifest)
    return result


def plan_cuda_nvfp4_w4a4(
    weight_plan: CudaQuantizationPlan,
    calibration: CudaActivationCalibration,
    *,
    activation_headroom: float = 1.25,
) -> CudaW4A4Plan:
    """Require exact weight identity, observed inputs and complete activation coverage."""
    if calibration.weight_plan_sha256 != stable_sha256(weight_plan):
        raise PlanningError("CUDA activation calibration does not bind this weight plan")
    if calibration.source_precision != weight_plan.activation_dtype:
        raise PlanningError("CUDA activation calibration precision differs from the source")
    selected = {
        item.tensor_name: item for item in weight_plan.allocations if item.method == "nvfp4"
    }
    stats = {item.tensor_name: item for item in calibration.statistics}
    if set(stats) != set(selected):
        raise PlanningError("CUDA activation calibration must cover every selected tensor exactly")
    for name, allocation in selected.items():
        if stats[name].input_columns != allocation.shape[1]:
            raise PlanningError(f"CUDA activation calibration input shape differs: {name}")
        _global_scale(stats[name].absolute_maximum)
    return CudaW4A4Plan(
        weight_plan=weight_plan,
        calibration=calibration,
        activation_headroom=activation_headroom,
    )


def convert_cuda_nvfp4_w4a4(
    model_dir: str | Path,
    plan: CudaW4A4Plan,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    rows_per_chunk: int = 256,
    allow_unmeasured: bool = False,
) -> CudaW4A4PackManifest:
    """Export calibrated W4A4 with shared global scales for fused expert tables."""
    verified = plan_cuda_nvfp4_w4a4(
        plan.weight_plan,
        plan.calibration,
        activation_headroom=plan.activation_headroom,
    )
    if stable_sha256(verified) != stable_sha256(plan):
        raise PlanningError("CUDA W4A4 plan differs from its calibration binding")
    result = _convert_cuda_nvfp4(
        model_dir,
        plan.weight_plan,
        output_dir,
        device=device,
        rows_per_chunk=rows_per_chunk,
        allow_unmeasured=allow_unmeasured,
        calibration=plan.calibration,
        activation_headroom=plan.activation_headroom,
    )
    assert isinstance(result, CudaW4A4PackManifest)
    return result


def _convert_cuda_nvfp4(
    model_dir: str | Path,
    plan: CudaQuantizationPlan,
    output_dir: str | Path,
    *,
    device: str,
    rows_per_chunk: int,
    allow_unmeasured: bool,
    calibration: CudaActivationCalibration | None = None,
    activation_headroom: float = 1.25,
) -> CudaPackManifest | CudaW4A4PackManifest:
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
    input_scales: dict[str, float] = {}
    artifact_plan: CudaQuantizationPlan | CudaW4A4Plan = plan
    if calibration is not None:
        artifact_plan = plan_cuda_nvfp4_w4a4(
            plan,
            calibration,
            activation_headroom=activation_headroom,
        )
        maxima: dict[str, float] = {}
        for stat in calibration.statistics:
            group = _w4a4_scale_group(stat.tensor_name)
            maxima[group] = max(maxima.get(group, 0), stat.absolute_maximum)
        input_scales = {
            group: _global_scale(maximum * activation_headroom) for group, maximum in maxima.items()
        }
        allocations = {
            name: item.model_copy(update={"scale_group": _w4a4_scale_group(name)})
            if name in selected
            else item
            for name, item in allocations.items()
        }
    group_maxima: dict[str, float] = {}
    grouped_files = sorted({item.source_file for item in allocations.values() if item.scale_group})
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
    suffixes: tuple[str, ...] = (".weight_packed", ".weight_scale", ".weight_global_scale")
    if calibration is not None:
        suffixes += (".input_global_scale",)
    generated = {name.removesuffix(".weight") + suffix for name in selected for suffix in suffixes}
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
                    if calibration is not None:
                        transformed[prefix + ".input_global_scale"] = torch.tensor(
                            input_scales[_w4a4_scale_group(name)], dtype=torch.float32
                        )
                output_names = (
                    [name]
                    if item.method == "preserve"
                    else [name.removesuffix(".weight") + suffix for suffix in suffixes]
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
        if calibration is not None:
            config["quantization_config"]["config_groups"]["nvfp4"]["input_activations"] = {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor_group",
                "group_size": 16,
                "dynamic": "local",
                "scale_dtype": "float8_e4m3fn",
            }
        write_data(staging / "config.json", config)
        write_data(
            staging / _INDEX_NAME,
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        )
        write_data(staging / PLAN_NAME, artifact_plan)
        manifest_data: dict[str, Any] = dict(
            activation_dtype=plan.activation_dtype,
            backend="numpy-reference" if device == "cpu" else "torch-cuda",
            plan_sha256=stable_sha256(artifact_plan),
            quantized_tensors=sorted(selected),
            files=[
                _digest(staging, path.relative_to(staging).as_posix())
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ],
        )
        manifest: CudaPackManifest | CudaW4A4PackManifest
        if calibration is None:
            manifest = CudaPackManifest(**manifest_data)
        else:
            manifest = CudaW4A4PackManifest(
                **manifest_data,
                calibration_sha256=stable_sha256(calibration),
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
