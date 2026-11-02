"""MLX matrix packing for Unlimited-OCR-2 storage, without an inference graph."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from axquant.errors import ArtifactError, PlanningError
from axquant.inspector import inspect_model
from axquant.predicate import PlanPredicate, allocation_quant_params
from axquant.schema import Inventory, QuantizationPlan, QuantMethod, SupportTier, TensorRole
from axquant.serde import file_sha256, read_data, write_data

_QUANTIZED_ROLES = {TensorRole.ATTENTION, TensorRole.MLP, TensorRole.EXPERT}
_ASSETS = (
    "*.py",
    "tokenizer*.json",
    "special_tokens_map.json",
    "generation_config.json",
    "*processor_config.json",
    "*.model",
    "merges.txt",
    "vocab.json",
    "chat_template*.jinja",
    "LICENSE*",
    "NOTICE*",
)


def preflight_source_mx(source: Path, plan: QuantizationPlan) -> Inventory:
    """Require exact source coverage and protected state before opening arrays."""
    inventory = inspect_model(
        source, model_id=plan.source_model.model_id, revision=plan.source_model.revision
    )
    if (
        inventory.architecture_profile.adapter_id != "unlimited-ocr2-v1"
        or inventory.architecture_profile.support_tier is SupportTier.INSPECT_ONLY
        or inventory.quantized_source
    ):
        raise PlanningError(
            "source MX packing requires the supported unquantized Unlimited-OCR-2 source"
        )
    assignments = {a.tensor: a for a in plan.assignments}
    if len(assignments) != len(plan.assignments) or set(assignments) != {
        t.name for t in inventory.tensors
    }:
        raise PlanningError("source MX plan must cover every source tensor exactly once")
    for tensor in inventory.tensors:
        allocation = assignments[tensor.name]
        if allocation.role != tensor.role or allocation.parameters != tensor.parameters:
            raise PlanningError(f"source MX allocation disagrees with source: {tensor.name}")
        if allocation.bits == 16 and allocation.method is QuantMethod.BF16:
            continue
        if (
            tensor.role not in _QUANTIZED_ROLES
            or not tensor.quantizable
            or tensor.protected_recommendation
            or tensor.tied_to is not None
            or len(tensor.shape) != 2
            or tensor.shape[-1] % 32
            or not tensor.name.endswith(".weight")
            or allocation.group_size != 32
            or (allocation.bits, allocation.method)
            not in {
                (4, QuantMethod.MXFP4),
                (8, QuantMethod.AFFINE),
            }
        ):
            raise PlanningError(f"unsupported or unprotected source MX allocation: {tensor.name}")
    return inventory


def convert_source_mx(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    *,
    q_mode: str,
) -> None:
    """Pack original matrix names; preserve all other values/dtypes and graph metadata.

    The caller owns staging, atomic publication and normal artifact verification.
    No model, remote code, tokenizer or inference runtime is instantiated here.
    """
    if q_mode not in {"mxfp4", "mxfp8"}:
        raise PlanningError("Unlimited-OCR-2 source packing requires explicit mxfp4 or mxfp8")
    inventory = preflight_source_mx(source, plan)
    assignments = {a.tensor: a for a in plan.assignments}
    # Include only permitted source members; never copy cache/credentials/runtime receipts.
    assets = {p.name: p for pattern in _ASSETS for p in source.glob(pattern)}
    members = assets | {t.file: source / t.file for t in inventory.tensors}
    members["config.json"] = source / "config.json"
    if (source / "model.safetensors.index.json").is_file():
        members["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for name, path in members.items():
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ArtifactError(f"unsafe source MX member: {name}")
    hashes = {name: file_sha256(path) for name, path in sorted(members.items())}
    mx: Any = importlib.import_module("mlx.core")
    destination.mkdir(parents=True, exist_ok=False)
    config = read_data(source / "config.json")
    bits = 4 if q_mode == "mxfp4" else 8
    quantization: dict[str, Any] = {"bits": bits, "group_size": 32, "mode": q_mode}
    index: dict[str, str] = {}
    visited: set[str] = set()
    total_bytes = 0
    for filename in sorted({t.file for t in inventory.tensors}):
        arrays = mx.load(str(source / filename))
        output: dict[str, Any] = {}
        for name, array in arrays.items():
            if name not in assignments or name in visited:
                raise PlanningError(f"unexpected or repeated source MX tensor: {name}")
            allocation = assignments[name]
            visited.add(name)
            if allocation.bits == 16:
                output[name] = array
                continue
            params = allocation_quant_params(allocation, q_mode)
            if params != {"group_size": 32, "bits": bits, "mode": q_mode}:
                raise PlanningError(f"source MX plan mixes physical formats: {name}")
            if not bool(mx.all(mx.isfinite(array)).item()):
                raise ArtifactError(f"source MX weight is non-finite: {name}")
            decision = predicate(allocation.module_path, SimpleNamespace(weight=array))
            if decision != params:
                raise PlanningError(f"source MX predicate disagrees with allocation: {name}")
            packed, scales = mx.quantize(array, **params)
            mx.eval(packed, scales)
            if (
                packed.dtype != mx.uint32
                or scales.dtype != mx.uint8
                or not bool(mx.all(scales != 255).item())
                or not bool(
                    mx.all(
                        mx.isfinite(mx.dequantize(packed, scales, **params, dtype=mx.float32))
                    ).item()
                )
            ):
                raise ArtifactError(f"source MX packing produced invalid data: {name}")
            prefix = name.removesuffix(".weight")
            for key, value in ((name, packed), (prefix + ".scales", scales)):
                if key in output or (key != name and key in assignments):
                    raise PlanningError(f"source MX generated tensor collision: {key}")
                output[key] = value
            quantization[allocation.module_path] = params
        mx.save_safetensors(str(destination / filename), output, metadata={"format": "mlx"})
        total_bytes += sum(int(v.nbytes) for v in output.values())
        index.update({key: filename for key in output})
        del arrays, output
        mx.clear_cache()
    if visited != set(assignments) or predicate.unmatched_quantized_modules():
        raise PlanningError("source MX backend did not visit every planned tensor")
    for name, path in assets.items():
        shutil.copyfile(path, destination / name)
    config["quantization"] = quantization
    config["axquant_storage_only"] = {
        "layout": "source-tensors-v1",
        "architecture": "unlimited-ocr2-v1",
        "reason": "SR-SWA inference graph is not implemented in the MLX runtimes",
    }
    write_data(destination / "config.json", config)
    write_data(
        destination / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_bytes},
            "weight_map": index,
        },
    )
    for name, path in members.items():
        if file_sha256(path) != hashes[name]:
            raise ArtifactError(f"source MX member changed during conversion: {name}")
    (destination / "SOURCE_SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8"
    )
