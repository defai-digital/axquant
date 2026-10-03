"""Staged, source-bound MXFP6 reference export and verified tensor readback."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from axquant.errors import ArtifactError, PlanningError
from axquant.identity import semantic_plan_sha256
from axquant.inspector import inspect_model
from axquant.mxfp6 import (
    Mxfp6Format,
    Mxfp6Tensor,
    _require_numpy,
    dequantize_mxfp6,
    mxfp6_storage_bytes,
    quantize_mxfp6,
)
from axquant.schema import (
    OutlierStrategy,
    QuantizationPlan,
    QuantMethod,
    SourcePlanBinding,
    SupportTier,
    TensorRole,
    TensorSpec,
)
from axquant.schema.loading import load_versioned
from axquant.schema.mxfp6 import Mxfp6File, Mxfp6PackManifest, Mxfp6TensorRecord
from axquant.serde import file_sha256, write_data
from axquant.source_binding import source_binding_issues

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

MANIFEST_NAME = "axquant_mxfp6.json"
_CHUNK_BYTES = 4 * 1024 * 1024
_QUANTIZABLE_ROLES = frozenset({TensorRole.ATTENTION, TensorRole.MLP, TensorRole.EXPERT})
_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def _header(path: Path) -> tuple[int, dict[str, Any]]:
    from safetensors import SafetensorError, safe_open

    # Let the public parser validate offsets, dtypes, lengths, and overlap first.
    try:
        with safe_open(path, framework="numpy"):
            pass
    except SafetensorError as exc:
        raise ArtifactError("invalid Safetensors payload in MXFP6 artifact") from exc
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        data: dict[str, Any] = json.loads(handle.read(size))
    return 8 + size, data


def _read_exact(handle: BinaryIO, length: int) -> bytes:
    data = handle.read(length)
    if len(data) != length:
        raise ArtifactError("Safetensors payload ended during MXFP6 export/readback")
    return data


def _float_array(data: bytes, dtype: str) -> NDArray[np.generic]:
    import numpy as np

    if dtype == "BF16":
        words = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        return words.view(np.float32)
    if dtype in {"F16", "F32"}:
        return cast(
            "NDArray[np.generic]", np.frombuffer(data, dtype="<f2" if dtype == "F16" else "<f4")
        )
    raise ArtifactError(f"unsupported MXFP6 reference floating dtype: {dtype}")


def _file_record(path: Path, root: Path) -> Mxfp6File:
    return Mxfp6File(
        path=path.relative_to(root).as_posix(),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _validate_source(
    source: Path, plan: QuantizationPlan, binding: SourcePlanBinding | None
) -> list[TensorSpec]:
    if binding is None:
        raise PlanningError("MXFP6 export requires the plan-bound source binding sidecar")
    issues = source_binding_issues(binding=binding, plan=plan, source_dir=source)
    if issues:
        raise PlanningError("MXFP6 source binding mismatch: " + "; ".join(issues))
    inventory = inspect_model(
        source, model_id=plan.source_model.model_id, revision=plan.source_model.revision
    )
    if inventory.architecture_profile.support_tier is SupportTier.INSPECT_ONLY:
        raise PlanningError("MXFP6 export requires a convertible architecture adapter")
    expected = {item.tensor: item for item in plan.assignments}
    actual = {item.name: item for item in inventory.tensors}
    if len(expected) != len(plan.assignments) or expected.keys() != actual.keys():
        raise PlanningError("MXFP6 source tensor coverage does not match the plan")
    if inventory.architecture_profile != plan.architecture_profile:
        raise PlanningError("MXFP6 source architecture does not match the plan")
    candidates: list[TensorSpec] = []
    for name, tensor in actual.items():
        allocation = expected[name]
        if (tensor.module_path, tensor.role, tensor.parameters) != (
            allocation.module_path,
            allocation.role,
            allocation.parameters,
        ):
            raise PlanningError(f"MXFP6 source tensor no longer matches its allocation: {name}")
        if allocation.bits >= 8:
            continue
        if (
            allocation.bits != 6
            or allocation.method is not QuantMethod.AFFINE
            or allocation.strategy_metadata.get("physical_mode") not in {None, "affine"}
            or allocation.outlier_strategy is not OutlierStrategy.NONE
            or tensor.role not in _QUANTIZABLE_ROLES
            or not tensor.quantizable
            or tensor.protected_recommendation
            or tensor.dtype not in _FLOAT_BYTES
            or len(tensor.shape) < 2
            or not tensor.shape[-1]
            or tensor.shape[-1] % 32
        ):
            raise PlanningError(
                f"MXFP6 export requires unrefined affine6 assignments on eligible "
                f"block-aligned language weights; incompatible allocation: {name}"
            )
        candidates.append(tensor)
    if not candidates:
        raise PlanningError("the plan has no eligible affine6 tensors to export as MXFP6")
    return inventory.tensors


def _write_shard(
    source: Path,
    destination: Path,
    records: list[Mxfp6TensorRecord],
    element_format: Mxfp6Format,
) -> None:
    data_start, source_header = _header(source)
    if all(item.encoding == "preserved" for item in records):
        shutil.copyfile(source, destination)
        return
    header: dict[str, Any] = {"__metadata__": {"format": "axquant.mxfp6-pack.v1"}}
    offset = 0
    for item in records:
        if item.encoding == "preserved":
            descriptor = dict(source_header[item.name])
            length = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
            descriptor["data_offsets"] = [offset, offset + length]
            header[item.data_key] = descriptor
            offset += length
        else:
            for key, shape in (
                (item.data_key, (*item.shape[:-1], item.shape[-1] * 3 // 4)),
                (item.scales_key, (*item.shape[:-1], item.shape[-1] // 32)),
            ):
                assert key is not None
                length = math.prod(shape)
                header[key] = {
                    "dtype": "U8",
                    "shape": shape,
                    "data_offsets": [offset, offset + length],
                }
                offset += length
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    output_start = 8 + len(encoded)
    with source.open("rb") as reader, destination.open("wb") as writer:
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        for item in records:
            start, end = source_header[item.name]["data_offsets"]
            reader.seek(data_start + start)
            if item.encoding == "preserved":
                writer.seek(output_start + header[item.data_key]["data_offsets"][0])
                remaining = end - start
                while remaining:
                    chunk = _read_exact(reader, min(remaining, _CHUNK_BYTES))
                    writer.write(chunk)
                    remaining -= len(chunk)
                continue
            width = item.shape[-1]
            row_bytes = width * _FLOAT_BYTES[item.source_dtype]
            chunk_rows = max(1, _CHUNK_BYTES // row_bytes)
            rows = math.prod(item.shape[:-1])
            assert item.scales_key is not None
            packed_start = output_start + header[item.data_key]["data_offsets"][0]
            scales_start = output_start + header[item.scales_key]["data_offsets"][0]
            for row in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - row)
                weights = _float_array(_read_exact(reader, count * row_bytes), item.source_dtype)
                quantized = quantize_mxfp6(
                    weights.reshape(count, width), element_format=element_format
                )
                writer.seek(packed_start + row * width * 3 // 4)
                writer.write(quantized.packed.tobytes())
                writer.seek(scales_start + row * width // 32)
                writer.write(quantized.scales.tobytes())
        writer.truncate(output_start + offset)
    _header(destination)


def export_mxfp6(
    model: str | Path,
    plan: QuantizationPlan,
    output: str | Path,
    *,
    element_format: Mxfp6Format = "e2m3",
    allow_unmeasured: bool = False,
    source_binding: SourcePlanBinding | None = None,
) -> Mxfp6PackManifest:
    """Export a local source as reference-only MXFP6; never mutate the input plan."""
    if not allow_unmeasured:
        raise PlanningError("experimental MXFP6 export requires --allow-unmeasured")
    if element_format not in {"e2m3", "e3m2"}:
        raise PlanningError(f"unsupported MXFP6 element format: {element_format}")
    _require_numpy()
    source = Path(model).expanduser().resolve()
    destination = Path(output).expanduser().absolute()
    if not source.is_dir():
        raise ArtifactError("MXFP6 export requires a local checkpoint directory")
    if destination.exists() or destination.is_symlink():
        raise ArtifactError("MXFP6 export destination already exists")
    if source == destination.resolve() or source in destination.resolve().parents:
        raise ArtifactError("MXFP6 output must be outside the source checkpoint")
    tensors = _validate_source(source, plan, source_binding)
    assert source_binding is not None
    sources = [_file_record(source / name, source) for name in sorted({t.file for t in tensors})]
    before_config = file_sha256(source / "config.json")
    assignments = {item.tensor: item for item in plan.assignments}
    records: list[Mxfp6TensorRecord] = []
    for tensor in tensors:
        quantized = assignments[tensor.name].bits == 6
        records.append(
            Mxfp6TensorRecord(
                name=tensor.name,
                source_file=tensor.file,
                output_file=tensor.file,
                shape=tensor.shape,
                source_dtype=tensor.dtype,
                encoding="mxfp6" if quantized else "preserved",
                data_key=tensor.name + ".mxfp6" if quantized else tensor.name,
                scales_key=tensor.name + ".mxfp6_scales" if quantized else None,
                storage_bytes=mxfp6_storage_bytes(tensor.shape)
                if quantized
                else tensor.storage_bytes,
            )
        )
    by_file: dict[str, list[Mxfp6TensorRecord]] = defaultdict(list)
    for record in records:
        by_file[record.source_file].append(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.mxfp6-", dir=destination.parent))
    try:
        for name, members in by_file.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_shard(source / name, target, members, element_format)
        for item in sources:
            if _file_record(source / item.path, source) != item:
                raise ArtifactError(f"MXFP6 source changed during export: {item.path}")
            if all(record.encoding == "preserved" for record in by_file[item.path]) and (
                file_sha256(staging / item.path) != item.sha256
            ):
                raise ArtifactError("protected-only shard changed during MXFP6 export")
        if file_sha256(source / "config.json") != before_config:
            raise ArtifactError("MXFP6 source config changed during export")
        issues = source_binding_issues(binding=source_binding, plan=plan, source_dir=source)
        if issues:
            raise ArtifactError("MXFP6 source binding changed during export: " + "; ".join(issues))
        manifest = Mxfp6PackManifest(
            element_format=element_format,
            source_plan_sha256=semantic_plan_sha256(plan),
            source_config_sha256=before_config,
            source_index_sha256=source_binding.index_sha256,
            source_files=sources,
            output_files=[_file_record(staging / name, staging) for name in sorted(by_file)],
            tensors=records,
        )
        write_data(staging / MANIFEST_NAME, manifest)
        if destination.exists() or destination.is_symlink():
            raise ArtifactError("MXFP6 export destination appeared during export")
        os.replace(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_mxfp6_tensor(directory: str | Path, name: str) -> NDArray[np.generic]:
    """Checksum-verified reference readback of one MXFP6 or preserved tensor."""
    _require_numpy()
    import numpy as np

    root = Path(directory).expanduser().resolve()
    manifest = load_versioned(root / MANIFEST_NAME, Mxfp6PackManifest)
    tensor = next((item for item in manifest.tensors if item.name == name), None)
    if tensor is None:
        raise ArtifactError(f"tensor is not in the MXFP6 manifest: {name}")
    member = next(item for item in manifest.output_files if item.path == tensor.output_file)
    path = root / member.path
    if root not in path.resolve().parents:
        raise ArtifactError("MXFP6 payload resolves outside its artifact directory")
    if _file_record(path, root) != member:
        raise ArtifactError(f"MXFP6 payload checksum/size mismatch: {member.path}")
    start, header = _header(path)
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    for item in manifest.tensors:
        if item.output_file == member.path:
            expected[item.data_key] = (
                ("U8", (*item.shape[:-1], item.shape[-1] * 3 // 4))
                if item.encoding == "mxfp6"
                else (item.source_dtype, item.shape)
            )
            if item.scales_key is not None:
                expected[item.scales_key] = ("U8", (*item.shape[:-1], item.shape[-1] // 32))
    if set(header) - {"__metadata__"} != expected.keys():
        raise ArtifactError("MXFP6 payload tensor coverage differs from its manifest")
    for key, (dtype, shape) in expected.items():
        entry = header[key]
        if entry["dtype"] != dtype or tuple(entry["shape"]) != shape:
            raise ArtifactError(f"MXFP6 payload layout differs from its manifest: {key}")
    with path.open("rb") as reader:
        entry = header[tensor.data_key]
        lo, hi = entry["data_offsets"]
        reader.seek(start + lo)
        data = _read_exact(reader, hi - lo)
        if tensor.encoding == "preserved":
            if tensor.source_dtype in _FLOAT_BYTES:
                return _float_array(data, tensor.source_dtype).reshape(tensor.shape)
            from safetensors import safe_open

            with safe_open(path, framework="numpy") as handle:
                return np.asarray(handle.get_tensor(tensor.data_key))
        assert tensor.scales_key is not None
        scales_entry = header[tensor.scales_key]
        lo, hi = scales_entry["data_offsets"]
        reader.seek(start + lo)
        scales = np.frombuffer(_read_exact(reader, hi - lo), dtype=np.uint8)
        packed = np.frombuffer(data, dtype=np.uint8)
    return dequantize_mxfp6(
        Mxfp6Tensor(
            packed=packed.reshape(entry["shape"]),
            scales=scales.reshape(scales_entry["shape"]),
            shape=tensor.shape,
            element_format=manifest.element_format,
        )
    )
