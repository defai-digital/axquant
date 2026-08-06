from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from axquant.errors import ArtifactError
from axquant.schema import (
    AxEngineMtpCapabilityCheck,
    BytePreservedMtpSidecarManifest,
    ModelIdentity,
    MtpSidecarFileBinding,
    PreparedMtpInputBinding,
    PreparedMtpOutputBinding,
    PreparedMtpSidecarManifest,
    PreparedMtpTensorPayload,
    PreparedMtpTransform,
    QuantizedMtpSidecarManifest,
    QuantizedMtpTensorRecord,
)
from axquant.serde import file_sha256, load_model, write_data

# Recognized root-level filenames for an externally-shipped MTP head sidecar,
# in preference order. Kept as one canonical constant because callers across
# the pipeline (inspect, convert, probe, release audit) must agree on every
# recognized name, or a checkpoint shipped under an alternate name is
# silently mishandled by whichever caller falls out of sync. A tuple (not a
# frozenset) is required: `converter._resolve_external_mtp_sidecar_file`
# picks the first match when a sidecar directory ships both names, and
# Python's per-process string-hash randomization would otherwise make that
# choice non-deterministic across runs.
EXTERNAL_MTP_SIDECAR_FILENAMES: tuple[str, ...] = ("mtp.safetensors", "mtp_head.safetensors")

QWEN36_MTP_NORM_TENSORS = frozenset(
    {
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
)

QWEN36_MTP_PROJECTION_TENSORS = frozenset(
    {
        "mtp.fc.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
    }
)

QWEN36_MTP_TENSORS = QWEN36_MTP_NORM_TENSORS | QWEN36_MTP_PROJECTION_TENSORS
QWEN36_MTP_LAYOUT = "ax-engine-qwen36-v1"
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _TensorSlice:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def byte_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class _SafetensorsLayout:
    data_base: int
    payload_bytes: int
    tensors: dict[str, _TensorSlice]


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_qwen36_layout(path: Path) -> _SafetensorsLayout:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as source:
            header_size_bytes = source.read(8)
            if len(header_size_bytes) != 8:
                raise ArtifactError(f"invalid Safetensors header: {path}")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            if header_size <= 1 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
                raise ArtifactError(f"unsafe Safetensors header size in {path}")
            header_bytes = source.read(header_size)
            if len(header_bytes) != header_size:
                raise ArtifactError(f"truncated Safetensors header: {path}")
    except OSError as exc:
        raise ArtifactError(f"cannot read MTP sidecar {path}: {exc}") from exc
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid MTP Safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ArtifactError(f"MTP Safetensors header is not an object: {path}")

    data_base = 8 + header_size
    payload_bytes = file_size - data_base
    if payload_bytes <= 0:
        raise ArtifactError(f"MTP sidecar has no tensor payload: {path}")
    tensors: dict[str, _TensorSlice] = {}
    for name, raw_entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_entry, dict):
            raise ArtifactError(f"invalid MTP tensor header entry in {path}")
        dtype = raw_entry.get("dtype")
        shape = raw_entry.get("shape")
        offsets = raw_entry.get("data_offsets")
        if dtype != "BF16":
            raise ArtifactError(f"Qwen 3.6 MTP tensor {name} must use BF16, found {dtype!r}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape
            )
        ):
            raise ArtifactError(f"invalid shape for Qwen 3.6 MTP tensor {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ArtifactError(f"invalid offsets for Qwen 3.6 MTP tensor {name}")
        start, end = offsets
        if start < 0 or end <= start or end > payload_bytes:
            raise ArtifactError(f"out-of-range payload for Qwen 3.6 MTP tensor {name}")
        elements = math.prod(shape)
        if end - start != elements * 2:
            raise ArtifactError(f"BF16 payload size does not match shape for MTP tensor {name}")
        tensors[name] = _TensorSlice(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            start=start,
            end=end,
        )

    actual_names = set(tensors)
    if actual_names != QWEN36_MTP_TENSORS:
        missing = sorted(QWEN36_MTP_TENSORS - actual_names)
        unexpected = sorted(actual_names - QWEN36_MTP_TENSORS)
        raise ArtifactError(
            f"Qwen 3.6 MTP tensor coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name in QWEN36_MTP_NORM_TENSORS:
        if len(tensors[name].shape) != 1:
            raise ArtifactError(f"Qwen 3.6 MTP norm tensor must be one-dimensional: {name}")
    for name in QWEN36_MTP_PROJECTION_TENSORS:
        if len(tensors[name].shape) != 2:
            raise ArtifactError(f"Qwen 3.6 MTP projection tensor must be two-dimensional: {name}")

    intervals = sorted((tensor.start, tensor.end, tensor.name) for tensor in tensors.values())
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            raise ArtifactError(
                f"Qwen 3.6 MTP payload is not contiguous before {name}: {start} != {cursor}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise ArtifactError(f"Qwen 3.6 MTP payload coverage mismatch: {cursor} != {payload_bytes}")
    return _SafetensorsLayout(
        data_base=data_base,
        payload_bytes=payload_bytes,
        tensors=tensors,
    )


def _payload_sha256(path: Path, layout: _SafetensorsLayout, tensor: _TensorSlice) -> str:
    digest = hashlib.sha256()
    remaining = tensor.byte_count
    try:
        with path.open("rb") as source:
            source.seek(layout.data_base + tensor.start)
            while remaining:
                chunk = source.read(min(_COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ArtifactError(f"unexpected end of MTP payload for {tensor.name}")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot hash MTP tensor {tensor.name}: {exc}") from exc
    return digest.hexdigest()


def _binding_matches(path: Path, binding: MtpSidecarFileBinding, label: str) -> None:
    relative = Path(binding.path)
    if relative.is_absolute() or relative.parts != (path.name,) or relative.name != path.name:
        raise ArtifactError(f"{label} provenance binds a different path")
    if not path.is_file():
        raise ArtifactError(f"{label} provenance target is not a regular file")
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise ArtifactError(f"cannot inspect {label}: {exc}") from exc
    if size_bytes != binding.size_bytes:
        raise ArtifactError(f"{label} provenance size does not match")
    if file_sha256(path) != binding.sha256:
        raise ArtifactError(f"{label} provenance checksum does not match")


def _validate_source_model(
    manifest: BytePreservedMtpSidecarManifest | PreparedMtpSidecarManifest,
    source_model: ModelIdentity,
) -> None:
    if (
        manifest.source.model.model_id != source_model.model_id
        or manifest.source.model.revision != source_model.revision
    ):
        raise ArtifactError("MTP sidecar source model does not match the quantization plan")


def _validated_source_shards(
    manifest: BytePreservedMtpSidecarManifest | PreparedMtpSidecarManifest,
) -> dict[str, Any]:
    shards = {shard.name: shard for shard in manifest.source.shards}
    if len(shards) != len(manifest.source.shards):
        raise ArtifactError("MTP provenance contains duplicate source shard records")
    for shard_name in shards:
        relative = Path(shard_name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
            raise ArtifactError(f"MTP provenance contains an unsafe source shard: {shard_name}")
    return shards


def _load_raw_manifest(
    manifest_path: Path,
    source: Path,
    source_model: ModelIdentity,
) -> tuple[BytePreservedMtpSidecarManifest, _SafetensorsLayout, dict[str, str]]:
    try:
        manifest = load_model(manifest_path, BytePreservedMtpSidecarManifest)
    except (ArtifactError, ValidationError) as exc:
        raise ArtifactError(f"invalid byte-preserved MTP provenance: {manifest_path}") from exc
    _validate_source_model(manifest, source_model)
    _binding_matches(source, manifest.output.mtp, "raw MTP sidecar")
    layout = _parse_qwen36_layout(source)
    if manifest.tensor_count != len(layout.tensors):
        raise ArtifactError("raw MTP provenance tensor count does not match the sidecar")
    if manifest.total_payload_bytes != layout.payload_bytes:
        raise ArtifactError("raw MTP provenance payload size does not match the sidecar")
    records = {record.name: record for record in manifest.tensor_payloads}
    if len(records) != len(manifest.tensor_payloads) or set(records) != set(layout.tensors):
        raise ArtifactError("raw MTP provenance tensor records do not match the sidecar")
    shards = _validated_source_shards(manifest)
    payload_sha256: dict[str, str] = {}
    source_intervals: dict[str, list[tuple[int, int, str]]] = {}
    for name, tensor in layout.tensors.items():
        record = records[name]
        source_shard = shards.get(record.source_shard)
        source_start, source_end = record.source_data_range
        if (
            record.dtype != tensor.dtype
            or tuple(record.shape) != tensor.shape
            or record.byte_count != tensor.byte_count
            or source_shard is None
            or source_start < 0
            or source_end <= source_start
            or source_end - source_start != tensor.byte_count
            or source_end > source_shard.size_bytes
        ):
            raise ArtifactError(f"raw MTP provenance metadata mismatch for {name}")
        source_intervals.setdefault(record.source_shard, []).append(
            (source_start, source_end, name)
        )
        digest = _payload_sha256(source, layout, tensor)
        if digest != record.sha256:
            raise ArtifactError(f"raw MTP provenance payload checksum mismatch for {name}")
        payload_sha256[name] = digest
    for shard_name, intervals in source_intervals.items():
        previous_end = -1
        previous_name = ""
        for start, end, name in sorted(intervals):
            if start < previous_end:
                raise ArtifactError(
                    "raw MTP provenance source ranges overlap in "
                    f"{shard_name}: {previous_name}, {name}"
                )
            previous_end = end
            previous_name = name
    return manifest, layout, payload_sha256


def _bf16_add_one(payload: bytes, tensor_name: str) -> bytes:
    if len(payload) % 2:
        raise ArtifactError(f"odd BF16 payload size for {tensor_name}")
    output = bytearray(len(payload))
    for offset in range(0, len(payload), 2):
        raw_word = struct.unpack_from("<H", payload, offset)[0]
        raw_float = struct.unpack("<f", struct.pack("<I", raw_word << 16))[0]
        if not math.isfinite(raw_float):
            raise ArtifactError(f"non-finite BF16 norm value in {tensor_name}")
        shifted = raw_float + 1.0
        shifted_bits = struct.unpack("<I", struct.pack("<f", shifted))[0]
        rounded = shifted_bits + 0x7FFF + ((shifted_bits >> 16) & 1)
        struct.pack_into("<H", output, offset, (rounded >> 16) & 0xFFFF)
    return bytes(output)


def _transform_norm_payloads(
    source: Path,
    destination: Path,
    layout: _SafetensorsLayout,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as output:
            for name in sorted(QWEN36_MTP_NORM_TENSORS):
                tensor = layout.tensors[name]
                output.seek(layout.data_base + tensor.start)
                payload = output.read(tensor.byte_count)
                if len(payload) != tensor.byte_count:
                    raise ArtifactError(f"unexpected end of MTP norm payload for {name}")
                shifted = _bf16_add_one(payload, name)
                output.seek(layout.data_base + tensor.start)
                output.write(shifted)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _runtime_contract(source_model: ModelIdentity) -> dict[str, Any]:
    return {
        "schema_version": "axquant.mtp-runtime.v1",
        "arch_id": "qwen3-next-mtp",
        "layout": QWEN36_MTP_LAYOUT,
        # The prepared layout has already applied the +1.0 HF-delta -> MLX
        # multiplier shift; AX Engine must not shift the norms again.
        "mtp_norm_layout": "mlx_multiplier",
        "mtp_depth_max": 1,
        "mtp_tensor_count": len(QWEN36_MTP_TENSORS),
        "recommended_draft_sampler": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.95,
        },
        "release_status": "development-only",
        "source_model": {
            "model_id": source_model.model_id,
            "revision": source_model.revision,
        },
    }


def _validate_prepared_manifest(
    manifest_path: Path,
    sidecar: Path,
    source_model: ModelIdentity,
) -> PreparedMtpSidecarManifest:
    try:
        manifest = load_model(manifest_path, PreparedMtpSidecarManifest)
    except (ArtifactError, ValidationError) as exc:
        raise ArtifactError(f"invalid prepared MTP provenance: {manifest_path}") from exc
    _validate_source_model(manifest, source_model)
    _validated_source_shards(manifest)
    _binding_matches(sidecar, manifest.output.mtp, "prepared MTP sidecar")
    runtime = sidecar.parent / "mtplx_runtime.json"
    if not runtime.is_file():
        raise ArtifactError("prepared MTP sidecar is missing mtplx_runtime.json")
    _binding_matches(runtime, manifest.output.runtime, "prepared MTP runtime")
    try:
        runtime_value = json.loads(
            runtime.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid prepared MTP runtime contract: {exc}") from exc
    expected_runtime = _runtime_contract(source_model)
    if json.dumps(runtime_value, sort_keys=True, separators=(",", ":")) != json.dumps(
        expected_runtime,
        sort_keys=True,
        separators=(",", ":"),
    ):
        raise ArtifactError("prepared MTP runtime contract does not match ax-engine-qwen36-v1")
    input_manifest_path = Path(manifest.input.manifest.path)
    if input_manifest_path.parts != ("ax_mtp_sidecar_manifest.json",):
        raise ArtifactError("prepared MTP provenance binds an unsafe input manifest path")
    input_mtp_path = Path(manifest.input.mtp.path)
    if input_mtp_path.parts not in {(name,) for name in EXTERNAL_MTP_SIDECAR_FILENAMES}:
        raise ArtifactError("prepared MTP provenance binds an unsafe input sidecar path")
    if manifest.input.mtp.size_bytes != manifest.output.mtp.size_bytes:
        raise ArtifactError("prepared MTP provenance input/output sidecar sizes differ")
    if set(manifest.transform.transformed_tensors) != QWEN36_MTP_NORM_TENSORS:
        raise ArtifactError("prepared MTP provenance has the wrong transformed tensor set")
    if set(manifest.transform.unchanged_tensors) != QWEN36_MTP_PROJECTION_TENSORS:
        raise ArtifactError("prepared MTP provenance has the wrong unchanged tensor set")

    layout = _parse_qwen36_layout(sidecar)
    if manifest.tensor_count != len(layout.tensors):
        raise ArtifactError("prepared MTP provenance tensor count does not match")
    if manifest.total_payload_bytes != layout.payload_bytes:
        raise ArtifactError("prepared MTP provenance payload size does not match")
    records = {record.name: record for record in manifest.tensor_payloads}
    if len(records) != len(manifest.tensor_payloads) or set(records) != set(layout.tensors):
        raise ArtifactError("prepared MTP provenance tensor records do not match")
    for name, tensor in layout.tensors.items():
        record = records[name]
        operation = "add_one_bf16" if name in QWEN36_MTP_NORM_TENSORS else "byte_preserved"
        digest = _payload_sha256(sidecar, layout, tensor)
        if (
            tuple(record.shape) != tensor.shape
            or record.byte_count != tensor.byte_count
            or record.sha256 != digest
            or record.operation != operation
        ):
            raise ArtifactError(f"prepared MTP provenance payload mismatch for {name}")
        if operation == "byte_preserved" and record.sha256 != record.source_sha256:
            raise ArtifactError(f"prepared MTP projection changed unexpectedly: {name}")
        if operation == "add_one_bf16" and record.sha256 == record.source_sha256:
            raise ArtifactError(f"prepared MTP norm was not transformed: {name}")
    return manifest


def _copy_verified(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ArtifactError(f"cannot copy missing MTP bundle file: {source}")
    if destination.is_symlink():
        raise ArtifactError(f"MTP bundle destination must not be a symlink: {destination}")
    source_digest = file_sha256(source)
    if destination.exists():
        if not destination.is_file():
            raise ArtifactError(f"MTP bundle destination is not a file: {destination}")
        if file_sha256(destination) != source_digest:
            raise ArtifactError(f"conversion output contains a different {destination.name}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != source_digest:
            raise ArtifactError(f"{source.name} checksum changed during copy")
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _copy_prepared_bundle(
    sidecar: Path,
    manifest_path: Path,
    output_dir: Path,
    source_model: ModelIdentity,
) -> PreparedMtpSidecarManifest:
    _validate_prepared_manifest(manifest_path, sidecar, source_model)
    _copy_verified(sidecar, output_dir / "mtp.safetensors")
    _copy_verified(sidecar.parent / "mtplx_runtime.json", output_dir / "mtplx_runtime.json")
    _copy_verified(manifest_path, output_dir / "ax_mtp_sidecar_manifest.json")
    return _validate_prepared_manifest(
        output_dir / "ax_mtp_sidecar_manifest.json",
        output_dir / "mtp.safetensors",
        source_model,
    )


def prepare_qwen36_mtp_sidecar(
    sidecar: str | Path,
    output_dir: str | Path,
    *,
    source_model: ModelIdentity,
) -> PreparedMtpSidecarManifest:
    source_value = Path(sidecar).expanduser().resolve()
    if source_value.is_dir():
        source = next(
            (
                candidate
                for name in EXTERNAL_MTP_SIDECAR_FILENAMES
                if (candidate := source_value / name).is_file()
            ),
            source_value / EXTERNAL_MTP_SIDECAR_FILENAMES[0],
        )
    else:
        source = source_value
    if not source.is_file():
        raise ArtifactError(f"MTP sidecar does not exist: {source}")
    provenance_path = source.parent / "ax_mtp_sidecar_manifest.json"
    if not provenance_path.is_file():
        raise ArtifactError("ax-engine-qwen36-v1 requires checksum-bound MTP source provenance")
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    try:
        provenance_value = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"MTP sidecar provenance is unreadable: {provenance_path}") from exc
    transform = provenance_value.get("transform") if isinstance(provenance_value, dict) else None
    mode = transform.get("mode") if isinstance(transform, dict) else None
    if mode == QWEN36_MTP_LAYOUT:
        return _copy_prepared_bundle(
            source,
            provenance_path,
            destination_dir,
            source_model,
        )
    if mode != "byte_preserved":
        raise ArtifactError(
            "ax-engine-qwen36-v1 requires byte-preserved or AXQuant-prepared provenance"
        )

    raw_manifest, raw_layout, source_payloads = _load_raw_manifest(
        provenance_path,
        source,
        source_model,
    )
    output_sidecar = destination_dir / "mtp.safetensors"
    runtime_path = destination_dir / "mtplx_runtime.json"
    manifest_path = destination_dir / "ax_mtp_sidecar_manifest.json"
    for output_path in (output_sidecar, runtime_path, manifest_path):
        if output_path.exists() or output_path.is_symlink():
            raise ArtifactError(f"MTP output already exists: {output_path}")
    _transform_norm_payloads(source, output_sidecar, raw_layout)
    output_layout = _parse_qwen36_layout(output_sidecar)
    write_data(runtime_path, _runtime_contract(source_model))

    tensor_payloads: list[PreparedMtpTensorPayload] = []
    for name in sorted(output_layout.tensors):
        tensor = output_layout.tensors[name]
        digest = _payload_sha256(output_sidecar, output_layout, tensor)
        operation = "add_one_bf16" if name in QWEN36_MTP_NORM_TENSORS else "byte_preserved"
        if operation == "byte_preserved" and digest != source_payloads[name]:
            raise ArtifactError(f"Qwen 3.6 MTP projection changed during transform: {name}")
        if operation == "add_one_bf16" and digest == source_payloads[name]:
            raise ArtifactError(f"Qwen 3.6 MTP norm was not transformed: {name}")
        tensor_payloads.append(
            PreparedMtpTensorPayload(
                name=name,
                dtype="BF16",
                shape=list(tensor.shape),
                byte_count=tensor.byte_count,
                sha256=digest,
                source_sha256=source_payloads[name],
                operation=operation,
            )
        )

    manifest = PreparedMtpSidecarManifest(
        source=raw_manifest.source,
        input=PreparedMtpInputBinding(
            manifest=MtpSidecarFileBinding(
                path=provenance_path.name,
                size_bytes=provenance_path.stat().st_size,
                sha256=file_sha256(provenance_path),
            ),
            mtp=MtpSidecarFileBinding(
                path=source.name,
                size_bytes=source.stat().st_size,
                sha256=file_sha256(source),
            ),
        ),
        output=PreparedMtpOutputBinding(
            mtp=MtpSidecarFileBinding(
                path=output_sidecar.name,
                size_bytes=output_sidecar.stat().st_size,
                sha256=file_sha256(output_sidecar),
            ),
            runtime=MtpSidecarFileBinding(
                path=runtime_path.name,
                size_bytes=runtime_path.stat().st_size,
                sha256=file_sha256(runtime_path),
            ),
        ),
        transform=PreparedMtpTransform(
            mode=QWEN36_MTP_LAYOUT,
            implementation="axquant",
            operation="add_one_to_qwen36_mtp_norms_bf16",
            transformed_tensors=sorted(QWEN36_MTP_NORM_TENSORS),
            unchanged_tensors=sorted(QWEN36_MTP_PROJECTION_TENSORS),
        ),
        tensor_count=len(output_layout.tensors),
        tensor_payloads=tensor_payloads,
        total_payload_bytes=output_layout.payload_bytes,
    )
    write_data(manifest_path, manifest)
    return _validate_prepared_manifest(manifest_path, output_sidecar, source_model)


def _f32_to_bf16_bytes(np: Any, values: Any) -> bytes:
    """Round-to-nearest-even BF16 serialization of a float32 array."""
    as_bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = as_bits + 0x7FFF + ((as_bits >> 16) & 1)
    return bytes((rounded >> 16).astype(np.uint16).tobytes())


# Packing label shared with AX Engine's `mtp-capability` report: MLX-native
# affine packing (uint32-packed codes + BF16 group scales/biases), exactly
# what `mx.quantize` emits and the engine's `mtp_take_weight` executes.
MLX_PACKED_MTP_PACKING = "mlx-affine-packed-u32"
# Bits AXQuant emits for quantized sidecar projections; a subset of the
# engine loader's accepted set restricted to the `mtp_sidecar_bits` runtime
# contract (2/4/6/8 — 16 means byte-preserved, 3 is out of contract).
_QUANTIZED_MTP_BITS = (2, 4, 6, 8)


def quantize_qwen36_mtp_sidecar(
    source_path: str | Path,
    output_path: str | Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    capability: AxEngineMtpCapabilityCheck,
    generated_by: str = "axquant",
) -> QuantizedMtpSidecarManifest:
    """Emit an opt-in quantized MTP sidecar next to the byte-preserved default.

    ADR-0005 / RM-40: the byte-preserved sidecar stays the default and is
    never touched; this writes a *separate* artifact in AX Engine's executable
    MLX-packed layout — `mx.quantize` uint32-packed codes plus BF16 group
    scales/biases under the engine's `<base>.scales` / `<base>.biases` key
    convention — so the engine's `mtp_take_weight` consumes it directly.
    Every packed tensor is verified by an `mx.dequantize` round trip before
    writing. It fails closed unless the supplied AX Engine capability check
    proves the runtime executes the quantized layout with MTP enabled. Norm
    tensors and any projection whose input dimension does not divide the
    group size are preserved at BF16 inside the quantized sidecar.

    The consuming runtime resolves bits from `mtp_sidecar_bits` in
    `mtplx_runtime.json`; use ``annotate_mtp_runtime_sidecar_bits`` to stamp
    it when packaging.
    """
    if not (capability.ok and capability.mtp_enabled):
        raise ArtifactError(
            "quantized MTP sidecar requires a passing AX Engine capability "
            "check with MTP enabled; byte-preserved remains the default"
        )
    if bits not in _QUANTIZED_MTP_BITS:
        raise ArtifactError(
            f"quantized MTP sidecar bits must be one of {_QUANTIZED_MTP_BITS}, got {bits}"
        )
    if capability.supported_bits and bits not in capability.supported_bits:
        raise ArtifactError(
            f"AX Engine capability reports supported bits "
            f"{sorted(capability.supported_bits)}; {bits}-bit is not executable"
        )
    if capability.packing and capability.packing != MLX_PACKED_MTP_PACKING:
        raise ArtifactError(
            f"AX Engine capability reports packing {capability.packing!r}; "
            f"this writer emits {MLX_PACKED_MTP_PACKING!r}"
        )
    # Path gates before optional MLX import so non-MLX callers (and Ubuntu CI)
    # still get the ADR-0005 overwrite contract without requiring the backend.
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination == source:
        raise ArtifactError("quantized MTP sidecar must not overwrite the byte-preserved sidecar")
    try:
        import numpy as np
    except ImportError as exc:
        raise ArtifactError("quantized MTP sidecar requires numpy") from exc
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ArtifactError(
            "quantized MTP sidecar requires MLX: the engine-executable packing "
            "is produced by mx.quantize itself"
        ) from exc

    layout = _parse_qwen36_layout(source)

    records: list[QuantizedMtpTensorRecord] = []
    header: dict[str, Any] = {
        "__metadata__": {
            "format": MLX_PACKED_MTP_PACKING,
            "default_bits": str(bits),
            "group_size": str(group_size),
        }
    }
    payloads: list[bytes] = []
    offset = 0

    def _emit(name: str, dtype: str, shape: tuple[int, ...], payload: bytes) -> None:
        nonlocal offset
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)

    with source.open("rb") as stream:
        for name in sorted(layout.tensors):
            tensor = layout.tensors[name]
            stream.seek(layout.data_base + tensor.start)
            raw = stream.read(tensor.byte_count)
            if len(raw) != tensor.byte_count:
                raise ArtifactError(f"unexpected end of MTP payload for {name}")
            in_features = tensor.shape[-1]
            quantizable = (
                len(tensor.shape) == 2
                and name not in QWEN36_MTP_NORM_TENSORS
                and in_features % group_size == 0
            )
            if not quantizable:
                _emit(name, tensor.dtype, tensor.shape, raw)
                records.append(
                    QuantizedMtpTensorRecord(
                        name=name,
                        quantized=False,
                        bits=16,
                        group_size=None,
                        payload_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
                continue
            as_uint16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            weight_f32 = (as_uint16 << 16).view(np.float32).reshape(tensor.shape)
            weight_mx = mx.array(np.ascontiguousarray(weight_f32)).astype(mx.bfloat16)
            try:
                packed, scales, biases = mx.quantize(weight_mx, group_size=group_size, bits=bits)
                reconstructed = mx.dequantize(
                    packed, scales, biases, group_size=group_size, bits=bits
                )
                mx.eval(packed, scales, biases, reconstructed)
            except (ValueError, RuntimeError) as exc:
                raise ArtifactError(
                    f"quantized MTP sidecar cannot pack {name} at {bits}-bit "
                    f"group {group_size}: {exc}"
                ) from exc

            out_features = tensor.shape[0]
            groups = in_features // group_size
            packed_np = np.asarray(packed)
            if packed_np.dtype != np.uint32 or packed_np.shape[0] != out_features:
                raise ArtifactError(f"unexpected MLX packing for {name}")
            # The engine infers group size as (packed_cols * 32 / bits) /
            # scale_cols; assert the inference resolves to our group so an
            # MLX packing change can never ship a silently mis-typed sidecar.
            inferred_group = (packed_np.shape[1] * 32 // bits) // groups
            if inferred_group != group_size:
                raise ArtifactError(
                    f"engine group inference would resolve {inferred_group} "
                    f"for {name}, expected {group_size}"
                )
            # Round-trip guard: the dequantized grid must stay within one grid
            # step of the quantizer's BF16-cast input, plus one BF16 ulp — at
            # 8-bit the grid step drops below BF16's own resolution (2^-7
            # relative), so representation error legitimately dominates. A
            # mis-typed packing produces errors on the order of the weight
            # range itself, far beyond this bound.
            error = mx.abs(reconstructed.astype(mx.float32) - weight_mx.astype(mx.float32))
            max_error = float(mx.max(error).item())
            max_step = float(mx.max(mx.abs(scales.astype(mx.float32))).item())
            max_magnitude = float(mx.max(mx.abs(weight_mx.astype(mx.float32))).item())
            bound = max_step + max_magnitude * 2.0**-7 + 1e-6
            if not math.isfinite(max_error) or max_error > bound:
                raise ArtifactError(
                    f"quantized MTP round-trip error {max_error:.6f} exceeds "
                    f"the packing bound {bound:.6f} for {name}"
                )

            base = name.removesuffix(".weight")
            scales_f32 = np.asarray(scales.astype(mx.float32))
            biases_f32 = np.asarray(biases.astype(mx.float32))
            codes_payload = np.ascontiguousarray(packed_np).tobytes()
            _emit(name, "U32", tuple(int(v) for v in packed_np.shape), codes_payload)
            _emit(
                f"{base}.scales",
                "BF16",
                (out_features, groups),
                _f32_to_bf16_bytes(np, scales_f32.reshape(out_features, groups)),
            )
            _emit(
                f"{base}.biases",
                "BF16",
                (out_features, groups),
                _f32_to_bf16_bytes(np, biases_f32.reshape(out_features, groups)),
            )
            records.append(
                QuantizedMtpTensorRecord(
                    name=name,
                    quantized=True,
                    bits=bits,
                    group_size=group_size,
                    payload_sha256=hashlib.sha256(codes_payload).hexdigest(),
                )
            )

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Safetensors convention: pad the header with spaces so the payload region
    # begins on an 8-byte boundary (mirrors the official writers).
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            for payload in payloads:
                output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise

    return QuantizedMtpSidecarManifest(
        generated_by=generated_by,
        source_sidecar=MtpSidecarFileBinding(
            path=source.name,
            size_bytes=source.stat().st_size,
            sha256=file_sha256(source),
        ),
        output=MtpSidecarFileBinding(
            path=destination.name,
            size_bytes=destination.stat().st_size,
            sha256=file_sha256(destination),
        ),
        default_bits=bits,
        group_size=group_size,
        tensors=records,
        capability=capability,
    )


def probe_ax_engine_mtp_capability(
    command: Sequence[str],
    *,
    timeout: float = 120.0,
) -> AxEngineMtpCapabilityCheck:
    """Execute a real AX Engine capability probe for the quantized MTP layout.

    Runs the given command (an AX Engine doctor/capability invocation) as a
    subprocess and parses the last stdout line as a JSON object carrying
    ``ok``, ``mtp_enabled``, ``layout``, and ``ax_engine_version``. Every
    failure mode — missing binary, timeout, non-zero exit, malformed output,
    missing fields — fails closed: the recorded capability is what the
    runtime reported, never what a caller asserted.
    """
    if not command:
        raise ArtifactError("AX Engine capability probe requires a command")
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        raise ArtifactError(f"AX Engine capability probe could not run: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ArtifactError(
            f"AX Engine capability probe timed out after {timeout} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:500]
        raise ArtifactError(f"AX Engine capability probe exited {completed.returncode}: {detail}")
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise ArtifactError("AX Engine capability probe emitted no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ArtifactError("AX Engine capability probe emitted no JSON object") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("AX Engine capability probe output is not a JSON object")
    reported_bits = payload.get("supported_bits")
    supported_bits = (
        [bits for bits in reported_bits if isinstance(bits, int) and not isinstance(bits, bool)]
        if isinstance(reported_bits, list)
        else []
    )
    try:
        return AxEngineMtpCapabilityCheck(
            ok=payload.get("ok") is True,
            mtp_enabled=payload.get("mtp_enabled") is True,
            layout=str(payload.get("layout") or ""),
            ax_engine_version=str(payload.get("ax_engine_version") or ""),
            supported_bits=supported_bits,
            packing=str(payload.get("packing") or ""),
        )
    except ValidationError as exc:
        raise ArtifactError(f"AX Engine capability probe output is incomplete: {exc}") from exc


def annotate_mtp_runtime_sidecar_bits(runtime_path: str | Path, bits: int) -> None:
    """Stamp `mtp_sidecar_bits` into an existing `mtplx_runtime.json` atomically.

    The engine resolves quantized sidecar bit width from this hint; without
    it, packed projections dequantize under the default 4-bit assumption and
    an 8-bit sidecar silently expands to the wrong shape.
    """
    if bits not in _QUANTIZED_MTP_BITS:
        raise ArtifactError(f"mtp_sidecar_bits must be one of {_QUANTIZED_MTP_BITS}, got {bits}")
    path = Path(runtime_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read MTP runtime config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"MTP runtime config is not a JSON object: {path}")
    payload["mtp_sidecar_bits"] = bits
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
