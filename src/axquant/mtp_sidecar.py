from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from axquant.errors import ArtifactError
from axquant.schema import (
    BytePreservedMtpSidecarManifest,
    ModelIdentity,
    MtpSidecarFileBinding,
    PreparedMtpInputBinding,
    PreparedMtpOutputBinding,
    PreparedMtpSidecarManifest,
    PreparedMtpTensorPayload,
    PreparedMtpTransform,
)
from axquant.serde import file_sha256, load_model, write_data

# Recognized root-level filenames for an externally-shipped MTP head sidecar.
# Kept as one canonical constant because callers across the pipeline (inspect,
# convert, probe, release audit) must agree on every recognized name, or a
# checkpoint shipped under an alternate name is silently mishandled by
# whichever caller falls out of sync.
EXTERNAL_MTP_SIDECAR_FILENAMES: frozenset[str] = frozenset(
    {"mtp.safetensors", "mtp_head.safetensors"}
)

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
            raise ArtifactError(f"duplicate Safetensors header key: {key}")
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
    if relative.is_absolute() or ".." in relative.parts or relative.name != path.name:
        raise ArtifactError(f"{label} provenance binds a different path")
    if path.stat().st_size != binding.size_bytes:
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
    shard_names = {shard.name for shard in manifest.source.shards}
    payload_sha256: dict[str, str] = {}
    for name, tensor in layout.tensors.items():
        record = records[name]
        if (
            record.dtype != tensor.dtype
            or tuple(record.shape) != tensor.shape
            or record.byte_count != tensor.byte_count
            or record.source_data_range[1] - record.source_data_range[0] != tensor.byte_count
            or record.source_shard not in shard_names
        ):
            raise ArtifactError(f"raw MTP provenance metadata mismatch for {name}")
        digest = _payload_sha256(source, layout, tensor)
        if digest != record.sha256:
            raise ArtifactError(f"raw MTP provenance payload checksum mismatch for {name}")
        payload_sha256[name] = digest
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
    _binding_matches(sidecar, manifest.output.mtp, "prepared MTP sidecar")
    runtime = sidecar.parent / "mtplx_runtime.json"
    if not runtime.is_file():
        raise ArtifactError("prepared MTP sidecar is missing mtplx_runtime.json")
    _binding_matches(runtime, manifest.output.runtime, "prepared MTP runtime")
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
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ArtifactError(f"conversion output contains a different {destination.name}")
        return
    shutil.copy2(source, destination)
    if file_sha256(destination) != file_sha256(source):
        raise ArtifactError(f"{source.name} checksum changed during copy")


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
    source = source_value / "mtp.safetensors" if source_value.is_dir() else source_value
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
    if output_sidecar.exists():
        raise ArtifactError(f"MTP output already exists: {output_sidecar}")
    _transform_norm_payloads(source, output_sidecar, raw_layout)
    output_layout = _parse_qwen36_layout(output_sidecar)
    runtime_path = destination_dir / "mtplx_runtime.json"
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
    manifest_path = destination_dir / "ax_mtp_sidecar_manifest.json"
    write_data(manifest_path, manifest)
    return _validate_prepared_manifest(manifest_path, output_sidecar, source_model)
