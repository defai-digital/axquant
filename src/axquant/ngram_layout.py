"""Byte-preserving n-gram table relayout for MTPLX-compatible pack variants.

Qwen4-exp packs carry the hashed n-gram PLE table as ShardedEmbedding keys
(``...ngram_embedding.shards.<i>...`` plus the HF alias ``shard_<i>``) inside
``model.safetensors.index.json``. The inspected MTPLX 2.11.3 loader expects a
standalone ``ngram-table.safetensors`` and treats the in-index keys as
extraneous parameters, so the pack fails to load.

``relayout_ngram_table`` builds an MTPLX-targeted variant pack in a NEW
directory and concatenates shard rows in numeric order under the canonical
``ngram.weight/scales/biases`` names. Dtypes and payload bytes are preserved
without requantization, using bounded-memory copies. The source is never modified.
Layout compatibility is
not exactness: the MTPLX Forge baseline of a relaid-out pack stays unverified.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axquant.artifact_paths import artifact_member_path, artifact_tree_files
from axquant.errors import ArtifactError
from axquant.module_paths import is_ngram_shard_key, mlx_module_aliases
from axquant.serde import file_sha256

NGRAM_TABLE_FILENAME = "ngram-table.safetensors"
NGRAM_RELAYOUT_MANIFEST_FILENAME = "axquant_ngram_relayout_manifest.json"
NGRAM_RELAYOUT_SCHEMA = "axquant.ngram-relayout.v2"
NGRAM_LAYOUT_CONTRACT_KEY = "ngram_layout"
NGRAM_LAYOUT_SHARDED = "sharded-index"
NGRAM_LAYOUT_STANDALONE = "standalone-table"
INDEX_FILENAME = "model.safetensors.index.json"
RUNTIME_CONTRACT_FILENAME = "mtplx_runtime.json"
MTPLX_NGRAM_PROFILE_VERSION = "2.11.3"
_SOURCE_ONLY_EVIDENCE = frozenset(
    {
        "axquant_manifest.json",
        "axquant_runtime.json",
        "axquant_compatibility.json",
        "axquant_runtime_export.json",
        "ax_engine_manifest.json",
        "ax_expert_stream.json",
        "runtime_check.json",
        "credentials.json",
        "README.md",
    }
)

_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_COPY_CHUNK_BYTES = 8 * 1024 * 1024

_DTYPE_BYTE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


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
class _SafetensorsFile:
    """Header-only layout of one Safetensors file (dtype-agnostic)."""

    data_base: int
    payload_bytes: int
    metadata: dict[str, Any]
    tensors: dict[str, _TensorSlice]


@dataclass(frozen=True)
class _TensorPart:
    source: Path
    layout: _SafetensorsFile
    tensor: _TensorSlice
    payload: bytes | None = None


def _serialize_tensor_parts(
    metadata: dict[str, Any],
    entries: list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]],
    destination: Path,
) -> None:
    """Concatenate raw row ranges without materializing a table or shard."""
    header: dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, parts in entries:
        size = sum(part.tensor.byte_count for part in parts)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with destination.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for _, _, _, parts in entries:
            for part in parts:
                if part.payload is not None:
                    if len(part.payload) != part.tensor.byte_count:
                        raise ArtifactError("transformed tensor byte count mismatch")
                    output.write(part.payload)
                    continue
                with part.source.open("rb") as source:
                    source.seek(part.layout.data_base + part.tensor.start)
                    remaining = part.tensor.byte_count
                    while remaining:
                        chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
                        if not chunk:
                            raise ArtifactError(f"truncated source tensor {part.tensor.name}")
                        output.write(chunk)
                        remaining -= len(chunk)
        output.flush()
        os.fsync(output.fileno())


@dataclass(frozen=True)
class NgramRelayoutReport:
    """Operator-facing summary of one relayout plan or execution."""

    source: Path
    output: Path
    moved_tensor_count: int
    moved_bytes: int
    rebuilt_shard_files: tuple[str, ...]
    removed_shard_files: tuple[str, ...]
    copied_file_count: int
    total_size_before: int | None
    total_size_after: int | None
    dry_run: bool


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ArtifactError(f"{label} must be a JSON object: {path}")
    return document


def _parse_safetensors_file(path: Path) -> _SafetensorsFile:
    """Parse any Safetensors header with strict contiguity validation."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            header_size_bytes = stream.read(8)
            if len(header_size_bytes) != 8:
                raise ArtifactError(f"invalid Safetensors header: {path}")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            if header_size <= 1 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
                raise ArtifactError(f"unsafe Safetensors header size in {path}")
            header_bytes = stream.read(header_size)
            if len(header_bytes) != header_size:
                raise ArtifactError(f"truncated Safetensors header: {path}")
    except OSError as exc:
        raise ArtifactError(f"cannot read Safetensors file {path}: {exc}") from exc
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid Safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ArtifactError(f"Safetensors header is not an object: {path}")

    metadata_entry = header.get("__metadata__", {})
    metadata = metadata_entry if isinstance(metadata_entry, dict) else {}
    data_base = 8 + header_size
    payload_bytes = file_size - data_base
    if payload_bytes <= 0:
        raise ArtifactError(f"Safetensors file has no tensor payload: {path}")

    tensors: dict[str, _TensorSlice] = {}
    for name, raw_entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_entry, dict):
            raise ArtifactError(f"invalid Safetensors tensor header entry in {path}")
        dtype = raw_entry.get("dtype")
        shape = raw_entry.get("shape")
        offsets = raw_entry.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTE_SIZES:
            raise ArtifactError(f"unknown or missing dtype for tensor {name} in {path}")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        ):
            raise ArtifactError(f"invalid shape for tensor {name} in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ArtifactError(f"invalid offsets for tensor {name} in {path}")
        start, end = offsets
        if start < 0 or end < start or end > payload_bytes:
            raise ArtifactError(f"out-of-range payload for tensor {name} in {path}")
        expected = math.prod(shape) * _DTYPE_BYTE_SIZES[dtype]
        if end - start != expected:
            raise ArtifactError(
                f"payload size does not match dtype/shape for tensor {name} in {path}"
            )
        tensors[name] = _TensorSlice(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            start=start,
            end=end,
        )

    intervals = sorted((tensor.start, tensor.end, tensor.name) for tensor in tensors.values())
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            raise ArtifactError(f"Safetensors payload is not contiguous before {name}: {path}")
        cursor = end
    if cursor != payload_bytes:
        raise ArtifactError(f"Safetensors payload coverage mismatch in {path}")
    return _SafetensorsFile(
        data_base=data_base,
        payload_bytes=payload_bytes,
        metadata=metadata,
        tensors=tensors,
    )


def _copy_file_verified(source: Path, destination: Path) -> str:
    """Chunked byte copy; returns the verified sha256 of both sides."""

    expected = file_sha256(source)
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("wb") as dst:
        while chunk := src.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            dst.write(chunk)
    copied = digest.hexdigest()
    if copied != expected:
        raise ArtifactError(f"byte copy mismatch for {source.name}: {copied} != {expected}")
    return copied


def _iter_pack_files(pack: Path) -> Iterator[Path]:
    try:
        yield from artifact_tree_files(pack)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc


def _load_index(pack: Path) -> dict[str, Any]:
    return _read_json_object(pack / INDEX_FILENAME, INDEX_FILENAME)


def _load_runtime_contract(pack: Path) -> dict[str, Any]:
    path = pack / RUNTIME_CONTRACT_FILENAME
    if not path.is_file():
        raise ArtifactError(
            f"{RUNTIME_CONTRACT_FILENAME} missing; relayout targets MTPLX-facing packs only"
        )
    return _read_json_object(path, RUNTIME_CONTRACT_FILENAME)


def _validate_runtime_contract_layout(contract: dict[str, Any]) -> str:
    explicit = contract.get(NGRAM_LAYOUT_CONTRACT_KEY)
    if explicit is None:
        return NGRAM_LAYOUT_SHARDED
    if explicit == NGRAM_LAYOUT_STANDALONE:
        raise ArtifactError(
            f"{NGRAM_LAYOUT_CONTRACT_KEY} is already {NGRAM_LAYOUT_STANDALONE}; "
            "the pack is relaid out"
        )
    if explicit != NGRAM_LAYOUT_SHARDED:
        raise ArtifactError(
            f"unknown explicit {NGRAM_LAYOUT_CONTRACT_KEY} {explicit!r}; "
            f"expected {NGRAM_LAYOUT_SHARDED!r} or {NGRAM_LAYOUT_STANDALONE!r}"
        )
    return NGRAM_LAYOUT_SHARDED


@dataclass(frozen=True)
class _RelayoutPlan:
    source: Path
    moved: dict[str, str]
    shard_layouts: dict[str, _SafetensorsFile]
    affected_shards: tuple[str, ...]
    removed_shards: tuple[str, ...]
    total_size_before: int | None
    total_size_after: int | None
    source_pack_files: tuple[str, ...]
    copied_file_count: int
    table_groups: dict[str, tuple[str, ...]]
    table_metadata: dict[str, str]


def _table_contract(
    source: Path, moved: dict[str, str], layouts: dict[str, _SafetensorsFile]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    config = _read_json_object(source / "config.json", "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ArtifactError("text_config must be an object")
    quantization = config.get("quantization", config.get("quantization_config", {}))
    if not isinstance(quantization, dict):
        raise ArtifactError("quantization must be an object")
    groups: dict[str, dict[int, str]] = {}
    prefixes: set[str] = set()
    recipes: set[tuple[int, int]] = set()
    pattern = re.compile(
        r"^(.*\.ngram_embedding)\.(?:shards\.(\d+)|shard_(\d+))\.(weight|scales|biases)$"
    )
    for name, shard in moved.items():
        match = pattern.fullmatch(name)
        if match is None:
            raise ArtifactError(f"unsupported n-gram tensor name: {name}")
        prefix, number, alias_number, component = match.groups()
        index = int(number if number is not None else alias_number)
        prefixes.add(prefix)
        members = groups.setdefault(component, {})
        if index in members:
            raise ArtifactError(f"duplicate n-gram shard index {index} for {component}")
        members[index] = name
        if component != "weight":
            continue
        tensor = layouts[shard].tensors[name]
        if tensor.dtype == "U32":
            specs = [
                quantization[key]
                for key in mlx_module_aliases(name.removesuffix(".weight"))
                if key in quantization
            ]
            if not specs:
                specs = [quantization]
            for spec in specs:
                if not isinstance(spec, dict) or spec.get("mode", "affine") != "affine":
                    raise ArtifactError("n-gram table requires explicit affine quantization")
                bits, group = spec.get("bits"), spec.get("group_size")
                if (
                    type(bits) is not int
                    or bits not in {2, 4, 6, 8}
                    or type(group) is not int
                    or group not in {32, 64, 128}
                ):
                    raise ArtifactError("n-gram table requires explicit bits and group_size")
                recipes.add((bits, group))
        elif tensor.dtype in {"F16", "BF16", "F32"}:
            recipes.add((0, 32))
        else:
            raise ArtifactError(f"unsupported n-gram weight dtype {tensor.dtype}")
    if len(prefixes) != 1 or len(recipes) != 1 or "weight" not in groups:
        raise ArtifactError("n-gram table requires one embedding and a uniform quantization recipe")
    bits, group_size = next(iter(recipes))
    expected_components = {"weight", "scales", "biases"} if bits else {"weight"}
    if set(groups) != expected_components:
        raise ArtifactError("n-gram table component set does not match its quantization")
    indices = set(groups["weight"])
    count = text_config.get("split_ngram_parts")
    if type(count) is not int or count <= 0 or indices != set(range(count)):
        raise ArtifactError("n-gram shard indices must cover split_ngram_parts contiguously")
    ordered: dict[str, tuple[str, ...]] = {}
    row_counts: tuple[int, ...] | None = None
    for component, members in groups.items():
        if set(members) != indices:
            raise ArtifactError(f"missing n-gram {component} shard")
        names = tuple(members[i] for i in range(count))
        tensors = [layouts[moved[name]].tensors[name] for name in names]
        first = tensors[0]
        if len(first.shape) != 2 or any(
            len(t.shape) != 2 or t.dtype != first.dtype or t.shape[1:] != first.shape[1:]
            for t in tensors
        ):
            raise ArtifactError(f"n-gram {component} shard dtype/width mismatch")
        rows = tuple(t.shape[0] for t in tensors)
        if row_counts is not None and rows != row_counts:
            raise ArtifactError("n-gram component row counts differ")
        row_counts = rows
        ordered[component] = names
    if bits:
        w = layouts[moved[ordered["weight"][0]]].tensors[ordered["weight"][0]]
        s = layouts[moved[ordered["scales"][0]]].tensors[ordered["scales"][0]]
        b = layouts[moved[ordered["biases"][0]]].tensors[ordered["biases"][0]]
        if (
            w.dtype != "U32"
            or s.dtype not in {"F16", "BF16", "F32"}
            or s.dtype != b.dtype
            or s.shape != b.shape
            or w.shape[1] * 32 != s.shape[1] * group_size * bits
        ):
            raise ArtifactError("n-gram packed width, scales, or biases disagree with recipe")
    return ordered, {"format": "mlx", "ngram_bits": str(bits), "ngram_group_size": str(group_size)}


def _plan_relayout(source: Path) -> _RelayoutPlan:
    """Discover n-gram shard keys and validate every gate before any write."""

    index = _load_index(source)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError(f"{INDEX_FILENAME} has no non-empty weight_map in {source}")
    if (source / NGRAM_TABLE_FILENAME).is_file():
        raise ArtifactError(f"{NGRAM_TABLE_FILENAME} already exists in {source}")
    moved = {
        name: str(shard)
        for name, shard in sorted(weight_map.items())
        if isinstance(name, str) and isinstance(shard, str) and is_ngram_shard_key(name)
    }
    if not moved:
        raise ArtifactError(
            f"no sharded n-gram keys in {INDEX_FILENAME}; relayout does not apply to {source}"
        )

    contract = _load_runtime_contract(source)
    _validate_runtime_contract_layout(contract)

    shards_of_moved: dict[str, list[str]] = {}
    for name, shard in moved.items():
        shards_of_moved.setdefault(shard, []).append(name)
    shard_layouts: dict[str, _SafetensorsFile] = {}
    removed_shards: list[str] = []
    for shard, names in sorted(shards_of_moved.items()):
        try:
            shard_path = artifact_member_path(source, shard)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        if not shard_path.is_file():
            raise ArtifactError(f"index references missing shard {shard} in {source}")
        layout = _parse_safetensors_file(shard_path)
        for name in names:
            tensor = layout.tensors.get(name)
            if tensor is None:
                raise ArtifactError(f"indexed n-gram tensor {name} is absent from {shard}")
        strays = sorted(
            tensor.name
            for tensor in layout.tensors.values()
            if is_ngram_shard_key(tensor.name) and tensor.name not in moved
        )
        if strays:
            raise ArtifactError(f"{shard} contains unindexed n-gram tensors: {strays}")
        shard_layouts[shard] = layout
        if set(layout.tensors) == set(names):
            removed_shards.append(shard)

    metadata = index.get("metadata")
    total_size_before: int | None = None
    if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
        total_size_before = metadata["total_size"]
    total_size_after: int | None = None
    if total_size_before is not None:
        moved_bytes = sum(
            shard_layouts[shard].tensors[name].byte_count for name, shard in moved.items()
        )
        total_size_after = total_size_before - moved_bytes

    source_pack_files = tuple(
        str(path.relative_to(source).as_posix())
        for path in _iter_pack_files(source)
        if path.name != NGRAM_RELAYOUT_MANIFEST_FILENAME
        and path.name not in _SOURCE_ONLY_EVIDENCE
        and not any(
            part.startswith(".") or part == "certifications"
            for part in path.relative_to(source).parts
        )
    )
    affected = tuple(sorted(shards_of_moved))
    table_groups, table_metadata = _table_contract(source, moved, shard_layouts)
    copied_file_count = len(source_pack_files) - len(affected) - 3
    return _RelayoutPlan(
        source=source,
        moved=moved,
        shard_layouts=shard_layouts,
        affected_shards=affected,
        removed_shards=tuple(sorted(removed_shards)),
        total_size_before=total_size_before,
        total_size_after=total_size_after,
        source_pack_files=source_pack_files,
        copied_file_count=copied_file_count,
        table_groups=table_groups,
        table_metadata=table_metadata,
    )


def _rewritten_index(source: Path, plan: _RelayoutPlan) -> dict[str, Any]:
    index = _load_index(source)
    weight_map = {
        name: shard for name, shard in index.get("weight_map", {}).items() if name not in plan.moved
    }
    for removed in plan.removed_shards:
        if any(shard == removed for shard in weight_map.values()):
            raise ArtifactError(f"rewritten index still references removed shard {removed}")
    index["weight_map"] = weight_map
    metadata = index.get("metadata")
    if (
        isinstance(metadata, dict)
        and isinstance(metadata.get("total_size"), int)
        and plan.total_size_after is not None
    ):
        index["metadata"] = {**metadata, "total_size": plan.total_size_after}
    return index


def _rebuilt_shard_entries(
    shard: str, plan: _RelayoutPlan
) -> list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]]:
    layout = plan.shard_layouts[shard]
    shard_path = plan.source / shard
    moved_names = {name for name, moved_shard in plan.moved.items() if moved_shard == shard}
    entries: list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]] = []
    for name, tensor in layout.tensors.items():
        if name in moved_names:
            continue
        entries.append(
            (name, tensor.dtype, tensor.shape, (_TensorPart(shard_path, layout, tensor),))
        )
    return entries


def _ngram_table_entries(
    plan: _RelayoutPlan,
) -> list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]]:
    entries: list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]] = []
    for component, names in sorted(plan.table_groups.items()):
        parts = tuple(
            _TensorPart(
                plan.source / plan.moved[name],
                plan.shard_layouts[plan.moved[name]],
                plan.shard_layouts[plan.moved[name]].tensors[name],
            )
            for name in names
        )
        shape = (sum(part.tensor.shape[0] for part in parts), parts[0].tensor.shape[1])
        entries.append((f"ngram.{component}", parts[0].tensor.dtype, shape, parts))
    return entries


def _write_json_atomic(destination: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    finally:
        os.close(descriptor)


def _pack_tensor_digests(pack: Path) -> dict[str, str]:
    """Per-tensor payload sha256 map for every Safetensors file under ``pack``."""

    digests: dict[str, str] = {}
    for path in _iter_pack_files(pack):
        if path.suffix != ".safetensors":
            continue
        layout = _parse_safetensors_file(path)
        for name, tensor in layout.tensors.items():
            if name in digests:
                raise ArtifactError(f"tensor {name} exists in multiple Safetensors files")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                stream.seek(layout.data_base + tensor.start)
                remaining = tensor.byte_count
                while remaining > 0:
                    chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
                    if not chunk:
                        raise ArtifactError(f"unexpected end of payload for {name} in {path}")
                    digest.update(chunk)
                    remaining -= len(chunk)
            digests[name] = digest.hexdigest()
    return digests


def _verify_variant(source: Path, plan: _RelayoutPlan, staging: Path) -> None:
    """Structural full accounting: moved, zero lost, zero duplicated, zero strays."""

    expected_digests = _pack_tensor_digests(source)
    variant_digests = _pack_tensor_digests(staging)
    table = _parse_safetensors_file(staging / NGRAM_TABLE_FILENAME)
    for component, names in plan.table_groups.items():
        canonical = f"ngram.{component}"
        tensor = table.tensors.get(canonical)
        if tensor is None:
            raise ArtifactError(f"missing canonical table component {canonical}")
        variant_digests.pop(canonical)
        cursor = tensor.start
        for name in names:
            original = plan.shard_layouts[plan.moved[name]].tensors[name]
            segment = _TensorSlice(
                name, original.dtype, original.shape, cursor, cursor + original.byte_count
            )
            digest = hashlib.sha256()
            with (staging / NGRAM_TABLE_FILENAME).open("rb") as stream:
                stream.seek(table.data_base + segment.start)
                remaining = segment.byte_count
                while remaining:
                    chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
                    if not chunk:
                        raise ArtifactError(f"truncated variant tensor {name}")
                    digest.update(chunk)
                    remaining -= len(chunk)
            variant_digests[name] = digest.hexdigest()
            cursor = segment.end
        if cursor != tensor.end:
            raise ArtifactError(f"canonical table size mismatch for {canonical}")

    lost = sorted(set(expected_digests) - set(variant_digests))
    added = sorted(set(variant_digests) - set(expected_digests))
    changed = sorted(
        name
        for name in variant_digests
        if name in expected_digests and variant_digests[name] != expected_digests[name]
    )
    if lost or added or changed:
        raise ArtifactError(
            f"variant tensor accounting failed: lost={lost} added={added} payload_changed={changed}"
        )
    table_names = set(_parse_safetensors_file(staging / NGRAM_TABLE_FILENAME).tensors)
    if table_names != {f"ngram.{part}" for part in plan.table_groups}:
        raise ArtifactError("n-gram table tensor set does not match the moved keys")

    variant_index = _read_json_object(staging / INDEX_FILENAME, INDEX_FILENAME)
    variant_map = variant_index.get("weight_map")
    if not isinstance(variant_map, dict):
        raise ArtifactError("variant index has no weight_map")
    residual = sorted(name for name in variant_map if is_ngram_shard_key(name))
    if residual:
        raise ArtifactError(f"variant index still carries sharded n-gram keys: {residual}")
    source_keys = set(_load_index(source).get("weight_map", {}))
    expected_keys = source_keys - set(plan.moved)
    if set(variant_map) != expected_keys:
        raise ArtifactError("variant index weight_map does not match the planned rewrite")
    for shard in set(variant_map.values()):
        if not (staging / str(shard)).is_file():
            raise ArtifactError(f"variant index references missing shard {shard}")

    variant_contract = _read_json_object(
        staging / RUNTIME_CONTRACT_FILENAME, RUNTIME_CONTRACT_FILENAME
    )
    if variant_contract.get(NGRAM_LAYOUT_CONTRACT_KEY) != NGRAM_LAYOUT_STANDALONE:
        raise ArtifactError("variant runtime contract does not declare the standalone layout")


def _write_manifest(plan: _RelayoutPlan, staging: Path, output: Path) -> dict[str, str]:
    output_files: dict[str, str] = {}
    for path in _iter_pack_files(staging):
        if path.name == NGRAM_RELAYOUT_MANIFEST_FILENAME:
            continue
        output_files[str(path.relative_to(staging).as_posix())] = file_sha256(path)
    source_manifest = plan.source / "axquant_manifest.json"
    moved_tensors = {
        name: {
            "shard": plan.moved[name],
            "dtype": plan.shard_layouts[plan.moved[name]].tensors[name].dtype,
            "shape": list(plan.shard_layouts[plan.moved[name]].tensors[name].shape),
            "sha256": _source_tensor_digest(plan, name),
        }
        for name in sorted(plan.moved)
    }
    payload = {
        "schema_version": NGRAM_RELAYOUT_SCHEMA,
        "source": {
            "directory": plan.source.name,
            "axquant_manifest_sha256": (
                file_sha256(source_manifest) if source_manifest.is_file() else None
            ),
        },
        # Directory name only: the manifest ships inside the pack and must not
        # carry absolute private paths.
        "variant_output": output.name,
        "mtplx_target_version": MTPLX_NGRAM_PROFILE_VERSION,
        "ngram_layout": {
            "source": NGRAM_LAYOUT_SHARDED,
            "variant": NGRAM_LAYOUT_STANDALONE,
        },
        "accounting": {
            "moved": len(plan.moved),
            "lost": 0,
            "duplicated": 0,
            "removed_shard_files": list(plan.removed_shards),
        },
        "moved_tensors": moved_tensors,
        "table_metadata": plan.table_metadata,
        "row_order": {component: list(names) for component, names in plan.table_groups.items()},
        "output_files": output_files,
        "exactness_baseline": {
            "status": "unverified",
            "public_release_blocker": True,
            "scope": "compatibility-smoke-only",
            "notes": (
                "Layout compatibility only. No MTPLX Forge exactness baseline is bound "
                "to this variant; run an MTPLX load probe and record its receipt "
                "before any public claim."
            ),
        },
    }
    destination = staging / NGRAM_RELAYOUT_MANIFEST_FILENAME
    _write_json_atomic(destination, payload)
    return output_files


def _source_tensor_digest(plan: _RelayoutPlan, name: str) -> str:
    shard = plan.moved[name]
    layout = plan.shard_layouts[shard]
    tensor = layout.tensors[name]
    digest = hashlib.sha256()
    with (plan.source / shard).open("rb") as stream:
        stream.seek(layout.data_base + tensor.start)
        remaining = tensor.byte_count
        while remaining > 0:
            chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
            if not chunk:
                raise ArtifactError(f"unexpected end of payload for {name} in {shard}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _build_variant(plan: _RelayoutPlan, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.ngram-relayout.", dir=output.parent))
    try:
        for name in plan.source_pack_files:
            source_path = plan.source / name
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == INDEX_FILENAME:
                _write_json_atomic(destination, _rewritten_index(plan.source, plan))
            elif name == RUNTIME_CONTRACT_FILENAME:
                contract = _load_runtime_contract(plan.source)
                contract[NGRAM_LAYOUT_CONTRACT_KEY] = NGRAM_LAYOUT_STANDALONE
                _write_json_atomic(destination, contract)
            elif name == "config.json":
                config = _read_json_object(source_path, "config.json")
                text = config.get("text_config", config)
                text["ngram_sidecar"] = True
                for field in ("quantization", "quantization_config"):
                    quantization = config.get(field)
                    if isinstance(quantization, dict):
                        config[field] = {
                            key: value
                            for key, value in quantization.items()
                            if ".ngram_embedding." not in key
                        }
                _write_json_atomic(destination, config)
            elif name in plan.removed_shards:
                continue  # every tensor in this shard moved; the file disappears
            elif name in plan.affected_shards:
                _serialize_tensor_parts(
                    plan.shard_layouts[name].metadata,
                    _rebuilt_shard_entries(name, plan),
                    destination,
                )
            else:
                _copy_file_verified(source_path, destination)
        _serialize_tensor_parts(
            plan.table_metadata, _ngram_table_entries(plan), staging / NGRAM_TABLE_FILENAME
        )
        _verify_variant(plan.source, plan, staging)
        _write_manifest(plan, staging, output)
        if output.exists():
            raise ArtifactError(f"output appeared during relayout: {output}")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def relayout_ngram_table(
    source: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
) -> NgramRelayoutReport:
    """Build an MTPLX-targeted variant pack with a standalone n-gram table.

    Concatenates shard rows under canonical keys without changing payloads
    or precision. Config and index change to describe the table; this alone
    does not adapt the MTP head or norms. The source remains untouched.
    """

    source_dir = Path(source).expanduser().resolve()
    if not source_dir.is_dir():
        raise ArtifactError(f"source pack directory does not exist: {source_dir}")
    output_dir = Path(output).expanduser().resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ArtifactError("output must be a directory outside the source pack")
    if output_dir.exists():
        raise ArtifactError(f"output already exists: {output_dir}")

    plan = _plan_relayout(source_dir)
    report = NgramRelayoutReport(
        source=source_dir,
        output=output_dir,
        moved_tensor_count=len(plan.moved),
        moved_bytes=sum(
            plan.shard_layouts[shard].tensors[name].byte_count for name, shard in plan.moved.items()
        ),
        rebuilt_shard_files=tuple(
            shard for shard in plan.affected_shards if shard not in plan.removed_shards
        ),
        removed_shard_files=plan.removed_shards,
        copied_file_count=plan.copied_file_count,
        total_size_before=plan.total_size_before,
        total_size_after=plan.total_size_after,
        dry_run=True,
    )
    if dry_run:
        return report
    _build_variant(plan, output_dir)
    return NgramRelayoutReport(
        source=report.source,
        output=report.output,
        moved_tensor_count=report.moved_tensor_count,
        moved_bytes=report.moved_bytes,
        rebuilt_shard_files=report.rebuilt_shard_files,
        removed_shard_files=report.removed_shard_files,
        copied_file_count=report.copied_file_count,
        total_size_before=report.total_size_before,
        total_size_after=report.total_size_after,
        dry_run=False,
    )
