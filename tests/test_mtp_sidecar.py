from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from axquant.errors import ArtifactError
from axquant.mtp_sidecar import (
    QWEN36_MTP_NORM_TENSORS,
    QWEN36_MTP_PROJECTION_TENSORS,
    QWEN36_MTP_TENSORS,
    prepare_qwen36_mtp_sidecar,
)
from axquant.schema import ModelIdentity
from axquant.serde import file_sha256, write_data


def _float_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _bf16_payload(values: tuple[float, ...]) -> bytes:
    return b"".join(struct.pack("<H", _float_to_bf16(value)) for value in values)


def _write_raw_sidecar(path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
    cursor = 0
    for name in sorted(QWEN36_MTP_TENSORS):
        if name in QWEN36_MTP_NORM_TENSORS:
            shape = [2]
            payload = _bf16_payload((-0.5, 0.25))
        else:
            shape = [1, 2]
            payload = _bf16_payload((0.125, -2.0))
        payloads[name] = payload
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
    rendered = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rendered += b" " * ((-len(rendered)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(rendered))
        + rendered
        + b"".join(payloads[name] for name in sorted(payloads))
    )
    return payloads


def _write_raw_manifest(
    directory: Path,
    payloads: dict[str, bytes],
    *,
    model_id: str = "Qwen/Qwen3.6-27B",
    revision: str = "source-revision",
) -> Path:
    records: list[dict[str, object]] = []
    source_offset = 0
    for name in sorted(payloads):
        payload = payloads[name]
        records.append(
            {
                "name": name,
                "dtype": "BF16",
                "shape": [2] if name in QWEN36_MTP_NORM_TENSORS else [1, 2],
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_data_range": [source_offset, source_offset + len(payload)],
                "source_shard": "source.safetensors",
            }
        )
        source_offset += len(payload)
    sidecar = directory / "mtp.safetensors"
    manifest = directory / "ax_mtp_sidecar_manifest.json"
    write_data(
        manifest,
        {
            "schema_version": "axquant.mtp_sidecar_provenance.v2",
            "generated_by": "test",
            "source": {
                "model": {"model_id": model_id, "revision": revision},
                "path": "source-model",
                "index_sha256": "a" * 64,
                "shards": [
                    {
                        "name": "source.safetensors",
                        "size_bytes": 1,
                        "sha256": "b" * 64,
                    }
                ],
            },
            "output": {
                "mtp": {
                    "path": sidecar.name,
                    "size_bytes": sidecar.stat().st_size,
                    "sha256": file_sha256(sidecar),
                }
            },
            "transform": {
                "mode": "byte_preserved",
                "implementation": "test payload copy",
            },
            "tensor_count": len(payloads),
            "tensor_payloads": records,
            "total_payload_bytes": sum(len(payload) for payload in payloads.values()),
        },
    )
    return manifest


def _payloads(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    data_base = 8 + header_size
    return {
        name: data[data_base + entry["data_offsets"][0] : data_base + entry["data_offsets"][1]]
        for name, entry in header.items()
        if name != "__metadata__"
    }


def _source_model() -> ModelIdentity:
    return ModelIdentity(
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
    )


def test_prepare_qwen36_mtp_sidecar_transforms_only_norms(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    _write_raw_manifest(source, raw_payloads)
    output = tmp_path / "prepared"

    manifest = prepare_qwen36_mtp_sidecar(source, output, source_model=_source_model())

    prepared_payloads = _payloads(output / "mtp.safetensors")
    expected_norm = _bf16_payload((0.5, 1.25))
    for name in QWEN36_MTP_NORM_TENSORS:
        assert prepared_payloads[name] == expected_norm
        record = next(record for record in manifest.tensor_payloads if record.name == name)
        assert record.operation == "add_one_bf16"
        assert record.sha256 != record.source_sha256
    for name in QWEN36_MTP_PROJECTION_TENSORS:
        assert prepared_payloads[name] == raw_payloads[name]
        record = next(record for record in manifest.tensor_payloads if record.name == name)
        assert record.operation == "byte_preserved"
        assert record.sha256 == record.source_sha256
    runtime = json.loads((output / "mtplx_runtime.json").read_text(encoding="utf-8"))
    assert runtime["layout"] == "ax-engine-qwen36-v1"
    assert runtime["mtp_depth_max"] == 1
    assert runtime["release_status"] == "development-only"
    assert manifest.transform.transformed_tensors == sorted(QWEN36_MTP_NORM_TENSORS)
    assert manifest.transform.unchanged_tensors == sorted(QWEN36_MTP_PROJECTION_TENSORS)


def test_prepare_qwen36_mtp_sidecar_reuses_only_valid_prepared_bundle(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    _write_raw_manifest(source, raw_payloads)
    prepared = tmp_path / "prepared"
    prepare_qwen36_mtp_sidecar(source, prepared, source_model=_source_model())
    copied = tmp_path / "copied"

    manifest = prepare_qwen36_mtp_sidecar(prepared, copied, source_model=_source_model())

    assert (copied / "mtp.safetensors").read_bytes() == (prepared / "mtp.safetensors").read_bytes()
    assert (copied / "mtplx_runtime.json").read_bytes() == (
        prepared / "mtplx_runtime.json"
    ).read_bytes()
    assert manifest.output.mtp.sha256 == file_sha256(copied / "mtp.safetensors")


def test_prepare_qwen36_mtp_sidecar_rejects_payload_provenance_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    manifest_path = _write_raw_manifest(source, raw_payloads)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_payloads"][0]["sha256"] = "0" * 64
    write_data(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="payload checksum mismatch"):
        prepare_qwen36_mtp_sidecar(
            source,
            tmp_path / "prepared",
            source_model=_source_model(),
        )


def test_prepare_qwen36_mtp_sidecar_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    _write_raw_manifest(source, raw_payloads)

    with pytest.raises(ArtifactError, match="source model"):
        prepare_qwen36_mtp_sidecar(
            source,
            tmp_path / "prepared",
            source_model=ModelIdentity(
                model_id="Qwen/Qwen3.6-27B",
                revision="different-revision",
            ),
        )


def test_prepare_qwen36_mtp_sidecar_rejects_unrecognized_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    manifest_path = _write_raw_manifest(source, raw_payloads)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transform"]["mode"] = "legacy-shift"
    write_data(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="byte-preserved or AXQuant-prepared"):
        prepare_qwen36_mtp_sidecar(
            source,
            tmp_path / "prepared",
            source_model=_source_model(),
        )
