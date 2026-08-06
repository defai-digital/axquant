from __future__ import annotations

import hashlib
import json
import shlex
import struct
import sys
from pathlib import Path

import pytest

from axquant.errors import ArtifactError
from axquant.mtp_sidecar import (
    QWEN36_MTP_NORM_TENSORS,
    QWEN36_MTP_PROJECTION_TENSORS,
    QWEN36_MTP_TENSORS,
    prepare_qwen36_mtp_sidecar,
    probe_ax_engine_mtp_capability,
    quantize_qwen36_mtp_sidecar,
)
from axquant.schema import (
    AxEngineMtpCapabilityCheck,
    ModelIdentity,
    QuantizedMtpSidecarManifest,
)
from axquant.serde import file_sha256, load_model, write_data


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
                        "size_bytes": sum(len(payload) for payload in payloads.values()) + 1024,
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
    assert runtime["mtp_norm_layout"] == "mlx_multiplier"
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


def test_prepare_qwen36_mtp_sidecar_rejects_nested_output_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    manifest_path = _write_raw_manifest(source, raw_payloads)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["mtp"]["path"] = "nested/mtp.safetensors"
    write_data(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="different path"):
        prepare_qwen36_mtp_sidecar(
            source,
            tmp_path / "prepared",
            source_model=_source_model(),
        )


def test_prepare_qwen36_mtp_sidecar_rejects_out_of_bounds_source_range(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    manifest_path = _write_raw_manifest(source, raw_payloads)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    byte_count = manifest["tensor_payloads"][0]["byte_count"]
    manifest["tensor_payloads"][0]["source_data_range"] = [-byte_count, 0]
    write_data(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="metadata mismatch"):
        prepare_qwen36_mtp_sidecar(
            source,
            tmp_path / "prepared",
            source_model=_source_model(),
        )


def test_prepare_qwen36_mtp_sidecar_validates_runtime_contract_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    _write_raw_manifest(source, raw_payloads)
    prepared = tmp_path / "prepared"
    prepare_qwen36_mtp_sidecar(source, prepared, source_model=_source_model())

    runtime_path = prepared / "mtplx_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["mtp_depth_max"] = 2
    write_data(runtime_path, runtime)
    manifest_path = prepared / "ax_mtp_sidecar_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["runtime"]["size_bytes"] = runtime_path.stat().st_size
    manifest["output"]["runtime"]["sha256"] = file_sha256(runtime_path)
    write_data(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="runtime contract"):
        prepare_qwen36_mtp_sidecar(
            prepared,
            tmp_path / "copied",
            source_model=_source_model(),
        )


def test_prepare_qwen36_mtp_sidecar_accepts_alternate_root_filename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    manifest_path = _write_raw_manifest(source, raw_payloads)
    alternate = source / "mtp_head.safetensors"
    (source / "mtp.safetensors").rename(alternate)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["mtp"]["path"] = alternate.name
    write_data(manifest_path, manifest)

    result = prepare_qwen36_mtp_sidecar(
        source,
        tmp_path / "prepared",
        source_model=_source_model(),
    )

    assert result.output.mtp.path == "mtp.safetensors"


def test_copy_prepared_mtp_bundle_rejects_symlink_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    raw_payloads = _write_raw_sidecar(source / "mtp.safetensors")
    _write_raw_manifest(source, raw_payloads)
    prepared = tmp_path / "prepared"
    prepare_qwen36_mtp_sidecar(source, prepared, source_model=_source_model())
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "mtp.safetensors").symlink_to(prepared / "mtp.safetensors")

    with pytest.raises(ArtifactError, match="must not be a symlink"):
        prepare_qwen36_mtp_sidecar(
            prepared,
            copied,
            source_model=_source_model(),
        )


def _write_quantizable_sidecar(path: Path) -> None:
    """Sidecar whose projections (4 x 64) admit group-64 affine quantization."""
    header: dict[str, object] = {"__metadata__": {"format": "mlx"}}
    payload_chunks: list[bytes] = []
    cursor = 0
    for name in sorted(QWEN36_MTP_TENSORS):
        if name in QWEN36_MTP_NORM_TENSORS:
            shape = [2]
            values = (-0.5, 0.25)
        else:
            shape = [4, 64]
            values = tuple(((index * 37) % 23 - 11) / 7.0 for index in range(4 * 64))
        payload = _bf16_payload(values)
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [cursor, cursor + len(payload)],
        }
        payload_chunks.append(payload)
        cursor += len(payload)
    rendered = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rendered += b" " * ((-len(rendered)) % 8)
    path.write_bytes(struct.pack("<Q", len(rendered)) + rendered + b"".join(payload_chunks))


def _capability(*, ok: bool = True, mtp_enabled: bool = True) -> AxEngineMtpCapabilityCheck:
    return AxEngineMtpCapabilityCheck(
        ok=ok,
        mtp_enabled=mtp_enabled,
        layout="ax-engine-qwen36-v1",
        ax_engine_version="6.11.1",
    )


def test_quantized_sidecar_is_a_separate_gated_artifact(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    mx = pytest.importorskip("mlx.core")
    source = tmp_path / "mtp.safetensors"
    _write_quantizable_sidecar(source)
    source_digest = file_sha256(source)
    output = tmp_path / "mtp.quantized.safetensors"

    manifest = quantize_qwen36_mtp_sidecar(
        source,
        output,
        bits=4,
        group_size=64,
        capability=_capability(),
    )

    # The byte-preserved default is untouched (ADR-0005).
    assert file_sha256(source) == source_digest
    assert manifest.byte_preserved_default_retained is True
    assert manifest.packing == "mlx-affine-packed-u32"
    assert manifest.output.sha256 == file_sha256(output)
    assert manifest.source_sidecar.sha256 == source_digest

    by_name = {record.name: record for record in manifest.tensors}
    for name in QWEN36_MTP_NORM_TENSORS:
        assert not by_name[name].quantized and by_name[name].bits == 16
    for name in QWEN36_MTP_PROJECTION_TENSORS:
        assert by_name[name].quantized
        assert by_name[name].bits == 4 and by_name[name].group_size == 64

    # Round-trip one projection exactly the way the engine's mtp_take_weight
    # consumes it: `<name>` packed U32 codes plus `<base>.scales` /
    # `<base>.biases` BF16 arrays fed to MLX dequantize.
    blob = output.read_bytes()
    header_len = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    data_base = 8 + header_len
    name = sorted(QWEN36_MTP_PROJECTION_TENSORS)[0]
    base = name.removesuffix(".weight")

    def _payload(entry_name: str) -> tuple[bytes, list[int], str]:
        entry = header[entry_name]
        start, end = entry["data_offsets"]
        return blob[data_base + start : data_base + end], entry["shape"], entry["dtype"]

    codes_bytes, codes_shape, codes_dtype = _payload(name)
    assert codes_dtype == "U32"
    assert codes_shape == [4, 8]  # 64 columns * 4 bits / 32 bits per word
    scales_bytes, scales_shape, scales_dtype = _payload(f"{base}.scales")
    biases_bytes, _, biases_dtype = _payload(f"{base}.biases")
    assert scales_dtype == "BF16" and biases_dtype == "BF16"
    assert scales_shape == [4, 1]

    def _bf16_to_f32(payload: bytes, shape: list[int]) -> object:
        as_u32 = np.frombuffer(payload, dtype=np.uint16).astype(np.uint32) << 16
        return as_u32.view(np.float32).reshape(shape)

    codes = np.frombuffer(codes_bytes, dtype=np.uint32).reshape(codes_shape)
    scales = _bf16_to_f32(scales_bytes, scales_shape)
    biases = _bf16_to_f32(biases_bytes, scales_shape)
    reconstructed = np.asarray(
        mx.dequantize(
            mx.array(codes),
            mx.array(scales),
            mx.array(biases),
            group_size=64,
            bits=4,
        ).astype(mx.float32)
    )
    # Compare against the BF16-cast source — the quantizer's actual input —
    # so the one-grid-step bound is exact.
    original = _bf16_to_f32(
        _bf16_payload(tuple(((index * 37) % 23 - 11) / 7.0 for index in range(4 * 64))),
        [4, 64],
    )
    max_step = float(np.max(np.abs(scales)))
    assert np.max(np.abs(reconstructed - original)) <= max_step + 1e-6


def test_quantized_sidecar_respects_capability_bit_and_packing_limits(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mlx.core")
    source = tmp_path / "mtp.safetensors"
    _write_quantizable_sidecar(source)
    output = tmp_path / "mtp.quantized.safetensors"

    limited = _capability().model_copy(update={"supported_bits": [4, 8]})
    with pytest.raises(ArtifactError, match="not executable"):
        quantize_qwen36_mtp_sidecar(source, output, bits=2, capability=limited)
    wrong_packing = _capability().model_copy(update={"packing": "some-other-packing"})
    with pytest.raises(ArtifactError, match="packing"):
        quantize_qwen36_mtp_sidecar(source, output, bits=4, capability=wrong_packing)
    assert not output.exists()


def test_annotate_mtp_runtime_sidecar_bits_updates_atomically(tmp_path: Path) -> None:
    from axquant.mtp_sidecar import annotate_mtp_runtime_sidecar_bits

    runtime_path = tmp_path / "mtplx_runtime.json"
    runtime_path.write_text(
        json.dumps({"layout": "ax-engine-qwen36-v1", "mtp_depth_max": 1}),
        encoding="utf-8",
    )
    annotate_mtp_runtime_sidecar_bits(runtime_path, 8)
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert payload["mtp_sidecar_bits"] == 8
    assert payload["layout"] == "ax-engine-qwen36-v1"
    with pytest.raises(ArtifactError, match="must be one of"):
        annotate_mtp_runtime_sidecar_bits(runtime_path, 3)


def test_quantized_sidecar_fails_closed_without_capability(tmp_path: Path) -> None:
    source = tmp_path / "mtp.safetensors"
    _write_quantizable_sidecar(source)
    output = tmp_path / "mtp.quantized.safetensors"
    with pytest.raises(ArtifactError, match="capability"):
        quantize_qwen36_mtp_sidecar(source, output, capability=_capability(ok=False))
    with pytest.raises(ArtifactError, match="capability"):
        quantize_qwen36_mtp_sidecar(source, output, capability=_capability(mtp_enabled=False))
    assert not output.exists()


def test_quantized_sidecar_refuses_to_overwrite_the_default(tmp_path: Path) -> None:
    source = tmp_path / "mtp.safetensors"
    _write_quantizable_sidecar(source)
    with pytest.raises(ArtifactError, match="must not overwrite"):
        quantize_qwen36_mtp_sidecar(source, source, capability=_capability())


def _capability_json_command(tmp_path: Path, name: str, **overrides: object) -> str:
    """Command string for a stub capability probe (quoting-safe via a script).

    The payload mirrors the real `ax-engine mtp-capability` output shape.
    """
    payload = {
        "ok": True,
        "mtp_enabled": True,
        "layout": "ax-engine-qwen36-v1",
        "quantized_sidecar": True,
        "supported_bits": [2, 4, 6, 8, 16],
        "packing": "mlx-affine-packed-u32",
        "ax_engine_version": "6.13.1",
    }
    payload.update(overrides)
    script = tmp_path / f"capability_probe_{name}.py"
    script.write_text(
        f"import json\nprint(json.dumps({payload!r}))\n",
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def test_capability_probe_executes_a_real_command(tmp_path: Path) -> None:
    command = shlex.split(_capability_json_command(tmp_path, "ok"))
    check = probe_ax_engine_mtp_capability(command)
    assert check.ok and check.mtp_enabled
    assert check.layout == "ax-engine-qwen36-v1"
    assert check.supported_bits == [2, 4, 6, 8, 16]
    assert check.packing == "mlx-affine-packed-u32"
    assert check.ax_engine_version == "6.13.1"


def test_capability_probe_fails_closed_on_bad_output() -> None:
    with pytest.raises(ArtifactError, match="no JSON object"):
        probe_ax_engine_mtp_capability([sys.executable, "-c", "print('not json')"])
    with pytest.raises(ArtifactError, match="exited 3"):
        probe_ax_engine_mtp_capability([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(ArtifactError, match="incomplete"):
        probe_ax_engine_mtp_capability(
            [sys.executable, "-c", "import json; print(json.dumps({'ok': True}))"]
        )
    with pytest.raises(ArtifactError, match="could not run"):
        probe_ax_engine_mtp_capability(["/nonexistent/ax-engine-doctor"])


def test_quantize_mtp_sidecar_cli_runs_the_capability_probe(tmp_path: Path) -> None:
    from axquant.cli import main

    source = tmp_path / "mtp.safetensors"
    _write_quantizable_sidecar(source)
    output = tmp_path / "mtp.quantized.safetensors"
    manifest_path = tmp_path / "mtp_sidecar_quantized.json"

    assert (
        main(
            [
                "quantize-mtp-sidecar",
                "--sidecar",
                str(source),
                "--output",
                str(output),
                "--capability-command",
                _capability_json_command(tmp_path, "pass"),
                "--manifest-output",
                str(manifest_path),
            ]
        )
        == 0
    )
    manifest = load_model(manifest_path, QuantizedMtpSidecarManifest)
    assert manifest.capability.layout == "ax-engine-qwen36-v1"
    assert manifest.packing == "mlx-affine-packed-u32"
    assert output.exists()


def test_quantize_mtp_sidecar_cli_rejects_failed_probe_and_ambiguous_inputs(
    tmp_path: Path,
) -> None:
    from axquant.cli import main

    source = tmp_path / "mtp.safetensors"
    _write_quantizable_sidecar(source)
    output = tmp_path / "mtp.quantized.safetensors"

    assert (
        main(
            [
                "quantize-mtp-sidecar",
                "--sidecar",
                str(source),
                "--output",
                str(output),
                "--capability-command",
                _capability_json_command(tmp_path, "fail", ok=False),
            ]
        )
        == 2
    )
    assert not output.exists()
    # Exactly one capability source is allowed.
    assert (
        main(
            [
                "quantize-mtp-sidecar",
                "--sidecar",
                str(source),
                "--output",
                str(output),
            ]
        )
        == 2
    )
