from __future__ import annotations

import json
import struct
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors import safe_open
from safetensors.numpy import save_file

import axquant.mxfp6_export as exporter
from axquant.cli import main
from axquant.errors import ArtifactError, PlanningError, QuantizerError
from axquant.inspector import inspect_model
from axquant.manual import manual_quantization_plan
from axquant.mxfp6 import (
    dequantize_mxfp6,
    encode_fp6,
    fp6_values,
    mxfp6_storage_bytes,
    pack_fp6,
    quantize_mxfp6,
    unpack_fp6,
)
from axquant.schema import ManualPlanRecipe, QuantMethod, TensorRole
from axquant.schema.mxfp6 import Mxfp6File, Mxfp6PackManifest
from axquant.serde import file_sha256, write_data
from axquant.source_binding import build_source_plan_binding, write_source_plan_binding

MANIFEST_NAME = exporter.MANIFEST_NAME
export_mxfp6 = exporter.export_mxfp6
load_mxfp6_tensor = exporter.load_mxfp6_tensor


def _positive_fp6_values(element_format):
    # Independent, explicit OCP v1.0 code ordering; no production codec helpers.
    if element_format == "e2m3":
        return np.array(
            [i / 8 for i in range(8)]
            + [1 + i / 8 for i in range(8)]
            + [2 + i / 4 for i in range(8)]
            + [4 + i / 2 for i in range(8)]
        )
    return np.array(
        [0, 1 / 16, 1 / 8, 3 / 16]
        + [2**power * (1 + i / 4) for power in range(-2, 5) for i in range(4)]
    )


@pytest.mark.parametrize("element_format", ["e2m3", "e3m2"])
def test_every_fp6_code_matches_ocp_and_preserves_signed_zero(element_format):
    positive = _positive_fp6_values(element_format)
    expected = np.concatenate((positive, -positive))
    decoded = fp6_values(element_format)
    np.testing.assert_array_equal(decoded, expected)
    assert not np.signbit(decoded[0]) and np.signbit(decoded[32])
    np.testing.assert_array_equal(encode_fp6(expected, element_format), np.arange(64))


@pytest.mark.parametrize("element_format", ["e2m3", "e3m2"])
def test_midpoints_round_to_even_at_every_exponent_boundary(element_format):
    table = _positive_fp6_values(element_format)
    midpoints = (table[:-1] + table[1:]) / 2
    expected = np.arange(31, dtype=np.uint8)
    expected += expected & 1
    np.testing.assert_array_equal(encode_fp6(midpoints, element_format), expected)
    np.testing.assert_array_equal(encode_fp6(-midpoints, element_format), expected | 32)
    np.testing.assert_array_equal(
        encode_fp6(np.nextafter(midpoints, np.inf), element_format), np.arange(1, 32)
    )
    np.testing.assert_array_equal(
        encode_fp6(np.nextafter(midpoints, -np.inf), element_format), np.arange(31)
    )
    assert encode_fp6(np.array([1e30, -1e30]), element_format).tolist() == [31, 63]


def test_packing_has_independent_golden_bytes_and_no_padding():
    codes = np.tile(np.array([0, 1, 2, 63], dtype=np.uint8), 8).reshape(1, 32)
    packed = pack_fp6(codes)
    assert packed.tobytes() == bytes([0x40, 0x20, 0xFC]) * 8
    np.testing.assert_array_equal(unpack_fp6(packed), codes)
    assert mxfp6_storage_bytes((1, 32)) == 25


@pytest.mark.parametrize("element_format", ["e2m3", "e3m2"])
@pytest.mark.parametrize("shape", [(2, 64), (3, 4, 96)])
def test_quantization_storage_and_error_bound(element_format, shape):
    weights = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    quantized = quantize_mxfp6(weights, element_format=element_format)
    decoded = dequantize_mxfp6(quantized)
    assert quantized.packed.nbytes + quantized.scales.nbytes == mxfp6_storage_bytes(shape)
    assert decoded.shape == shape and decoded.dtype == np.float32
    # The maximum adjacent FP6 spacing is 0.5 / 4; error is at most half a step.
    step = 0.5 if element_format == "e2m3" else 4.0
    bound = np.ldexp(step / 2, quantized.scales.astype(np.int32) - 127)
    errors = np.abs(decoded - weights).reshape(*shape[:-1], -1, 32)
    assert np.all(errors <= bound[..., None])


@pytest.mark.parametrize("element_format", ["e2m3", "e3m2"])
def test_extreme_scales_zero_blocks_and_float32_saturation(element_format):
    weights = np.zeros((3, 32), dtype=np.float32)
    weights[0, ::2] = -0.0
    weights[1, :] = np.finfo(np.float32).smallest_subnormal
    weights[2, :] = np.finfo(np.float32).max
    quantized = quantize_mxfp6(weights, element_format=element_format)
    assert quantized.scales[0, 0] == 127
    assert quantized.scales[1, 0] == 0
    decoded = dequantize_mxfp6(quantized)
    assert np.all(np.isfinite(decoded))
    np.testing.assert_array_equal(np.signbit(decoded[0]), np.signbit(weights[0]))
    largest_scale = replace(quantized, scales=np.full((3, 1), 254, dtype=np.uint8))
    assert np.all(np.isfinite(dequantize_mxfp6(largest_scale)))


@pytest.mark.parametrize(
    "weights",
    [np.ones((2, 31)), np.ones(32), np.ones((0, 32)), np.ones((2, 32), dtype=np.int32)],
)
def test_reject_invalid_weights(weights):
    with pytest.raises(QuantizerError):
        quantize_mxfp6(weights)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.finfo(np.float64).max])
def test_nonfinite_and_out_of_float32_range_rejected(value):
    with pytest.raises(QuantizerError):
        quantize_mxfp6(np.full((2, 32), value))


def test_corrupt_layouts_and_e8m0_nan_are_rejected():
    tensor = quantize_mxfp6(np.ones((2, 32), dtype=np.float32))
    for corrupt in (
        replace(tensor, scales=np.full((2, 1), 255, dtype=np.uint8)),
        replace(tensor, scales=np.ones((2, 1), dtype=np.float32)),
        replace(tensor, packed=tensor.packed[:, :-1]),
        replace(tensor, shape=(1, 64)),
        replace(tensor, element_format="e9m9"),
    ):
        with pytest.raises(QuantizerError):
            dequantize_mxfp6(corrupt)
    with pytest.raises(QuantizerError):
        pack_fp6(np.full((1, 32), 64, dtype=np.uint8))


@pytest.fixture
def source_and_plan(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "tie_word_embeddings": False,
                "text_config": {
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "num_hidden_layers": 64,
                    "vocab_size": 248320,
                },
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(7)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(4, 64)).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": rng.normal(size=(4, 64)).astype(np.float16),
            "model.embed_tokens.weight": rng.normal(size=(8, 64)).astype(np.float32),
            "model.norm.weight": rng.normal(size=(64,)).astype(np.float32),
            "lm_head.weight": rng.normal(size=(8, 64)).astype(np.float32),
        },
        source / "model.safetensors",
    )
    save_file({"mtp.fc.weight": np.ones((4, 64), dtype=np.float32)}, source / "mtp.safetensors")
    inventory = inspect_model(source, model_id="Qwen/Qwen3.6-27B", revision="a" * 40)
    plan = manual_quantization_plan(
        inventory,
        ManualPlanRecipe(default_bits=6, group_size=32, target_bpw=16.0),
    )
    return source, plan


def _export(source_and_plan, output, **kwargs):
    source, plan = source_and_plan
    return export_mxfp6(
        source,
        plan,
        output,
        allow_unmeasured=True,
        source_binding=build_source_plan_binding(plan, source),
        **kwargs,
    )


@pytest.mark.parametrize("element_format", ["e2m3", "e3m2"])
def test_export_readback_preserves_protected_tensors_and_mtp_bytes(
    source_and_plan, tmp_path, element_format, monkeypatch
):

    # Force multiple chunks instead of covering only the single-chunk path.
    monkeypatch.setattr(exporter, "_CHUNK_BYTES", 256)
    source, plan = source_and_plan
    original_plan = plan.model_dump_json()
    output = tmp_path / "output"
    manifest = _export(source_and_plan, output, element_format=element_format)
    assert manifest.runtime_support == "reference-only"
    assert manifest.evidence_kind == "unmeasured_development"
    assert plan.model_dump_json() == original_plan
    assert file_sha256(source / "mtp.safetensors") == file_sha256(output / "mtp.safetensors")
    assert not (output / "config.json").exists()
    assert str(source) not in (output / MANIFEST_NAME).read_text()
    for record in manifest.tensors:
        with safe_open(source / record.source_file, framework="numpy") as reader:
            original = reader.get_tensor(record.name)
        decoded = load_mxfp6_tensor(output, record.name)
        if record.encoding == "preserved":
            assert original.tobytes() == decoded.tobytes()
        else:
            expected = dequantize_mxfp6(quantize_mxfp6(original, element_format=element_format))
            np.testing.assert_array_equal(decoded, expected)


def test_export_requires_opt_in_binding_and_exact_coverage(source_and_plan, tmp_path):
    source, plan = source_and_plan
    output = tmp_path / "output"
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        export_mxfp6(source, plan, output)
    with pytest.raises(PlanningError, match="binding"):
        export_mxfp6(source, plan, output, allow_unmeasured=True)
    corrupt = plan.model_copy(update={"assignments": plan.assignments[:-1]})
    with pytest.raises(PlanningError, match="coverage"):
        _export((source, corrupt), output)
    assert not output.exists()


def test_source_binding_config_drift_fails_before_writing(source_and_plan, tmp_path):
    source, plan = source_and_plan
    binding = build_source_plan_binding(plan, source)
    (source / "config.json").write_text('{"model_type":"qwen3_5","changed":true}')
    with pytest.raises(PlanningError, match="binding mismatch"):
        export_mxfp6(
            source, plan, tmp_path / "output", allow_unmeasured=True, source_binding=binding
        )


def test_protected_assignment_and_incompatible_bits_rejected(source_and_plan, tmp_path):
    source, plan = source_and_plan
    for target in (TensorRole.MTP_PROJECTION, TensorRole.MLP):
        assignments = [
            item.model_copy(
                update={"bits": 6 if item.role.is_mtp else 4, "method": QuantMethod.AFFINE}
            )
            if item.role == target
            else item
            for item in plan.assignments
        ]
        with pytest.raises(PlanningError, match="incompatible allocation"):
            _export(
                (source, plan.model_copy(update={"assignments": assignments})), tmp_path / "bad"
            )


def test_atomic_failure_cleanup_and_existing_destination(source_and_plan, tmp_path, monkeypatch):

    def fail(*args, **kwargs):
        raise ArtifactError("injected I/O failure")

    output = tmp_path / "output"
    monkeypatch.setattr(exporter, "_write_shard", fail)
    with pytest.raises(ArtifactError, match="injected"):
        _export(source_and_plan, output)
    assert not output.exists() and not list(tmp_path.glob(".output.mxfp6-*"))
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing data")
    with pytest.raises(ArtifactError, match="already exists"):
        _export(source_and_plan, output)
    assert marker.read_text() == "existing data"


def test_payload_corruption_and_manifest_coverage_fail_closed(source_and_plan, tmp_path):
    output = tmp_path / "output"
    manifest = _export(source_and_plan, output)
    record = next(item for item in manifest.tensors if item.encoding == "mxfp6")
    payload = output / record.output_file
    with payload.open("r+b") as writer:
        writer.seek(-1, 2)
        writer.write(b"\xff")
    with pytest.raises(ArtifactError, match="checksum"):
        load_mxfp6_tensor(output, record.name)
    # Even recomputing the outer checksum cannot hide a malformed physical layout.
    from safetensors.numpy import load_file

    arrays = load_file(payload)
    arrays[record.data_key] = arrays[record.data_key][:, :-1]
    save_file(arrays, payload)
    manifest.output_files = [
        item.model_copy(
            update={"sha256": file_sha256(payload), "size_bytes": payload.stat().st_size}
        )
        if item.path == record.output_file
        else item
        for item in manifest.output_files
    ]
    write_data(output / MANIFEST_NAME, manifest)
    with pytest.raises(ArtifactError, match="layout"):
        load_mxfp6_tensor(output, record.name)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../escape", "a\\b", "a//b"])
def test_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValidationError):
        Mxfp6File(path=path, sha256="a" * 64, size_bytes=1)


def test_cli_export_and_manifest_roundtrip(source_and_plan, tmp_path):
    source, plan = source_and_plan
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)
    write_source_plan_binding(tmp_path, plan, source)
    args = [
        "export-mxfp6",
        "--model",
        str(source),
        "--plan",
        str(plan_path),
        "--output",
        str(tmp_path / "output"),
    ]
    assert main(args) == 2
    assert main([*args, "--allow-unmeasured", "--element-format", "e3m2"]) == 0
    payload = json.loads((tmp_path / "output" / MANIFEST_NAME).read_text())
    manifest = Mxfp6PackManifest.model_validate(payload)
    assert manifest.element_format == "e3m2"
    with pytest.raises(ValidationError):
        Mxfp6PackManifest.model_validate({**payload, "runtime_support": "mlx"})
    with pytest.raises(ValidationError):
        Mxfp6PackManifest.model_validate({**payload, "schema_version": "axquant.mxfp6-pack.v2"})


def test_bf16_source_decodes_and_exports_without_torch(source_and_plan, tmp_path):
    source, _ = source_and_plan
    path = source / "model.safetensors"
    with safe_open(path, framework="numpy") as reader:
        keys = reader.keys()
        arrays = {name: reader.get_tensor(name).astype(np.float32) for name in keys}
    header = {}
    parts = []
    offset = 0
    for name, values in sorted(arrays.items()):
        data = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": values.shape,
            "data_offsets": [offset, offset + len(data)],
        }
        parts.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(parts))
    inventory = inspect_model(source, model_id="Qwen/Qwen3.6-27B", revision="a" * 40)
    plan = manual_quantization_plan(inventory, ManualPlanRecipe(default_bits=6, target_bpw=16.0))
    output = tmp_path / "output"
    manifest = _export((source, plan), output)
    for item in manifest.tensors:
        decoded = load_mxfp6_tensor(output, item.name)
        assert np.all(np.isfinite(decoded))
        if item.source_dtype == "BF16" and item.encoding == "preserved":
            expected = (arrays[item.name].view(np.uint32) & 0xFFFF0000).view(np.float32)
            np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize("shape", [(True, 32), (1.5, 32), (1, 32.0)])
def test_storage_estimate_rejects_non_integer_dimensions(shape):
    with pytest.raises(QuantizerError, match="shape"):
        mxfp6_storage_bytes(shape)


def test_readback_rejects_forged_preserved_storage_bytes(source_and_plan, tmp_path):
    output = tmp_path / "output"
    manifest = _export(source_and_plan, output)
    preserved = next(item for item in manifest.tensors if item.encoding == "preserved")
    preserved.storage_bytes += 1
    write_data(output / MANIFEST_NAME, manifest)
    with pytest.raises(ArtifactError, match="storage bytes"):
        load_mxfp6_tensor(output, preserved.name)
