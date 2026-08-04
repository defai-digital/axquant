from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.cli import main
from axquant.feasibility import ArtifactTarget, assess_feasibility, audit_artifact
from axquant.schema import BaselineKind, FeasibilityReport
from axquant.serde import file_sha256, load_model


def _qwen_config(bits: int | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        # These compact fixtures intentionally contain only the language path.
        # Do not declare a vision tower whose protected weights are absent.
        "language_model_only": True,
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
        },
    }
    if bits is not None:
        config["quantization"] = {
            "bits": bits,
            "group_size": 64,
            "mode": "affine",
        }
    return config


def _save_raw_safetensors(
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    path: Path,
) -> None:
    dtype_bytes = {"BF16": 2}
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * dtype_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _write_shared_files(model_dir: Path, shard_name: str) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "language_model.model.layers.0.mlp.down_proj.weight": shard_name,
                    "language_model.model.layers.0.input_layernorm.weight": shard_name,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def _quantized_baseline(tmp_path: Path, bits: int) -> Path:
    model_dir = tmp_path / f"qwen36-{bits}bit"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(_qwen_config(bits)),
        encoding="utf-8",
    )
    shard_name = "model-00001-of-00001.safetensors"
    packed_elements = 256 * bits // 32
    save_file(
        {
            "language_model.model.layers.0.mlp.down_proj.weight": np.zeros(
                (8, packed_elements // 8),
                dtype=np.uint32,
            ),
            "language_model.model.layers.0.input_layernorm.weight": np.zeros(
                (8,),
                dtype=np.float32,
            ),
        },
        model_dir / shard_name,
    )
    _save_raw_safetensors(
        {"mtp.fc.weight": ("BF16", (8, 8))},
        model_dir / "mtp.safetensors",
    )
    _write_shared_files(model_dir, shard_name)
    (model_dir / "model-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ax.native_model.v1",
                "tensors": [
                    {
                        "name": "language_model.model.layers.0.mlp.down_proj.weight",
                        "role": "mlp",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen-dense",
                "mtp_depth_max": 3,
                "mtp_tensor_count": 1,
            }
        ),
        encoding="utf-8",
    )
    sidecar = model_dir / "mtp.safetensors"
    (model_dir / "ax_mtp_sidecar_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "ax.mtp_sidecar_provenance.v1",
                "output": {
                    "mtp": {
                        "path": "mtp.safetensors",
                        "size_bytes": sidecar.stat().st_size,
                        "sha256": file_sha256(sidecar),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def _bf16_source(tmp_path: Path) -> Path:
    model_dir = tmp_path / "qwen36-bf16"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(_qwen_config()),
        encoding="utf-8",
    )
    shard_name = "model-00001-of-00001.safetensors"
    _save_raw_safetensors(
        {
            "language_model.model.layers.0.mlp.down_proj.weight": (
                "BF16",
                (16, 16),
            ),
            "language_model.model.layers.0.input_layernorm.weight": (
                "BF16",
                (8,),
            ),
        },
        model_dir / shard_name,
    )
    _save_raw_safetensors(
        {"mtp.fc.weight": ("BF16", (8, 8))},
        model_dir / "mtp.safetensors",
    )
    _write_shared_files(model_dir, shard_name)
    return model_dir


def _target(
    path: Path,
    kind: BaselineKind,
    revision: str,
) -> ArtifactTarget:
    return ArtifactTarget(
        model=path,
        kind=kind,
        model_id=f"AutomatosX/{path.name}",
        revision=revision,
    )


def test_quantized_baseline_audit_reports_logical_bpw(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    audit = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert audit.complete is True
    assert audit.logical_parameters == 328
    assert audit.mtp_logical_parameters == 64
    assert audit.precision_parameters == {"4bit": 256, "bf16": 64, "f32": 8}
    assert audit.effective_bpw > 4.0
    assert audit.main_effective_bpw > 4.0
    assert audit.integrity.mtp_provenance_valid is True


def test_feasibility_distinguishes_baseline_ready_from_conversion_ready(
    tmp_path: Path,
) -> None:
    four = _quantized_baseline(tmp_path, 4)
    six = _quantized_baseline(tmp_path, 6)
    baseline_report = assess_feasibility(
        reference_4bit=_target(four, BaselineKind.UNIFORM_4BIT, "a" * 40),
        reference_6bit=_target(six, BaselineKind.UNIFORM_6BIT, "b" * 40),
    )
    assert baseline_report.status == "baseline-ready"
    assert baseline_report.checks["logical_parameter_counts_match"] is True
    assert "a complete BF16 source checkpoint" in baseline_report.blockers[-1]

    source = _bf16_source(tmp_path)
    ready_report = assess_feasibility(
        reference_4bit=_target(four, BaselineKind.UNIFORM_4BIT, "a" * 40),
        reference_6bit=_target(six, BaselineKind.UNIFORM_6BIT, "b" * 40),
        source_bf16=_target(source, BaselineKind.BF16_SOURCE, "c" * 40),
    )
    assert ready_report.status == "ready-for-conversion"
    assert ready_report.blockers == []
    assert ready_report.checks["source_bf16_complete"] is True


def test_feasibility_blocks_incomplete_weight_index(tmp_path: Path) -> None:
    four = _quantized_baseline(tmp_path, 4)
    six = _quantized_baseline(tmp_path, 6)
    (six / "model-00001-of-00001.safetensors").unlink()
    report = assess_feasibility(
        reference_4bit=_target(four, BaselineKind.UNIFORM_4BIT, "a" * 40),
        reference_6bit=_target(six, BaselineKind.UNIFORM_6BIT, "b" * 40),
    )
    assert report.status == "blocked"
    assert report.checks["required_baselines_complete"] is False
    assert any("missing shards" in blocker for blocker in report.blockers)


def test_baseline_audit_rejects_invalid_mtp_checksum(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    provenance_path = model_dir / "ax_mtp_sidecar_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["output"]["mtp"]["sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    audit = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert audit.complete is False
    assert audit.integrity.mtp_provenance_valid is False
    assert any("MTP provenance" in issue for issue in audit.issues)


def test_baseline_audit_rejects_noncanonical_mtp_checksum(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    provenance_path = model_dir / "ax_mtp_sidecar_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["output"]["mtp"]["sha256"] = provenance["output"]["mtp"]["sha256"].upper()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    audit = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))

    assert audit.complete is False
    assert audit.integrity.mtp_provenance_valid is False
    assert any("MTP provenance" in issue for issue in audit.issues)


def test_baseline_audit_resolves_the_requested_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axquant.feasibility as feasibility

    model_dir = _quantized_baseline(tmp_path, 4)
    revision = "a" * 40
    original = feasibility.resolve_model_dir
    observed: dict[str, object] = {}

    def recording_resolve(
        model: str | Path,
        *,
        revision: str | None = None,
        allow_download: bool = False,
    ) -> Path:
        observed.update(revision=revision, allow_download=allow_download)
        return original(model, revision=revision, allow_download=allow_download)

    monkeypatch.setattr(feasibility, "resolve_model_dir", recording_resolve)
    audit = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, revision))
    assert audit.complete is True
    assert observed == {"revision": revision, "allow_download": False}


def test_baseline_audit_rejects_mutable_revision_and_symlinks(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    mutable = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "main"))
    assert mutable.complete is False
    assert any("revision is not pinned" in issue for issue in mutable.issues)

    external = tmp_path / "external-tokenizer.json"
    external.write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").unlink()
    (model_dir / "tokenizer.json").symlink_to(external)
    linked = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert linked.complete is False
    assert any("symbolic links" in issue for issue in linked.issues)

    root_link = tmp_path / "linked-model"
    root_link.symlink_to(model_dir, target_is_directory=True)
    linked_root = audit_artifact(_target(root_link, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert linked_root.complete is False
    assert any("root cannot be a symbolic link" in issue for issue in linked_root.issues)


def test_baseline_audit_supports_one_alternate_mtp_sidecar_only(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    alternate = model_dir / "mtp_head.safetensors"
    (model_dir / "mtp.safetensors").replace(alternate)
    provenance_path = model_dir / "ax_mtp_sidecar_manifest.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["output"]["mtp"] = {
        "path": alternate.name,
        "size_bytes": alternate.stat().st_size,
        "sha256": file_sha256(alternate),
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    alternate_audit = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert alternate_audit.complete is True
    assert alternate_audit.integrity.mtp_sidecar_present is True

    shutil.copyfile(alternate, model_dir / "mtp.safetensors")
    ambiguous = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert ambiguous.complete is False
    assert any("multiple external MTP sidecars" in issue for issue in ambiguous.issues)


def test_baseline_audit_rejects_boolean_mtp_runtime_counts(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    runtime_path = model_dir / "mtplx_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["mtp_depth_max"] = True
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    audit = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert audit.complete is False
    assert audit.integrity.mtp_runtime_valid is False


def test_baseline_audit_rejects_nonfinite_json_and_unsafe_index_paths(tmp_path: Path) -> None:
    model_dir = _quantized_baseline(tmp_path, 4)
    config_path = model_dir / "config.json"
    config_path.write_text('{"model_type": NaN}', encoding="utf-8")
    invalid_config = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert invalid_config.complete is False
    assert invalid_config.integrity.config_valid is False

    config_path.write_text(json.dumps(_qwen_config(4)), encoding="utf-8")
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["language_model.model.layers.0.mlp.down_proj.weight"] = (
        "nested\\shard.safetensors"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    unsafe_index = audit_artifact(_target(model_dir, BaselineKind.UNIFORM_4BIT, "a" * 40))
    assert unsafe_index.complete is False
    assert any("unsafe path" in issue for issue in unsafe_index.issues)


def test_feasibility_cli_refuses_outputs_inside_checkpoint_or_same_file(tmp_path: Path) -> None:
    four = _quantized_baseline(tmp_path, 4)
    six = _quantized_baseline(tmp_path, 6)
    config_path = four / "config.json"
    original_config = config_path.read_bytes()
    common = [
        "feasibility",
        "--reference-4bit",
        str(four),
        "--reference-4bit-revision",
        "a" * 40,
        "--reference-6bit",
        str(six),
        "--reference-6bit-revision",
        "b" * 40,
    ]

    assert (
        main(
            [
                *common,
                "--output",
                str(config_path),
                "--markdown-output",
                str(tmp_path / "report.md"),
            ]
        )
        == 2
    )
    assert config_path.read_bytes() == original_config

    same_output = tmp_path / "same-output"
    assert (
        main(
            [
                *common,
                "--output",
                str(same_output),
                "--markdown-output",
                str(same_output),
            ]
        )
        == 2
    )
    assert not same_output.exists()


def test_feasibility_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    four = _quantized_baseline(tmp_path, 4)
    six = _quantized_baseline(tmp_path, 6)
    output = tmp_path / "feasibility.json"
    markdown = tmp_path / "feasibility.md"
    result = main(
        [
            "feasibility",
            "--reference-4bit",
            str(four),
            "--reference-4bit-revision",
            "a" * 40,
            "--reference-6bit",
            str(six),
            "--reference-6bit-revision",
            "b" * 40,
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ]
    )
    assert result == 0
    report = load_model(output, FeasibilityReport)
    assert report.status == "baseline-ready"
    assert "# AXQuant Feasibility Report" in markdown.read_text(encoding="utf-8")
    assert (
        main(
            [
                "feasibility",
                "--reference-4bit",
                str(four),
                "--reference-4bit-revision",
                "a" * 40,
                "--reference-6bit",
                str(six),
                "--reference-6bit-revision",
                "b" * 40,
                "--require-ready",
                "--output",
                str(output),
                "--markdown-output",
                str(markdown),
            ]
        )
        == 1
    )
