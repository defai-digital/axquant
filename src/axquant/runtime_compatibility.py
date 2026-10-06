"""Conservative, header-only runtime export checks for every conversion.

Static checks never establish successful generation or MTP exactness. New
architectures must add an explicit export profile before targeting a peer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axquant.artifact_paths import artifact_member_path
from axquant.errors import ArtifactError
from axquant.module_paths import is_ngram_shard_key
from axquant.mtp_sidecar import QWEN4_MTP_ARCH_ID
from axquant.ngram_layout import (
    INDEX_FILENAME,
    NGRAM_TABLE_FILENAME,
    _parse_safetensors_file,
    _read_json_object,
    _SafetensorsFile,
)
from axquant.schema.artifacts import ArtifactManifest
from axquant.schema.runtime_export import RuntimeCompatibilityReport
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

COMPATIBILITY_FILENAME = "axquant_compatibility.json"
COMPATIBILITY_SCHEMA = "axquant.runtime-compatibility.v1"
MTPLX_QWEN4_BASE_ENV = {
    "MTPLX_NGRAM_RESIDENT": "0",
    **{
        name: "0"
        for name in (
            "MTPLX_FUSED_GATE_UP",
            "MTPLX_FUSED_GDN_OUT",
            "MTPLX_FUSED_GDN_INPROJ",
            "MTPLX_QWEN4_RELAXED_DRAFT_TIES",
            "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
            "MTPLX_QWEN4_FIXED_M4_VERIFY",
            "MTPLX_QWEN4_M4_STAGE3",
            "MTPLX_QWEN4_ROUTE_KERNEL",
            "MTPLX_QWEN4_OPDIET",
            "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
            "MTPLX_QWEN4_BLOCK_VERIFY",
            "MTPLX_QWEN4_VERIFY_GLUE",
            "MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS",
        )
    },
}


def _canonical_ngram_issues(table: _SafetensorsFile) -> list[str]:
    bits, group = table.metadata.get("ngram_bits"), table.metadata.get("ngram_group_size")
    if bits not in {"0", "2", "4", "6", "8"}:
        return ["Standalone n-gram table needs explicit actual ngram_bits metadata"]
    expected = (
        {"ngram.weight"}
        if bits == "0"
        else {
            "ngram.weight",
            "ngram.scales",
            "ngram.biases",
        }
    )
    if set(table.tensors) != expected:
        return ["Standalone n-gram table must use canonical concatenated ngram.* keys"]
    weight = table.tensors["ngram.weight"]
    if len(weight.shape) != 2 or any(dim <= 0 for dim in weight.shape):
        return ["Standalone n-gram weight must be a nonempty matrix"]
    if bits == "0":
        return (
            []
            if weight.dtype in {"F16", "BF16", "F32"}
            else ["Unquantized n-gram weight must be floating"]
        )
    if group not in {"32", "64", "128"}:
        return ["Standalone n-gram table has unsupported group metadata"]
    scales, biases = table.tensors["ngram.scales"], table.tensors["ngram.biases"]
    if (
        weight.dtype != "U32"
        or len(scales.shape) != 2
        or scales.shape != biases.shape
        or scales.dtype not in {"F16", "BF16", "F32"}
        or scales.dtype != biases.dtype
        or weight.shape[0] != scales.shape[0]
        or weight.shape[1] * 32 != scales.shape[1] * int(group) * int(bits)
    ):
        return ["Standalone n-gram dtype/shape disagrees with actual bits/group metadata"]
    return []


def inspect_runtime_export(directory: str | Path) -> dict[str, Any]:
    """Report blockers from current on-disk headers, never from an old report."""
    pack = Path(directory)
    config = _read_json_object(pack / "config.json", "config.json")
    index_path = pack / INDEX_FILENAME
    if index_path.is_file():
        index = _read_json_object(index_path, INDEX_FILENAME)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactError("runtime compatibility requires a non-empty weight_map")
    else:
        # Small public MLX checkpoints may be unindexed single-file exports.
        files = sorted(pack.glob("model*.safetensors"))
        if not files:
            raise ArtifactError("runtime compatibility requires model Safetensors")
        weight_map = {}
        for path in files:
            for name in _parse_safetensors_file(path).tensors:
                if name in weight_map:
                    raise ArtifactError(f"duplicate model tensor {name}")
                weight_map[name] = path.name
    layouts = {}
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ArtifactError("weight_map names and filenames must be strings")
        if filename not in layouts:
            try:
                path = artifact_member_path(pack, filename)
            except ValueError as exc:
                raise ArtifactError(str(exc)) from exc
            layouts[filename] = _parse_safetensors_file(path)
        if name not in layouts[filename].tensors:
            raise ArtifactError(f"indexed tensor {name} is absent from {filename}")
    text = config.get("text_config", config)
    model_type = str(config.get("model_type", ""))
    if isinstance(text, dict) and model_type == "":
        model_type = str(text.get("model_type", ""))
    sidecar_path = pack / "mtp.safetensors"
    sidecar = _parse_safetensors_file(sidecar_path) if sidecar_path.is_file() else None
    table_path = pack / NGRAM_TABLE_FILENAME
    table = _parse_safetensors_file(table_path) if table_path.is_file() else None
    runtime_path = pack / "mtplx_runtime.json"
    runtime = (
        _read_json_object(runtime_path, "mtplx_runtime.json") if runtime_path.is_file() else {}
    )
    binding = {"config.json": file_sha256(pack / "config.json")}
    if index_path.is_file():
        binding[INDEX_FILENAME] = file_sha256(index_path)
    if runtime_path.is_file():
        binding["mtplx_runtime.json"] = file_sha256(runtime_path)
    headers = {
        name: {key: {"dtype": t.dtype, "shape": list(t.shape)} for key, t in layout.tensors.items()}
        for name, layout in layouts.items()
    }
    if sidecar is not None:
        headers["mtp.safetensors"] = {
            key: {"dtype": t.dtype, "shape": list(t.shape)} for key, t in sidecar.tensors.items()
        }
    if table is not None:
        headers[NGRAM_TABLE_FILENAME] = {
            key: {"dtype": t.dtype, "shape": list(t.shape)} for key, t in table.tensors.items()
        }
    header_metadata = {name: layout.metadata for name, layout in layouts.items()}
    if sidecar is not None:
        header_metadata["mtp.safetensors"] = sidecar.metadata
    if table is not None:
        header_metadata[NGRAM_TABLE_FILENAME] = table.metadata
    binding["tensor_headers_sha256"] = stable_sha256(
        {"tensors": headers, "metadata": header_metadata}
    )
    targets: dict[str, Any] = {}
    qwen4 = model_type in {"qwen4_exp", "qwen4_exp_text"}
    quantization = config.get("quantization", config.get("quantization_config", {}))
    modes = (
        sorted({str(v.get("mode", "affine")) for v in quantization.values() if isinstance(v, dict)})
        if isinstance(quantization, dict)
        else []
    )
    if isinstance(quantization, dict) and "bits" in quantization:
        modes = sorted(set(modes) | {str(quantization.get("mode", "affine"))})
    foreign_quantization = config.get("quantization_config")
    if isinstance(foreign_quantization, dict) and foreign_quantization.get("quant_method"):
        modes.append("non-mlx:" + str(foreign_quantization["quant_method"]))
    ngram_shards = any(is_ngram_shard_key(name) for name in weight_map)
    mtp_names = tuple(sidecar.tensors) if sidecar is not None else ()
    for target in ("omlx", "mtplx"):
        blockers: list[str] = []
        actions: list[str] = []
        if not qwen4:
            blockers.append(f"No audited {target} export profile for model_type {model_type!r}")
        else:
            if not isinstance(text, dict):
                blockers.append("text_config must be an object")
            elif text.get("mtp_num_hidden_layers", 0) and sidecar is None:
                blockers.append("MTP is declared but its native sidecar is absent")
            if sidecar is not None and not any(name.startswith("mtp.") for name in mtp_names):
                blockers.append("MTP sidecar has no native mtp.* tensors")
            if sidecar is not None and runtime.get("arch_id") != QWEN4_MTP_ARCH_ID:
                actions.append(
                    "Replace the historical Qwen3-Next label with the native "
                    "Flash-Next MTP contract"
                )
            if target == "omlx":
                if any(mode not in {"affine", "mxfp4", "mxfp8"} for mode in modes):
                    blockers.append("oMLX profile requires a public MLX quantization mode")
                if not ngram_shards:
                    blockers.append("oMLX profile requires indexed ShardedEmbedding n-gram storage")
                if sidecar is not None and not set(mtp_names).issubset(weight_map):
                    actions.append("Include native MTP tensors in the target runtime weight index")
            else:
                if not ngram_shards and table is None:
                    blockers.append("MTPLX n-gram sidecar is absent")
                if table is not None:
                    blockers.extend(_canonical_ngram_issues(table))
                    if ngram_shards:
                        blockers.append("N-gram table and indexed shards overlap")
                    if isinstance(text, dict) and text.get("ngram_sidecar") is not True:
                        actions.append("Enable ngram_sidecar in text config")
                if any(mode not in {"affine", "mxfp4"} for mode in modes):
                    blockers.append(
                        "MTPLX Flash-Next profile supports affine/MXFP4 trunk only; "
                        "MXFP8 has no audited export profile; precision is not changed"
                    )
                if ngram_shards:
                    actions.append("Concatenate n-gram shard rows with actual bits/group metadata")
                if runtime.get("trunk_norm_layout") != "mlx_multiplier":
                    actions.append("Rebase raw-HF trunk RMSNorm deltas to direct multipliers")
                if sidecar is not None and runtime.get("mtp_layout") != "mtplx-qwen4-v1":
                    actions.append(
                        "Normalize native MTP expert paths and head RMSNorm representation"
                    )
        targets[target] = {
            "profile": ("qwen4-exp-v2" if target == "mtplx" else "qwen4-exp-v1") if qwen4 else None,
            "status": "unsupported"
            if blockers
            else "requires-export"
            if actions
            else "static-compatible",
            "blockers": blockers,
            "required_actions": actions,
            "runtime_verified": False,
            "mtp_verified": False,
            "launch_environment": MTPLX_QWEN4_BASE_ENV if qwen4 and target == "mtplx" else {},
            "model_settings": {"qwen4_ple_ssd_offload": True} if qwen4 and target == "omlx" else {},
            "modality_constraints": ["MTPLX image input requires MTP generation mode"]
            if qwen4 and target == "mtplx"
            else [],
        }
    report = {
        "schema_version": COMPATIBILITY_SCHEMA,
        "model_type": model_type,
        "binding": binding,
        "targets": targets,
        "quality_certified": False,
    }
    return RuntimeCompatibilityReport.model_validate(report).model_dump(mode="json")


def write_runtime_compatibility(directory: str | Path) -> dict[str, Any]:
    report = inspect_runtime_export(directory)
    write_data(Path(directory) / COMPATIBILITY_FILENAME, report)
    return report


def require_runtime_compatibility_record(directory: str | Path) -> dict[str, Any]:
    """Reject reused/publication inputs with absent, stale, or unbound checks."""
    pack = Path(directory)
    path = pack / COMPATIBILITY_FILENAME
    if not path.is_file():
        raise ArtifactError(
            "pack has no runtime compatibility record; rebuild with current AXQuant"
        )
    saved = _read_json_object(path, COMPATIBILITY_FILENAME)
    RuntimeCompatibilityReport.model_validate(saved)
    current = inspect_runtime_export(pack)
    stale_verdict = any(
        saved["targets"][target][key] != current["targets"][target][key]
        for target in ("omlx", "mtplx")
        for key in ("profile", "status")
    )
    if (
        saved["binding"] != current["binding"]
        or saved["model_type"] != current["model_type"]
        or stale_verdict
    ):
        raise ArtifactError("runtime compatibility record is stale; pack config or headers changed")
    manifest = _read_json_object(pack / "axquant_manifest.json", "axquant_manifest.json")
    records = manifest.get("files")
    if not isinstance(records, list) or not any(
        isinstance(record, dict)
        and record.get("path") == COMPATIBILITY_FILENAME
        and record.get("sha256") == file_sha256(path)
        for record in records
    ):
        raise ArtifactError("runtime compatibility record is not bound by the conversion manifest")
    return current


def validate_declared_runtime_compatibility(
    directory: Path,
    manifest: ArtifactManifest | None = None,
) -> None:
    """Gate newly declared reports while retaining historical artifact readers."""
    if manifest is None:
        path = directory / "axquant_manifest.json"
        if not path.is_file():
            return
        raw = _read_json_object(path, "axquant_manifest.json")
        files = raw.get("files", [])
        declares_report = isinstance(files, list) and any(
            isinstance(record, dict) and record.get("path") == COMPATIBILITY_FILENAME
            for record in files
        )
        if not declares_report and not (directory / COMPATIBILITY_FILENAME).exists():
            return
        manifest = load_model(path, ArtifactManifest)
    if (directory / COMPATIBILITY_FILENAME).exists() or any(
        record.path == COMPATIBILITY_FILENAME for record in manifest.files
    ):
        require_runtime_compatibility_record(directory)
