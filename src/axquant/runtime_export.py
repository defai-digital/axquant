"""Atomic, explicit runtime variants built through public tensor formats.

No peer implementation is imported. Precision assignments never change.
The original conversion manifest is source provenance, not variant evidence.
"""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from axquant.errors import ArtifactError
from axquant.mtp_sidecar import QWEN4_MTP_ARCH_ID
from axquant.ngram_layout import (
    INDEX_FILENAME,
    NGRAM_RELAYOUT_MANIFEST_FILENAME,
    _copy_file_verified,
    _iter_pack_files,
    _parse_safetensors_file,
    _plan_relayout,
    _read_json_object,
    _serialize_tensor_parts,
    _TensorPart,
    relayout_ngram_table,
)
from axquant.runtime_compatibility import (
    QWEN35_MTPLX_PROFILE,
    QWEN35_OMLX_PROFILE,
    inspect_runtime_export,
    write_runtime_compatibility,
)
from axquant.schema.runtime_export import RuntimeExportManifest
from axquant.serde import file_sha256, write_data

RuntimeExportTarget = Literal["omlx", "mtplx"]
EXPORT_MANIFEST_FILENAME = "axquant_runtime_export.json"
_NORM_SUFFIXES = (
    ".q_norm.weight",
    ".k_norm.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".hc_norm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
)
_SUPPORT_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "NOTICE",
    INDEX_FILENAME,
    "mtplx_runtime.json",
}


def _shift_norm(part: _TensorPart) -> _TensorPart:
    """Translate a zero-centered RMSNorm to gamma using float32 arithmetic."""
    tensor = part.tensor
    if len(tensor.shape) != 1 or tensor.dtype not in {"F32", "F16", "BF16"}:
        raise ArtifactError(f"RMSNorm requires a floating vector: {tensor.name}")
    with part.source.open("rb") as source:
        source.seek(part.layout.data_base + tensor.start)
        raw = source.read(tensor.byte_count)
    if len(raw) != tensor.byte_count:
        raise ArtifactError(f"truncated norm {tensor.name}")
    payload = bytearray()
    if tensor.dtype == "BF16":
        for (bits,) in struct.iter_unpack("<H", raw):
            value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            shifted = struct.unpack("<I", struct.pack("<f", value + 1.0))[0]
            rounded = (shifted + 0x7FFF + ((shifted >> 16) & 1)) >> 16
            payload.extend(struct.pack("<H", rounded & 0xFFFF))
    else:
        code = "f" if tensor.dtype == "F32" else "e"
        for (value,) in struct.iter_unpack("<" + code, raw):
            # MLX normalizes through float32 even when storage uses float16.
            shifted_value = struct.unpack("<f", struct.pack("<f", value + 1.0))[0]
            payload.extend(struct.pack("<" + code, shifted_value))
    return replace(part, payload=bytes(payload))


# RMSNorm gains of a Qwen-Next MTP head (Qwen 3.5, 3.6, 3.8; dense and MoE).
_QWEN35_HEAD_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_embedding.weight",
    ".pre_fc_norm_hidden.weight",
)
_QWEN35_HEAD_FINAL_NORM = "mtp.norm.weight"
QWEN35_HEAD_SHARD = "model-mtp.safetensors"
# A trunk stored as MLX multipliers has input norms near 1.0; a zero-centered
# (raw HF delta) trunk has them near 0. The head is only shifted onto a trunk
# that already uses multipliers.
_TRUNK_MULTIPLIER_MEAN_MIN = 0.3


def _is_qwen35_head_norm(name: str) -> bool:
    return name == _QWEN35_HEAD_FINAL_NORM or name.endswith(_QWEN35_HEAD_NORM_SUFFIXES)


def _vector_mean(part: _TensorPart) -> float:
    """Mean of a floating RMSNorm vector, read straight from the shard."""
    tensor = part.tensor
    if len(tensor.shape) != 1 or tensor.dtype not in {"F32", "F16", "BF16"}:
        raise ArtifactError(f"RMSNorm requires a floating vector: {tensor.name}")
    with part.source.open("rb") as source:
        source.seek(part.layout.data_base + tensor.start)
        raw = source.read(tensor.byte_count)
    if len(raw) != tensor.byte_count:
        raise ArtifactError(f"truncated norm {tensor.name}")
    if tensor.dtype == "BF16":
        values = [
            struct.unpack("<f", struct.pack("<I", bits << 16))[0]
            for (bits,) in struct.iter_unpack("<H", raw)
        ]
    else:
        values = [
            value
            for (value,) in struct.iter_unpack("<" + ("f" if tensor.dtype == "F32" else "e"), raw)
        ]
    return sum(values) / len(values)


def _check_qwen35_trunk_uses_multipliers(pack: Path, index: dict[str, Any]) -> None:
    """Refuse to shift a head onto a trunk that is not stored as multipliers."""
    sample = next(
        (
            (name, filename)
            for name, filename in sorted(index["weight_map"].items())
            if name.endswith("layers.0.input_layernorm.weight") and "mtp" not in name
        ),
        None,
    )
    if sample is None:
        raise ArtifactError("cannot sample a trunk RMSNorm to confirm its representation")
    name, filename = sample
    layout = _parse_safetensors_file(pack / filename)
    mean = _vector_mean(_TensorPart(pack / filename, layout, layout.tensors[name]))
    if mean < _TRUNK_MULTIPLIER_MEAN_MIN:
        raise ArtifactError(
            f"trunk RMSNorm {name} has mean {mean:.3f}: it looks zero-centered, "
            "not stored as MLX multipliers, so the head is not shifted"
        )


def _normalize_qwen35_omlx(pack: Path, runtime: dict[str, Any]) -> None:
    """Turn the raw-delta MTP sidecar into an oMLX-loadable, indexed head.

    Only the head's RMSNorm gains change (+1 in float32, rounded back to the
    stored dtype); every other tensor keeps its bytes. The head moves into
    ``model-mtp.safetensors`` under the trunk's key prefix and into the weight
    index, where oMLX finds MTP weights; this variant is separate from the
    stock pack, which must not index ``mtp.*`` (stock mlx-lm would shift the
    trunk norms a second time).
    """
    config_path = pack / "config.json"
    config = _read_json_object(config_path, "config.json")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ArtifactError("text_config must be an object")
    sidecar_path = pack / "mtp.safetensors"
    if not sidecar_path.is_file():
        raise ArtifactError("Qwen MTP export needs mtp.safetensors")
    if runtime.get("mtp_norm_layout") != "raw_hf_delta":
        raise ArtifactError("Qwen MTP export needs declared raw_hf_delta head RMSNorms")
    index = _read_json_object(pack / INDEX_FILENAME, INDEX_FILENAME)
    _check_qwen35_trunk_uses_multipliers(pack, index)
    prefix = (
        "language_model."
        if any(name.startswith("language_model.") for name in index["weight_map"])
        else ""
    )
    layout = _parse_safetensors_file(sidecar_path)
    if any(not name.startswith("mtp.") for name in layout.tensors):
        raise ArtifactError("MTP sidecar contains tensors outside the native mtp.* namespace")
    entries: list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]] = []
    shifted = 0
    for name, tensor in layout.tensors.items():
        part = _TensorPart(sidecar_path, layout, tensor)
        if _is_qwen35_head_norm(name):
            part = _shift_norm(part)
            shifted += 1
        elif len(tensor.shape) == 1:
            raise ArtifactError(f"unexpected 1-D tensor in the MTP head: {name}")
        entries.append((prefix + name, tensor.dtype, tensor.shape, (part,)))
    if shifted < 7:
        raise ArtifactError(f"expected 7 MTP-head RMSNorm gains, found {shifted}")
    shard = pack / QWEN35_HEAD_SHARD
    for target_name, *_ in entries:
        if target_name in index["weight_map"]:
            raise ArtifactError(f"MTP tensor overlaps the trunk: {target_name}")
    _serialize_tensor_parts(layout.metadata, entries, shard)
    sidecar_path.unlink()
    for target_name, *_ in entries:
        index["weight_map"][target_name] = QWEN35_HEAD_SHARD
    text["mtp_num_hidden_layers"] = max(int(text.get("mtp_num_hidden_layers", 0) or 0), 1)
    layouts = {
        filename: _parse_safetensors_file(pack / filename)
        for filename in set(index["weight_map"].values())
    }
    index.setdefault("metadata", {})["total_size"] = sum(
        layouts[filename].tensors[name].byte_count for name, filename in index["weight_map"].items()
    )
    runtime["mtp_norm_layout"] = "mlx_multiplier"
    runtime["mtp_layout"] = "native-qwen35-indexed"
    runtime["mtp_tensor_count"] = len(entries)
    runtime["runtime_verified"] = False
    runtime.pop("mtplx_version", None)
    runtime.pop("layout", None)
    write_data(config_path, config)
    write_data(pack / INDEX_FILENAME, index)
    write_data(pack / "mtplx_runtime.json", runtime)


def _adapt_mtplx_weights(pack: Path, config: dict[str, Any]) -> list[str]:
    """Translate norm representation and raw-HF MTP expert storage only."""
    text = config.get("text_config", config)
    hidden = text.get("hidden_size")
    intermediate = text.get("moe_intermediate_size")
    experts = text.get("num_experts")
    index = _read_json_object(pack / INDEX_FILENAME, INDEX_FILENAME)
    filenames = set(index["weight_map"].values())
    if (pack / "mtp.safetensors").is_file():
        filenames.add("mtp.safetensors")
    changed: list[str] = []
    for filename in sorted(filenames):
        path = pack / filename
        layout = _parse_safetensors_file(path)
        entries: list[tuple[str, str, tuple[int, ...], tuple[_TensorPart, ...]]] = []
        did_change = False
        for name, tensor in layout.tensors.items():
            part = _TensorPart(path, layout, tensor)
            if name.startswith(("language_model.", "mtp.")) and name.endswith(_NORM_SUFFIXES):
                part = _shift_norm(part)
                did_change = True
            if name.startswith("mtp.") and name.endswith(".mlp.experts.gate_up_proj"):
                if tensor.dtype not in {"F32", "F16", "BF16"} or tensor.shape != (
                    experts,
                    2 * intermediate,
                    hidden,
                ):
                    raise ArtifactError(
                        "MTP gate_up requires the native floating "
                        "[experts, 2*intermediate, hidden] layout"
                    )
                block_bytes = tensor.byte_count // (2 * experts)
                for projection, offset in (("gate_proj", 0), ("up_proj", block_bytes)):
                    target_name = (
                        name.removesuffix("experts.gate_up_proj")
                        + f"switch_mlp.{projection}.weight"
                    )
                    parts = tuple(
                        replace(
                            part,
                            tensor=replace(
                                tensor,
                                start=tensor.start + expert * 2 * block_bytes + offset,
                                end=tensor.start + expert * 2 * block_bytes + offset + block_bytes,
                            ),
                        )
                        for expert in range(experts)
                    )
                    entries.append(
                        (target_name, tensor.dtype, (experts, intermediate, hidden), parts)
                    )
                did_change = True
                continue
            if name.startswith("mtp.") and name.endswith(".mlp.experts.down_proj"):
                if tensor.dtype not in {"F32", "F16", "BF16"} or tensor.shape != (
                    experts,
                    hidden,
                    intermediate,
                ):
                    raise ArtifactError("MTP down_proj has an unsupported expert layout")
                name = name.removesuffix("experts.down_proj") + "switch_mlp.down_proj.weight"
                did_change = True
            entries.append((name, tensor.dtype, tensor.shape, (part,)))
        if did_change:
            destination = pack / f".{filename}.adapted"
            _serialize_tensor_parts(layout.metadata, entries, destination)
            os.replace(destination, path)
            changed.append(filename)
    return changed


def _validate_mtplx_source(pack: Path, text: dict[str, Any]) -> None:
    """Reject unknown head layouts and norm representations before copying."""
    index = _read_json_object(pack / INDEX_FILENAME, INDEX_FILENAME)
    filenames = set(index["weight_map"].values())
    if (pack / "mtp.safetensors").is_file():
        filenames.add("mtp.safetensors")
    for filename in filenames:
        layout = _parse_safetensors_file(pack / filename)
        for name, tensor in layout.tensors.items():
            if name.startswith("model.language_model."):
                raise ArtifactError("MTPLX export requires MLX-sanitized trunk paths")
            if (
                name.startswith(("language_model.", "mtp."))
                and name.endswith(_NORM_SUFFIXES)
                and (len(tensor.shape) != 1 or tensor.dtype not in {"F32", "F16", "BF16"})
            ):
                raise ArtifactError(f"RMSNorm requires a floating vector: {name}")
            if name.startswith("mtp.") and tensor.dtype not in {"F32", "F16", "BF16"}:
                raise ArtifactError(
                    "initial MTPLX Flash-Next profile requires floating MTP weights"
                )
            if (
                name.startswith("mtp.")
                and name.endswith(".mlp.experts.gate_up_proj")
                and tensor.shape
                != (
                    text["num_experts"],
                    2 * text["moe_intermediate_size"],
                    text["hidden_size"],
                )
            ):
                raise ArtifactError("MTP gate_up requires the native floating expert layout")
            if (
                name.startswith("mtp.")
                and name.endswith(".mlp.experts.down_proj")
                and tensor.shape
                != (
                    text["num_experts"],
                    text["hidden_size"],
                    text["moe_intermediate_size"],
                )
            ):
                raise ArtifactError("MTP down_proj has an unsupported expert layout")


def _normalize_contract(pack: Path, target: RuntimeExportTarget) -> None:
    config_path = pack / "config.json"
    config = _read_json_object(config_path, "config.json")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ArtifactError("text_config must be an object")
    runtime_path = pack / "mtplx_runtime.json"
    runtime = (
        _read_json_object(runtime_path, "mtplx_runtime.json") if runtime_path.is_file() else {}
    )
    sidecar_path = pack / "mtp.safetensors"
    sidecar = _parse_safetensors_file(sidecar_path) if sidecar_path.is_file() else None
    if sidecar is not None:
        if runtime.get("mtp_norm_layout") != "raw_hf_delta":
            raise ArtifactError("native Flash-Next export requires declared raw_hf_delta MTP norms")
        if any(not name.startswith("mtp.") for name in sidecar.tensors):
            raise ArtifactError("native Flash-Next sidecar contains foreign tensors")
    index = _read_json_object(pack / INDEX_FILENAME, INDEX_FILENAME)
    if target == "omlx":
        # oMLX's Qwen4 native loader discovers MTP through the weight index.
        # This is a target variant; the stock MLX source index stays separate.
        if sidecar is not None:
            for name in sidecar.tensors:
                current = index["weight_map"].get(name)
                if current is not None and current != "mtp.safetensors":
                    raise ArtifactError(f"MTP tensor overlaps the trunk: {name}")
                index["weight_map"][name] = "mtp.safetensors"
            text["mtp_num_hidden_layers"] = 1
            for scope in (config, text):
                extra = scope.get("mlx_lm_extra_tensors", [])
                if isinstance(extra, dict):
                    # Companion pointers do not exclude tensor names.
                    continue
                if not isinstance(extra, list) or any(not isinstance(name, str) for name in extra):
                    raise ArtifactError("mlx_lm_extra_tensors must be a list or companion object")
                scope["mlx_lm_extra_tensors"] = [
                    name for name in extra if name not in sidecar.tensors
                ]
        runtime["mtp_layout"] = "native-qwen4-exp-indexed"
    else:
        if runtime.get("trunk_norm_layout") != "raw_hf_delta":
            raise ArtifactError("MTPLX export requires declared raw_hf_delta trunk norms")
        for field in ("hidden_size", "moe_intermediate_size", "num_experts"):
            if type(text.get(field)) is not int or text[field] <= 0:
                raise ArtifactError(f"MTPLX export requires positive {field}")
        _adapt_mtplx_weights(pack, config)
        # MTPLX attaches MTP independently after strict trunk loading.
        index["weight_map"] = {
            name: file for name, file in index["weight_map"].items() if not name.startswith("mtp.")
        }
        runtime["trunk_norm_layout"] = "mlx_multiplier"
        runtime["mtp_layout"] = "mtplx-qwen4-v1"
        runtime["mtp_norm_layout"] = "mtplx_qwen4_mixed"
    runtime.update(arch_id=QWEN4_MTP_ARCH_ID, runtime_verified=False)
    runtime.pop("mtplx_version", None)
    runtime.pop("layout", None)
    if sidecar_path.is_file():
        runtime["mtp_tensor_count"] = len(_parse_safetensors_file(sidecar_path).tensors)
    # total_size describes indexed tensor payloads, including MTP only on oMLX.
    layouts = {
        filename: _parse_safetensors_file(pack / filename)
        for filename in set(index["weight_map"].values())
    }
    index.setdefault("metadata", {})["total_size"] = sum(
        layouts[filename].tensors[name].byte_count for name, filename in index["weight_map"].items()
    )
    write_data(config_path, config)
    write_data(pack / INDEX_FILENAME, index)
    write_data(runtime_path, runtime)


def export_runtime_pack(
    source: str | Path,
    output: str | Path,
    *,
    target: RuntimeExportTarget,
    dry_run: bool = False,
    source_trunk_norm_layout: Literal["raw_hf_delta", "mlx_multiplier"] | None = None,
) -> dict[str, Any]:
    """Build a distinct runtime variant; fail before writes on unknown profiles."""
    source_dir, output_dir = (
        Path(source).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    if target not in {"omlx", "mtplx"}:
        raise ArtifactError(f"unsupported runtime export target {target!r}")
    if output_dir == source_dir or source_dir in output_dir.parents or output_dir.exists():
        raise ArtifactError("runtime export output must be new and outside the source pack")
    report = inspect_runtime_export(source_dir)
    verdict = report["targets"][target]
    if verdict["blockers"]:
        raise ArtifactError("; ".join(verdict["blockers"]))
    profile = verdict["profile"]
    qwen35_omlx = profile == QWEN35_OMLX_PROFILE
    if target == "mtplx" and profile == QWEN35_MTPLX_PROFILE:
        raise ArtifactError(
            "MTPLX loads this Qwen MTP layout as shipped (declared raw_hf_delta head "
            "norms, sidecar outside the index); no export is needed"
        )
    if not (source_dir / INDEX_FILENAME).is_file():
        raise ArtifactError("runtime variant export requires a weight index")
    # Validate tree membership before making staging directories.
    files = list(_iter_pack_files(source_dir))
    config = _read_json_object(source_dir / "config.json", "config.json")
    runtime_path = source_dir / "mtplx_runtime.json"
    runtime = (
        _read_json_object(runtime_path, "mtplx_runtime.json") if runtime_path.is_file() else {}
    )
    if source_trunk_norm_layout is not None:
        if source_trunk_norm_layout not in {"raw_hf_delta", "mlx_multiplier"}:
            raise ArtifactError("unknown source trunk norm declaration")
        existing = runtime.get("trunk_norm_layout")
        if existing is not None and existing != source_trunk_norm_layout:
            raise ArtifactError("source norm declaration conflicts with the packaged contract")
        runtime["trunk_norm_layout"] = source_trunk_norm_layout
    if (source_dir / "mtp.safetensors").is_file() and runtime.get(
        "mtp_norm_layout"
    ) != "raw_hf_delta":
        raise ArtifactError("MTP export requires declared raw_hf_delta MTP norms")
    if qwen35_omlx and runtime.get("trunk_norm_layout") != "mlx_multiplier":
        raise ArtifactError(
            "the Qwen oMLX export shifts the head onto an MLX-multiplier trunk; declare it "
            "with --source-trunk-norm-layout mlx_multiplier (a raw_hf_delta trunk is refused)"
        )
    if target == "mtplx":
        _plan_relayout(source_dir)
        if runtime.get("trunk_norm_layout") != "raw_hf_delta":
            raise ArtifactError("MTPLX export requires declared raw_hf_delta trunk norms")
        text = config.get("text_config", config)
        for field in ("hidden_size", "moe_intermediate_size", "num_experts"):
            if not isinstance(text, dict) or type(text.get(field)) is not int or text[field] <= 0:
                raise ArtifactError(f"MTPLX export requires positive {field}")
        _validate_mtplx_source(source_dir, text)
    if dry_run:
        return report
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.export.", dir=output_dir.parent))
    staging = temporary / "pack"
    try:
        if target == "mtplx":
            relayout_ngram_table(source_dir, staging)
            # The layout receipt described the intermediate representation.
            # The final export manifest binds the adapted variant instead.
            (staging / NGRAM_RELAYOUT_MANIFEST_FILENAME).unlink()
        else:
            staging.mkdir()
            for path in files:
                if qwen35_omlx and path.name == "vision.safetensors":
                    continue  # recorded as a modality constraint below
                if path.parent == source_dir and (
                    path.name in _SUPPORT_FILES or path.suffix == ".safetensors"
                ):
                    _copy_file_verified(path, staging / path.name)
        # Do not transplant source checksums, certificates, or runtime receipts.
        for path in list(_iter_pack_files(staging)):
            if path.parent != staging or (
                path.name not in _SUPPORT_FILES and path.suffix != ".safetensors"
            ):
                path.unlink()
        if source_trunk_norm_layout is not None:
            contract_path = staging / "mtplx_runtime.json"
            contract = _read_json_object(contract_path, "mtplx_runtime.json")
            contract["trunk_norm_layout"] = source_trunk_norm_layout
            write_data(contract_path, contract)
        if qwen35_omlx:
            staged_contract = _read_json_object(
                staging / "mtplx_runtime.json", "mtplx_runtime.json"
            )
            _normalize_qwen35_omlx(staging, staged_contract)
        else:
            _normalize_contract(staging, target)
        result = write_runtime_compatibility(staging)
        verdict = result["targets"][target]
        if verdict["status"] != "static-compatible":
            raise ArtifactError(f"runtime export remains incomplete: {verdict}")
        manifest = {
            "schema_version": "axquant.runtime-export.v1",
            "target": target,
            "source_binding": report["binding"],
            "source_manifest_sha256": file_sha256(source_dir / "axquant_manifest.json")
            if (source_dir / "axquant_manifest.json").is_file()
            else None,
            "source_trunk_norm_layout": source_trunk_norm_layout,
            "files": {path.name: file_sha256(path) for path in _iter_pack_files(staging)},
            "runtime_verified": False,
            "quality_certified": False,
            "launch_environment": verdict["launch_environment"],
            "model_settings": verdict["model_settings"],
            "modality_constraints": sorted(
                {
                    *verdict["modality_constraints"],
                    *report["targets"][target]["modality_constraints"],
                }
            ),
        }
        write_data(
            staging / EXPORT_MANIFEST_FILENAME, RuntimeExportManifest.model_validate(manifest)
        )
        if output_dir.exists():
            raise ArtifactError("runtime export output appeared during staging")
        os.rename(staging, output_dir)
        return result
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
