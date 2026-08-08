from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from safetensors import safe_open

from axquant.architectures import adapter_for
from axquant.errors import ArtifactError
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    Inventory,
    ModelIdentity,
    OptimizationScope,
    QuantMethod,
    SourceConversionProvenance,
    SupportTier,
    TensorRole,
    TensorSpec,
)
from axquant.serde import load_model

_FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
_DTYPE_BYTES = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
_MTP_TOKEN = re.compile(r"(^|[./_-])(mtp|multi[_-]?token)([./_-]|$)")
_MTP_PRECISION = re.compile(r"INT(16|8|6|4)(?![0-9])", re.IGNORECASE)
_FUSED_EXPERT_PATH_TOKENS = ("switch_mlp", "switch_glu", ".experts.")


def resolve_model_dir(
    model: str | Path,
    *,
    revision: str | None = None,
    allow_download: bool = False,
) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()
    try:
        resolved = snapshot_download(
            str(model),
            revision=revision,
            local_files_only=not allow_download,
        )
    except Exception as exc:
        action = "download or resolve" if allow_download else "find in the local Hub cache"
        raise ArtifactError(f"cannot {action} model {model}: {exc}") from exc
    return Path(resolved).resolve()


def classify_tensor(name: str, source_file: str = "") -> TensorRole:
    name_value = name.lower()
    source_name = source_file.rsplit("/", 1)[-1].lower()
    protected_path_value = f"{source_file}/{name}".lower()
    if source_name in EXTERNAL_MTP_SIDECAR_FILENAMES or _MTP_TOKEN.search(name_value):
        if any(token in protected_path_value for token in ("output_head", "lm_head", "vocab_head")):
            return TensorRole.MTP_OUTPUT
        if any(token in protected_path_value for token in ("proj", "projection")):
            return TensorRole.MTP_PROJECTION
        return TensorRole.MTP_BLOCK
    if any(
        token in name_value
        for token in ("audio_tower", "audio_model", "audio_encoder", "audio_projector")
    ):
        return TensorRole.AUDIO
    if (
        any(
            token in name_value
            for token in (
                "vision",
                "visual",
                "image",
                "multimodal",
                "multi_modal",  # Mistral3 multi_modal_projector (underscore form)
                "patch_merger",
            )
        )
        or source_name == "vision.safetensors"
    ):
        return TensorRole.VISION
    if "norm" in name_value:
        return TensorRole.NORM
    if any(token in name_value for token in ("lm_head", "output.weight", "output_layer")):
        return TensorRole.LM_HEAD
    if any(token in name_value for token in ("embed_tokens", "token_embedding", "wte.weight")):
        return TensorRole.EMBEDDING
    if (
        "router" in name_value
        or ".mlp.gate." in name_value
        or "shared_expert_gate" in name_value
        or ("expert" in name_value and ".gate." in name_value)
    ):
        # Qwen-style MoE routers are named `mlp.gate` (distinct from the
        # `gate_proj` expert/MLP projections, which carry the `_proj` suffix);
        # `shared_expert_gate` is routing state, not an expert projection.
        return TensorRole.ROUTER
    if "expert" in name_value or "switch_mlp" in name_value or "switch_glu" in name_value:
        return TensorRole.EXPERT
    if any(
        token in name_value
        for token in (
            "self_attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv_proj",
        )
    ):
        return TensorRole.ATTENTION
    if any(
        token in name_value
        for token in (
            "mlp",
            "feed_forward",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        )
    ):
        return TensorRole.MLP
    return TensorRole.OTHER


def module_path_for(tensor_name: str) -> str:
    for suffix in (".weight", ".kernel"):
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _read_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ArtifactError(f"{model_dir} does not contain config.json")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{config_path} must contain a JSON object")
    return value


def _tensor_files(
    model_dir: Path,
) -> tuple[list[Path], dict[Path, frozenset[str]] | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return sorted(model_dir.glob("*.safetensors")), None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError(f"{index_path} must contain a non-empty weight_map")
    indexed_names: dict[Path, set[str]] = {}
    for tensor_name, value in weight_map.items():
        if not tensor_name:
            raise ArtifactError(f"{index_path} contains an empty tensor name")
        if not isinstance(value, str):
            raise ArtifactError(f"{index_path} contains a non-string shard reference")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError(f"{index_path} contains an unsafe path: {value}")
        shard = model_dir / relative
        if shard.suffix != ".safetensors":
            raise ArtifactError(f"{index_path} references a non-Safetensors shard: {value}")
        if not shard.is_file():
            raise ArtifactError(f"{index_path} references a missing shard: {value}")
        indexed_names.setdefault(shard, set()).add(tensor_name)
    paths = set(indexed_names)
    for name in EXTERNAL_MTP_SIDECAR_FILENAMES:
        mtp_sidecar = model_dir / name
        if mtp_sidecar.is_file():
            paths.add(mtp_sidecar)
    vision_sidecar = model_dir / "vision.safetensors"
    if vision_sidecar.is_file():
        paths.add(vision_sidecar)
    unexpected_shards = set(model_dir.glob("*.safetensors")) - paths
    if unexpected_shards:
        names = sorted(path.name for path in unexpected_shards)
        raise ArtifactError(f"{index_path} does not account for Safetensors files: {names}")
    return (
        sorted(paths),
        {path: frozenset(names) for path, names in indexed_names.items()},
    )


def _architecture(config: dict[str, Any]) -> str | None:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures and isinstance(architectures[0], str):
        return architectures[0]
    model_type = config.get("model_type")
    return str(model_type) if model_type else None


def _native_quantization(model_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = model_dir / "model-manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tensors = manifest.get("tensors") if isinstance(manifest, dict) else None
    if not isinstance(tensors, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        quantization = tensor.get("quantization")
        if isinstance(name, str) and isinstance(quantization, dict):
            result[name] = quantization
    return result


def _mtp_runtime_bits(model_dir: Path) -> int | None:
    runtime_path = model_dir / "mtplx_runtime.json"
    if not runtime_path.is_file():
        return None
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {runtime_path}: {exc}") from exc
    if not isinstance(runtime, dict):
        raise ArtifactError(f"{runtime_path} must contain a JSON object")
    if "mtp_sidecar_bits" in runtime:
        bits = runtime["mtp_sidecar_bits"]
        if isinstance(bits, bool) or not isinstance(bits, int) or bits not in {4, 6, 8, 16}:
            raise ArtifactError(f"{runtime_path} has invalid mtp_sidecar_bits")
        return bits
    description = runtime.get("mtp_sidecar")
    if description is None:
        return None
    if not isinstance(description, str):
        raise ArtifactError(f"{runtime_path} has a non-string mtp_sidecar description")
    match = _MTP_PRECISION.search(description)
    return int(match.group(1)) if match else None


def _deepseek_fp4_expert_weight(name: str, dtype: str, config: dict[str, Any]) -> bool:
    """True when a tensor is a DeepSeek V4 FP4 expert body packed in I8/U8.

    Official Flash/Pro exports store routed-expert weights as FP4 in int8
    containers (``expert_dtype: fp4``). Physical last-dim length is half the
    logical dequant width MLX-LM reconstructs after sanitize.
    """

    if dtype not in {"I8", "U8"} or not name.endswith(".weight"):
        return False
    if ".ffn.experts." not in name and ".mlp.experts." not in name:
        return False
    if ".shared_experts." in name:
        return False
    expert_dtype = str(config.get("expert_dtype", "")).lower()
    if expert_dtype in {"fp4", "f4", "nvfp4", "mxfp4"}:
        return True
    configured = config.get("quantization_config")
    if isinstance(configured, dict):
        fmt = str(configured.get("fmt", "")).lower()
        if "fp4" in fmt or fmt in {"f4", "e2m1"}:
            return True
    return expert_dtype == "" and str(config.get("model_type", "")).startswith("deepseek")


def _quantization_details(
    name: str,
    dtype: str,
    source_file: str,
    config: dict[str, Any],
    native_quantization: dict[str, dict[str, Any]],
    mtp_bits: int | None,
) -> tuple[int | None, int | None, QuantMethod | None]:
    if _deepseek_fp4_expert_weight(name, dtype, config):
        return 4, None, QuantMethod.AFFINE
    if dtype != "U32" or name.endswith((".scales", ".biases")):
        if dtype == "BF16":
            return 16, None, QuantMethod.BF16
        return None, None, None
    quantization = native_quantization.get(name)
    configured = config.get("quantization") or config.get("quantization_config")
    if quantization is None and isinstance(configured, dict):
        override = configured.get(module_path_for(name))
        quantization = override if isinstance(override, dict) else configured
    if quantization is None and _MTP_TOKEN.search(f"{source_file}/{name}".lower()):
        if mtp_bits is not None:
            return mtp_bits, None, QuantMethod.AFFINE
        return None, None, None
    if not isinstance(quantization, dict):
        return None, None, None
    bits = quantization.get("bits")
    group_size = quantization.get("group_size")
    mode = quantization.get("mode", "affine")
    parsed_bits = int(bits) if isinstance(bits, int) and not isinstance(bits, bool) else None
    parsed_group_size = (
        int(group_size)
        if isinstance(group_size, int) and not isinstance(group_size, bool)
        else None
    )
    try:
        method = QuantMethod(str(mode))
    except ValueError:
        method = None
    return parsed_bits, parsed_group_size, method


def inspect_model(
    model: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    allow_download: bool = False,
    allow_quantized: bool = False,
) -> Inventory:
    model_dir = resolve_model_dir(model, revision=revision, allow_download=allow_download)
    config = _read_config(model_dir)
    provenance_path = model_dir / "axquant_source.json"
    source_provenance = (
        load_model(provenance_path, SourceConversionProvenance)
        if provenance_path.is_file()
        else None
    )
    if source_provenance is not None:
        if model_id is not None and model_id != source_provenance.source_model:
            raise ArtifactError("explicit model ID differs from axquant_source.json")
        if revision is not None and revision.lower() != source_provenance.source_revision:
            raise ArtifactError("explicit revision differs from axquant_source.json")
        model_id = source_provenance.source_model
        revision = source_provenance.source_revision
    model_reference = model_id or str(model)
    adapter = adapter_for(model_reference, config)
    architecture_profile = (
        adapter.profile(model_reference, config)
        if adapter is not None
        else ArchitectureProfile(
            config_model_type=(str(config["model_type"]) if config.get("model_type") else None)
        )
    )
    quantized_source = bool(config.get("quantization") or config.get("quantization_config"))
    native_quantization = _native_quantization(model_dir)
    mtp_bits = _mtp_runtime_bits(model_dir)
    if quantized_source and not allow_quantized:
        raise ArtifactError(
            "the source checkpoint is already quantized; use a BF16 source or pass "
            "--allow-quantized for inventory-only work"
        )
    tensor_files, indexed_names = _tensor_files(model_dir)
    if not tensor_files:
        raise ArtifactError(f"{model_dir} does not contain Safetensors weights")

    tensors: list[TensorSpec] = []
    unclassified_adapter_tensors: list[str] = []
    seen: set[str] = set()
    source_files: list[str] = []
    for tensor_file in tensor_files:
        relative_file = tensor_file.relative_to(model_dir).as_posix()
        source_files.append(relative_file)
        try:
            with safe_open(tensor_file, framework="numpy") as handle:
                actual_names = frozenset(handle.keys())
                expected_names = (
                    indexed_names.get(tensor_file) if indexed_names is not None else None
                )
                if expected_names is not None and actual_names != expected_names:
                    missing = sorted(expected_names - actual_names)
                    unindexed = sorted(actual_names - expected_names)
                    details: list[str] = []
                    if missing:
                        details.append(f"missing tensors {missing[:10]}")
                    if unindexed:
                        details.append(f"unindexed tensors {unindexed[:10]}")
                    raise ArtifactError(
                        f"{relative_file} does not match model.safetensors.index.json: "
                        + "; ".join(details)
                    )
                for name in sorted(actual_names):
                    if name in seen:
                        raise ArtifactError(f"duplicate tensor name {name}")
                    seen.add(name)
                    tensor_slice = handle.get_slice(name)
                    shape = tuple(int(value) for value in tensor_slice.get_shape())
                    dtype = str(tensor_slice.get_dtype())
                    physical_elements = math.prod(shape)
                    adapter_role = (
                        adapter.classify_tensor(name, relative_file)
                        if adapter is not None
                        else None
                    )
                    unclassified_by_adapter = adapter is not None and adapter_role is None
                    if unclassified_by_adapter:
                        unclassified_adapter_tensors.append(name)
                    role = (
                        adapter_role
                        if adapter_role is not None
                        else TensorRole.OTHER
                        if adapter is not None
                        else classify_tensor(name, relative_file)
                    )
                    # Official DeepSeek V4 mixed-precision exports use singular
                    # ``.scale`` (and I8/F8 weight bodies) rather than MLX's
                    # ``.scales``/``.biases`` affine sidecar naming. HyperConnection
                    # / HyperHead learnable ``*.scale`` vectors are real params
                    # (mlx_lm deepseek_v4 sanitize maps hc_*_scale → *.scale).
                    is_hc_learnable_scale = (
                        name.endswith((".attn_hc.scale", ".ffn_hc.scale", ".hc_head.scale"))
                        or name
                        in {
                            "hc_head_scale",
                            "model.hc_head.scale",
                        }
                        or name.endswith((".hc_attn_scale", ".hc_ffn_scale"))
                    )
                    quantization_metadata = (
                        name.endswith((".scales", ".biases"))
                        or (name.endswith(".scale") and not is_hc_learnable_scale)
                    ) or (
                        quantized_source and name.endswith(".bias") and dtype not in _FLOAT_DTYPES
                    )
                    current_bits, current_group_size, current_method = _quantization_details(
                        name,
                        dtype,
                        relative_file,
                        config,
                        native_quantization,
                        mtp_bits,
                    )
                    # FP4 expert bodies pack two 4-bit values per I8/U8 element; MLX
                    # dequant doubles the trailing dim (deepseek_v4 sanitize/load).
                    fp4_expert = _deepseek_fp4_expert_weight(name, dtype, config)
                    logical_shape = list(shape)
                    if fp4_expert and shape:
                        logical_shape[-1] = int(shape[-1]) * 2
                    parameters = (
                        0
                        if quantization_metadata
                        else physical_elements * 2
                        if fp4_expert
                        else physical_elements * 32 // current_bits
                        if dtype == "U32" and current_bits is not None
                        else physical_elements
                    )
                    storage_bytes = physical_elements * _DTYPE_BYTES.get(dtype, 0)
                    current_precision = (
                        "bf16"
                        if current_bits == 16 and current_method == QuantMethod.BF16
                        else f"{current_bits}bit"
                        if current_bits is not None
                        else dtype.lower()
                    )
                    shape = logical_shape
                    # Packed MoE expert stacks are 3-D ([experts, out, in]);
                    # MLX-LM quantizes them as fused switch modules with the
                    # same per-group affine layout as 2-D linears, so they are
                    # quantizable. Every other non-2-D tensor stays preserved.
                    module_path = module_path_for(name)
                    # Re-quant of mixed FP4/FP8 sources (DeepSeek V4 Flash): after
                    # MLX sanitize/dequant, these become ordinary Linear modules.
                    # Inventory must still mark them quantizable so architecture-
                    # prior plans can assign 2/3/4/6-bit budgets.
                    requant_weight = (
                        allow_quantized
                        and quantized_source
                        and name.endswith(".weight")
                        and role
                        in {
                            TensorRole.EXPERT,
                            TensorRole.MLP,
                            TensorRole.ATTENTION,
                            TensorRole.EMBEDDING,
                            TensorRole.LM_HEAD,
                            TensorRole.ROUTER,
                        }
                    )
                    quantizable = (
                        (len(shape) == 2 or (len(shape) == 3 and role == TensorRole.EXPERT))
                        and (dtype in _FLOAT_DTYPES or requant_weight)
                        and role != TensorRole.NORM
                        and not quantization_metadata
                        and not unclassified_by_adapter
                        # Nemotron-H MoEGate is a custom Module with a raw weight
                        # matrix and no to_quantized(); MLX-LM never visits it.
                        # Mark non-quantizable so plans stay BF16 (fail-closed
                        # coverage would otherwise abort convert).
                        and not (
                            module_path.endswith(".mixer.gate") or ".mixer.gate." in module_path
                        )
                        # DeepSeek V4 compressor APE is an array attribute, not a
                        # Linear module, so MLX-LM convert never visits it.
                        and not name.endswith(".ape")
                        and ".ape." not in name
                        # MoE router gates and MultiLinear wo_a are not visited by
                        # nn.quantize (no to_quantized after load/dequant).
                        and not module_path.endswith(".ffn.gate")
                        and not module_path.endswith(".attn.wo_a")
                    )
                    tensors.append(
                        TensorSpec(
                            name=name,
                            module_path=module_path,
                            shape=shape,
                            dtype=dtype,
                            parameters=parameters,
                            physical_elements=physical_elements,
                            storage_bytes=storage_bytes,
                            role=role,
                            quantizable=quantizable,
                            file=relative_file,
                            current_precision=current_precision,
                            current_bits=current_bits,
                            current_group_size=current_group_size,
                            current_method=current_method,
                            quantization_metadata=quantization_metadata,
                            protected_recommendation=role
                            in {
                                TensorRole.EMBEDDING,
                                TensorRole.NORM,
                                TensorRole.LM_HEAD,
                                TensorRole.ROUTER,
                                TensorRole.MTP_PROJECTION,
                                TensorRole.MTP_BLOCK,
                                TensorRole.MTP_OUTPUT,
                                TensorRole.VISION,
                                TensorRole.AUDIO,
                            },
                            protection_reason=(
                                f"default {role.value} protection policy"
                                if role
                                in {
                                    TensorRole.EMBEDDING,
                                    TensorRole.NORM,
                                    TensorRole.LM_HEAD,
                                    TensorRole.ROUTER,
                                    TensorRole.MTP_PROJECTION,
                                    TensorRole.MTP_BLOCK,
                                    TensorRole.MTP_OUTPUT,
                                    TensorRole.VISION,
                                    TensorRole.AUDIO,
                                }
                                else None
                            ),
                        )
                    )
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"cannot inspect {tensor_file}: {exc}") from exc

    tensors.sort(key=lambda tensor: (tensor.file, tensor.name))
    total_parameters = sum(tensor.parameters for tensor in tensors)
    quantizable_parameters = sum(tensor.parameters for tensor in tensors if tensor.quantizable)
    # A supported MoE family must never silently preserve packed expert stacks
    # because an adapter classified their 3-D weights as ordinary MLPs.  Keep
    # inspection available, but revoke conversion scope until classification
    # coverage is restored.  This backstop prevents a nominal low-bit model
    # whose effective BPW is actually close to BF16.
    uncovered_fused_experts = [
        tensor.name
        for tensor in tensors
        if len(tensor.shape) == 3
        and tensor.dtype in _FLOAT_DTYPES
        and tensor.role != TensorRole.EXPERT
        and not tensor.role.is_mtp
        and any(token in tensor.name.lower() for token in _FUSED_EXPERT_PATH_TOKENS)
    ]
    vision_tensors_present = any(tensor.role == TensorRole.VISION for tensor in tensors)
    uncovered_declared_vision = architecture_profile.vision_present and not vision_tensors_present
    audio_tensors_present = any(tensor.role == TensorRole.AUDIO for tensor in tensors)
    uncovered_declared_audio = architecture_profile.audio_present and not audio_tensors_present
    unprepared_qwen3_asr = architecture_profile.adapter_id == "qwen3-asr-v1" and any(
        tensor.name.startswith("thinker.") for tensor in tensors
    )
    classification_coverage_notes: list[str] = []
    if unclassified_adapter_tensors:
        classification_coverage_notes.append(
            "Adapter tensor classification is incomplete; conversion is disabled."
        )
    if uncovered_fused_experts:
        classification_coverage_notes.append(
            "Fused expert tensors are not fully classified; conversion is disabled."
        )
    if uncovered_declared_vision:
        classification_coverage_notes.append(
            "The config declares a vision tower but no vision tensors were classified; "
            "conversion is disabled."
        )
    if uncovered_declared_audio:
        classification_coverage_notes.append(
            "The config declares an audio tower but no audio tensors were classified; "
            "conversion is disabled."
        )
    if unprepared_qwen3_asr:
        classification_coverage_notes.append(
            "The upstream Qwen3-ASR tensor layout must first be normalized to an MLX-Audio "
            "BF16 checkpoint with scripts/hf_to_mlx_bf16.py; conversion is disabled."
        )
    if classification_coverage_notes and (
        architecture_profile.support_level == ArchitectureSupportLevel.SUPPORTED
    ):
        architecture_profile = architecture_profile.model_copy(
            update={
                "support_level": ArchitectureSupportLevel.INVENTORY_ONLY,
                "support_tier": SupportTier.INSPECT_ONLY,
                "optimization_scope": OptimizationScope.INVENTORY_ONLY,
                "notes": [
                    *architecture_profile.notes,
                    *classification_coverage_notes,
                ],
            }
        )
    tied_weight_groups: list[list[str]] = []
    if bool(config.get("tie_word_embeddings")):
        embeddings = [tensor for tensor in tensors if tensor.role == TensorRole.EMBEDDING]
        heads = [tensor for tensor in tensors if tensor.role == TensorRole.LM_HEAD]
        if embeddings and heads:
            embedding = max(embeddings, key=lambda tensor: tensor.parameters)
            head = max(heads, key=lambda tensor: tensor.parameters)
            embedding.tied_to = head.name
            head.tied_to = embedding.name
            tied_weight_groups.append([embedding.name, head.name])
    warnings: list[str] = []
    if unclassified_adapter_tensors:
        preview = unclassified_adapter_tensors[:10]
        warnings.append(
            "adapter tensor classification is incomplete; tensors remain protected and "
            f"conversion is disabled: {preview}"
        )
    if uncovered_fused_experts:
        preview = uncovered_fused_experts[:10]
        warnings.append(
            "fused expert coverage is incomplete; tensors must classify as expert before "
            f"conversion: {preview}"
        )
    if unprepared_qwen3_asr:
        warnings.append(
            "upstream Qwen3-ASR thinker.* tensors include a duplicated tied LM head and "
            "runtime-specific layouts; prepare the pinned source with "
            "scripts/hf_to_mlx_bf16.py before planning or conversion"
        )
    if quantized_source:
        warnings.append(
            "logical parameter counts were reconstructed from packed quantization metadata"
        )
    mtp_tensors_present = any(tensor.role.is_mtp for tensor in tensors)
    if architecture_profile.mtp_declared and not mtp_tensors_present:
        warnings.append(
            "the model config declares MTP but no MTP tensors were found; "
            "an external sidecar is required"
        )
    if adapter is None:
        warnings.append("No supported architecture adapter matched; this report is inventory-only.")
    warnings.extend(architecture_profile.notes)
    if vision_tensors_present:
        architecture_profile = architecture_profile.model_copy(update={"vision_present": True})
    elif uncovered_declared_vision:
        # The config declares a vision tower (e.g. `vision_config`) but no
        # tensor classified as VISION. This is the fail-closed backstop for
        # dense-family vision-token coverage (AXQ-018): a future spec's
        # vision-tower naming that `_VISION_TOKENS`/`extra_role_patterns`
        # does not cover would otherwise fall through to a generic
        # attention/MLP role and quantize inline instead of being routed to
        # the protected vision sidecar, with no other signal that anything
        # went wrong.
        warnings.append(
            "config declares a vision tower but no tensor classified as vision; "
            "vision-tower naming may not be covered by the role classifier"
        )
    if audio_tensors_present:
        architecture_profile = architecture_profile.model_copy(update={"audio_present": True})
    elif uncovered_declared_audio:
        warnings.append(
            "config declares an audio tower but no tensor classified as audio; "
            "audio-tower naming may not be covered by the role classifier"
        )
    weight_bytes = sum(path.stat().st_size for path in tensor_files)
    mtp_weight_bytes = sum(
        path.stat().st_size
        for path in tensor_files
        if _MTP_TOKEN.search(path.relative_to(model_dir).as_posix().lower())
    )
    precision_parameters: dict[str, int] = {}
    for tensor in tensors:
        if tensor.parameters <= 0:
            continue
        precision_parameters[tensor.current_precision] = (
            precision_parameters.get(tensor.current_precision, 0) + tensor.parameters
        )
    return Inventory(
        model=ModelIdentity(
            model_id=model_reference,
            revision=revision,
            architecture=_architecture(config),
            local_path=str(model_dir),
        ),
        tensors=tensors,
        total_parameters=total_parameters,
        quantizable_parameters=quantizable_parameters,
        weight_bytes=weight_bytes,
        mtp_weight_bytes=mtp_weight_bytes,
        precision_parameters=precision_parameters,
        mtp_present=mtp_tensors_present or architecture_profile.mtp_declared,
        quantized_source=quantized_source,
        source_files=source_files,
        architecture_profile=architecture_profile,
        tied_weight_groups=tied_weight_groups,
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        warnings=warnings,
    )
