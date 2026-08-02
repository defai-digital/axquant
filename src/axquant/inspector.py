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
from axquant.schema import (
    ArchitectureProfile,
    Inventory,
    ModelIdentity,
    QuantMethod,
    TensorRole,
    TensorSpec,
)

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
_MTP_PRECISION = re.compile(r"INT(4|6|8|16)", re.IGNORECASE)


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
    value = f"{source_file}/{name}".lower()
    if _MTP_TOKEN.search(value):
        if any(token in value for token in ("output_head", "lm_head", "vocab_head")):
            return TensorRole.MTP_OUTPUT
        if any(token in value for token in ("proj", "projection")):
            return TensorRole.MTP_PROJECTION
        return TensorRole.MTP_BLOCK
    if any(
        token in value
        for token in (
            "vision",
            "visual",
            "image",
            "multimodal",
            "multi_modal",  # Mistral3 multi_modal_projector (underscore form)
            "patch_merger",
        )
    ):
        return TensorRole.VISION
    if "norm" in value:
        return TensorRole.NORM
    if any(token in value for token in ("lm_head", "output.weight", "output_layer")):
        return TensorRole.LM_HEAD
    if any(token in value for token in ("embed_tokens", "token_embedding", "wte.weight")):
        return TensorRole.EMBEDDING
    if "router" in value or ("expert" in value and ".gate." in value):
        return TensorRole.ROUTER
    if "expert" in value or "switch_mlp" in value or "switch_glu" in value:
        return TensorRole.EXPERT
    if any(
        token in value
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
        token in value
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


def _tensor_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return sorted(model_dir.glob("*.safetensors"))
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError(f"{index_path} must contain a non-empty weight_map")
    paths: set[Path] = set()
    for value in weight_map.values():
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
        paths.add(shard)
    mtp_sidecar = model_dir / "mtp.safetensors"
    if mtp_sidecar.is_file():
        paths.add(mtp_sidecar)
    vision_sidecar = model_dir / "vision.safetensors"
    if vision_sidecar.is_file():
        paths.add(vision_sidecar)
    return sorted(paths)


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
    except (OSError, json.JSONDecodeError):
        return None
    description = runtime.get("mtp_sidecar") if isinstance(runtime, dict) else None
    if not isinstance(description, str):
        return None
    match = _MTP_PRECISION.search(description)
    return int(match.group(1)) if match else None


def _quantization_details(
    name: str,
    dtype: str,
    source_file: str,
    config: dict[str, Any],
    native_quantization: dict[str, dict[str, Any]],
    mtp_bits: int | None,
) -> tuple[int | None, int | None, QuantMethod | None]:
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
    parsed_bits = int(bits) if isinstance(bits, int) else None
    parsed_group_size = int(group_size) if isinstance(group_size, int) else None
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
    tensor_files = _tensor_files(model_dir)
    if not tensor_files:
        raise ArtifactError(f"{model_dir} does not contain Safetensors weights")

    tensors: list[TensorSpec] = []
    seen: set[str] = set()
    source_files: list[str] = []
    for tensor_file in tensor_files:
        relative_file = tensor_file.relative_to(model_dir).as_posix()
        source_files.append(relative_file)
        try:
            with safe_open(tensor_file, framework="numpy") as handle:
                for name in sorted(handle.keys()):
                    if name in seen:
                        raise ArtifactError(f"duplicate tensor name {name}")
                    seen.add(name)
                    tensor_slice = handle.get_slice(name)
                    shape = tuple(int(value) for value in tensor_slice.get_shape())
                    dtype = str(tensor_slice.get_dtype())
                    physical_elements = math.prod(shape)
                    role = (
                        adapter.classify_tensor(name, relative_file)
                        if adapter is not None
                        else None
                    ) or classify_tensor(name, relative_file)
                    quantization_metadata = name.endswith((".scales", ".biases"))
                    current_bits, current_group_size, current_method = _quantization_details(
                        name,
                        dtype,
                        relative_file,
                        config,
                        native_quantization,
                        mtp_bits,
                    )
                    parameters = (
                        0
                        if quantization_metadata
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
                    # Packed MoE expert stacks are 3-D ([experts, out, in]);
                    # MLX-LM quantizes them as fused switch modules with the
                    # same per-group affine layout as 2-D linears, so they are
                    # quantizable. Every other non-2-D tensor stays preserved.
                    module_path = module_path_for(name)
                    quantizable = (
                        (len(shape) == 2 or (len(shape) == 3 and role == TensorRole.EXPERT))
                        and dtype in _FLOAT_DTYPES
                        and role != TensorRole.NORM
                        and not quantization_metadata
                        # Nemotron-H MoEGate is a custom Module with a raw weight
                        # matrix and no to_quantized(); MLX-LM never visits it.
                        # Mark non-quantizable so plans stay BF16 (fail-closed
                        # coverage would otherwise abort convert).
                        and not (
                            module_path.endswith(".mixer.gate")
                            or ".mixer.gate." in module_path
                        )
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
    if quantized_source:
        warnings.append(
            "logical parameter counts were reconstructed from packed quantization metadata"
        )
    mtp_tensors_present = any(tensor.role.is_mtp for tensor in tensors)
    if architecture_profile.mtp_declared and not mtp_tensors_present:
        warnings.append(
            "Qwen config declares MTP but no MTP tensors were found; "
            "an external sidecar is required"
        )
    if adapter is None:
        warnings.append("No supported Qwen 3.6 adapter matched; this report is inventory-only.")
    warnings.extend(architecture_profile.notes)
    if any(tensor.role == TensorRole.VISION for tensor in tensors):
        architecture_profile = architecture_profile.model_copy(update={"vision_present": True})
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
