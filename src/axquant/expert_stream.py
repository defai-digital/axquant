from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from axquant.errors import ArtifactError, PlanningError
from axquant.inspector import inspect_model
from axquant.schema import (
    ExpertStreamManifest,
    ExpertStreamProjection,
    ExpertStreamTensor,
    Inventory,
    QuantizationPlan,
    TensorRole,
    TensorSpec,
)
from axquant.serde import read_data, write_data

ExpertStreamSetting = Literal["off", "auto", "required"]

EXPERT_STREAM_MANIFEST_NAME = "ax_expert_stream.json"
AUTO_REQUIRED_BYTES = 256 * 1024**3
STREAM_CAPABLE_ADAPTERS = frozenset({"qwen38-moe-v1", "deepseek-v4-v1"})
# Packs whose full-resident size exceeds every shipping Mac (512 GB). Flash
# 2/3-bit (~115 GB) can still resident-load on 192 GB hosts, so `off` stays legal.
STREAM_OFF_FORBIDDEN_ADAPTERS = frozenset({"qwen38-moe-v1"})

_LAYER = re.compile(r"(?:^|\.)layers\.(?P<layer>[0-9]+)(?:\.|$)")
_INDEXED_EXPERT = re.compile(r"\.experts\.[0-9]+\.")
_PACKED_EXPERT_TOKENS = ("switch_mlp", "switch_glu", ".experts.", "ffn.w1", "ffn.w2", "ffn.w3")
_STORAGE_SUFFIXES = (".weight", ".scales", ".biases", ".bias", ".scale")
_RESIDENT_ROLES = [
    "embedding",
    "attention",
    "router",
    "shared_expert",
    "norm",
    "lm_head",
    "mtp",
]


def validate_expert_stream_request(adapter_id: str, setting: ExpertStreamSetting) -> None:
    if setting not in {"off", "auto", "required"}:
        raise PlanningError(f"unsupported expert stream setting: {setting}")
    if adapter_id in STREAM_OFF_FORBIDDEN_ADAPTERS and setting == "off":
        raise PlanningError(
            "--expert-stream off is unsafe for this Super-class pack: a 512 GB Mac still "
            "cannot run it fully resident; use auto or required"
        )


def _projection(name: str) -> ExpertStreamProjection | None:
    value = name.lower()
    if "gate_up_proj" in value or ".switch_mlp.fc1" in value:
        return "gate_up"
    if "gate_proj" in value:
        return "gate"
    if "up_proj" in value:
        return "up"
    if "down_proj" in value or ".switch_mlp.fc2" in value:
        return "down"
    # DeepSeek V4 source / sanitize names: w1=gate, w3=up, w2=down; stacked
    # ``ffn.experts.{gate,up,down}`` is the unfused packed-stack alias.
    if ".w1." in value or value.endswith(".w1") or value.endswith(".w1.weight"):
        return "gate"
    if ".w3." in value or value.endswith(".w3") or value.endswith(".w3.weight"):
        return "up"
    if ".w2." in value or value.endswith(".w2") or value.endswith(".w2.weight"):
        return "down"
    if ".experts.gate." in value or value.endswith(".experts.gate.weight"):
        return "gate"
    if ".experts.up." in value or value.endswith(".experts.up.weight"):
        return "up"
    if ".experts.down." in value or value.endswith(".experts.down.weight"):
        return "down"
    return None


def _module_key(name: str) -> str:
    for suffix in _STORAGE_SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def _packed_expert_tensors(
    inventory: Inventory,
) -> list[tuple[TensorSpec, int, ExpertStreamProjection]]:
    candidates: list[tuple[TensorSpec, int, ExpertStreamProjection]] = []
    for tensor in inventory.tensors:
        value = tensor.name.lower()
        if tensor.role is not TensorRole.EXPERT:
            continue
        if _INDEXED_EXPERT.search(value):
            continue
        if not any(token in value for token in _PACKED_EXPERT_TOKENS):
            continue
        layer_match = _LAYER.search(value)
        projection = _projection(value)
        if layer_match is None or projection is None:
            continue
        if not tensor.shape:
            raise ArtifactError(f"packed expert tensor has no expert axis: {tensor.name}")
        candidates.append((tensor, int(layer_match.group("layer")), projection))
    return candidates


def build_expert_stream_manifest(
    inventory: Inventory,
    *,
    experts_per_tok: int,
    requirement: Literal["auto", "required"],
    default_group_size: int,
) -> ExpertStreamManifest:
    if requirement not in {"auto", "required"}:
        raise PlanningError("manifest construction requires expert stream auto or required")
    if experts_per_tok < 1:
        raise ArtifactError("expert stream manifest requires a positive experts-per-token value")
    if default_group_size < 1:
        raise ArtifactError("expert stream manifest requires a positive group size")

    candidates = _packed_expert_tensors(inventory)
    if not candidates:
        raise ArtifactError("converted checkpoint contains no packed layer-stack expert tensors")

    expert_counts = {int(tensor.shape[0]) for tensor, _, _ in candidates}
    if len(expert_counts) != 1 or next(iter(expert_counts)) < 1:
        raise ArtifactError(
            f"packed expert tensors disagree on expert-axis size: {sorted(expert_counts)}"
        )
    num_experts = next(iter(expert_counts))

    packing: dict[str, tuple[int, int]] = {}
    for tensor, _, _ in candidates:
        if not tensor.name.endswith(".weight"):
            continue
        bits = tensor.current_bits
        if bits is None:
            raise ArtifactError(f"packed expert weight has no precision metadata: {tensor.name}")
        packing[_module_key(tensor.name)] = (
            bits,
            tensor.current_group_size or default_group_size,
        )

    layers_with_separate_up = {layer for _, layer, projection in candidates if projection == "up"}
    records: list[ExpertStreamTensor] = []
    expert_bytes = 0
    layer_bytes: dict[int, int] = defaultdict(int)
    for tensor, layer, projection in candidates:
        if projection == "gate" and layer not in layers_with_separate_up:
            projection = "gate_up"
        module_key = _module_key(tensor.name)
        module_packing = packing.get(module_key)
        if module_packing is None:
            raise ArtifactError(
                f"packed expert tensor has no matching weight precision: {tensor.name}"
            )
        bits, group_size = module_packing
        records.append(
            ExpertStreamTensor(
                name=tensor.name,
                file=tensor.file,
                layer=layer,
                proj=projection,
                num_experts=num_experts,
                bits=bits,
                group_size=group_size,
            )
        )
        expert_bytes += tensor.storage_bytes
        layer_bytes[layer] += tensor.storage_bytes

    records.sort(key=lambda item: (item.layer, item.proj, item.name))
    full_resident_bytes = sum(tensor.storage_bytes for tensor in inventory.tensors)
    if full_resident_bytes < expert_bytes:
        raise ArtifactError("expert tensor bytes exceed full checkpoint tensor bytes")
    required = requirement == "required" or full_resident_bytes > AUTO_REQUIRED_BYTES
    return ExpertStreamManifest(
        required=required,
        num_experts=num_experts,
        experts_per_tok=experts_per_tok,
        estimated_resident_bytes=full_resident_bytes - expert_bytes,
        estimated_full_resident_bytes=full_resident_bytes,
        estimated_max_layer_expert_bytes=max(layer_bytes.values()),
        resident_roles=list(_RESIDENT_ROLES),
        streamed_roles=["expert"],
        tensors=records,
    )


def _positive_config_int(config: dict[str, Any], names: tuple[str, ...]) -> int | None:
    scopes = [config]
    for key in ("text_config", "language_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            scopes.insert(0, nested)
    for scope in scopes:
        for name in names:
            value = scope.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def emit_expert_stream_manifest(
    output_dir: str | Path,
    plan: QuantizationPlan,
    *,
    setting: ExpertStreamSetting = "auto",
) -> ExpertStreamManifest | None:
    adapter_id = plan.architecture_profile.adapter_id
    validate_expert_stream_request(adapter_id, setting)
    if setting == "off":
        return None
    if not any(allocation.role is TensorRole.EXPERT for allocation in plan.assignments):
        if setting == "required" or adapter_id in STREAM_CAPABLE_ADAPTERS:
            raise ArtifactError("expert streaming requires packed expert allocations in the plan")
        return None

    directory = Path(output_dir).expanduser().resolve()
    inventory = inspect_model(
        directory,
        model_id=plan.source_model.model_id,
        revision=plan.source_model.revision,
        allow_quantized=True,
    )
    candidates = _packed_expert_tensors(inventory)
    if not candidates:
        raise ArtifactError(
            "expert streaming was requested but conversion emitted no packed layer-stack "
            "expert tensors"
        )

    config_value = read_data(directory / "config.json")
    if not isinstance(config_value, dict):
        raise ArtifactError("converted config.json must contain an object")
    experts_per_tok = _positive_config_int(
        config_value,
        ("num_experts_per_tok", "num_experts_per_token", "experts_per_token", "top_k"),
    )
    if experts_per_tok is None:
        raise ArtifactError("converted config does not declare experts per token")
    manifest = build_expert_stream_manifest(
        inventory,
        experts_per_tok=experts_per_tok,
        requirement=setting,
        default_group_size=plan.group_size,
    )
    configured_experts = _positive_config_int(
        config_value,
        ("num_experts", "num_local_experts", "n_routed_experts"),
    )
    if configured_experts is not None and configured_experts != manifest.num_experts:
        raise ArtifactError(
            "converted config expert count does not match packed expert tensors: "
            f"{configured_experts} != {manifest.num_experts}"
        )
    write_data(directory / EXPERT_STREAM_MANIFEST_NAME, manifest)
    return manifest
