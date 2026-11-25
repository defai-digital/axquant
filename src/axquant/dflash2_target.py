"""DFlash2 target identity and declaration (AXQ-067 DF-1 / DF-1b).

AX Engine's primary greedy-text route drafts with an upstream DFlash2
checkpoint that it selects by the pack's recorded ``source_model.model_id``
and that reuses the target's token embedding and LM head. A pack built from
an admitted source under a local path or the wrong identity silently loses
that route, so conversion fails closed for checkpoints shaped like an admitted
target until the operator either supplies the Hub identity or declines.

Every converted pack whose config matches an admitted signature carries
``axquant_dflash2_target.json``. It records the identity decision, the
mirrored drafter pin, the format lane and the digests of the drafter-shared
tensors. It is development metadata, never runtime, speed or quality evidence.
No MLX import.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from axquant.errors import ArtifactError, PlanningError, ValidationGateError
from axquant.schema.dflash2 import (
    Dflash2AdmittedTarget,
    Dflash2AdmittedTargets,
    Dflash2DrafterPin,
    Dflash2MirrorReference,
    Dflash2SharedTensor,
    Dflash2SharedTensorPart,
    Dflash2TargetDeclaration,
)
from axquant.source_binding import path_shaped_model_id

MIRROR_FILE = "dflash2_admitted_targets.json"
DECLARATION_NAME = "axquant_dflash2_target.json"

_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
# Tensors the drafter never reads and that do not set the trunk format lane.
_SIDE_TOKENS = ("mtp", "visual", "vision", "audio")
_HASH_CHUNK = 16 * 1024 * 1024

Status = Literal["target", "non_target", "declared_non_target"]


@cache
def admitted_targets() -> Dflash2AdmittedTargets:
    """Load the packaged mirror of AX Engine's admitted DFlash2 targets."""
    resource = files("axquant.data").joinpath(MIRROR_FILE)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
        return Dflash2AdmittedTargets.model_validate(payload)
    except (OSError, ValueError, ValidationError) as exc:
        raise ArtifactError(f"invalid DFlash2 admitted-target mirror: {exc}") from exc


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def _model_types(config: Mapping[str, Any]) -> set[str]:
    types = {config.get("model_type"), _text_config(config).get("model_type")}
    return {value for value in types if isinstance(value, str)}


def _num_experts(config: Mapping[str, Any]) -> int | None:
    for scope in (_text_config(config), config):
        for key in ("num_experts", "n_routed_experts", "num_local_experts"):
            value = scope.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def matching_target(config: Mapping[str, Any]) -> Dflash2AdmittedTarget | None:
    """Return the admitted target whose architecture signature ``config`` matches."""
    text = _text_config(config)
    types = _model_types(config)
    for target in admitted_targets().targets:
        signature = target.signature
        if (
            types & set(signature.model_types)
            and text.get("hidden_size") == signature.hidden_size
            and text.get("num_hidden_layers") == signature.num_hidden_layers
            and text.get("vocab_size") == signature.vocab_size
            and _num_experts(config) == signature.num_experts
        ):
            return target
    return None


def is_hub_identity(
    model_id: str,
    local_path: str | None = None,
    source_argument: str | None = None,
) -> bool:
    """True when ``model_id`` names a Hub repository rather than a location.

    Without ``--model-id`` the inventory records the source argument as typed,
    so a relative directory such as ``models/Qwen3.8-27B`` is Hub-shaped text.
    It is a location when it is the local directory being converted or when
    it resolves to the recorded source directory. A Hub download passes its ID
    as the argument; that is not a directory and stays a Hub identity.
    """
    if path_shaped_model_id(model_id) or not _HUB_ID.fullmatch(model_id):
        return False
    if model_id == local_path:
        return False
    try:
        # The converter resolves its source argument the same way, at this moment.
        if model_id == source_argument and Path(source_argument).expanduser().is_dir():
            return False
        if local_path is not None and (
            Path(model_id).resolve() == Path(local_path).expanduser().resolve()
        ):
            return False
    except OSError:
        # Cannot tell a location from an ID; fail closed toward "location".
        return False
    return True


# Absolute, home, file-URI and Windows drive paths must not enter a reason.
_PATH_IN_TEXT = re.compile(r"(^|[\s'\"(=])(/[^\s]|~[/\\]|file:|[A-Za-z]:[\\/])")


@dataclass(frozen=True)
class Dflash2Classification:
    target: Dflash2AdmittedTarget
    status: Status
    source_model_id: str | None
    source_revision: str | None
    reason: str | None


def classify_source(
    config: Mapping[str, Any],
    *,
    model_id: str,
    revision: str | None,
    local_path: str | None = None,
    source_argument: str | None = None,
    not_target_reason: str | None = None,
) -> Dflash2Classification | None:
    """Decide the DFlash2 status of a conversion source; fail closed when unknown.

    Returns ``None`` when the checkpoint does not match an admitted signature.
    """
    target = matching_target(config)
    if target is None:
        return None
    admitted = target.source_model_id
    # The manifest records model_id verbatim and AX Engine matches on it, so the
    # admitted ID text is the admitted identity whatever the directory layout.
    is_admitted_id = model_id.lower() == admitted.lower()
    hub = is_admitted_id or is_hub_identity(model_id, local_path, source_argument)
    if is_admitted_id and model_id != admitted:
        raise PlanningError(
            f"use the exact admitted model ID {admitted}; AX Engine applies its format "
            f"policy by exact ID, so {model_id} would bypass it"
        )
    normalized_revision = revision.lower() if isinstance(revision, str) else None
    if normalized_revision is not None and not _REVISION.fullmatch(normalized_revision):
        normalized_revision = None
    if not_target_reason is not None:
        reason = not_target_reason.strip()
        if not reason:
            raise PlanningError("--not-dflash2-target requires a non-empty reason")
        if _PATH_IN_TEXT.search(reason) or (local_path and local_path in reason):
            raise PlanningError(
                "the --not-dflash2-target reason is published with the pack; remove "
                "filesystem paths from it"
            )
        if is_admitted_id:
            raise PlanningError(
                f"{model_id} is the admitted DFlash2 source; AX Engine will select its "
                "drafter, so the pack cannot be declared a non-target"
            )
        return Dflash2Classification(
            target=target,
            status="declared_non_target",
            source_model_id=model_id if hub else None,
            source_revision=normalized_revision if hub else None,
            reason=reason,
        )
    if not hub:
        raise PlanningError(
            "this checkpoint matches the architecture of the AX Engine DFlash2 target "
            f"{admitted}, but its source identity is not a Hub repository ID "
            "(it names the local source). "
            f"Pass --model-id {admitted} with its immutable --revision when it is that "
            "checkpoint, --model-id <org/name> for a fine-tune or sibling, or "
            "--not-dflash2-target REASON to convert without a DFlash2 identity (AXQ-067)"
        )
    if is_admitted_id:
        if normalized_revision is None:
            raise PlanningError(
                f"a DFlash2 target built from {admitted} requires an immutable 40-character "
                "source --revision (AXQ-067)"
            )
        return Dflash2Classification(
            target=target,
            status="target",
            source_model_id=model_id,
            source_revision=normalized_revision,
            reason=None,
        )
    return Dflash2Classification(
        target=target,
        status="non_target",
        source_model_id=model_id,
        source_revision=normalized_revision,
        reason=None,
    )


def format_lane(mode: str, bits: int, group_size: int | None) -> str:
    """Name a trunk format lane the way AX Engine's format scope does."""
    mode = mode.lower()
    # MLX packs microscaling formats in fixed 32-value blocks.
    if mode in {"mxfp4", "mxfp8"}:
        return mode
    return f"{mode}{bits}_g{group_size if group_size is not None else 0}"


def _is_side_module(module: str) -> bool:
    segments = module.lower().split(".")
    return any(segment.startswith(_SIDE_TOKENS) for segment in segments)


def _is_shared_module(module: str) -> bool:
    last = module.rsplit(".", 1)[-1]
    return last in {"embed_tokens", "lm_head"}


def trunk_lane_weights(
    allocations: Iterable[Any],
    *,
    q_mode: str,
    group_size: int | None,
) -> dict[str, int]:
    """Parameters per trunk format lane of a plan; BF16 trunk is ``unquantized``."""
    from axquant.predicate import allocation_physical_mode

    weights: dict[str, int] = {}
    for allocation in allocations:
        module = str(allocation.module_path)
        if _is_side_module(module) or _is_shared_module(module):
            continue
        if allocation.bits >= 16:
            lane = "unquantized"
        else:
            mode = allocation_physical_mode(allocation, q_mode)
            lane = format_lane(mode, allocation.bits, allocation.group_size or group_size)
        weights[lane] = weights.get(lane, 0) + int(allocation.parameters)
    return weights


def predicted_format_lane(
    allocations: Iterable[Any],
    *,
    q_mode: str,
    group_size: int | None,
) -> str | None:
    """Parameter-weighted dominant trunk lane of a plan, before conversion."""
    weights = trunk_lane_weights(allocations, q_mode=q_mode, group_size=group_size)
    if not weights:
        return None
    return max(sorted(weights), key=lambda lane: weights[lane])


def require_admitted_lane(classification: Dflash2Classification, lane: str | None) -> None:
    """A DFlash2 target must use a format lane AX Engine loads for that target."""
    allowed = classification.target.admitted_format_lanes
    if classification.status != "target" or allowed is None or lane is None:
        return
    if lane not in allowed:
        raise PlanningError(
            f"format lane {lane} is not admitted by AX Engine for the DFlash2 target "
            f"{classification.target.source_model_id}; admitted lanes: {', '.join(allowed)}"
        )


@dataclass(frozen=True)
class _StoredTensor:
    name: str
    file: Path
    dtype: str
    shape: list[int]
    start: int
    end: int


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ArtifactError(f"truncated Safetensors file: {path.name}")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ArtifactError(f"invalid Safetensors header length in {path.name}")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ArtifactError(f"invalid Safetensors header in {path.name}")
    header["__data_offset__"] = 8 + length
    return header


def _stored_tensors(pack_dir: Path) -> dict[str, _StoredTensor]:
    index = pack_dir / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not all(
            isinstance(name, str) for name in weight_map.values()
        ):
            raise ArtifactError("model.safetensors.index.json has no valid weight_map")
        shard_names = sorted(set(weight_map.values()))
    elif (pack_dir / "model.safetensors").is_file():
        weight_map = None
        shard_names = ["model.safetensors"]
    else:
        raise ArtifactError("converted pack has no indexed Safetensors weights")
    stored: dict[str, _StoredTensor] = {}
    for shard_name in shard_names:
        # Shards are plain files beside the index; never follow a path out of the pack.
        if any(mark in shard_name for mark in ("/", "\\", "\x00")) or shard_name in {".", ".."}:
            raise ArtifactError(f"Safetensors shard name is not a plain file name: {shard_name}")
        shard = pack_dir / shard_name
        header = _read_header(shard)
        base = int(header.pop("__data_offset__"))
        header.pop("__metadata__", None)
        size = shard.stat().st_size
        for name, spec in header.items():
            if weight_map is not None and weight_map.get(name) != shard_name:
                continue
            try:
                begin, end = (int(value) for value in spec["data_offsets"])
                shape = [int(value) for value in spec["shape"]]
                dtype = str(spec["dtype"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactError(f"invalid Safetensors entry {name} in {shard_name}") from exc
            if not 0 <= begin <= end or base + end > size or any(dim < 0 for dim in shape):
                raise ArtifactError(f"invalid Safetensors data range for {name} in {shard_name}")
            stored[name] = _StoredTensor(
                name=name,
                file=shard,
                dtype=dtype,
                shape=shape,
                start=base + begin,
                end=base + end,
            )
    return stored


def _tensor_sha256(tensor: _StoredTensor) -> str:
    digest = hashlib.sha256()
    with tensor.file.open("rb") as handle:
        handle.seek(tensor.start)
        remaining = tensor.end - tensor.start
        while remaining > 0:
            chunk = handle.read(min(_HASH_CHUNK, remaining))
            if not chunk:
                raise ArtifactError(f"truncated tensor data for {tensor.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _quantization(pack_config: Mapping[str, Any]) -> Mapping[str, Any]:
    quantization = pack_config.get("quantization")
    if not isinstance(quantization, Mapping):
        quantization = pack_config.get("quantization_config")
    return quantization if isinstance(quantization, Mapping) else {}


def _module_quantization(
    quantization: Mapping[str, Any], module: str
) -> tuple[str, int, int | None] | None:
    entry = quantization.get(module)
    override = entry if isinstance(entry, Mapping) else {}

    # MLX-LM passes a per-module dict as overrides of the top-level defaults.
    def field(key: str) -> Any:
        return override[key] if key in override else quantization.get(key)

    bits = field("bits")
    if not isinstance(bits, int) or bits <= 0:
        return None
    group = field("group_size")
    return (
        str(field("mode") or "affine"),
        bits,
        group if isinstance(group, int) else None,
    )


def pack_format_lane(pack_dir: Path) -> str:
    """Parameter-weighted dominant trunk lane of a converted MLX pack."""
    pack_config = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    quantization = _quantization(pack_config)
    stored = _stored_tensors(pack_dir)
    weights: dict[str, int] = {}
    for name, tensor in stored.items():
        if not name.endswith(".weight"):
            continue
        module = name[: -len(".weight")]
        if _is_side_module(module) or _is_shared_module(module):
            continue
        if f"{module}.scales" not in stored:
            # Unquantized matrices count too, so a small quantized island cannot
            # name the lane of a mostly unquantized trunk. Norms and biases are 1-D.
            if len(tensor.shape) >= 2:
                count = 1
                for dim in tensor.shape:
                    count *= dim
                weights["unquantized"] = weights.get("unquantized", 0) + count
            continue
        params = _module_quantization(quantization, module)
        weight = tensor
        if params is None:
            continue
        lane = format_lane(*params)
        # Packed weights hold ``bits`` per parameter; weight by parameters, as the plan does.
        weights[lane] = weights.get(lane, 0) + (weight.end - weight.start) * 8 // params[1]
    if not weights:
        return "unquantized"
    return max(sorted(weights), key=lambda lane: weights[lane])


def _shared_module(stored: Mapping[str, _StoredTensor], suffix: str) -> str | None:
    candidates = sorted(
        name[: -len(".weight")]
        for name in stored
        if (name == f"{suffix}.weight" or name.endswith(f".{suffix}.weight"))
        and not _is_side_module(name)
    )
    if len(candidates) > 1:
        raise ArtifactError(f"ambiguous drafter-shared module {suffix}: {candidates}")
    return candidates[0] if candidates else None


def _tied(pack_config: Mapping[str, Any]) -> bool:
    for scope in (_text_config(pack_config), pack_config):
        value = scope.get("tie_word_embeddings")
        if isinstance(value, bool):
            return value
    return False


def shared_tensors(pack_dir: Path, *, required: bool = True) -> list[Dflash2SharedTensor]:
    """Record the token embedding and LM head the drafter reuses, with digests.

    ``required`` is False for non-targets: a missing module is then skipped,
    since no drafter will read it.
    """
    pack_config = json.loads((pack_dir / "config.json").read_text(encoding="utf-8"))
    quantization = _quantization(pack_config)
    stored = _stored_tensors(pack_dir)
    records: list[Dflash2SharedTensor] = []
    for role, suffix in (("token_embedding", "embed_tokens"), ("lm_head", "lm_head")):
        module = _shared_module(stored, suffix)
        if module is None:
            if role == "lm_head" and _tied(pack_config) and records:
                records.append(
                    Dflash2SharedTensor(
                        role="lm_head", module=records[0].module, tied_to_embedding=True
                    )
                )
                continue
            if not required:
                continue
            raise ArtifactError(f"converted pack has no drafter-shared {suffix} tensor")
        parts = [
            Dflash2SharedTensorPart(
                name=tensor.name,
                file=tensor.file.name,
                dtype=tensor.dtype,
                shape=tensor.shape,
                sha256=_tensor_sha256(tensor),
            )
            for part in ("weight", "scales", "biases")
            if (tensor := stored.get(f"{module}.{part}")) is not None
        ]
        params = (
            _module_quantization(quantization, module) if f"{module}.scales" in stored else None
        )
        records.append(
            Dflash2SharedTensor(
                role=role,
                module=module,
                mode=params[0] if params else None,
                bits=params[1] if params else None,
                group_size=params[2] if params else None,
                parts=parts,
            )
        )
    return records


def build_declaration(
    pack_dir: Path,
    classification: Dflash2Classification,
    *,
    target_num_hidden_layers: int,
) -> Dflash2TargetDeclaration:
    """Build the declaration from a converted (staged) pack directory."""
    target = classification.target
    mirror = admitted_targets()
    lane = pack_format_lane(pack_dir)
    require_admitted_lane(classification, lane)
    drafter = (
        Dflash2DrafterPin(
            repo=target.drafter_repo,
            revision=target.drafter_revision,
            config_sha256=target.drafter_config_sha256,
            weights_sha256=target.drafter_weights_sha256,
        )
        if classification.status == "target"
        else None
    )
    try:
        return Dflash2TargetDeclaration(
            status=classification.status,
            source_model_id=classification.source_model_id,
            source_revision=classification.source_revision,
            admitted_source_model_id=target.source_model_id,
            mirror=Dflash2MirrorReference(
                observed_at=mirror.observed_at, ax_engine_commit=mirror.ax_engine_commit
            ),
            drafter=drafter,
            tap_layer_ids=list(target.tap_layer_ids),
            target_num_hidden_layers=target_num_hidden_layers,
            format_lane=lane,
            shared_tensors=shared_tensors(pack_dir, required=classification.status == "target"),
            reason=classification.reason,
        )
    except ValidationError as exc:
        raise ArtifactError(f"invalid DFlash2 target declaration: {exc}") from exc


def _non_mlx_pack(pack_config: Mapping[str, Any]) -> bool:
    """True for packs AX Engine does not load (for example compressed-tensors CUDA)."""
    if isinstance(pack_config.get("quantization"), Mapping):
        return False
    config = pack_config.get("quantization_config")
    return isinstance(config, Mapping) and "quant_method" in config


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def require_publication_declaration(directory: str | Path) -> Dflash2TargetDeclaration | None:
    """Publication gate: an MLX pack shaped like a DFlash2 target must declare itself.

    The declaration is checked against the manifest, the current mirror and
    the pack bytes, so a declaration cannot vouch for a pack that changed after
    conversion. Returns ``None`` for packs outside the admitted signatures and
    for non-MLX packs, which AX Engine does not load.
    """
    root = Path(directory).expanduser()
    try:
        return _check_publication_declaration(root)
    except ValidationGateError:
        raise
    except (ArtifactError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValidationGateError(f"DFlash2 target declaration check failed: {exc}") from exc


def _check_publication_declaration(root: Path) -> Dflash2TargetDeclaration | None:
    config_path = root / "config.json"
    if not config_path.is_file():
        return None
    pack_config = _read_json_object(config_path)
    if _non_mlx_pack(pack_config):
        return None
    target = matching_target(pack_config)
    if target is None:
        return None
    declaration_path = root / DECLARATION_NAME
    if not declaration_path.is_file():
        raise ValidationGateError(
            f"pack matches the AX Engine DFlash2 target signature of {target.source_model_id} "
            f"but has no {DECLARATION_NAME}; reconvert with this toolkit (AXQ-067)"
        )
    raw = declaration_path.read_bytes()
    try:
        declaration = Dflash2TargetDeclaration.model_validate_json(raw)
    except ValidationError as exc:
        raise ValidationGateError(f"invalid {DECLARATION_NAME}: {exc}") from exc
    manifest = _read_json_object(root / "axquant_manifest.json")
    files = manifest.get("files")
    bound = {
        entry.get("path"): entry.get("sha256")
        for entry in (files if isinstance(files, list) else [])
        if isinstance(entry, dict)
    }
    if bound.get(DECLARATION_NAME) != hashlib.sha256(raw).hexdigest():
        raise ValidationGateError(f"{DECLARATION_NAME} is not bound by axquant_manifest.json")

    # Identity: the manifest is what AX Engine reads.
    source = manifest.get("source_model")
    if not isinstance(source, dict):
        raise ValidationGateError("axquant_manifest.json has no source_model")
    model_id = str(source.get("model_id") or "")
    revision = source.get("revision")
    revision = revision.lower() if isinstance(revision, str) else None
    if declaration.admitted_source_model_id != target.source_model_id:
        raise ValidationGateError(
            f"{DECLARATION_NAME} names {declaration.admitted_source_model_id}, but the "
            f"current mirror admits {target.source_model_id}"
        )
    if model_id.lower() == target.source_model_id.lower() and model_id != target.source_model_id:
        raise ValidationGateError(
            f"manifest source {model_id!r} is a case variant of {target.source_model_id}"
        )
    if (model_id == target.source_model_id) != (declaration.status == "target"):
        raise ValidationGateError(
            f"manifest source {model_id!r} disagrees with DFlash2 status {declaration.status}"
        )
    # A declared non-target without a Hub identity records none; the status check
    # above already keeps the admitted ID out of it.
    if declaration.source_model_id is not None and declaration.source_model_id != model_id:
        raise ValidationGateError(f"{DECLARATION_NAME} source differs from the manifest source")
    if declaration.source_revision is not None and declaration.source_revision != revision:
        raise ValidationGateError(f"{DECLARATION_NAME} revision differs from the manifest")

    # Mirror-derived fields must match the current mirror.
    layers = _text_config(pack_config).get("num_hidden_layers")
    if (
        declaration.tap_layer_ids != list(target.tap_layer_ids)
        or declaration.target_num_hidden_layers != target.signature.num_hidden_layers
        or declaration.target_num_hidden_layers != layers
    ):
        raise ValidationGateError(
            f"{DECLARATION_NAME} tap layers or layer count differ from the current mirror"
        )
    if declaration.status == "target":
        pin = declaration.drafter
        if pin is None or (
            pin.repo,
            pin.revision,
            pin.config_sha256,
            pin.weights_sha256,
        ) != (
            target.drafter_repo,
            target.drafter_revision,
            target.drafter_config_sha256,
            target.drafter_weights_sha256,
        ):
            raise ValidationGateError(
                f"{DECLARATION_NAME} drafter pin differs from the current AX Engine mirror; "
                "re-declare against the current pin"
            )

    # Pack bytes: recompute rather than trust the recorded values.
    lane = pack_format_lane(root)
    if lane != declaration.format_lane:
        raise ValidationGateError(
            f"pack format lane {lane} differs from {DECLARATION_NAME} ({declaration.format_lane})"
        )
    allowed = target.admitted_format_lanes
    if declaration.status == "target" and allowed is not None and lane not in allowed:
        raise ValidationGateError(f"format lane {lane} is not admitted for this DFlash2 target")
    actual = shared_tensors(root, required=declaration.status == "target")
    if actual != declaration.shared_tensors:
        raise ValidationGateError(
            f"drafter-shared tensors (token embedding, LM head) differ from {DECLARATION_NAME}"
        )
    return declaration
