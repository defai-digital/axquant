"""Graft a Qwen3.5/3.6 MoE MTP head onto a Holo3-class trunk pack.

Holo3-35B-A3B declares MTP in config but ships **no** ``mtp.*`` weights.
This module extracts MTP from a shape-compatible donor (prefer the fine-tune
parent ``Qwen/Qwen3.5-35B-A3B``), restacks per-expert tensors into the packed
19-tensor layout used by certified Qwen3.6 35B AXQ MTP sidecars, and composes
the sidecar onto an existing non-MTP AXQ pack without mutating main weights.

Claims must disclose the graft: MTP is **not** co-trained on Holo3.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.schema import ArtifactFile, ModelIdentity, ProtectedTensorSidecarManifest
from axquant.schema._base import utc_now
from axquant.serde import file_sha256, stable_sha256, write_data

# Packed MTP tensor set shipped by AutomatosX Qwen3.6-35B-A3B AXQ-*-MTP packs.
QWEN35_MOE_PACKED_MTP_SHAPES: dict[str, tuple[int, ...]] = {
    "mtp.fc.weight": (2048, 4096),
    "mtp.layers.0.input_layernorm.weight": (2048,),
    "mtp.layers.0.mlp.experts.down_proj": (256, 2048, 512),
    "mtp.layers.0.mlp.experts.gate_up_proj": (256, 1024, 2048),
    "mtp.layers.0.mlp.gate.weight": (256, 2048),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (2048, 512),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (512, 2048),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (512, 2048),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2048),
    "mtp.layers.0.post_attention_layernorm.weight": (2048,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (512, 2048),
    "mtp.layers.0.self_attn.o_proj.weight": (2048, 4096),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (8192, 2048),
    "mtp.layers.0.self_attn.v_proj.weight": (512, 2048),
    "mtp.norm.weight": (2048,),
    "mtp.pre_fc_norm_embedding.weight": (2048,),
    "mtp.pre_fc_norm_hidden.weight": (2048,),
}

_MTP_UNPACKED_EXPERT = re.compile(
    r"^(?P<prefix>mtp\.layers\.\d+\.mlp)\.experts\.(?P<index>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

_GRAFT_KIND = "parent-qwen35-moe-mtp"
_MTP_SIDECAR = "mtp.safetensors"
_MTP_MANIFEST = "axquant_mtp_sidecar_manifest.json"
_GRAFT_RECORD = "axquant_mtp_graft.json"


@dataclass(frozen=True)
class GraftedMtpBundle:
    directory: Path
    sidecar: Path
    manifest: ProtectedTensorSidecarManifest
    graft_record: Path
    donor: ModelIdentity
    trunk: ModelIdentity


def prepare_grafted_qwen_moe_mtp(
    donor_dir: str | Path,
    *,
    output_dir: str | Path,
    trunk: ModelIdentity,
    donor: ModelIdentity,
) -> GraftedMtpBundle:
    """Build a packed MTP sidecar from a donor checkpoint for a Holo3-class trunk."""
    source = Path(donor_dir).expanduser().resolve()
    if not source.is_dir():
        raise ArtifactError(f"donor is not a directory: {source}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sidecar_path = destination / _MTP_SIDECAR
    if sidecar_path.exists() or (destination / _MTP_MANIFEST).exists():
        raise ArtifactError(f"graft output already exists under {destination}")

    weight_map = _donor_weight_map(source)
    mtp_keys = sorted(name for name in weight_map if name.startswith("mtp."))
    if not mtp_keys:
        raise ArtifactError(f"donor has no mtp.* tensors: {source}")

    mx = _mlx_core()
    loaded = _load_named_tensors(mx, source, weight_map, mtp_keys)
    packed = _to_packed_mtp_weights(mx, loaded)
    _validate_packed_shapes(packed)
    mx.save_safetensors(str(sidecar_path), packed)
    _stamp_sidecar_metadata(
        sidecar_path,
        trunk=trunk,
        donor=donor,
    )

    parameters = sum(int(_numel(array)) for array in packed.values())
    dtypes = tuple(sorted({str(array.dtype).split(".")[-1].upper() for array in packed.values()}))
    # Prefer BF16 label used on Hub sidecars.
    dtypes = tuple("BF16" if dtype in {"BFLOAT16", "BF16"} else dtype for dtype in dtypes)
    names = sorted(packed)
    source_files = sorted({Path(weight_map[name]).name for name in mtp_keys})
    manifest = ProtectedTensorSidecarManifest(
        source_model=trunk,
        role="mtp",
        tensor_count=len(names),
        parameters=parameters,
        dtypes=dtypes or ("BF16",),
        tensor_names_sha256=stable_sha256(names),
        source_files=[
            ArtifactFile(
                path=name,
                size_bytes=(source / name).stat().st_size if (source / name).is_file() else 0,
                sha256=file_sha256(source / name) if (source / name).is_file() else ("0" * 64),
            )
            for name in source_files
            if (source / name).is_file()
        ],
        output=ArtifactFile(
            path=_MTP_SIDECAR,
            size_bytes=sidecar_path.stat().st_size,
            sha256=file_sha256(sidecar_path),
        ),
        created_at=utc_now(),
    )
    write_data(destination / _MTP_MANIFEST, manifest)
    graft_path = destination / _GRAFT_RECORD
    write_data(
        graft_path,
        {
            "schema_version": "axquant.mtp-graft.v1",
            "graft_kind": _GRAFT_KIND,
            "trunk_model": {
                "model_id": trunk.model_id,
                "revision": trunk.revision,
            },
            "donor_model": {
                "model_id": donor.model_id,
                "revision": donor.revision,
            },
            "tensor_names": names,
            "tensor_shapes": {name: list(QWEN35_MOE_PACKED_MTP_SHAPES[name]) for name in names},
            "notes": [
                "MTP head grafted from donor; not co-trained on the trunk checkpoint.",
                "Acceleration claims require separate exactness/speedup evidence.",
            ],
            "created_at": utc_now().isoformat(),
        },
    )
    return GraftedMtpBundle(
        directory=destination,
        sidecar=sidecar_path,
        manifest=manifest,
        graft_record=graft_path,
        donor=donor,
        trunk=trunk,
    )


def compose_grafted_mtp_onto_pack(
    pack_dir: str | Path,
    mtp_bundle: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Copy a grafted MTP sidecar onto an AXQ pack directory.

    When ``output_dir`` is set, the pack is copied first so the source pack is
    left unchanged (preferred for preserving certified trunk digests).
    """
    source_pack = Path(pack_dir).expanduser().resolve()
    bundle = Path(mtp_bundle).expanduser().resolve()
    if not source_pack.is_dir():
        raise ArtifactError(f"pack is not a directory: {source_pack}")
    sidecar = bundle / _MTP_SIDECAR
    manifest = bundle / _MTP_MANIFEST
    if not sidecar.is_file() or not manifest.is_file():
        raise ArtifactError(f"mtp bundle incomplete under {bundle}")

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists():
            raise ArtifactError(f"compose output already exists: {destination}")
        shutil.copytree(source_pack, destination)
    else:
        destination = source_pack

    for name in (_MTP_SIDECAR, _MTP_MANIFEST, _GRAFT_RECORD):
        candidate = bundle / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)

    _patch_pack_mtp_flags(destination)
    return destination


def _patch_pack_mtp_flags(pack_dir: Path) -> None:
    manifest_path = pack_dir / "axquant_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["mtp_present"] = True
        sidecar = pack_dir / _MTP_SIDECAR
        if sidecar.is_file():
            payload["mtp_weight_file_size_bytes"] = sidecar.stat().st_size
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    runtime_path = pack_dir / "axquant_runtime.json"
    if runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        mtp = runtime.get("mtp") if isinstance(runtime.get("mtp"), dict) else {}
        mtp = dict(mtp)
        mtp["detected"] = True
        mtp["sidecar_file"] = _MTP_SIDECAR
        mtp["enabled_by_default"] = False
        mtp["optimized"] = False
        runtime["mtp"] = mtp
        runtime_path.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _donor_weight_map(source: Path) -> dict[str, str]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactError(f"invalid weight_map in {index_path}")
        return {str(k): str(v) for k, v in weight_map.items()}
    # Single-file or pre-extracted sidecar directory.
    single = source / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="np") as handle:
            return {name: single.name for name in handle.keys()}
    mtp_only = source / _MTP_SIDECAR
    if mtp_only.is_file():
        from safetensors import safe_open

        with safe_open(str(mtp_only), framework="np") as handle:
            return {name: mtp_only.name for name in handle.keys()}
    raise ArtifactError(f"donor has no index or model/mtp safetensors: {source}")


def _load_named_tensors(
    mx: Any,
    source: Path,
    weight_map: dict[str, str],
    names: list[str],
) -> dict[str, Any]:
    by_shard: dict[str, list[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if shard is None:
            raise ArtifactError(f"donor index missing tensor {name}")
        by_shard.setdefault(shard, []).append(name)
    loaded: dict[str, Any] = {}
    for shard_name, tensor_names in sorted(by_shard.items()):
        shard_path = source / shard_name
        if not shard_path.is_file():
            raise ArtifactError(f"missing donor shard: {shard_path}")
        weights = mx.load(str(shard_path))
        for name in tensor_names:
            if name not in weights:
                raise ArtifactError(f"tensor {name} missing from {shard_path}")
            loaded[name] = weights[name]
        del weights
    return loaded


def _to_packed_mtp_weights(mx: Any, loaded: dict[str, Any]) -> dict[str, Any]:
    # Already packed (Qwen3.6-style or extracted sidecar).
    if "mtp.layers.0.mlp.experts.gate_up_proj" in loaded:
        packed = {name: loaded[name] for name in QWEN35_MOE_PACKED_MTP_SHAPES if name in loaded}
        missing = sorted(set(QWEN35_MOE_PACKED_MTP_SHAPES) - set(packed))
        if missing:
            raise ArtifactError(f"packed donor MTP incomplete: missing {missing}")
        return packed

    experts: dict[str, dict[int, Any]] = {
        "gate_proj": {},
        "up_proj": {},
        "down_proj": {},
    }
    non_expert: dict[str, Any] = {}
    for name, array in loaded.items():
        match = _MTP_UNPACKED_EXPERT.match(name)
        if match is None:
            non_expert[name] = array
            continue
        experts[match.group("proj")][int(match.group("index"))] = array

    for proj in ("gate_proj", "up_proj", "down_proj"):
        if not experts[proj]:
            raise ArtifactError(f"donor MTP missing unpacked experts.{proj}")
    gate_idx = sorted(experts["gate_proj"])
    if gate_idx != sorted(experts["up_proj"]) or gate_idx != sorted(experts["down_proj"]):
        raise ArtifactError("donor MTP expert index sets are incomplete")
    if gate_idx != list(range(len(gate_idx))):
        raise ArtifactError("donor MTP expert indices must be contiguous from 0")
    expected_experts = int(QWEN35_MOE_PACKED_MTP_SHAPES["mtp.layers.0.mlp.experts.down_proj"][0])
    if len(gate_idx) != expected_experts:
        raise ArtifactError(f"expected {expected_experts} MTP experts, found {len(gate_idx)}")

    gate = mx.stack([experts["gate_proj"][i] for i in gate_idx], axis=0)
    up = mx.stack([experts["up_proj"][i] for i in gate_idx], axis=0)
    down = mx.stack([experts["down_proj"][i] for i in gate_idx], axis=0)
    if gate.shape != up.shape:
        raise ArtifactError(f"MTP gate/up shape mismatch: {gate.shape} vs {up.shape}")
    # Official pack: concat on intermediate axis (-2).
    gate_up = mx.concatenate([gate, up], axis=-2)

    packed: dict[str, Any] = {
        "mtp.layers.0.mlp.experts.gate_up_proj": gate_up,
        "mtp.layers.0.mlp.experts.down_proj": down,
    }
    for name in QWEN35_MOE_PACKED_MTP_SHAPES:
        if name in packed:
            continue
        if name not in non_expert:
            raise ArtifactError(f"donor MTP missing non-expert tensor {name}")
        packed[name] = non_expert[name]
    return packed


def _validate_packed_shapes(packed: dict[str, Any]) -> None:
    if set(packed) != set(QWEN35_MOE_PACKED_MTP_SHAPES):
        missing = sorted(set(QWEN35_MOE_PACKED_MTP_SHAPES) - set(packed))
        extra = sorted(set(packed) - set(QWEN35_MOE_PACKED_MTP_SHAPES))
        raise ArtifactError(f"packed MTP name mismatch: missing={missing} extra={extra}")
    for name, expected in QWEN35_MOE_PACKED_MTP_SHAPES.items():
        actual = tuple(int(x) for x in packed[name].shape)
        if actual != expected:
            raise ArtifactError(f"packed MTP shape mismatch for {name}: {actual} != {expected}")


def _stamp_sidecar_metadata(
    sidecar_path: Path,
    *,
    trunk: ModelIdentity,
    donor: ModelIdentity,
) -> None:
    """Best-effort metadata stamp; payload tensors already validated."""
    # mx.save_safetensors may not preserve custom metadata across versions.
    # Graft honesty lives primarily in axquant_mtp_graft.json.
    _ = (sidecar_path, trunk, donor)


def _numel(array: Any) -> int:
    shape = getattr(array, "shape", ())
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _mlx_core() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx is required to pack grafted MTP tensors") from exc
    return mx
