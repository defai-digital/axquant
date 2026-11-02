"""AX Engine weight-layout declaration for converted packs (AXQ-066).

AX Engine admits a ``qwen4_exp`` (Qwen 3.8 Flash Next) pack from a newer
exporter only when ``axquant_manifest.json`` declares the on-disk layout under
``axquant.artifact.v3`` / ``axquant.runtime.v2``. The declaration is emitted
only after the staged pack is checked: trunk norms must be declared as raw
HF deltas and every trunk conv1d kernel must be in MLX ``(channels, kernel, 1)``
order. Any other pack, or a check that does not hold, gets no declaration and
keeps the frozen v2 / v1 envelopes, so AX Engine stays fail-closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.ngram_layout import (
    INDEX_FILENAME,
    RUNTIME_CONTRACT_FILENAME,
    _parse_safetensors_file,
    _read_json_object,
)
from axquant.schema import (
    AxEngineOptimizationMetadataV2,
    AxEngineWeightLayout,
    RuntimeMetadata,
    RuntimeMetadataV2,
)

DECLARED_LAYOUT_MODEL_TYPES = frozenset({"qwen4_exp"})
_MTP_SIDECAR_FILENAME = "mtp.safetensors"


def _trunk_conv1d_shapes(pack: Path) -> dict[str, tuple[int, ...]]:
    index = _read_json_object(pack / INDEX_FILENAME, INDEX_FILENAME)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ArtifactError(f"{INDEX_FILENAME} must contain a weight_map object")
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ArtifactError(f"{INDEX_FILENAME} weight_map entries must be strings")
        if shard == _MTP_SIDECAR_FILENAME or name.startswith("mtp."):
            continue
        if name.endswith("conv1d.weight"):
            names_by_shard.setdefault(shard, []).append(name)
    shapes: dict[str, tuple[int, ...]] = {}
    for shard, names in sorted(names_by_shard.items()):
        parsed = _parse_safetensors_file(pack / shard)
        for name in names:
            tensor = parsed.tensors.get(name)
            if tensor is None:
                raise ArtifactError(f"{shard} does not contain indexed tensor {name}")
            shapes[name] = tensor.shape
    return shapes


def _is_mlx_depthwise_conv1d(shape: tuple[int, ...]) -> bool:
    return len(shape) == 3 and shape[0] > 0 and shape[1] > 1 and shape[2] == 1


def verified_weight_layout(pack: str | Path, model_type: str | None) -> AxEngineWeightLayout | None:
    """Return the layout declaration the staged pack supports, or ``None``.

    ``None`` means "do not declare": the family is out of scope, or the pack
    does not record raw-delta trunk norms. A recorded raw-delta contract whose
    conv1d kernels are not in MLX order is a conversion defect and raises.
    """

    if model_type not in DECLARED_LAYOUT_MODEL_TYPES:
        return None
    directory = Path(pack)
    contract_path = directory / RUNTIME_CONTRACT_FILENAME
    if not contract_path.is_file():
        return None
    contract: dict[str, Any] = _read_json_object(contract_path, RUNTIME_CONTRACT_FILENAME)
    if contract.get("trunk_norm_layout") != "raw_hf_delta":
        return None
    shapes = _trunk_conv1d_shapes(directory)
    if not shapes:
        raise ArtifactError("qwen4_exp pack declares raw-delta norms but has no trunk conv1d")
    misordered = sorted(
        name for name, shape in shapes.items() if not _is_mlx_depthwise_conv1d(shape)
    )
    if misordered:
        raise ArtifactError(
            "qwen4_exp conv1d kernels are not in MLX (channels, kernel, 1) order: "
            + ", ".join(misordered[:5])
        )
    return AxEngineWeightLayout(norm_weights="raw_delta", conv1d_axes="mlx")


def declare_weight_layout(
    runtime: RuntimeMetadata,
    layout: AxEngineWeightLayout,
) -> RuntimeMetadataV2:
    """Promote runtime v1 metadata to v2 with a verified layout declaration."""

    if isinstance(runtime, RuntimeMetadataV2):
        raise ArtifactError("runtime metadata already carries a weight-layout declaration")
    ax_engine = AxEngineOptimizationMetadataV2(
        **runtime.ax_engine.model_dump(),
        weight_sanitize="hf_norm_only",
        weight_layout=layout,
    )
    payload = runtime.model_dump(exclude={"schema_version", "ax_engine"})
    return RuntimeMetadataV2(**payload, ax_engine=ax_engine)
