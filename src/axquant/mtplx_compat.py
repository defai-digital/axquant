"""MTPLX load contract for a converted MLX pack.

Youssofal's Qwen3.8-Flash-Next MTPLX pack is the reference for the n-gram
sidecar: ``ngram-table.safetensors`` holds ``ngram.weight``, ``ngram.scales``,
and ``ngram.biases``, with ``ngram_bits`` / ``ngram_group_size`` metadata, and
``config.json`` points at it through ``mlx_lm_extra_tensors.ngram_file``.
Qwen3.8-27B has no n-gram table. Its MTPLX pack points only at
``mtp.safetensors`` and keeps the quantization container's ``mode`` equal to
the tensors on disk.

AXQuant's MLX-LM convert path used to leave the container ``mode`` at the
library default ``affine`` while per-module entries said ``mxfp8``. Loaders
that trust the container then ask for affine biases that MXFP8 does not store.
This module rewrites that contract in place. It does not invent an n-gram
table and it does not relabel MXFP8 weights as affine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.serde import write_data

_PHYSICAL_MODES = frozenset({"affine", "mxfp4", "mxfp8"})
_NGRAM_TABLE = "ngram-table.safetensors"
_MTP_FILE = "mtp.safetensors"
_QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
_QWEN4_MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})

# Wiring published by Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality. Quant
# bits stay unset: a BF16 sidecar must not be marked prequantized.
_QWEN35_BF16_MTP_CONTRACT = {
    "base_hidden_variant": "post_norm",
    "concat_order": "embedding_hidden",
    "hidden_variant": "post_norm",
    "mtp_position_mode": "local",
}


def write_mtplx_load_contract(directory: str | Path) -> list[str]:
    """Align ``config.json`` with the MTPLX sidecar contract. Return change notes."""

    pack = Path(directory)
    config_path = pack / "config.json"
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {config_path.name} for MTPLX contract: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ArtifactError("config.json must be a JSON object")
    notes = apply_mtplx_load_contract(loaded, pack)
    if notes:
        write_data(config_path, loaded)
    return notes


def apply_mtplx_load_contract(config: dict[str, Any], directory: str | Path) -> list[str]:
    """Mutate ``config`` so MTPLX and MLX loaders agree with the pack directory."""

    pack = Path(directory)
    notes: list[str] = []
    if _align_quantization_containers(config):
        notes.append("quantization container mode matches per-module packing")
    extra_note = _align_extra_tensors(config, pack)
    if extra_note:
        notes.append(extra_note)
    contract_note = _align_qwen35_mtp_contract(config, pack)
    if contract_note:
        notes.append(contract_note)
    return notes


def _align_quantization_containers(config: dict[str, Any]) -> bool:
    changed = False
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if isinstance(block, dict) and _align_quant_block(block):
            changed = True
    return changed


def _align_quant_block(block: dict[str, Any]) -> bool:
    modes: list[str] = []
    bits: list[int] = []
    groups: list[int] = []
    for value in block.values():
        if not isinstance(value, dict):
            continue
        mode = value.get("mode")
        if isinstance(mode, str):
            modes.append(mode)
        raw_bits = value.get("bits")
        raw_group = value.get("group_size")
        if isinstance(raw_bits, int) and not isinstance(raw_bits, bool):
            bits.append(raw_bits)
        if isinstance(raw_group, int) and not isinstance(raw_group, bool):
            groups.append(raw_group)
    if not modes or len(set(modes)) != 1:
        return False
    mode = modes[0]
    if mode not in _PHYSICAL_MODES:
        return False
    changed = False
    if block.get("mode") != mode:
        block["mode"] = mode
        changed = True
    if len(set(bits)) == 1 and block.get("bits") != bits[0]:
        block["bits"] = bits[0]
        changed = True
    if len(set(groups)) == 1 and block.get("group_size") != groups[0]:
        block["group_size"] = groups[0]
        changed = True
    return changed


def _align_extra_tensors(config: dict[str, Any], pack: Path) -> str | None:
    extra = config.get("mlx_lm_extra_tensors")
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ArtifactError("mlx_lm_extra_tensors must be a JSON object")
    changed = False
    if (pack / _MTP_FILE).is_file() and extra.get("mtp_file") != _MTP_FILE:
        extra["mtp_file"] = _MTP_FILE
        changed = True
    ngram = pack / _NGRAM_TABLE
    declared = extra.get("ngram_file")
    if ngram.is_file():
        if not ngram_table_matches_mtplx(ngram):
            raise ArtifactError(
                f"{_NGRAM_TABLE} is not an MTPLX n-gram table: expected "
                "ngram.weight plus ngram.scales and ngram.biases when ngram_bits is not 0"
            )
        if declared != _NGRAM_TABLE:
            extra["ngram_file"] = _NGRAM_TABLE
            changed = True
        if _mark_qwen4_ngram_sidecar(config):
            changed = True
    elif declared:
        raise ArtifactError(
            f"config.json declares ngram_file {declared!r} but {_NGRAM_TABLE} is absent"
        )
    if not extra:
        return None
    if config.get("mlx_lm_extra_tensors") != extra:
        config["mlx_lm_extra_tensors"] = extra
        changed = True
    elif changed:
        config["mlx_lm_extra_tensors"] = extra
    return "mlx_lm_extra_tensors sidecar pointers" if changed else None


def _mark_qwen4_ngram_sidecar(config: dict[str, Any]) -> bool:
    if not _model_types(config) & _QWEN4_MODEL_TYPES:
        return False
    text = config.get("text_config")
    if not isinstance(text, dict):
        return False
    if text.get("ngram_sidecar") is True:
        return False
    text["ngram_sidecar"] = True
    return True


def _align_qwen35_mtp_contract(config: dict[str, Any], pack: Path) -> str | None:
    if not _model_types(config) & _QWEN35_MODEL_TYPES:
        return None
    sidecar = pack / _MTP_FILE
    if not sidecar.is_file():
        return None
    if "mtplx_mtp_contract" in config:
        return None
    if _sidecar_has_scales(sidecar):
        return None
    config["mtplx_mtp_contract"] = dict(_QWEN35_BF16_MTP_CONTRACT)
    return "qwen3_5 BF16 MTP contract"


def _model_types(config: dict[str, Any]) -> set[str]:
    types = {str(config.get("model_type") or "")}
    text = config.get("text_config")
    if isinstance(text, dict):
        types.add(str(text.get("model_type") or ""))
    types.discard("")
    return types


def _sidecar_has_scales(path: Path) -> bool:
    header = _safetensors_header(path)
    return any(name.endswith(".scales") for name in header if name != "__metadata__")


def ngram_table_matches_mtplx(path: str | Path) -> bool:
    """True when the table uses the MTPLX ``ngram.*`` names and metadata."""

    header = _safetensors_header(Path(path))
    meta = header.get("__metadata__", {})
    if not isinstance(meta, dict):
        return False
    try:
        bits = int(str(meta.get("ngram_bits", "4")))
        group_size = int(str(meta.get("ngram_group_size", "32")))
    except (TypeError, ValueError):
        return False
    if group_size <= 0 or bits < 0:
        return False
    required = ("ngram.weight",) if bits == 0 else ("ngram.weight", "ngram.scales", "ngram.biases")
    return all(name in header for name in required)


def _safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ArtifactError(f"{path.name} is not a safetensors file")
            size = int.from_bytes(raw_size, "little")
            if size <= 0 or size > 64 * 1024 * 1024:
                raise ArtifactError(f"{path.name} has an invalid safetensors header length")
            loaded = json.loads(handle.read(size))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ArtifactError(f"{path.name} safetensors header must be a JSON object")
    return loaded
