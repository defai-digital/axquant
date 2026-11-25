#!/usr/bin/env python3
"""Bring an existing MLX pack to the current runtime layout, metadata only.

Repairs packs converted before the layout rules landed:

* Qwen 3.5-family packs: index the protected ``vision.safetensors`` sidecar so
  MLX-VLM loads the vision tower (byte-preserved ``model.visual.*`` /
  ``vision_tower.*`` names only; other layouts are refused).
* Raw-delta MTP heads: declare ``mtp_norm_encoding: delta`` in ``config.json``
  (via the current MTPLX load contract) for oMLX's sidecar import and MTPLX.
* Qwen MTP sidecars: declare the ``qwen3-next-mtp`` execution contract
  (``arch_id`` and ``mtp_tensor_count``) in ``mtplx_runtime.json``, which oMLX
  requires; older packs shipped the file without it.
* Verify that no MTP tensor is indexed.

Tensor bytes never change. The original small files are copied to a backup
directory first; ``axquant_manifest.json`` file records, the runtime
compatibility record and the AX Engine ``model-manifest.json`` are refreshed
so the pack stays self-consistent. Older runtime receipts become stale and a
new publication smoke receipt is required before publishing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.converter import (  # noqa: E402
    INDEXED_VISION_MODEL_TYPES,
    _index_vision_sidecar,
    _require_mtp_outside_index,
)
from axquant.model_support import require_model_supported  # noqa: E402
from axquant.mtp_sidecar import (  # noqa: E402
    QWEN_NEXT_MTP_ARCH_ID,
    QWEN_NEXT_MTP_LEGACY_ARCH_IDS,
)
from axquant.mtplx_compat import apply_mtplx_load_contract  # noqa: E402
from axquant.runtime_compatibility import (  # noqa: E402
    COMPATIBILITY_FILENAME,
    write_runtime_compatibility,
)
from axquant.schema.loading import load_artifact_manifest  # noqa: E402
from axquant.serde import file_sha256, write_data  # noqa: E402

INDEX = "model.safetensors.index.json"
RUNTIME = "mtplx_runtime.json"
_BACKED_UP = (
    "config.json",
    RUNTIME,
    INDEX,
    "axquant_manifest.json",
    COMPATIBILITY_FILENAME,
    "model-manifest.json",
)
_VISION_PREFIXES = ("model.visual.", "visual.", "vision_tower.")
_QWEN_MTP_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"})


def _runtime_contract_fix(pack: Path, model_type: object) -> dict[str, Any] | None:
    """Return the corrected MTPLX runtime contract, or None when it is current."""
    runtime = pack / RUNTIME
    sidecar = pack / "mtp.safetensors"
    if model_type not in _QWEN_MTP_MODEL_TYPES or not runtime.is_file():
        return None
    if not sidecar.is_file():
        return None
    contract = _read(runtime)
    names = [name for name in _header(sidecar) if name != "__metadata__"]
    # oMLX strips a VLM ``language_model.`` / ``model.`` prefix on import.
    bare = [name.removeprefix("language_model.").removeprefix("model.") for name in names]
    if not names or not all(name.startswith("mtp.") for name in bare):
        raise ValueError("Qwen MTP sidecar must contain only mtp.* tensors")
    arch = contract.get("arch_id")
    if arch not in (None, "", QWEN_NEXT_MTP_ARCH_ID, *QWEN_NEXT_MTP_LEGACY_ARCH_IDS):
        raise ValueError(f"mtplx_runtime.json declares a foreign arch_id {arch!r}")
    count = contract.get("mtp_tensor_count")
    if count is not None and count != len(names):
        raise ValueError(f"mtp_tensor_count {count!r} != sidecar tensors {len(names)}")
    if arch == QWEN_NEXT_MTP_ARCH_ID and count == len(names):
        return None
    return {**contract, "arch_id": QWEN_NEXT_MTP_ARCH_ID, "mtp_tensor_count": len(names)}


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        value = json.loads(handle.read(size))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} header is not an object")
    return value


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def plan_repair(pack: Path) -> dict[str, Any]:
    config = _read(pack / "config.json")
    model_type = config.get("model_type")
    actions: list[str] = []
    refusals: list[str] = []
    index_path = pack / INDEX
    vision = pack / "vision.safetensors"
    if vision.is_file() and index_path.is_file():
        weight_map = _read(index_path).get("weight_map", {})
        if "vision.safetensors" not in set(weight_map.values()):
            names = [name for name in _header(vision) if name != "__metadata__"]
            if model_type not in INDEXED_VISION_MODEL_TYPES:
                refusals.append(f"vision sidecar for model_type {model_type!r} is not indexable")
            elif not all(name.startswith(_VISION_PREFIXES) for name in names):
                refusals.append("vision sidecar uses an unrecognized tensor-name layout")
            else:
                actions.append("index vision sidecar")
    try:
        if _runtime_contract_fix(pack, model_type) is not None:
            actions.append("mtplx_runtime.json: declare qwen3-next-mtp arch_id")
    except ValueError as exc:
        refusals.append(f"mtplx_runtime.json: {exc}")
    preview = json.loads(json.dumps(config))
    notes = apply_mtplx_load_contract(preview, pack)
    actions.extend(f"config.json: {note}" for note in notes)
    return {"actions": actions, "refusals": refusals}


def _refresh_derived(pack: Path, ax_engine_bench: str) -> None:
    """Regenerate derived records, then bind them in the AXQuant manifest.

    The compatibility record is (re)written from current headers, including on
    older packs that never had one; the AX Engine manifest is regenerated before
    the file records are refreshed, since some manifests list it.
    """
    write_runtime_compatibility(pack)
    if (pack / "model-manifest.json").is_file():
        completed = subprocess.run(
            [ax_engine_bench, "generate-manifest", str(pack), "--force", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"AX Engine manifest regeneration failed: {completed.stderr.strip()[-500:]}"
            )
    path = pack / "axquant_manifest.json"
    manifest = _read(path)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("axquant_manifest.json has no files list")
    listed = {record.get("path") for record in records}
    if COMPATIBILITY_FILENAME not in listed:
        records.append({"path": COMPATIBILITY_FILENAME})
    for record in records:
        if record.get("path") in {
            "config.json",
            RUNTIME,
            INDEX,
            COMPATIBILITY_FILENAME,
            "model-manifest.json",
        }:
            member = pack / record["path"]
            record["size_bytes"] = member.stat().st_size
            record["sha256"] = file_sha256(member)
    records.sort(key=lambda record: record["path"])
    write_data(path, manifest)
    load_artifact_manifest(path)  # the refreshed manifest must still validate


def repair(pack: Path, backup_root: Path, ax_engine_bench: str) -> dict[str, Any]:
    require_model_supported(pack)
    plan = plan_repair(pack)
    if plan["refusals"] or not plan["actions"]:
        return {**plan, "applied": False}
    backup = backup_root / pack.name
    backup.mkdir(parents=True, exist_ok=False)
    before = {}
    for name in _BACKED_UP:
        if (pack / name).is_file():
            shutil.copy2(pack / name, backup / name)
            before[name] = file_sha256(pack / name)
    if "index vision sidecar" in plan["actions"]:
        header = _header(pack / "vision.safetensors")
        names = tuple(sorted(name for name in header if name != "__metadata__"))
        data_size = sum(
            header[name]["data_offsets"][1] - header[name]["data_offsets"][0] for name in names
        )
        parameters = 0
        for name in names:
            count = 1
            for dimension in header[name]["shape"]:
                count *= dimension
            parameters += count
        _index_vision_sidecar(
            pack,
            tensor_names=names,
            data_size=data_size,
            parameters=parameters,
            family="Qwen 3.5",
        )
    config = _read(pack / "config.json")
    runtime_fix = _runtime_contract_fix(pack, config.get("model_type"))
    if runtime_fix is not None:
        write_data(pack / RUNTIME, runtime_fix)
    if apply_mtplx_load_contract(config, pack):
        write_data(pack / "config.json", config)
    _require_mtp_outside_index(pack)
    _refresh_derived(pack, ax_engine_bench)
    tracked = set(before) | {COMPATIBILITY_FILENAME}
    after = {name: file_sha256(pack / name) for name in sorted(tracked) if (pack / name).is_file()}
    record = {
        "schema_version": "axquant.pack-layout-repair.v1",
        "pack": pack.name,
        "repaired_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "actions": plan["actions"],
        "before_sha256": before,
        "after_sha256": after,
        "tensor_bytes_changed": False,
    }
    write_data(backup / "repair.json", record)
    return {
        **plan,
        "applied": True,
        "changed": sorted(n for n in after if after[n] != before.get(n)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--ax-engine-bench", default="ax-engine-bench")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: report)")
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Only regenerate derived records and manifest bindings of already repaired packs",
    )
    args = parser.parse_args()
    status = 0
    for pack in args.packs:
        pack = pack.resolve()
        try:
            if args.refresh_only:
                require_model_supported(pack)
                _refresh_derived(pack, args.ax_engine_bench)
                result = {"refreshed": True}
            elif args.apply:
                result = repair(pack, args.backup_root.resolve(), args.ax_engine_bench)
            else:
                result = plan_repair(pack)
        except Exception as exc:  # report every pack; never stop half-way silently
            result = {"error": f"{type(exc).__name__}: {exc}"}
            status = 1
        print(json.dumps({"pack": pack.name, **result}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
