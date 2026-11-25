#!/usr/bin/env python3
"""Apply the current runtime layout to a published pack, starting from the Hub.

The Hub copy is the authority for everything except tensor bytes: it may carry
card, config or evidence edits that the factory copy lacks. The workflow is

``prepare``
    Build ``<mirror-root>/<pack>`` from the Hub's non-weight files plus hard
    links to the factory ``*.safetensors``, after checking that every factory
    weight matches the Hub LFS SHA-256. Then apply
    ``scripts/repair_pack_layout.py`` to the mirror.
``publication_smoke.py --directory <mirror>``
    Run the publication smoke on exactly what the Hub will hold.
``push``
    Upload only the non-weight files that differ from the Hub, in one commit.
    The receipt must pass, weights are never uploaded, nothing is deleted or
    created on the Hub, and only allow-listed files may be added or replaced.
``sync-local``
    Make the factory pack's non-weight files equal to the mirror; replaced and
    removed factory files are backed up first.

Without ``--yes`` ``push`` and ``sync-local`` only report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from axquant.model_support import require_model_supported  # noqa: E402
from axquant.publication_smoke import (  # noqa: E402
    SMOKE_RECEIPT_FILENAME,
    require_publication_smoke,
)
from axquant.runtime_compatibility import COMPATIBILITY_FILENAME  # noqa: E402
from axquant.serde import file_sha256, write_data  # noqa: E402

OWNER = "AutomatosX"
REPLACEABLE = frozenset(
    {
        "config.json",
        "mtplx_runtime.json",
        "model.safetensors.index.json",
        "axquant_manifest.json",
        COMPATIBILITY_FILENAME,
        "model-manifest.json",
        SMOKE_RECEIPT_FILENAME,
    }
)
ADDABLE = frozenset({COMPATIBILITY_FILENAME, SMOKE_RECEIPT_FILENAME})
PRIVATE_MARKERS = ("/Volumes/", "/Users/", "/home/", "/private/", "df-macstudio", "macbookpro")
COMMIT_TITLE = "Refresh runtime layout metadata"
COMMIT_BODY = (
    "Metadata-only update; tensor bytes are unchanged. The protected vision "
    "sidecar is listed in model.safetensors.index.json so MLX-VLM loads the "
    "vision tower, config.json declares mtp_norm_encoding: delta for raw-delta "
    "MTP heads where it was missing (oMLX sidecar import, MTPLX), "
    "mtplx_runtime.json declares the qwen3-next-mtp arch_id oMLX requires "
    "where it was missing, and the "
    "AXQuant manifest and runtime compatibility record are refreshed to match. "
    "axquant_publication_smoke.json records load-and-generate checks on AX "
    "Engine, MLX-LM, MLX-VLM and oMLX where applicable, run on this repository "
    "content; for packs with a decision head, oMLX served the chat backbone "
    "without that head, as the receipt states. It is not a quality, speed or "
    "MTP acceptance claim."
)


def _blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _same_as_hub(sibling: Any, path: Path) -> bool:
    """Compare content: LFS files by SHA-256, plain files by git blob id."""
    if sibling.lfs is not None:
        return bool(sibling.lfs.sha256 == file_sha256(path))
    return bool(sibling.blob_id == _blob_sha1(path))


def _remote(api: Any, name: str) -> dict[str, Any]:
    info = api.model_info(f"{OWNER}/{name}", files_metadata=True)
    return {sibling.rfilename: sibling for sibling in info.siblings or []}


def _tree(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    }


def prepare(api: Any, pack: Path, mirror_root: Path, ax_engine_bench: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from repair_pack_layout import repair

    require_model_supported(pack)
    mirror = mirror_root / pack.name
    if mirror.exists():
        raise RuntimeError(f"mirror already exists: {mirror}")
    remote = _remote(api, pack.name)
    weights = sorted(name for name in remote if name.endswith(".safetensors"))
    local_weights = sorted(path.name for path in pack.glob("*.safetensors"))
    if weights != local_weights:
        raise RuntimeError(f"weight sets differ: hub={weights} local={local_weights}")
    for name in weights:
        lfs = remote[name].lfs
        if lfs is None or lfs.sha256 != file_sha256(pack / name):
            raise RuntimeError(f"factory weight {name} differs from the Hub; refusing")
    base_commit = api.model_info(f"{OWNER}/{pack.name}").sha
    staging_root = Path(tempfile.mkdtemp(prefix=f".{pack.name}.", dir=mirror_root))
    staging = staging_root / pack.name
    try:
        staging.mkdir()
        for name in weights:
            os.link(pack / name, staging / name)
        with tempfile.TemporaryDirectory(dir=mirror_root) as downloads:
            for name in sorted(set(remote) - set(weights)):
                fetched = hf_hub_download(
                    f"{OWNER}/{pack.name}", name, local_dir=downloads, revision=base_commit
                )
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fetched, target)
        # Repair and validate in staging; only a successful mirror is promoted.
        result = repair(staging, mirror_root / "_repair-records", ax_engine_bench)
        if result.get("refusals"):
            raise RuntimeError(f"repair refused: {result['refusals']}")
        write_data(staging_root / "base.json", {"repo": pack.name, "base_commit": base_commit})
        os.rename(staging, mirror)
        os.rename(staging_root / "base.json", mirror_root / f"{pack.name}.base.json")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {"mirror": str(mirror), "base_commit": base_commit, "repair": result}


def _require_weights_match(remote: dict[str, Any], directory: Path) -> None:
    """Every local weight must be the Hub LFS object, compared by SHA-256."""
    local = sorted(path.name for path in directory.glob("*.safetensors"))
    hub = sorted(name for name in remote if name.endswith(".safetensors"))
    if local != hub:
        raise RuntimeError(f"weight sets differ: hub={hub} local={local}")
    for name in local:
        lfs = remote[name].lfs
        if lfs is None or lfs.sha256 != file_sha256(directory / name):
            raise RuntimeError(f"weight {name} is not the Hub weight; refusing")


def _require_regenerable_ax_manifest(mirror: Path, ax_engine_bench: str) -> None:
    """model-manifest.json is outside the smoke digest; prove it is derived."""
    path = mirror / "model-manifest.json"
    before = file_sha256(path)
    completed = subprocess.run(
        [ax_engine_bench, "generate-manifest", str(mirror), "--force", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode or file_sha256(path) != before:
        raise RuntimeError(
            "model-manifest.json is not what AX Engine regenerates from the smoked payload"
        )


def plan_push(api: Any, mirror: Path, ax_engine_bench: str) -> dict[str, Any]:
    require_model_supported(mirror)
    if require_publication_smoke(mirror) is None:
        # Every pack this tool repairs is a generative MLX pack; a config edit
        # that makes the gate inapplicable must not bypass it.
        raise RuntimeError("publication smoke gate is not applicable to the mirror; refusing")
    remote = _remote(api, mirror.name)
    local = _tree(mirror)
    missing = sorted(set(remote) - local)
    if missing:
        raise RuntimeError(f"mirror lacks Hub files (would not delete them): {missing}")
    _require_weights_match(remote, mirror)
    uploads: list[str] = []
    for name in sorted(local):
        path = mirror / name
        if name.endswith(".safetensors"):
            continue
        if name in remote and _same_as_hub(remote[name], path):
            continue
        if name == "model-manifest.json" and name not in remote:
            continue  # AX Engine regenerates it on load; do not add a cache file
        if name not in REPLACEABLE or (name not in remote and name not in ADDABLE):
            raise RuntimeError(f"{name} differs from the Hub but is not an allowed update")
        text = path.read_text(encoding="utf-8")
        markers = [marker for marker in PRIVATE_MARKERS if marker in text]
        if markers:
            raise RuntimeError(f"{name} contains private location markers {markers}")
        json.loads(text)
        if name == "model-manifest.json":
            _require_regenerable_ax_manifest(mirror, ax_engine_bench)
        uploads.append(name)
    base = mirror.parent / f"{mirror.name}.base.json"
    if not base.is_file():
        raise RuntimeError(f"missing {base.name}; re-run prepare")
    parent = json.loads(base.read_text(encoding="utf-8"))["base_commit"]
    return {"repo_id": f"{OWNER}/{mirror.name}", "uploads": uploads, "parent_commit": parent}


def push(api: Any, mirror: Path, execute: bool, ax_engine_bench: str) -> dict[str, Any]:
    from huggingface_hub import CommitOperationAdd

    plan = plan_push(api, mirror, ax_engine_bench)
    if execute and plan["uploads"]:
        commit = api.create_commit(
            plan["repo_id"],
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(mirror / name))
                for name in plan["uploads"]
            ],
            commit_message=COMMIT_TITLE,
            commit_description=COMMIT_BODY,
            # The Hub rejects the commit if the branch moved since prepare.
            parent_commit=plan["parent_commit"],
        )
        plan["commit"] = commit.oid
        after = _remote(api, mirror.name)
        stale = [n for n in plan["uploads"] if not _same_as_hub(after[n], mirror / n)]
        if stale:
            raise RuntimeError(f"Hub content does not match the mirror after commit: {stale}")
    return plan


def sync_local(
    api: Any, pack: Path, mirror: Path, backup_root: Path, execute: bool
) -> dict[str, Any]:
    require_model_supported(pack)
    # Only a mirror whose exact payload passed the smoke, over the same weights
    # as the factory pack and the Hub, may overwrite factory metadata.
    require_publication_smoke(mirror)
    remote = _remote(api, pack.name)
    _require_weights_match(remote, pack)
    _require_weights_match(remote, mirror)
    mirror_files = {n for n in _tree(mirror) if not n.endswith(".safetensors")}
    pack_files = {n for n in _tree(pack) if not n.endswith(".safetensors")}
    replace = sorted(
        n
        for n in mirror_files
        if not (pack / n).is_file() or file_sha256(pack / n) != file_sha256(mirror / n)
    )
    remove = sorted(pack_files - mirror_files)
    plan = {"pack": pack.name, "replace_or_add": replace, "move_to_backup": remove}
    if execute and (replace or remove):
        backup = backup_root / pack.name / "pre-hub-sync"
        backup.mkdir(parents=True, exist_ok=False)
        for name in sorted(set(replace) | set(remove)):
            if (pack / name).is_file():
                (backup / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pack / name, backup / name)
        for name in replace:
            (pack / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mirror / name, pack / name)
        for name in remove:
            (pack / name).unlink()
        write_data(backup / "sync.json", plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "push", "sync-local"))
    parser.add_argument("names", nargs="+", help="Pack directory names")
    parser.add_argument("--models-root", type=Path, default=Path("/Volumes/Ext16TR0/models"))
    parser.add_argument(
        "--mirror-root", type=Path, default=Path("/Volumes/Ext16TR0/models-hub-mirror")
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("/Volumes/Ext16TR0/models-repair-backup/layout-2026-11-20"),
    )
    parser.add_argument("--ax-engine-bench", default="ax-engine-bench")
    parser.add_argument("--yes", action="store_true", help="Write to the Hub or factory pack")
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    if api.whoami()["name"] != OWNER:
        raise SystemExit(f"Hub token is not the {OWNER} account")
    args.mirror_root.mkdir(parents=True, exist_ok=True)
    status = 0
    for name in args.names:
        pack = args.models_root / name
        mirror = args.mirror_root / name
        try:
            if args.command == "prepare":
                result = prepare(api, pack, args.mirror_root, args.ax_engine_bench)
            elif args.command == "push":
                result = push(api, mirror, args.yes, args.ax_engine_bench)
            else:
                result = sync_local(api, pack, mirror, args.backup_root, args.yes)
        except Exception as exc:  # report every pack
            result = {"error": f"{type(exc).__name__}: {exc}"}
            status = 1
        print(json.dumps({"pack": name, **result}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
