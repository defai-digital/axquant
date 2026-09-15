#!/usr/bin/env python3
"""Annotate AutomatosX Hub MTP packs with axquant_omlx_compat.json.

Downloads only sidecar + runtime JSON (not language shards), writes the
companion file, and uploads it. Does not requantize. Gemma assistant-MTP
and DeepSeek nextn sidecars without mtp.* names are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# Local tree (unreleased annotate_qwen_mtp_omlx_compat).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from axquant.errors import ArtifactError
from axquant.mtp_sidecar import OMLX_COMPAT_FILENAME, annotate_qwen_mtp_omlx_compat

SIDECAR_NAMES = ("mtp.safetensors", "mtp_head.safetensors")
RUNTIME_NAME = "mtplx_runtime.json"
COMMIT_MESSAGE = (
    "Add axquant_omlx_compat.json for OMLX sidecar import "
    "(does not change language weights)"
)


def _list_mtp_repos(api: HfApi, author: str) -> list[str]:
    return sorted(
        model.id
        for model in api.list_models(author=author)
        if "MTP" in model.id.upper()
    )


def _annotate_repo(api: HfApi, repo_id: str, work: Path) -> dict[str, str]:
    files = set(list_repo_files(repo_id=repo_id, repo_type="model"))
    sidecar_name = next((name for name in SIDECAR_NAMES if name in files), None)
    if sidecar_name is None:
        return {"repo": repo_id, "status": "skip", "reason": "no mtp.safetensors sidecar"}
    if RUNTIME_NAME not in files:
        return {
            "repo": repo_id,
            "status": "skip",
            "reason": "mtplx_runtime.json missing; not a Qwen oMLX/MTPLX sidecar pack",
        }
    dest = work / repo_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    runtime_path = hf_hub_download(
        repo_id=repo_id,
        filename=RUNTIME_NAME,
        local_dir=dest,
        repo_type="model",
    )
    runtime_before = Path(runtime_path).read_bytes()
    try:
        contract = json.loads(runtime_before.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"repo": repo_id, "status": "skip", "reason": "mtplx_runtime.json is not JSON"}
    arch = contract.get("arch_id") if isinstance(contract, dict) else None
    from axquant.mtp_sidecar import QWEN_NEXT_MTP_ARCH_ID, QWEN_NEXT_MTP_LEGACY_ARCH_IDS

    if not isinstance(arch, str) or (
        arch != QWEN_NEXT_MTP_ARCH_ID and arch not in QWEN_NEXT_MTP_LEGACY_ARCH_IDS
    ):
        return {
            "repo": repo_id,
            "status": "skip",
            "reason": f"mtplx_runtime.json arch_id {arch!r} is not a Qwen sidecar",
        }
    hf_hub_download(
        repo_id=repo_id,
        filename=sidecar_name,
        local_dir=dest,
        repo_type="model",
    )
    try:
        written = annotate_qwen_mtp_omlx_compat(dest)
    except ArtifactError as exc:
        return {"repo": repo_id, "status": "skip", "reason": str(exc)}
    api.upload_file(
        path_or_fileobj=str(written),
        path_in_repo=OMLX_COMPAT_FILENAME,
        repo_id=repo_id,
        repo_type="model",
        commit_message=COMMIT_MESSAGE,
    )
    runtime_local = dest / RUNTIME_NAME
    if (
        runtime_local.is_file()
        and RUNTIME_NAME in files
        and runtime_before is not None
        and runtime_local.read_bytes() != runtime_before
    ):
        api.upload_file(
            path_or_fileobj=str(runtime_local),
            path_in_repo=RUNTIME_NAME,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Normalize mtplx_runtime.json arch_id to qwen3-next-mtp",
        )
    payload = json.loads(written.read_text(encoding="utf-8"))
    return {
        "repo": repo_id,
        "status": "ok",
        "sidecar": sidecar_name,
        "tensors": str(payload.get("lightning_mtp_tensor_count")),
        "arch_id": str(payload.get("arch_id")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="AutomatosX")
    parser.add_argument("--repo", action="append", dest="repos")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    repos = args.repos or _list_mtp_repos(api, args.author)
    print(f"repos={len(repos)} dry_run={args.dry_run}", flush=True)
    results: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="axquant-omlx-") as tmp:
        work = Path(tmp)
        for repo_id in repos:
            print(f"start {repo_id}", flush=True)
            if args.dry_run:
                files = set(list_repo_files(repo_id=repo_id, repo_type="model"))
                sidecar = next((name for name in SIDECAR_NAMES if name in files), None)
                results.append(
                    {
                        "repo": repo_id,
                        "status": "dry-run",
                        "sidecar": sidecar or "none",
                    }
                )
                continue
            try:
                row = _annotate_repo(api, repo_id, work)
            except Exception as exc:  # noqa: BLE001 — campaign log, keep going
                row = {"repo": repo_id, "status": "error", "reason": str(exc)}
            results.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    ok = sum(1 for row in results if row["status"] == "ok")
    skip = sum(1 for row in results if row["status"] == "skip")
    err = sum(1 for row in results if row["status"] == "error")
    print(f"done ok={ok} skip={skip} error={err}", flush=True)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
