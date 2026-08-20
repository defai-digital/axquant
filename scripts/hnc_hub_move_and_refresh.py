#!/usr/bin/env python3
"""Move the 2.4T hobby pack to ...-2bit-MTP and refresh its custom card.

Hard-coded D1 pair only. Default is dry-run. Auth via HF_TOKEN or
huggingface-cli. Never print the token.

After --apply, run: python scripts/sync_hf_collections.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from axquant.errors import ArtifactError
from axquant.naming import require_mtp_suffix_matches_packaging

FROM_ID = "AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit"
TO_ID = "AutomatosX/AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP"
CARD = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "hub-cards"
    / "AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP.md"
)
HUB_URL = "https://huggingface.co"


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


def _list_files(api: HfApi, repo_id: str) -> list[str]:
    info = api.model_info(repo_id)
    return [sibling.rfilename for sibling in (info.siblings or [])]


def _repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.model_info(repo_id)
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return False
        raise
    return True


def _head_redirect(url: str) -> tuple[int, str | None]:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:  # public Hub URL
            return int(response.status), response.geturl()
    except HTTPError as exc:
        location = exc.headers.get("Location") if exc.headers is not None else None
        return int(exc.code), location
    except URLError as exc:
        _die(f"HEAD {url} failed: {exc}")


def _expect_suffix_fail(leaf: str, filenames: list[str]) -> str:
    try:
        require_mtp_suffix_matches_packaging(leaf, filenames=filenames)
    except ArtifactError as exc:
        return str(exc)
    _die(f"{leaf} unexpectedly passed the MTP suffix lint")


def _preflight(api: HfApi) -> tuple[str, list[str]]:
    if not CARD.is_file():
        _die(f"missing hobby card: {CARD}")
    old_exists = _repo_exists(api, FROM_ID)
    new_exists = _repo_exists(api, TO_ID)
    if old_exists:
        files = _list_files(api, FROM_ID)
        if "mtp.safetensors" not in files:
            _die(f"{FROM_ID} is missing mtp.safetensors")
        old_error = _expect_suffix_fail(FROM_ID.rsplit("/", 1)[-1], files)
        try:
            require_mtp_suffix_matches_packaging(TO_ID.rsplit("/", 1)[-1], filenames=files)
        except ArtifactError as exc:
            _die(f"new leaf would fail suffix lint: {exc}")
        if new_exists:
            _die(f"{TO_ID} already exists while {FROM_ID} is still present")
        print(f"old-leaf lint (expected fail): {old_error}", flush=True)
        print("new-leaf lint: pass", flush=True)
        return "move", files
    if new_exists:
        files = _list_files(api, TO_ID)
        if "mtp.safetensors" not in files:
            _die(f"{TO_ID} is missing mtp.safetensors")
        require_mtp_suffix_matches_packaging(TO_ID.rsplit("/", 1)[-1], filenames=files)
        print("already moved; card refresh only", flush=True)
        return "refresh", files
    _die(f"neither {FROM_ID} nor {TO_ID} exists")


def _apply(api: HfApi, action: str) -> None:
    if action == "move":
        print(f"moving {FROM_ID} -> {TO_ID}", flush=True)
        api.move_repo(from_id=FROM_ID, to_id=TO_ID, repo_type="model")
    print(f"uploading hobby card to {TO_ID}", flush=True)
    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=TO_ID,
        repo_type="model",
        commit_message="Refresh hobby card for packaged MTP name",
    )
    status, location = _head_redirect(f"{HUB_URL}/{FROM_ID}")
    print(f"HEAD {FROM_ID} -> {status} {location or ''}".rstrip(), flush=True)
    print("next: python scripts/sync_hf_collections.py --apply", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the hard-coded 2.4T move and card upload",
    )
    args = parser.parse_args(argv)
    api = HfApi()
    action, files = _preflight(api)
    print(f"plan: {action} {FROM_ID} -> {TO_ID}", flush=True)
    print(f"upload_file {CARD} as README.md on {TO_ID}", flush=True)
    print(f"source files: {len(files)} (includes mtp.safetensors)", flush=True)
    print("then: python scripts/sync_hf_collections.py --apply", flush=True)
    if not args.apply:
        print("dry-run only; pass --apply to execute", flush=True)
        return 0
    _apply(api, action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
