#!/usr/bin/env python3
"""Replace a live AutomatosX repository with a rebuilt development pack.

Republish in place: one Hub commit on top of ``--expected-parent`` that
uploads changed files, deletes files the rebuilt pack no longer has, and is
verified against the local digests. Download counts, likes and URLs stay;
earlier revisions stay in the repository history.

Gates: the shared publication preconditions (support, MTP naming, runtime
compatibility, runtime smoke receipt, DFlash2 declaration, path-free
identity), the privacy scan, and a byte check of every file against
``axquant_manifest.json``. No certification gate runs; the pack stays
development evidence. Without ``--yes`` only the change set is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.errors import AxquantError  # noqa: E402
from axquant.publisher import republish_development_pack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--repo", required=True, help="owner/name of the live repository")
    parser.add_argument(
        "--expected-parent", required=True, help="40-hex Hub commit that main must still be"
    )
    parser.add_argument(
        "--receipt", type=Path, help="Write the change set (and commit) as JSON here"
    )
    parser.add_argument("--yes", action="store_true", help="Create the Hub commit")
    args = parser.parse_args()
    if args.receipt is not None and args.pack.resolve() in args.receipt.resolve().parents:
        parser.error("--receipt must be outside the pack")
    try:
        result = republish_development_pack(
            model_dir=args.pack,
            repo_id=args.repo,
            expected_parent=args.expected_parent,
            execute=args.yes,
        )
    except AxquantError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    payload = {
        "schema_version": "axquant.internal.development_republication.v1",
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "executed": args.yes,
        **asdict(result),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
