#!/usr/bin/env python3
"""Render or check frozen schema contract snapshots.

Usage::

    python scripts/render_schema_contracts.py --write
    python scripts/render_schema_contracts.py --check
    python scripts/render_schema_contracts.py --check --base-ref origin/main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from axquant.schema_contracts import (  # noqa: E402
    check_base_ref_immutability,
    check_schema_contracts,
    write_schema_contracts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate schemas/*.schema.json, schemas/manifest.json, docs/schema-catalog.md",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if snapshots drift from live models",
    )
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Git ref for immutability check (default: origin/main or main)",
    )
    parser.add_argument(
        "--skip-base-ref",
        action="store_true",
        help="Skip base-ref immutability comparison",
    )
    args = parser.parse_args(argv)

    if args.write:
        written = write_schema_contracts(root=_ROOT)
        if written:
            for path in written:
                print(f"wrote {path.relative_to(_ROOT)}")
        else:
            print("already up to date")
        return 0

    messages = check_schema_contracts(root=_ROOT)
    if not args.skip_base_ref:
        messages.extend(check_base_ref_immutability(root=_ROOT, base_ref=args.base_ref))
    if messages:
        for message in messages:
            print(message, file=sys.stderr)
        return 1
    print("schema contracts match live models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
