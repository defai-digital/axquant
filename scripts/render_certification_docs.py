#!/usr/bin/env python3
"""Regenerate or check public certification matrices from certificate JSON.

Certificate records under ``docs/certifications/*.json`` are the single source
of truth. This script writes:

* the marked matrix in ``README.md``
* the marked matrix in ``docs/certifications/README.md``
* ``docs/releases/certification-matrix.md``
* ``docs/certifications/full-list.md`` (every certificate record)

Usage::

    python scripts/render_certification_docs.py --write
    python scripts/render_certification_docs.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from axquant.public_cert_index import check_documents, write_documents  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate README, certification index, and release matrices",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated docs drift from certificate JSON",
    )
    args = parser.parse_args(argv)

    if args.write:
        written = write_documents(root=_ROOT)
        if written:
            for path in written:
                print(f"wrote {path.relative_to(_ROOT)}")
        else:
            print("already up to date")
        return 0

    messages = check_documents(root=_ROOT)
    if messages:
        for message in messages:
            print(message, file=sys.stderr)
        return 1
    print("certification docs match docs/certifications/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
