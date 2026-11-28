#!/usr/bin/env python3
"""Build or verify a local DFlash2 composite target + drafter pack (AXQ-067 DF-5).

``compose`` writes a new directory: the target pack's files unchanged (hard
links when the filesystem allows) plus the admitted drafter's files under
``dflash2/`` and ``axquant_dflash2_companion.json``. ``verify`` re-checks an
existing composite. Local pilot only: publishing a ``-DFlash2`` repository
needs a separate owner direction (AXQ-071). No drafter bytes are changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.dflash2_composite import (  # noqa: E402
    compose_dflash2_composite,
    composite_issues,
    composite_summary,
)
from axquant.errors import AxquantError  # noqa: E402
from axquant.naming import dflash2_composite_name  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    compose = commands.add_parser("compose")
    compose.add_argument("--target", type=Path, required=True)
    compose.add_argument("--drafter", type=Path, required=True, help="Pinned drafter snapshot")
    compose.add_argument(
        "--output-root", type=Path, required=True, help="Directory that receives the composite"
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "compose":
            output = args.output_root / dflash2_composite_name(args.target.resolve().name)
            companion = compose_dflash2_composite(
                target_dir=args.target, drafter_dir=args.drafter, output_dir=output
            )
            print(json.dumps({"output": str(output), **composite_summary(companion)}, indent=2))
            return 0
        issues = composite_issues(args.directory)
    except AxquantError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    for issue in issues:
        print(f"issue: {issue}", file=sys.stderr)
    print("composite verified" if not issues else f"{len(issues)} issue(s)")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
