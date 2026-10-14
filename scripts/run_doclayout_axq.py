#!/usr/bin/env python3
"""Factory driver for source-bound DocLayout development exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from axquant.doclayout import REVISION, convert_doclayout

BASE = Path("/Volumes/Ext16TR0/models/axdoclayout-axq-20261014")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["nvfp4", "mxfp4", "mxfp8", "all"])
    args = parser.parse_args()
    for mode in ["nvfp4", "mxfp4", "mxfp8"] if args.mode == "all" else [args.mode]:
        convert_doclayout(
            BASE / "source",
            BASE / f"AXDocLayout-M-v1.0-AXQ-{mode.upper()}",
            q_mode=mode,
            revision=REVISION,
            allow_unmeasured=True,
        )


if __name__ == "__main__":
    main()
