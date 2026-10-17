#!/usr/bin/env python3
"""Factory driver for source-bound DocLayout development exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from axquant.doclayout import REVISION, convert_doclayout

BASE = Path("/Volumes/Ext16TR0/models/axdoclayout-axq-20261014")

# DocLayout packs follow the operator-guide naming, not the language-model CUDA/MLX
# lanes: no AX- prefix (the base name already carries AX), no CUDA/MLX tag, no
# activation suffix. This matches sync_hf_collections.py and docs/guides/doclayout.md.
SOURCE_VERSION = "v1.0"


def _pack_dir(mode: str) -> Path:
    return BASE / f"AXDocLayout-M-{SOURCE_VERSION}-AXQ-{mode.upper()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["nvfp4", "mxfp4", "mxfp8", "all"])
    args = parser.parse_args()
    for mode in ["nvfp4", "mxfp4", "mxfp8"] if args.mode == "all" else [args.mode]:
        convert_doclayout(
            BASE / "source",
            _pack_dir(mode),
            q_mode=mode,
            revision=REVISION,
            allow_unmeasured=True,
        )


if __name__ == "__main__":
    main()
