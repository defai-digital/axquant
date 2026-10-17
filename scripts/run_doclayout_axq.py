#!/usr/bin/env python3
"""Factory driver for source-bound DocLayout development exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from axquant.doclayout import REVISION, convert_doclayout
from axquant.naming import cuda_pack_name, model_name
from axquant.doclayout import SOURCE

BASE = Path("/Volumes/Ext16TR0/models/axdoclayout-axq-20261014")


def _pack_dir(mode: str) -> Path:
    if mode == "nvfp4":
        name = cuda_pack_name(SOURCE, lane="NVFP4-W4A16")
    else:
        name = model_name(SOURCE, target_class=mode.upper(), include_mlx=True)
    return BASE / name


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
