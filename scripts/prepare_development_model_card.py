#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from axquant.model_card import prepare_development_model_card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanitize AXQ development metadata and render a Hugging Face model card."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--product-class", choices=("4bit", "6bit", "8bit"), default=None)
    args = parser.parse_args()
    files = prepare_development_model_card(
        artifact_dir=args.artifact,
        repo_id=args.repo_id,
        product_class=args.product_class,
    )
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
