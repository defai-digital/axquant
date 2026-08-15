#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from axquant.model_card import prepare_development_model_card


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize AXQ development metadata and render a Hugging Face model card. "
            "By default, binds a public docs/certifications Tier 1 certificate when the "
            "artifact digest matches (fails closed on mismatch)."
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--product-class",
        choices=(
            "2bit",
            "2bit-experimental",
            "3bit",
            "3bit-experimental",
            "4bit",
            "6bit",
            "8bit",
            "MXFP4",
        ),
        default=None,
    )
    parser.add_argument("--artifact-edition", type=int, default=None)
    parser.add_argument(
        "--no-public-certification",
        action="store_true",
        help="Skip looking up docs/certifications for this repo_id (always development banner)",
    )
    parser.add_argument(
        "--certifications-dir",
        type=Path,
        default=None,
        help="Override path to public certificate JSON (default: repo docs/certifications)",
    )
    args = parser.parse_args()
    files = prepare_development_model_card(
        artifact_dir=args.artifact,
        repo_id=args.repo_id,
        product_class=args.product_class,
        artifact_edition=args.artifact_edition,
        use_public_certification=not args.no_public_certification,
        certifications_dir=args.certifications_dir,
    )
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
