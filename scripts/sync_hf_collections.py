#!/usr/bin/env python3
"""Create and keep AutomatosX Hugging Face collections method-first, then family.

The previous single dump mixed uniform MLX, QAT, OptiQ, and AXQ across every
family. This script keeps the org collections the way users browse: a
certified starting list, one collection per precision line (MXFP4, MXFP8, NVFP4),
one MTP collection,
one collection per live model family, plus a complete index of every public
AutomatosX model.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

NAMESPACE = "AutomatosX"
CATALOG_TITLE = "MLX"
# Pinned after the 2026-10-04 apply (all 14 embedding repos + Embeddings /
# Nemotron collections deleted; catalog collection renamed to MLX). A slug
# that 404s is recreated.
KNOWN_SLUGS: dict[str, str] = {
    "CUDA": "AutomatosX/cuda-6ac1b59d9199d121c6c7e2e3",
    "Certified AXQ": "AutomatosX/certified-axq-6ab432f02d50e1d27eaeb250",
    "MXFP4": "AutomatosX/mxfp4-6ac15f9e6eda0bc49dee0a32",
    "MXFP8": "AutomatosX/mxfp8-6ac15fa0b73470ecf715faec",
    "NVFP4": "AutomatosX/nvfp4-6ac1a489659e8a86ec4280fa",
    "MTP": "AutomatosX/mtp-6ac1a4e2daa11b99c2da9338",
    "Qwen": "AutomatosX/qwen-6ab4341bed3654678bbc5f72",
    "Nemotron": "AutomatosX/nemotron-6ac1b99e003f6638a7513f59",
    "Qwen3.8": "AutomatosX/qwen38-6ac15fa1ba7bf169f7539343",
    "Tiel Coder": "AutomatosX/tiel-coder-6ab432f2fdec85ba0ecd515d",
    "Qwen3-VL": "AutomatosX/qwen3-vl-6ac1af94ba7bf169f75c1ab5",
    "DeepSeek": "AutomatosX/deepseek-6ac15fa3aeb86a300e188fdf",
    "OCR": "AutomatosX/ocr-6ab432f8bed2124120a3b782",
    "Unlimited-OCR": "AutomatosX/unlimited-ocr-6ac15fa87eb43e9e3ef61c97",
    CATALOG_TITLE: "AutomatosX/mlx-6ab432f9724c4d8a1037df1c",
}

NOTE_T1 = "Checkpoint Tier 1 certified. See the certificate for the bound host."
NOTE_T1_T2 = (
    "Checkpoint Tier 1 and scoped MTP Tier 2 certified. See the certificates for the bound host."
)
NOTE_T1_NO_T2 = "Checkpoint Tier 1 certified. MTP Tier 2 is not certified."
NOTE_T1_NOMTP = (
    "Checkpoint Tier 1 certified. No-MTP sibling of the certified MTP pack; language path matches."
)
NOTE_T1_EXP = (
    "Experimental pack. Checkpoint Tier 1 certified (generation viability); MTP Tier 2 not claimed."
)
CERTIFIED_NOTES = frozenset({NOTE_T1, NOTE_T1_T2, NOTE_T1_NO_T2, NOTE_T1_NOMTP, NOTE_T1_EXP})
NOTE_OPTIQ = "OptiQ mixed 4/8-bit."
NOTE_UNIFORM = "Uniform MLX quantization."
NOTE_QAT = "Official QAT 4-bit, converted to MLX."
NOTE_QAT_OPTIQ = "QAT source, then OptiQ mixed 4/8-bit."
NOTE_AXQ_DEV = "AXQ development artifact. Not certified; see the model card."
NOTE_AXQ_ASR = "AXQ language-decoder PTQ with a protected BF16 audio tower. Not certified."
NOTE_AXQ_VL = "AXQ language-path PTQ with a protected BF16 vision tower. Not certified."
NOTE_AXQ_EMBED = "AXQ retrieval pack (feature-extraction only). Not certified."
NOTE_DWQ = "Uniform MLX 4-bit with DWQ."
NOTE_CUDA = "Native AXQuant CUDA NVFP4 W4A16. Development artifact; not certified."
NOTE_CUDA_W4A4 = (
    "Native AXQuant CUDA NVFP4 W4A4. Two-GPU development smoke; no MTP or quality certification."
)
NOTE_MXFP8 = "MLX MXFP8 OCR pack."
NOTE_24T = (
    "Experimental 2-bit of the 2.4T MoE with a packaged native MTP sidecar. "
    "SSD paging is too slow for practical serving; this revision will not be "
    "certified. MTP acceleration is not claimed."
)
NOTE_PRO_0813 = (
    "Experimental 2-bit Super-class stream pack of DeepSeek-V4-Pro-0813. "
    "SSD paging required; not certified. DSpark sidecar packaged; acceleration "
    "not claimed. No 4-bit sibling."
)
NOTE_KIMI_K3 = (
    "Experimental Super-class stream pack of Kimi-K3. SSD paging required; "
    "not certified. Native MXFP4 dequantized to affine 2-bit. No packaged MTP."
)


@dataclass(frozen=True)
class Item:
    repo: str
    note: str | None = None


@dataclass(frozen=True)
class Spec:
    title: str
    description: str
    items: tuple[Item, ...]
    existing_slug: str | None = None


def _ax(name: str, note: str | None = None) -> Item:
    return Item(f"AutomatosX/{name}", note)


COLLECTIONS: tuple[Spec, ...] = (
    Spec(
        title="Certified AXQ",
        description="Measured AXQ packs with a public Tier 1 certificate. Start here.",
        items=(_ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),),
    ),
    Spec(
        title="MXFP4",
        description="Every AXQ MXFP4 pack: certified and development, MTP and no-MTP.",
        items=(
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
        ),
    ),
    Spec(
        title="MXFP8",
        description="MXFP8-format packs. MLX OCR, Qwen3-VL, and embeddings.",
        items=(
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
        ),
    ),
    Spec(
        title="NVFP4",
        description="Native AXQuant CUDA NVFP4 W4A16 and calibrated W4A4 development checkpoints.",
        items=(
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-4B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-8B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
        ),
    ),
    Spec(
        title="MTP",
        description="AXQ packs with packaged MTP sidecars. -MTP ships assets; speed needs Tier 2.",
        items=(
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="CUDA",
        description="Native AXQuant CUDA NVFP4 OCR and Qwen3-VL development checkpoints.",
        items=(
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-4B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-8B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
        ),
    ),
    Spec(
        title="Qwen",
        description="All public Qwen packs: Qwen3.8, VL, and embeddings.",
        items=(
            _ax("AX-Qwen3-VL-4B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-8B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
        ),
    ),
    Spec(
        title="Nemotron",
        description="Nemotron-3-Embed 1B/8B AXQ MXFP4/MXFP8 retrieval packs. Not certified.",
        items=(
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
        ),
    ),
    Spec(
        title="Qwen3.8",
        description="Qwen3.8 MLX: 27B MXFP4 with packaged MTP, and Flash-Next MXFP4 MTP.",
        items=(
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Tiel Coder",
        description="Tiel Coder 35B-A3B MXFP4 with packaged MTP, including the Cyber variant.",
        items=(
            _ax("AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Qwen3-VL",
        description="Qwen3-VL 4B/8B MLX MXFP4/MXFP8 plus CUDA NVFP4 W4A4. Vision BF16, no MTP.",
        items=(
            _ax("AX-Qwen3-VL-4B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-8B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
        ),
    ),
    Spec(
        title="DeepSeek",
        description="DeepSeek-OCR-2 AXQ MXFP4/MXFP8/NVFP4.",
        items=(
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
        ),
    ),
    Spec(
        title="OCR",
        description="DeepSeek-OCR-2 and Unlimited-OCR AXQ MX packs plus CUDA NVFP4.",
        items=(
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Unlimited-OCR",
        description="Unlimited-OCR 3B MoE: native AXQ CUDA NVFP4 and MLX MXFP8/MXFP4 packs.",
        items=(
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title=CATALOG_TITLE,
        description=(
            "Complete index of every public AutomatosX model. Prefer the family collections above."
        ),
        items=(
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-4B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3-VL-8B-Instruct-CUDA-AXQ-NVFP4-W4A4", NOTE_CUDA_W4A4),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP4", NOTE_AXQ_DEV),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-AXQ-MXFP8", NOTE_AXQ_DEV),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A16", NOTE_CUDA),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-4B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-MXFP8", NOTE_AXQ_VL),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-MXFP4", NOTE_AXQ_EMBED),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-MXFP8", NOTE_AXQ_EMBED),
        ),
    ),
)


def _validate() -> None:
    if not COLLECTIONS or COLLECTIONS[-1].title != CATALOG_TITLE:
        raise SystemExit("complete catalog must be the last collection")
    for spec in COLLECTIONS:
        if len(spec.description) > 150:
            raise SystemExit(f"description too long ({len(spec.description)}): {spec.title!r}")
        seen: set[str] = set()
        for item in spec.items:
            if item.repo in seen:
                raise SystemExit(f"duplicate {item.repo} in {spec.title!r}")
            seen.add(item.repo)
            if item.note is not None and len(item.note) > 500:
                raise SystemExit(f"note too long for {item.repo} in {spec.title!r}")
            if spec.title == "Certified AXQ" and item.note not in CERTIFIED_NOTES:
                raise SystemExit(
                    f"Certified AXQ has a non-certificate note for {item.repo}: {item.note!r}"
                )
    catalog = {item.repo for item in COLLECTIONS[-1].items}
    family = {item.repo for spec in COLLECTIONS[:-1] for item in spec.items}
    missing_from_catalog = sorted(family - catalog)
    if missing_from_catalog:
        raise SystemExit(f"complete catalog missing: {missing_from_catalog}")
    only_in_catalog = sorted(catalog - family)
    if only_in_catalog:
        raise SystemExit(f"models missing a family collection: {only_in_catalog}")


def _retry(fn, *, retries: int = 6):
    delay = 1.0
    for attempt in range(retries):
        try:
            return fn()
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in {408, 409, 429, 500, 502, 503, 504} and attempt + 1 < retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError("unreachable")


def _find_existing(api: HfApi, title: str) -> str | None:
    for collection in api.list_collections(owner=NAMESPACE):
        if collection.title == title:
            return collection.slug
    return None


def _collection_exists(api: HfApi, slug: str) -> bool:
    try:
        _retry(lambda: api.get_collection(slug))
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {401, 404}:
            return False
        raise
    return True


def _ensure_collection(api: HfApi, spec: Spec) -> str:
    slug = spec.existing_slug or KNOWN_SLUGS.get(spec.title) or _find_existing(api, spec.title)
    if slug is not None and not _collection_exists(api, slug):
        print(f"missing pinned slug for {spec.title}: {slug}")
        slug = _find_existing(api, spec.title)
    if slug is None:
        created = _retry(
            lambda: api.create_collection(
                title=spec.title,
                namespace=NAMESPACE,
                description=spec.description,
                exists_ok=True,
            )
        )
        slug = created.slug
        print(f"created {spec.title}: {slug}")
    else:
        print(f"existing {spec.title}: {slug}")
    _retry(
        lambda: api.update_collection_metadata(
            collection_slug=slug,
            title=spec.title,
            description=spec.description,
        )
    )
    return slug


def _sync_items(api: HfApi, slug: str, spec: Spec) -> None:
    collection = _retry(lambda: api.get_collection(slug))
    desired = [item.repo for item in spec.items]
    desired_set = set(desired)

    for extra in collection.items:
        if extra.item_id not in desired_set:
            print(f"  remove {extra.item_id}")
            _retry(
                lambda item=extra: api.delete_collection_item(
                    collection_slug=slug,
                    item_object_id=item.item_object_id,
                )
            )

    collection = _retry(lambda: api.get_collection(slug))
    by_id = {item.item_id: item for item in collection.items}

    for position, item in enumerate(spec.items):
        existing = by_id.get(item.repo)
        if existing is None:
            print(f"  add [{position}] {item.repo}")
            _retry(
                lambda item=item, position=position: api.add_collection_item(
                    collection_slug=slug,
                    item_id=item.repo,
                    item_type="model",
                    note=item.note,
                    exists_ok=True,
                )
            )
            collection = _retry(lambda: api.get_collection(slug))
            by_id = {entry.item_id: entry for entry in collection.items}
            existing = by_id[item.repo]
        note_changed = (existing.note or None) != item.note
        position_changed = existing.position != position
        if note_changed or position_changed:
            print(f"  update [{position}] {item.repo}")
            _retry(
                lambda existing=existing, item=item, position=position: api.update_collection_item(
                    collection_slug=slug,
                    item_object_id=existing.item_object_id,
                    note=item.note,
                    position=position,
                )
            )


def sync(*, apply: bool) -> int:
    _validate()
    complete_repos = {item.repo for item in COLLECTIONS[-1].items}

    print(f"{len(COLLECTIONS)} collections, {len(complete_repos)} complete-index models")
    for spec in COLLECTIONS:
        print(f"  {spec.title:22} {len(spec.items):3}  {spec.description}")
    if not apply:
        print("dry-run only; pass --apply to write the Hub")
        return 0

    api = HfApi()
    me = api.whoami()
    print(f"authenticated as {me['name']}")
    orgs = {org.get("name") for org in me.get("orgs") or []}
    if me.get("name") != NAMESPACE and NAMESPACE not in orgs:
        raise SystemExit(f"token cannot write {NAMESPACE} collections")
    live = {model.id for model in api.list_models(author=NAMESPACE, limit=None)}
    if live != complete_repos:
        raise SystemExit(
            "catalog drifted from the live org: "
            f"missing={sorted(live - complete_repos)} extra={sorted(complete_repos - live)}"
        )
    slugs: list[str] = []
    for spec in COLLECTIONS:
        slug = _ensure_collection(api, spec)
        slugs.append(slug)
        _sync_items(api, slug, spec)
    for position, slug in enumerate(slugs):
        print(f"position {position}: {slug}")
        _retry(
            lambda slug=slug, position=position: api.update_collection_metadata(
                collection_slug=slug,
                position=position,
            )
        )
    print("done")
    for spec, slug in zip(COLLECTIONS, slugs, strict=True):
        print(f"https://huggingface.co/collections/{slug}  {spec.title}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create/update collections on the Hub (default is a dry run).",
    )
    args = parser.parse_args()
    return sync(apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
