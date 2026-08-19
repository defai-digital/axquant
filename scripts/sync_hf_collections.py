#!/usr/bin/env python3
"""Create and keep AutomatosX Hugging Face collections family-first.

The previous single dump mixed uniform MLX, QAT, OptiQ, and AXQ across every
family. This script rebuilds the org collections the way users browse: a
certified starting list, one collection per model family, plus a complete
index that preserves the original catalog URL.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

NAMESPACE = "AutomatosX"
COMPLETE_SLUG = "AutomatosX/automatosx-mlx-model-catalog-6a604acfe036cdf7e0993092"
KNOWN_SLUGS = {
    "Certified AXQ": "AutomatosX/certified-axq-6a7f53947a289f25b70b8290",
    "Qwen3.8": "AutomatosX/qwen38-6a7f539725d8d5b6f524c9a0",
    "Qwen3.6": "AutomatosX/qwen36-6a7f539837353d885ac950e5",
    "Gemma 4": "AutomatosX/gemma-4-6a7f539ab25fa2c2c40a3bc9",
    "Qwen3.5": "AutomatosX/qwen35-6a7f539d7b63c4177df8db8b",
    "Qwen3-Coder-Next": "AutomatosX/qwen3-coder-next-6a7f539ea514d4eed9308e2d",
    "Qwen3-VL": "AutomatosX/qwen3-vl-6a7f539fef1f1bbc02191224",
    "DeepSeek": "AutomatosX/deepseek-6a7f539f6dbb163f53e10230",
    "GPT-OSS": "AutomatosX/gpt-oss-6a7f53a1b91ffbc9d632a263",
    "Holo3": "AutomatosX/holo3-6a7f53a1cf47102943335f3e",
    "Holo-3.1": "AutomatosX/holo-31-6a819f5921908d636ae8b2a8",
    "Devstral": "AutomatosX/devstral-6a7f53a28f83a5c088373faf",
    "Ornith 1.0": "AutomatosX/ornith-10-6a7f53a328beff762aced1c9",
    "Mistral": "AutomatosX/mistral-6a7f53a3c81397e9339a5b75",
    "Embeddings": "AutomatosX/embeddings-6a7f53a515672c27b985e679",
    "OCR": "AutomatosX/ocr-6a7f53a89dfa008d21861df4",
    "Muse-Glimmer": "AutomatosX/muse-glimmer-6a7f53a96446b0696dabdbaa",
    "Nemotron": "AutomatosX/nemotron-6a7f53a98c678e2ffa59ea33",
    "Speech": "AutomatosX/speech-6a7f53aa5ff26cff62787cf6",
    "AutomatosX MLX Model Catalog": COMPLETE_SLUG,
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
NOTE_0731 = (
    "Flash-0731 source revision. AXQ development artifact. Not certified; see the model card."
)
NOTE_0731_MEM = (
    "Flash-0731 ship SKU. Not certified on the 192 GB factory Studio; recert on a larger Mac."
)
CERTIFIED_NOTES = frozenset({NOTE_T1, NOTE_T1_T2, NOTE_T1_NO_T2, NOTE_T1_NOMTP, NOTE_T1_EXP})
NOTE_OPTIQ = "OptiQ mixed 4/8-bit."
NOTE_UNIFORM = "Uniform MLX quantization."
NOTE_QAT = "Official QAT 4-bit, converted to MLX."
NOTE_QAT_OPTIQ = "QAT source, then OptiQ mixed 4/8-bit."
NOTE_AXQ_DEV = "AXQ development artifact. Not certified; see the model card."
NOTE_AXQ_ASR = "AXQ language-decoder PTQ with a protected BF16 audio tower. Not certified."
NOTE_AXQ_VL = "AXQ language-path PTQ with a protected BF16 vision tower. Not certified."
NOTE_DWQ = "Uniform MLX 4-bit with DWQ."
NOTE_CUDA = "CUDA AWQ W4A16. Not an MLX pack."
NOTE_MXFP8 = "MLX MXFP8 OCR pack."
NOTE_24T = (
    "Experimental 2-bit of the 2.4T MoE. SSD paging is too slow for practical "
    "serving; this revision will not be certified."
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
        items=(
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-8bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-6bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-4bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Holo3-35B-A3B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Holo3-35B-A3B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Ornith-1.0-35B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Ornith-1.0-35B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-gpt-oss-120b-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-gpt-oss-20b-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-gpt-oss-20b-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-gemma-4-31b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-31b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-12b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-12b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP", NOTE_T1_EXP),
        ),
    ),
    Spec(
        title="Qwen3.8",
        description=(
            "Qwen3.8 MLX: certified 27B AXQ MXFP4/4/6/8-bit ± MTP, plus experimental 2.4T 2-bit."
        ),
        items=(
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-8bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit", NOTE_24T),
        ),
    ),
    Spec(
        title="Qwen3.6",
        description=(
            "Qwen3.6 27B and 35B-A3B: uniform, OptiQ, certified AXQ 4/6-bit MTP, "
            "and no-MTP siblings."
        ),
        items=(
            _ax("AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-6bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-4bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP", NOTE_OPTIQ),
            _ax("AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP", NOTE_OPTIQ),
            _ax("AX-Qwen3.6-27B-MLX-6bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.6-27B-MLX-4bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.6-35B-A3B-MLX-6bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.6-35B-A3B-MLX-4bit-MTP", NOTE_UNIFORM),
        ),
    ),
    Spec(
        title="Gemma 4",
        description=(
            "Gemma 4 12B/26B/31B plus DiffusionGemma: uniform, QAT, OptiQ, and certified AXQ."
        ),
        items=(
            _ax("AX-gemma-4-31b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-31b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP", NOTE_OPTIQ),
            _ax("AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP", NOTE_QAT),
            _ax("AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP", NOTE_UNIFORM),
            _ax("AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP", NOTE_OPTIQ),
            _ax("AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP", NOTE_QAT),
            _ax("AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP", NOTE_UNIFORM),
            _ax("AX-gemma-4-12b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-12b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP", NOTE_QAT_OPTIQ),
            _ax("AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP", NOTE_QAT),
            _ax("AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP", NOTE_UNIFORM),
            _ax("AX-DiffusionGemma-26B-A4B-IT-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-DiffusionGemma-26B-A4B-IT-MLX-4bit", NOTE_UNIFORM),
        ),
    ),
    Spec(
        title="Qwen3.5",
        description=(
            "Qwen3.5 9B MTP: uniform, OptiQ, and AXQ. No distinct AXQ-4bit (floor-collapsed)."
        ),
        items=(
            _ax("AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP", NOTE_AXQ_DEV),
            _ax("AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP", NOTE_OPTIQ),
            _ax("AX-Qwen3.5-9B-MLX-6bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.5-9B-MLX-4bit-MTP", NOTE_UNIFORM),
        ),
    ),
    Spec(
        title="Qwen3-Coder-Next",
        description="Qwen3-Coder-Next: uniform, OptiQ, and certified AXQ MXFP4/4/6-bit.",
        items=(
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Qwen3-Coder-Next-MLX-6bit", NOTE_UNIFORM),
            _ax("AX-Qwen3-Coder-Next-MLX-4bit", NOTE_UNIFORM),
        ),
    ),
    Spec(
        title="Qwen3-VL",
        description="Qwen3-VL Instruct AXQ: certified 30B-A3B plus 8B and 32B-Thinking packs.",
        items=(
            _ax("AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-32B-Thinking-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit", NOTE_AXQ_VL),
        ),
    ),
    Spec(
        title="DeepSeek",
        description=(
            "DeepSeek V4 Flash AXQ 2/4/6-bit MTP, Flash-0731 2/4-bit MTP plus "
            "MXFP4/6-bit stubs, and DeepSeek-OCR-2. 3-bit withdrawn."
        ),
        items=(
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-6bit-MTP", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-4bit-MTP", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP", NOTE_T1_EXP),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP", NOTE_0731),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP", NOTE_0731),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-MXFP4", NOTE_0731_MEM),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-6bit", NOTE_0731_MEM),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-4bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="GPT-OSS",
        description="OpenAI gpt-oss 20B and 120B certified AXQ packs.",
        items=(
            _ax("AX-gpt-oss-120b-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-gpt-oss-20b-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-gpt-oss-20b-MLX-AXQ-4bit", NOTE_T1),
        ),
    ),
    Spec(
        title="Holo3",
        description="Holo3-35B-A3B certified AXQ 4/6-bit.",
        items=(
            _ax("AX-Holo3-35B-A3B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Holo3-35B-A3B-MLX-AXQ-4bit", NOTE_T1),
        ),
    ),
    Spec(
        title="Holo-3.1",
        description="Holo-3.1-35B-A3B AXQ MXFP4 (Tier 1) plus 6/8-bit eval packs.",
        items=(
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Devstral",
        description="Devstral Small coding models: OptiQ 2512 and AXQ 2505.",
        items=(
            _ax("AX-Devstral-Small-2-24B-Instruct-2512-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Devstral-Small-2505-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Devstral-Small-2505-MLX-AXQ-4bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Ornith 1.0",
        description="Ornith 1.0 35B AXQ 4/6-bit coding packs.",
        items=(
            _ax("AX-Ornith-1.0-35B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Ornith-1.0-35B-MLX-AXQ-4bit", NOTE_T1),
        ),
    ),
    Spec(
        title="Mistral",
        description=(
            "Ministral 3 and Mistral Small: OptiQ and AXQ. No distinct Ministral-3-8B AXQ-4bit."
        ),
        items=(
            _ax("AX-Ministral-3-14B-Instruct-2512-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Ministral-3-8B-Instruct-2512-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Embeddings",
        description=(
            "Qwen3-Embedding, EmbeddingGemma, and Nemotron-3-Embed (uniform, DWQ, and AXQ)."
        ),
        items=(
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-8B-MLX-4bit-DWQ", NOTE_DWQ),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-4B-MLX-4bit-DWQ", NOTE_DWQ),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-8bit", NOTE_UNIFORM),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-EmbeddingGemma-300M-MLX-8bit", NOTE_UNIFORM),
        ),
    ),
    Spec(
        title="OCR",
        description="Unlimited-OCR (MLX MXFP8 and CUDA AWQ) plus DeepSeek-OCR-2 AXQ.",
        items=(
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-MXFP8", NOTE_MXFP8),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AWQ-W4A16", NOTE_CUDA),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-4bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Muse-Glimmer",
        description="Muse-Glimmer 30B AXQ 4/6-bit image-text packs.",
        items=(
            _ax("AX-Muse-Glimmer-30B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Muse-Glimmer-30B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Nemotron",
        description="Nemotron-3-Nano and Qwen3-Nemotron GenRM AXQ packs.",
        items=(
            _ax("AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-4bit", NOTE_AXQ_DEV),
        ),
    ),
    Spec(
        title="Speech",
        description="Qwen3-ASR 1.7B AXQ 4/6-bit with a protected BF16 audio tower.",
        items=(
            _ax("AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit", NOTE_AXQ_ASR),
            _ax("AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit", NOTE_AXQ_ASR),
        ),
    ),
    Spec(
        title="AutomatosX MLX Model Catalog",
        description="Complete index of every Hub pack. Prefer the family collections above.",
        existing_slug=COMPLETE_SLUG,
        items=(
            # Family order, then leftover small models.
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-8bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit", NOTE_24T),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-6bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-27B-MLX-AXQ-4bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP", NOTE_T1_T2),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit", NOTE_T1_NOMTP),
            _ax("AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP", NOTE_OPTIQ),
            _ax("AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP", NOTE_OPTIQ),
            _ax("AX-Qwen3.6-27B-MLX-6bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.6-27B-MLX-4bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.6-35B-A3B-MLX-6bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.6-35B-A3B-MLX-4bit-MTP", NOTE_UNIFORM),
            _ax("AX-gemma-4-31b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-31b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP", NOTE_OPTIQ),
            _ax("AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP", NOTE_QAT),
            _ax("AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP", NOTE_UNIFORM),
            _ax("AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP", NOTE_OPTIQ),
            _ax("AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP", NOTE_QAT),
            _ax("AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP", NOTE_UNIFORM),
            _ax("AX-gemma-4-12b-MLX-AXQ-6bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-gemma-4-12b-MLX-AXQ-4bit-MTP", NOTE_T1_NO_T2),
            _ax("AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP", NOTE_QAT_OPTIQ),
            _ax("AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP", NOTE_QAT),
            _ax("AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP", NOTE_UNIFORM),
            _ax("AX-DiffusionGemma-26B-A4B-IT-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-DiffusionGemma-26B-A4B-IT-MLX-4bit", NOTE_UNIFORM),
            _ax("AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP", NOTE_AXQ_DEV),
            _ax("AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP", NOTE_OPTIQ),
            _ax("AX-Qwen3.5-9B-MLX-6bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3.5-9B-MLX-4bit-MTP", NOTE_UNIFORM),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3-Coder-Next-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Qwen3-Coder-Next-MLX-6bit", NOTE_UNIFORM),
            _ax("AX-Qwen3-Coder-Next-MLX-4bit", NOTE_UNIFORM),
            _ax("AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Qwen3-VL-32B-Thinking-MLX-AXQ-6bit", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-32B-Thinking-MLX-AXQ-MXFP4", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit", NOTE_AXQ_VL),
            _ax("AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit", NOTE_AXQ_VL),
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-6bit-MTP", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-4bit-MTP", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP", NOTE_T1_EXP),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP", NOTE_0731),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP", NOTE_0731),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-MXFP4", NOTE_0731_MEM),
            _ax("AX-DeepSeek-V4-Flash-0731-MLX-AXQ-6bit", NOTE_0731_MEM),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-DeepSeek-OCR-2-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-gpt-oss-120b-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-gpt-oss-20b-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-gpt-oss-20b-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Holo3-35B-A3B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Holo3-35B-A3B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-MXFP4", NOTE_T1),
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Holo-3.1-35B-A3B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Devstral-Small-2-24B-Instruct-2512-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Devstral-Small-2505-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Devstral-Small-2505-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Ornith-1.0-35B-MLX-AXQ-6bit", NOTE_T1),
            _ax("AX-Ornith-1.0-35B-MLX-AXQ-4bit", NOTE_T1),
            _ax("AX-Ministral-3-14B-Instruct-2512-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Ministral-3-8B-Instruct-2512-MLX-OptiQ-4bit", NOTE_OPTIQ),
            _ax("AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-8B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-8B-MLX-4bit-DWQ", NOTE_DWQ),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-4B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-4B-MLX-4bit-DWQ", NOTE_DWQ),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Embedding-0.6B-MLX-8bit", NOTE_UNIFORM),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Embed-8B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Embed-1B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-EmbeddingGemma-300M-MLX-8bit", NOTE_UNIFORM),
            _ax("AX-Unlimited-OCR-3B-MoE-MLX-MXFP8", NOTE_MXFP8),
            _ax("AX-Unlimited-OCR-3B-MoE-CUDA-AWQ-W4A16", NOTE_CUDA),
            _ax("AX-Muse-Glimmer-30B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Muse-Glimmer-30B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-6bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-4bit", NOTE_AXQ_DEV),
            _ax("AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit", NOTE_AXQ_ASR),
            _ax("AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit", NOTE_AXQ_ASR),
            _ax("AX-MiniCPM5-1B-MLX-AXQ-6bit", NOTE_AXQ_DEV),
        ),
    ),
)


def _validate() -> None:
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


def _ensure_collection(api: HfApi, spec: Spec) -> str:
    slug = spec.existing_slug or KNOWN_SLUGS.get(spec.title) or _find_existing(api, spec.title)
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
    by_id = {item.item_id: item for item in collection.items}
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
    all_family_repos = {
        item.repo for spec in COLLECTIONS if spec.existing_slug is None for item in spec.items
    }
    complete_repos = {item.repo for item in COLLECTIONS[-1].items}
    missing_from_complete = sorted(all_family_repos - complete_repos)
    if missing_from_complete:
        raise SystemExit(f"complete catalog missing: {missing_from_complete}")

    print(f"{len(COLLECTIONS)} collections, {len(complete_repos)} complete-index models")
    for spec in COLLECTIONS:
        print(f"  {spec.title:22} {len(spec.items):3}  {spec.description}")
    if not apply:
        print("dry-run only; pass --apply to write the Hub")
        return 0

    api = HfApi()
    me = api.whoami()
    print(f"authenticated as {me['name']}")
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
