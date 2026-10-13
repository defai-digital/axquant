#!/usr/bin/env python3
"""CUDA NVFP4 W4A16 pack for EmbeddingGemma 2, from the pinned BF16 snapshot.

Attention and MLP matrices use native NVFP4. Vision, audio, embeddings, norms,
and the final projection stay at source precision. Activations stay BF16.
W4A4 is not used: this checkpoint has no activation calibration, and float16
activations are unsafe. Conversion refuses a numpy-reference manifest.

  PYTHONPATH=src python scripts/run_embeddinggemma2_nvfp4.py convert
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.cuda import convert_cuda_nvfp4, plan_cuda_nvfp4  # noqa: E402
from axquant.naming import cuda_pack_name  # noqa: E402
from axquant.schema.cuda import CudaPackManifest  # noqa: E402
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data  # noqa: E402

SOURCE_ID = "google/embeddinggemma-2"
SOURCE_REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"
LANE = "NVFP4-W4A16"
PACK_NAME = cuda_pack_name("EmbeddingGemma-2", lane=LANE)
NOTICE = """This directory is a quantized CUDA derivative of google/embeddinggemma-2.

Changes from the upstream BF16 release:
- Eligible language attention and MLP weights were quantized to NVFP4 W4A16
  by AXQuant (round-to-nearest, block 16, no AWQ).
- Vision, audio, norms, token embeddings, and the final embedding projection
  stay at source precision.
- Activations stay BF16. Do not cast this model to float16.

The upstream checkpoint is Apache-2.0, Copyright Google DeepMind.
Deployments must follow the Gemma Prohibited Use Policy.
"""


def log(message: str) -> None:
    print(message, flush=True)


def source_dir() -> Path:
    return Path(
        os.environ.get(
            "EGEMMA_SRC",
            "/Volumes/Ext16TR0/axquant-certification/embeddinggemma-2/src",
        )
    ).expanduser()


def pack_dir() -> Path:
    override = os.environ.get("EGEMMA_NVFP4_PACK")
    if override:
        return Path(override)
    return Path(os.environ.get("FACTORY_MODELS", "/Volumes/Ext16TR0/models")) / PACK_NAME


def card() -> str:
    return f"""---
library_name: transformers
pipeline_tag: feature-extraction
base_model: {SOURCE_ID}
base_model_relation: quantized
license: apache-2.0
tags:
- axquant
- cuda
- nvfp4
- embedding
- feature-extraction
---

# {PACK_NAME}

Development AXQuant CUDA NVFP4 W4A16 pack of `{SOURCE_ID}` at `{SOURCE_REV}`.
Not a certified retrieval result.

## Precision

Language attention and MLP weights are NVFP4, block 16, E4M3 block scales.
Vision, audio, embeddings, norms, and the 512 to 768 projection stay BF16.
`quantization_config` has no input activation quantization. The recorded
activation dtype is BF16. Do not cast this model to float16.

## Embedding model

The checkpoint returns 768-d embeddings and supports Matryoshka widths 768, 512,
256, and 128. Re-normalize after truncation. Use the task prefixes in
`config_sentence_transformers.json`. This card does not claim a vLLM load or an
MTEB score.
"""


def reject_local_paths(pack: Path) -> None:
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path}")


def finalize_pack(src: Path, pack: Path, manifest: CudaPackManifest) -> None:
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing non-CUDA manifest backend {manifest.backend}")
    for member in manifest.files:
        if file_sha256(pack / member.path) != member.sha256:
            raise SystemExit(f"pack digest mismatch: {member.path}")
    config = read_data(pack / "config.json")
    quant = config.get("quantization_config") or {}
    nvfp4 = (quant.get("config_groups") or {}).get("nvfp4") or {}
    weights_cfg = nvfp4.get("weights") or {}
    if weights_cfg.get("num_bits") != 4 or nvfp4.get("input_activations") is not None:
        raise SystemExit("pack is not NVFP4 W4A16")
    if manifest.activation_dtype != "bfloat16":
        raise SystemExit(f"refusing activation dtype {manifest.activation_dtype}")
    if not (pack / "modules.json").is_file() or not (pack / "1_Pooling" / "config.json").is_file():
        raise SystemExit("sentence-transformers pooling files are missing")
    (pack / "NOTICE").write_text(NOTICE, encoding="utf-8")
    (pack / "README.md").write_text(card(), encoding="utf-8")
    license_src = src / "LICENSE"
    if license_src.is_file():
        (pack / "LICENSE").write_bytes(license_src.read_bytes())
    reject_local_paths(pack)
    log(f"built {PACK_NAME} backend={manifest.backend} quantized={len(manifest.quantized_tensors)}")


def cmd_convert() -> None:
    src = source_dir()
    weights = list(src.glob("*.safetensors"))
    if not (src / "config.json").is_file() or not weights:
        raise SystemExit(f"missing BF16 source {src}")
    pack = pack_dir()
    manifest_path = pack / "axquant_cuda_manifest.json"
    if manifest_path.is_file():
        finalize_pack(src, pack, load_model(manifest_path, CudaPackManifest))
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    plan = plan_cuda_nvfp4(
        src,
        model_id=SOURCE_ID,
        revision=SOURCE_REV,
        allow_unmeasured=True,
    )
    if plan.activation_dtype != "bfloat16":
        raise SystemExit(f"refusing activation dtype {plan.activation_dtype}")
    selected = sum(item.method == "nvfp4" for item in plan.allocations)
    preserved = sum(item.method == "preserve" for item in plan.allocations)
    log(f"plan nvfp4={selected} preserve={preserved} activation={plan.activation_dtype}")
    if selected < 1:
        raise SystemExit("NVFP4 plan selected no tensors")
    work = Path(os.environ.get("EGEMMA_NVFP4_WORK", str(src.parent / "nvfp4-work")))
    work.mkdir(parents=True, exist_ok=True)
    write_data(work / "plan.json", plan)
    manifest = convert_cuda_nvfp4(
        src,
        plan,
        pack,
        device=os.environ.get("EGEMMA_NVFP4_DEVICE", "cuda:0"),
        rows_per_chunk=int(os.environ.get("EGEMMA_NVFP4_ROWS", "2048")),
        allow_unmeasured=True,
    )
    if manifest.plan_sha256 != stable_sha256(plan):
        raise SystemExit("NVFP4 manifest is not bound to this plan")
    finalize_pack(src, pack, manifest)


def cmd_publish() -> None:
    from huggingface_hub import HfApi

    pack = pack_dir()
    manifest = load_model(pack / "axquant_cuda_manifest.json", CudaPackManifest)
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing to publish backend {manifest.backend}")
    for member in manifest.files:
        if file_sha256(pack / member.path) != member.sha256:
            raise SystemExit(f"pack digest mismatch: {member.path}")
    reject_local_paths(pack)
    repo = f"AutomatosX/{PACK_NAME}"
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo,
        folder_path=str(pack),
        commit_message=f"Publish {PACK_NAME}",
    )
    log(f"published {repo} {commit.oid}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["convert", "publish"])
    args = parser.parse_args()
    if args.stage == "convert":
        cmd_convert()
    else:
        cmd_publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
