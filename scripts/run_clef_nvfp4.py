#!/usr/bin/env python3
"""Build Clef and Clef-Flash CUDA NVFP4 W4A16 packs from the pinned BF16 snapshots.

Language attention and MLP matrices use native NVFP4. Vision, norms, embeddings,
the LM head, and the joint schema head stay at source precision. Conversion
requires CUDA and refuses a numpy-reference manifest.

  PYTHONPATH=src python scripts/run_clef_nvfp4.py --pack flash convert
  PYTHONPATH=src python scripts/run_clef_nvfp4.py --pack all publish
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.cuda import convert_cuda_nvfp4, plan_cuda_nvfp4  # noqa: E402
from axquant.naming import cuda_pack_name  # noqa: E402
from axquant.schema.cuda import CudaPackManifest  # noqa: E402
from axquant.serde import file_sha256, load_model, stable_sha256, write_data  # noqa: E402

SOURCES: dict[str, dict[str, str]] = {
    "clef": {
        "id": "Cloudflare/clef",
        "rev": "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
        "base": "Clef-27B",
    },
    "flash": {
        "id": "Cloudflare/clef-flash",
        "rev": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        "base": "Clef-Flash-9B",
    },
}
JOINT_HEAD = "joint_head.safetensors"
LANE = "NVFP4-W4A16"
NOTICE = """This directory is a quantized CUDA derivative of {source}.

Changes from the upstream BF16 release:
- Eligible language attention and MLP weights were quantized to NVFP4 W4A16
  by AXQuant (round-to-nearest, block 16, no AWQ).
- Vision, norms, token embeddings, and the LM head stay at source precision.
- joint_head.safetensors is the original BF16 joint schema head, copied unchanged.
- Activations are not quantized.

The upstream work is Apache-2.0. See LICENSE (Copyright 2026 Alibaba Cloud).
"""


def log(message: str) -> None:
    print(message, flush=True)


def work_root() -> Path:
    return Path(os.environ.get("CLEF_NVFP4_ROOT", "/home/devop/work/clef-nvfp4")).expanduser()


def source_dir(size: str) -> Path:
    override = os.environ.get(f"CLEF_{size.upper()}_SRC")
    if override:
        return Path(override)
    return work_root() / "src" / SOURCES[size]["id"].split("/")[-1]


def pack_name(size: str) -> str:
    return cuda_pack_name(SOURCES[size]["base"], lane=LANE)


def pack_dir(size: str) -> Path:
    override = os.environ.get("CLEF_NVFP4_MODELS")
    root = Path(override) if override else work_root() / "packs"
    return root / pack_name(size)


def park_joint_head(size: str) -> None:
    src = source_dir(size) / JOINT_HEAD
    parked = work_root() / "decision-sidecar" / size / JOINT_HEAD
    if not src.is_file():
        return
    parked.parent.mkdir(parents=True, exist_ok=True)
    if parked.is_file():
        parked.unlink()
    shutil.move(src, parked)
    log(f"parked {JOINT_HEAD}")


def restore_joint_head(size: str) -> None:
    src = source_dir(size) / JOINT_HEAD
    parked = work_root() / "decision-sidecar" / size / JOINT_HEAD
    if src.is_file() or not parked.is_file():
        return
    shutil.move(parked, src)
    log(f"restored {JOINT_HEAD}")


def card(size: str, backend: str) -> str:
    source = SOURCES[size]
    name = pack_name(size)
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-generation\n"
        f"base_model: {source['id']}\n"
        "license: apache-2.0\n"
        "tags:\n"
        "- axquant\n"
        "- cuda\n"
        "- nvfp4\n"
        "- clef\n"
        "- decision-model\n"
        "---\n\n"
        f"# {name}\n\n"
        "Native AXQuant RTN NVFP4 W4A16 development checkpoint. Eligible language "
        "attention and MLP matrices use E2M1 weights, E4M3FN block scales of 16, "
        "and an FP32 inverse global scale. Activations stay BF16. There is no AWQ "
        "and no activation quantization. The checkpoint uses the public "
        "compressed-tensors NVFP4 layout.\n\n"
        f"Source: [{source['id']}](https://huggingface.co/{source['id']}) at "
        f"`{source['rev']}`. Serialization backend: `{backend}`.\n\n"
        "## Decision model\n\n"
        "Clef is not a chat model. A request is a state plus a schema of typed "
        "questions (`noul`, `choice`, `score`). One prefill returns a probability "
        "for every allowed option. `joint_head.safetensors` is the original BF16 "
        "joint schema head, copied unchanged, with `joint_schema_model.py` and "
        "`joint_head_config.json`. Text generation does not run that head.\n\n"
        "## Precision\n\n"
        "Vision, norms, token embeddings, and the LM head stay at source precision. "
        "The joint schema head is not part of the language index and is not quantized. "
        "This package is development evidence. It is not a Decision Index certificate "
        "and it does not claim parity with the upstream BF16 checkpoint.\n\n"
        "The upstream license is Apache-2.0. See LICENSE and NOTICE.\n"
    )


def attach(size: str, backend: str) -> None:
    src = source_dir(size)
    pack = pack_dir(size)
    parked = work_root() / "decision-sidecar" / size / JOINT_HEAD
    head = src / JOINT_HEAD if (src / JOINT_HEAD).is_file() else parked
    if not head.is_file():
        raise SystemExit(f"missing {JOINT_HEAD}")
    shutil.copy2(head, pack / JOINT_HEAD)
    for name in ("joint_head_config.json", "joint_schema_model.py", "LICENSE"):
        origin = src / name
        if origin.is_file() and not (pack / name).is_file():
            shutil.copy2(origin, pack / name)
    (pack / "NOTICE").write_text(
        NOTICE.format(source=SOURCES[size]["id"]),
        encoding="utf-8",
    )
    (pack / "README.md").write_text(card(size, backend), encoding="utf-8")
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")


def cmd_convert(size: str) -> None:
    source = SOURCES[size]
    src = source_dir(size)
    if not (src / "config.json").is_file() or not any(src.glob("model-*.safetensors")):
        raise SystemExit(f"missing BF16 source {src}")
    pack = pack_dir(size)
    if (pack / "axquant_cuda_manifest.json").is_file():
        log(f"reuse pack {pack}")
        manifest = load_model(pack / "axquant_cuda_manifest.json", CudaPackManifest)
        if manifest.backend != "torch-cuda":
            raise SystemExit(f"refusing non-CUDA manifest backend {manifest.backend}")
        attach(size, manifest.backend)
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    work = work_root() / "work" / size
    work.mkdir(parents=True, exist_ok=True)
    park_joint_head(size)
    try:
        plan = plan_cuda_nvfp4(
            src,
            model_id=source["id"],
            revision=source["rev"],
            allow_unmeasured=True,
        )
        selected = sum(item.method == "nvfp4" for item in plan.allocations)
        preserved = sum(item.method == "preserve" for item in plan.allocations)
        log(f"plan {size}: nvfp4={selected} preserve={preserved}")
        if selected < 1:
            raise SystemExit("NVFP4 plan selected no tensors")
        write_data(work / "plan.json", plan)
        device = os.environ.get("CLEF_NVFP4_DEVICE", "cuda:0")
        manifest = convert_cuda_nvfp4(
            src,
            plan,
            pack,
            device=device,
            rows_per_chunk=int(os.environ.get("CLEF_NVFP4_ROWS", "2048")),
            allow_unmeasured=True,
        )
    finally:
        restore_joint_head(size)
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing non-CUDA manifest backend {manifest.backend}")
    if manifest.plan_sha256 != stable_sha256(plan):
        raise SystemExit("NVFP4 manifest is not bound to this plan")
    for member in manifest.files:
        if file_sha256(pack / member.path) != member.sha256:
            raise SystemExit(f"pack digest mismatch: {member.path}")
    attach(size, manifest.backend)
    head_src = source_dir(size) / JOINT_HEAD
    head_dst = pack / JOINT_HEAD
    if file_sha256(head_src) != file_sha256(head_dst):
        raise SystemExit("joint head bytes changed")
    log(
        f"built {pack_name(size)} backend={manifest.backend} "
        f"quantized={len(manifest.quantized_tensors)}"
    )


def cmd_publish(size: str) -> None:
    from huggingface_hub import HfApi

    pack = pack_dir(size)
    manifest = load_model(pack / "axquant_cuda_manifest.json", CudaPackManifest)
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing to publish backend {manifest.backend}")
    for member in manifest.files:
        if file_sha256(pack / member.path) != member.sha256:
            raise SystemExit(f"pack digest mismatch: {member.path}")
    if not (pack / JOINT_HEAD).is_file() or not (pack / "LICENSE").is_file():
        raise SystemExit("pack is missing the joint head or LICENSE")
    repo = f"AutomatosX/{pack_name(size)}"
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo,
        folder_path=str(pack),
        commit_message=f"Publish {pack_name(size)}",
    )
    log(f"published {repo} {commit.oid}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=["clef", "flash", "all"], required=True)
    parser.add_argument("stage", choices=["convert", "publish"])
    args = parser.parse_args()
    sizes = ["flash", "clef"] if args.pack == "all" else [args.pack]
    stage = cmd_convert if args.stage == "convert" else cmd_publish
    for size in sizes:
        stage(size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
