#!/usr/bin/env python3
"""Build the Clef-Omni CUDA NVFP4 W4A16 pack from the pinned BF16 snapshot.

Eligible thinker and talker attention, MLP, and expert matrices use native
NVFP4. Vision, the audio encoder, code2wav, norms, embeddings, LM heads, and
the joint schema head stay at source precision. Activations stay BF16.
Conversion requires CUDA and refuses a numpy-reference manifest.

  PYTHONPATH=src python scripts/run_clef_omni_nvfp4.py convert
"""

from __future__ import annotations

import json
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

SOURCE = {
    "id": "Cloudflare/clef-omni",
    "rev": "0db1cd2607d76a7bdb2a382f659e7b313079f84b",
}
JOINT_HEAD = "joint_head.safetensors"
LANE = "NVFP4-W4A16"
PACK_NAME = cuda_pack_name("Clef-Omni-30B-A3B", lane=LANE)


def log(message: str) -> None:
    print(message, flush=True)


def work_root() -> Path:
    default = "/home/devop/work/clef-omni-nvfp4"
    return Path(os.environ.get("CLEF_OMNI_NVFP4_ROOT", default)).expanduser()


def source_dir() -> Path:
    override = os.environ.get("CLEF_OMNI_SRC")
    if override:
        return Path(override)
    return work_root() / "src"


def pack_dir() -> Path:
    override = os.environ.get("CLEF_OMNI_NVFP4_MODELS")
    root = Path(override) if override else work_root() / "packs"
    return root / PACK_NAME


def park_joint_head() -> None:
    src = source_dir() / JOINT_HEAD
    parked = work_root() / "decision-sidecar" / JOINT_HEAD
    if not src.is_file():
        return
    parked.parent.mkdir(parents=True, exist_ok=True)
    if parked.is_file():
        parked.unlink()
    shutil.move(src, parked)
    log(f"parked {JOINT_HEAD}")


def restore_joint_head() -> None:
    src = source_dir() / JOINT_HEAD
    parked = work_root() / "decision-sidecar" / JOINT_HEAD
    if src.is_file() or not parked.is_file():
        return
    shutil.move(parked, src)
    log(f"restored {JOINT_HEAD}")


def card(backend: str) -> str:
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-generation\n"
        f"base_model: {SOURCE['id']}\n"
        "license: apache-2.0\n"
        "tags:\n"
        "- axquant\n"
        "- cuda\n"
        "- nvfp4\n"
        "- clef\n"
        "- clef-omni\n"
        "- decision-model\n"
        "- qwen3-omni\n"
        "---\n\n"
        f"# {PACK_NAME}\n\n"
        "Native AXQuant RTN NVFP4 W4A16 development checkpoint. Eligible thinker "
        "and talker attention, MLP, and expert matrices use E2M1 weights, E4M3FN "
        "block scales of 16, and an FP32 inverse global scale. Activations stay "
        "BF16. There is no W4A4, no AWQ, and no activation quantization. The "
        "checkpoint uses the public compressed-tensors NVFP4 layout.\n\n"
        f"Source: [{SOURCE['id']}](https://huggingface.co/{SOURCE['id']}) at "
        f"`{SOURCE['rev']}`. Serialization backend: `{backend}`.\n\n"
        "## Decision model\n\n"
        "Clef-Omni is not a chat model. A request is a state plus a schema of typed "
        "questions (`noul`, `choice`, `score`). One prefill returns a probability "
        "for every allowed option. `joint_head.safetensors` is the original BF16 "
        "joint schema head, copied unchanged, with `joint_schema_model.py` and "
        "`joint_head_config.json`. Text generation does not run that head. Upstream "
        "`load_release_model` does not load the talker or code2wav, and it does not "
        "load this NVFP4 checkpoint.\n\n"
        "## Precision\n\n"
        "Vision, the audio encoder, code2wav, norms, token embeddings, and the LM "
        "heads stay at source precision. Talker experts are quantized with the "
        "thinker trunk. Speech output was not measured. The joint schema head is "
        "not part of the language index and is not quantized. This package is "
        "development evidence. It is not a Decision Index certificate, and it does "
        "not claim parity with the upstream BF16 checkpoint. No CUDA decision-head "
        "forward was run.\n\n"
        "The upstream license is Apache-2.0. See LICENSE and NOTICE.\n"
    )


def attach(backend: str) -> None:
    src = source_dir()
    pack = pack_dir()
    parked = work_root() / "decision-sidecar" / JOINT_HEAD
    head = src / JOINT_HEAD if (src / JOINT_HEAD).is_file() else parked
    if not head.is_file():
        raise SystemExit(f"missing {JOINT_HEAD}")
    shutil.copy2(head, pack / JOINT_HEAD)
    sidecars = (
        "joint_head_config.json",
        "joint_schema_model.py",
        "LICENSE",
        "chat_template.jinja",
    )
    for name in sidecars:
        origin = src / name
        if origin.is_file() and not (pack / name).is_file():
            shutil.copy2(origin, pack / name)
    notice = f"""This directory is a quantized CUDA derivative of {SOURCE["id"]}.

Changes from the upstream BF16 release:
- Eligible thinker and talker attention, MLP, and expert weights were quantized
  to NVFP4 W4A16 by AXQuant (round-to-nearest, block 16, no AWQ).
- Vision, the audio encoder, code2wav, norms, token embeddings, and the LM heads
  stay at source precision.
- joint_head.safetensors is the original BF16 joint schema head, copied unchanged.
- Activations are not quantized. Speech output was not measured.

The upstream work is Apache-2.0. See LICENSE.
"""
    (pack / "NOTICE").write_text(notice, encoding="utf-8")
    (pack / "README.md").write_text(card(backend), encoding="utf-8")
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")


def cmd_convert() -> None:
    src = source_dir()
    if not (src / "config.json").is_file() or not any(src.glob("model-*.safetensors")):
        raise SystemExit(f"missing BF16 source {src}")
    pack = pack_dir()
    if (pack / "axquant_cuda_manifest.json").is_file():
        log(f"reuse pack {pack}")
        manifest = load_model(pack / "axquant_cuda_manifest.json", CudaPackManifest)
        if manifest.backend != "torch-cuda":
            raise SystemExit(f"refusing non-CUDA manifest backend {manifest.backend}")
        attach(manifest.backend)
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    work = work_root() / "work"
    work.mkdir(parents=True, exist_ok=True)
    park_joint_head()
    try:
        plan = plan_cuda_nvfp4(
            src,
            model_id=SOURCE["id"],
            revision=SOURCE["rev"],
            allow_unmeasured=True,
        )
        selected = sum(item.method == "nvfp4" for item in plan.allocations)
        preserved = sum(item.method == "preserve" for item in plan.allocations)
        log(f"plan nvfp4={selected} preserve={preserved}")
        if selected < 1:
            raise SystemExit("NVFP4 plan selected no tensors")
        write_data(work / "plan.json", plan)
        device = os.environ.get("CLEF_OMNI_NVFP4_DEVICE", "cuda:0")
        manifest = convert_cuda_nvfp4(
            src,
            plan,
            pack,
            device=device,
            rows_per_chunk=int(os.environ.get("CLEF_OMNI_NVFP4_ROWS", "2048")),
            allow_unmeasured=True,
        )
    finally:
        restore_joint_head()
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing non-CUDA manifest backend {manifest.backend}")
    if manifest.plan_sha256 != stable_sha256(plan):
        raise SystemExit("NVFP4 manifest is not bound to this plan")
    config = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    group = config["quantization_config"]["config_groups"]["nvfp4"]
    weights = group["weights"]
    if weights["num_bits"] != 4 or weights["group_size"] != 16:
        raise SystemExit("pack is not NVFP4 group 16")
    if group.get("input_activations") not in (None, {}):
        raise SystemExit("NVFP4 pack must keep activations unquantized")
    if manifest.activation_dtype != "bfloat16":
        raise SystemExit(f"activations are {manifest.activation_dtype}, expected bfloat16")
    for member in manifest.files:
        if file_sha256(pack / member.path) != member.sha256:
            raise SystemExit(f"pack digest mismatch: {member.path}")
    attach(manifest.backend)
    head_src = source_dir() / JOINT_HEAD
    if file_sha256(head_src) != file_sha256(pack / JOINT_HEAD):
        raise SystemExit("joint head bytes changed")
    log(f"built {PACK_NAME} backend={manifest.backend} quantized={len(manifest.quantized_tensors)}")


def cmd_publish() -> None:
    from huggingface_hub import HfApi

    pack = pack_dir()
    manifest = load_model(pack / "axquant_cuda_manifest.json", CudaPackManifest)
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing to publish backend {manifest.backend}")
    for member in manifest.files:
        if file_sha256(pack / member.path) != member.sha256:
            raise SystemExit(f"pack digest mismatch: {member.path}")
    if not (pack / JOINT_HEAD).is_file() or not (pack / "LICENSE").is_file():
        raise SystemExit("pack is missing the joint head or LICENSE")
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
    if len(sys.argv) != 2 or sys.argv[1] not in {"convert", "publish"}:
        raise SystemExit("usage: run_clef_omni_nvfp4.py convert|publish")
    if sys.argv[1] == "convert":
        cmd_convert()
    else:
        cmd_publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
