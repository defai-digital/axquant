#!/usr/bin/env python3
"""CUDA NVFP4 W4A16 pack of the merged Qwen3.5-0.8B-decision BF16 snapshot.

Quantize the merged backbone. Do not requantize an MXFP pack and do not leave
the LoRA unmerged. Vision, MTP, embeddings, norms, and the joint schema head
stay at source precision. Activations stay BF16.

  PYTHONPATH=src python scripts/run_qwen35_decision_nvfp4.py convert
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
    "id": "mlx-community/Qwen3.5-0.8B-decision",
    "rev": "36e166ca10c4d16471a231d66cfc22d80039fe34",
}
PACK_NAME = cuda_pack_name("Qwen3.5-0.8B-decision", lane="NVFP4-W4A16")
JOINT_HEAD = "joint_head.safetensors"
NOTICE = f"""This directory is a quantized CUDA derivative of {SOURCE["id"]}.

Changes from the upstream release:
- The published rank-16 LoRA (scale 1.0) was merged into the BF16 language
  weights before quantization.
- Eligible language attention and MLP weights were quantized to NVFP4 W4A16
  by AXQuant (round-to-nearest, block 16, no AWQ).
- Vision, MTP, norms, and token embeddings stay at source precision.
- joint_head.safetensors is the original BF16 joint schema head, copied unchanged.
- Activations are not quantized.

The upstream work is Apache-2.0. See LICENSE.
This pack does not repeat the upstream OptiQ decision scores.
"""


def log(message: str) -> None:
    print(message, flush=True)


def work_root() -> Path:
    default = "/home/devop/work/qwen35-decision-nvfp4"
    return Path(os.environ.get("QWEN35_DECISION_NVFP4_ROOT", default))


def source_dir() -> Path:
    override = os.environ.get("QWEN35_DECISION_MERGED")
    if override:
        return Path(override)
    return work_root() / "merged"


def sidecar_dir() -> Path:
    override = os.environ.get("QWEN35_DECISION_SIDECAR")
    if override:
        return Path(override)
    return work_root() / "src"


def pack_dir() -> Path:
    override = os.environ.get("QWEN35_DECISION_NVFP4_MODELS")
    root = Path(override) if override else work_root() / "packs"
    return root / PACK_NAME


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
        "- qwen3.5\n"
        "- decision-model\n"
        "---\n\n"
        f"# {PACK_NAME}\n\n"
        "Native AXQuant RTN NVFP4 W4A16 development checkpoint. Eligible language "
        "attention and MLP matrices use E2M1 weights, E4M3FN block scales of 16, "
        "and an FP32 inverse global scale. Activations stay BF16. There is no AWQ "
        "and no activation quantization.\n\n"
        f"Source: [{SOURCE['id']}](https://huggingface.co/{SOURCE['id']}) at "
        f"`{SOURCE['rev']}`. The rank-16 LoRA was merged into the BF16 language "
        f"weights before this quantization. Serialization backend: `{backend}`.\n\n"
        "## Decision model\n\n"
        "This checkpoint scores a joint schema head. It is not a chat model. "
        "`joint_head.safetensors` is the original BF16 head, copied unchanged, with "
        "`joint_head_config.json` and `decision_config.json`. Text generation does "
        "not run that head.\n\n"
        "## Precision\n\n"
        "Vision, MTP, norms, and token embeddings stay at source precision. "
        "This package is development evidence. It does not claim the upstream "
        "OptiQ decision scores, MTP acceleration, or parity with the BF16 checkpoint.\n\n"
        "The upstream license is Apache-2.0. See LICENSE and NOTICE.\n"
    )


def attach(backend: str) -> None:
    src = sidecar_dir()
    pack = pack_dir()
    head = src / JOINT_HEAD
    if not head.is_file():
        raise SystemExit(f"missing {JOINT_HEAD} in {src}")
    shutil.copy2(head, pack / JOINT_HEAD)
    for name in (
        "joint_head_config.json",
        "decision_config.json",
        "joint_schema_model.py",
        "LICENSE",
        "chat_template.jinja",
    ):
        origin = src / name
        if origin.is_file():
            shutil.copy2(origin, pack / name)
    if not (pack / "LICENSE").is_file() or not (pack / "decision_config.json").is_file():
        raise SystemExit("pack is missing LICENSE or decision_config.json")
    (pack / "NOTICE").write_text(NOTICE, encoding="utf-8")
    (pack / "README.md").write_text(card(backend), encoding="utf-8")
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")
        if "0.772" in text:
            raise SystemExit(f"upstream score leaked into {path.name}")


def cmd_convert() -> None:
    src = source_dir()
    if (src / "adapters.safetensors").is_file():
        raise SystemExit("refusing to quantize an unmerged LoRA snapshot")
    if not (src / "lora_merge.json").is_file():
        raise SystemExit(f"missing lora_merge.json in {src}")
    if not (src / "config.json").is_file() or not (src / "model.safetensors.index.json").is_file():
        raise SystemExit(f"missing merged BF16 source {src}")
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
    mtp_quantized = [
        item.tensor_name
        for item in plan.allocations
        if item.tensor_name.startswith("mtp.") and item.method == "nvfp4"
    ]
    if mtp_quantized:
        raise SystemExit("NVFP4 plan quantized an MTP tensor")
    write_data(work / "plan.json", plan)
    device = os.environ.get("QWEN35_DECISION_NVFP4_DEVICE", "cuda:0")
    manifest = convert_cuda_nvfp4(
        src,
        plan,
        pack,
        device=device,
        rows_per_chunk=int(os.environ.get("QWEN35_DECISION_NVFP4_ROWS", "2048")),
        allow_unmeasured=True,
    )
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
    if file_sha256(sidecar_dir() / JOINT_HEAD) != file_sha256(pack / JOINT_HEAD):
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
        raise SystemExit("usage: run_qwen35_decision_nvfp4.py convert|publish")
    if sys.argv[1] == "convert":
        cmd_convert()
    else:
        cmd_publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
