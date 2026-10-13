#!/usr/bin/env python3
"""CUDA NVFP4 W4A16 pack of tencent/Hy-MT2-30B-A3B from the pinned BF16 snapshot.

Language attention, the dense MLP, routed experts, and the shared expert use
native NVFP4. Norms, embeddings, the LM head, the router, and expert_bias stay
at source precision. Do not requantize an MLX pack. Publish from the Mac
project environment; do not copy a Hub token onto the CUDA host.

  PYTHONPATH=src python scripts/run_hy_mt2_nvfp4.py convert
  PYTHONPATH=src python scripts/run_hy_mt2_nvfp4.py publish
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
    "id": "tencent/Hy-MT2-30B-A3B",
    "rev": "d3ead4dba61c09aac60a261a96ad1df3e705febb",
}
LANE = "NVFP4-W4A16"
PACK_NAME = cuda_pack_name("Hy-MT2-30B-A3B", lane=LANE)
LICENSE_NAME = "LICENSE.txt"


def log(message: str) -> None:
    print(message, flush=True)


def work_root() -> Path:
    default = "/home/devop/work/hy-mt2-nvfp4"
    return Path(os.environ.get("HY_MT2_NVFP4_ROOT", default)).expanduser()


def source_dir() -> Path:
    override = os.environ.get("HY_MT2_NVFP4_SRC")
    if override:
        return Path(override).expanduser()
    return work_root() / "src"


def pack_dir() -> Path:
    override = os.environ.get("HY_MT2_NVFP4_PACK")
    if override:
        return Path(override).expanduser()
    return work_root() / "packs" / PACK_NAME


def card(backend: str) -> str:
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: translation\n"
        f"base_model: {SOURCE['id']}\n"
        "license: apache-2.0\n"
        "tags:\n"
        "- axquant\n"
        "- cuda\n"
        "- nvfp4\n"
        "- hy-mt2\n"
        "---\n\n"
        f"# {PACK_NAME}\n\n"
        "Native AXQuant RTN NVFP4 W4A16 development checkpoint. Eligible language "
        "attention, dense MLP, routed expert, and shared-expert matrices use E2M1 "
        "weights, E4M3FN block scales of 16, and an FP32 inverse global scale. "
        "Activations stay BF16. There is no AWQ and no activation quantization. "
        "The checkpoint uses the public compressed-tensors NVFP4 layout.\n\n"
        f"Source: [{SOURCE['id']}](https://huggingface.co/{SOURCE['id']}) at "
        f"`{SOURCE['rev']}`. Serialization backend: `{backend}`.\n\n"
        "## Precision\n\n"
        "Norms, token embeddings, the LM head, the router, and expert_bias stay "
        "at source precision. This checkpoint has no MTP tensors. Translation "
        "quality was not measured. This package is development evidence. It does "
        "not claim parity with the upstream BF16 checkpoint and it does not "
        "repeat upstream benchmark numbers.\n\n"
        "## Related packs\n\n"
        "| Pack | Role |\n"
        "| --- | --- |\n"
        "| [MLX MXFP4](https://huggingface.co/AutomatosX/"
        "AX-Hy-MT2-30B-A3B-MLX-AXQ-MXFP4) | Apple Silicon AXQ MXFP4. "
        "Norms and expert_bias stay BF16. |\n"
        "| [MLX MXFP8](https://huggingface.co/AutomatosX/"
        "AX-Hy-MT2-30B-A3B-MLX-AXQ-MXFP8) | Apple Silicon AXQ MXFP8. "
        "Same protection. |\n\n"
        "## License\n\n"
        "The upstream work is Apache License 2.0. See LICENSE.txt.\n"
    )


def _reject_local_paths(pack: Path) -> None:
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")


def attach(backend: str) -> None:
    src = source_dir()
    pack = pack_dir()
    origin = src / LICENSE_NAME
    if not origin.is_file():
        raise SystemExit(f"missing source {LICENSE_NAME}")
    shutil.copy2(origin, pack / LICENSE_NAME)
    license_text = (pack / LICENSE_NAME).read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise SystemExit("LICENSE.txt is not Apache-2.0")
    if (pack / "benchmarks").exists() or (pack / "README_CN.md").exists():
        raise SystemExit("pack contains upstream readme or benchmark files")
    (pack / "README.md").write_text(card(backend), encoding="utf-8")
    _reject_local_paths(pack)
    readme = (pack / "README.md").read_text(encoding="utf-8")
    front = readme.split("---", 2)[1]
    if "license: apache-2.0" not in front:
        raise SystemExit("model card front matter is not Apache-2.0")
    if "Translation quality was not measured" not in readme:
        raise SystemExit("model card omits the unmeasured-quality statement")


def _assert_plan(plan) -> None:
    selected = [item for item in plan.allocations if item.method == "nvfp4"]
    if not selected:
        raise SystemExit("NVFP4 plan selected no tensors")
    leaked = [
        item.tensor_name
        for item in selected
        if "expert_bias" in item.tensor_name
        or ".router." in item.tensor_name
        or item.tensor_name.startswith("lm_head")
        or "embed" in item.tensor_name
        or item.tensor_name.endswith("norm.weight")
    ]
    if leaked:
        raise SystemExit(f"NVFP4 plan quantized a protected tensor: {leaked[0]}")
    experts = [item for item in plan.allocations if ".experts." in item.tensor_name]
    if not experts or any(item.method != "nvfp4" for item in experts):
        raise SystemExit("routed experts were not all selected for NVFP4")
    if plan.activation_dtype != "bfloat16":
        raise SystemExit(f"activations are {plan.activation_dtype}, expected bfloat16")
    if plan.model_type != "hy_v3":
        raise SystemExit(f"unexpected model_type {plan.model_type}")
    if plan.revision != SOURCE["rev"]:
        raise SystemExit(f"source revision {plan.revision} is not {SOURCE['rev']}")


def cmd_convert() -> None:
    src = source_dir()
    if not (src / "config.json").is_file() or not (src / "model.safetensors.index.json").is_file():
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
    plan = plan_cuda_nvfp4(
        src,
        model_id=SOURCE["id"],
        revision=SOURCE["rev"],
        allow_unmeasured=True,
    )
    _assert_plan(plan)
    selected = sum(item.method == "nvfp4" for item in plan.allocations)
    preserved = sum(item.method == "preserve" for item in plan.allocations)
    log(f"plan nvfp4={selected} preserve={preserved}")
    write_data(work / "plan.json", plan)
    device = os.environ.get("HY_MT2_NVFP4_DEVICE", "cuda:0")
    manifest = convert_cuda_nvfp4(
        src,
        plan,
        pack,
        device=device,
        rows_per_chunk=int(os.environ.get("HY_MT2_NVFP4_ROWS", "2048")),
        allow_unmeasured=True,
    )
    if manifest.backend != "torch-cuda":
        raise SystemExit(f"refusing non-CUDA manifest backend {manifest.backend}")
    if manifest.plan_sha256 != stable_sha256(plan):
        raise SystemExit("NVFP4 manifest is not bound to this plan")
    config = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "hy_v3":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
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
    attach(manifest.backend)
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
        raise SystemExit("usage: run_hy_mt2_nvfp4.py convert|publish")
    if sys.argv[1] == "convert":
        cmd_convert()
    else:
        cmd_publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
