#!/usr/bin/env python3
"""CUDA NVFP4 W4A16 pack of ukisai/Swift-1.5-Qwen3.8-27b from the pinned BF16 snapshot.

Language attention and MLP matrices use native NVFP4. Vision, MTP, norms,
embeddings, and the LM head stay at source precision. Do not requantize an MLX
pack. The Swift Open License v1.0 travels with the pack. Publish from the Mac
project environment; do not copy a Hub token onto the CUDA host.

  PYTHONPATH=src python scripts/run_swift15_qwen38_nvfp4.py convert
  PYTHONPATH=src python scripts/run_swift15_qwen38_nvfp4.py publish
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
    "id": "ukisai/Swift-1.5-Qwen3.8-27b",
    "rev": "b4c84d42903a8646b25857eb2827288d92ed85a4",
}
LANE = "NVFP4-W4A16"
PACK_NAME = cuda_pack_name("Swift-1.5-Qwen3.8-27B", lane=LANE, mtp=True)
LICENSE_FILES = ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE")
NOTICE = f"""AXQuant changes

This directory is a quantized CUDA derivative of {SOURCE["id"]} at {SOURCE["rev"]}.
AutomatosX changed the weight files:
- Eligible language attention and MLP weights were quantized to NVFP4 W4A16
  (round-to-nearest, block 16, no AWQ).
- Vision, MTP, norms, token embeddings, and the LM head stay at source precision.
- Activations are not quantized.
- MTP acceleration was not measured. Generation quality was not measured.

The Swift contribution remains under the Swift Open License v1.0 (LICENSE).
The Qwen3.8-27B base remains Apache License 2.0 (LICENSE-APACHE-2.0).
Commercial use by a legal entity with gross revenue of US$1,000,000 or more
in its most recent fiscal year is not licensed.
"""


def log(message: str) -> None:
    print(message, flush=True)


def work_root() -> Path:
    default = "/home/devop/work/swift-nvfp4"
    return Path(os.environ.get("SWIFT_NVFP4_ROOT", default)).expanduser()


def source_dir() -> Path:
    override = os.environ.get("SWIFT_NVFP4_SRC")
    if override:
        return Path(override).expanduser()
    return work_root() / "src"


def pack_dir() -> Path:
    override = os.environ.get("SWIFT_NVFP4_PACK")
    if override:
        return Path(override).expanduser()
    return work_root() / "packs" / PACK_NAME


def card(backend: str) -> str:
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: image-text-to-text\n"
        f"base_model: {SOURCE['id']}\n"
        "license: other\n"
        "license_name: swift-open-license-1.0\n"
        "license_link: LICENSE\n"
        "tags:\n"
        "- axquant\n"
        "- cuda\n"
        "- nvfp4\n"
        "- qwen3.5\n"
        "- mtp\n"
        "---\n\n"
        f"# {PACK_NAME}\n\n"
        "Native AXQuant RTN NVFP4 W4A16 development checkpoint. Eligible language "
        "attention and MLP matrices use E2M1 weights, E4M3FN block scales of 16, "
        "and an FP32 inverse global scale. Activations stay BF16. There is no AWQ "
        "and no activation quantization. The checkpoint uses the public "
        "compressed-tensors NVFP4 layout.\n\n"
        f"Source: [{SOURCE['id']}](https://huggingface.co/{SOURCE['id']}) at "
        f"`{SOURCE['rev']}`. Serialization backend: `{backend}`.\n\n"
        "## Precision\n\n"
        "Vision, MTP, norms, token embeddings, and the LM head stay at source "
        "precision. MTP weights are packaged. MTP acceleration was not measured. "
        "Generation quality and vision quality were not measured. This package is "
        "development evidence. It does not claim parity with the upstream BF16 "
        "checkpoint and it does not repeat upstream benchmark numbers.\n\n"
        "## Related packs\n\n"
        "| Pack | Role |\n"
        "| --- | --- |\n"
        "| [MLX MXFP4](https://huggingface.co/AutomatosX/"
        "AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP) | Apple Silicon AXQ "
        "MXFP4 with MTP. Vision, MTP, and norms stay BF16. |\n"
        "| [MLX MXFP8](https://huggingface.co/AutomatosX/"
        "AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-MXFP8-MTP) | Apple Silicon AXQ "
        "MXFP8 with MTP. Same protection. |\n\n"
        "## License\n\n"
        "The Swift contribution in this checkpoint is licensed under the Swift Open "
        "License v1.0 ([`LICENSE`](LICENSE)). Quantized weights are an Object-form "
        f"derivative of [{SOURCE['id']}](https://huggingface.co/{SOURCE['id']}). "
        "Commercial use by a legal entity with gross revenue of US$1,000,000 or more "
        "in its most recent fiscal year is not licensed under that text. The "
        "Qwen3.8-27B base remains Apache License 2.0 "
        "([`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0)). The names UkisAI and Swift "
        "identify the upstream work.\n"
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
    for name in LICENSE_FILES:
        origin = src / name
        if not origin.is_file():
            raise SystemExit(f"missing source {name}")
        if name == "NOTICE":
            upstream = origin.read_text(encoding="utf-8", errors="replace").rstrip()
            body = upstream if "AXQuant changes" in upstream else f"{upstream}\n\n{NOTICE}"
            (pack / name).write_text(body.rstrip() + "\n", encoding="utf-8")
            continue
        shutil.copy2(origin, pack / name)
    if (pack / "benchmarks").exists():
        raise SystemExit("pack contains upstream benchmark files")
    (pack / "README.md").write_text(card(backend), encoding="utf-8")
    _reject_local_paths(pack)
    readme = (pack / "README.md").read_text(encoding="utf-8")
    front = readme.split("---", 2)[1]
    if "license: other" not in front or "swift-open-license-1.0" not in front:
        raise SystemExit("model card does not carry the Swift Open License")
    if "license: apache-2.0" in front:
        raise SystemExit("model card front matter is Apache-2.0")
    if "US$1,000,000" not in readme:
        raise SystemExit("model card omits the commercial-use threshold")
    if "Swift Open License v1.0" not in (pack / "LICENSE").read_text(encoding="utf-8"):
        raise SystemExit("LICENSE text is not the Swift Open License")
    if not (pack / "LICENSE-APACHE-2.0").is_file():
        raise SystemExit("Apache-2.0 base license is missing")


def _assert_plan(plan) -> None:
    selected = [item for item in plan.allocations if item.method == "nvfp4"]
    if not selected:
        raise SystemExit("NVFP4 plan selected no tensors")
    leaked = [
        item.tensor_name
        for item in selected
        if item.tensor_name.startswith("mtp.") or ".visual." in item.tensor_name
    ]
    if leaked:
        raise SystemExit(f"NVFP4 plan quantized a protected tensor: {leaked[0]}")
    if plan.activation_dtype != "bfloat16":
        raise SystemExit(f"activations are {plan.activation_dtype}, expected bfloat16")
    if plan.revision != SOURCE["rev"]:
        raise SystemExit(f"source revision {plan.revision} is not {SOURCE['rev']}")


def cmd_convert() -> None:
    src = source_dir()
    if not (src / "config.json").is_file() or not (src / "model.safetensors.index.json").is_file():
        raise SystemExit(f"missing BF16 source {src}")
    if (src / "benchmarks").exists():
        raise SystemExit(f"refusing source that still contains benchmarks: {src}")
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
    device = os.environ.get("SWIFT_NVFP4_DEVICE", "cuda:0")
    manifest = convert_cuda_nvfp4(
        src,
        plan,
        pack,
        device=device,
        rows_per_chunk=int(os.environ.get("SWIFT_NVFP4_ROWS", "2048")),
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
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    mtp_layers = text_cfg.get("mtp_num_hidden_layers", config.get("mtp_num_hidden_layers"))
    if mtp_layers != 1:
        raise SystemExit("MTP layer count was not preserved")
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
        raise SystemExit("usage: run_swift15_qwen38_nvfp4.py convert|publish")
    if sys.argv[1] == "convert":
        cmd_convert()
    else:
        cmd_publish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
