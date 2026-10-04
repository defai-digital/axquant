#!/usr/bin/env python3
"""Build native NVFP4A16 Nemotron packs with integrated, source-precision MTP.

Run with the CUDA extra on the factory host. CPU serialization emits the same
checkpoint format and records numpy-reference; it is not CUDA runtime evidence.
Publish only after the four MLX campaign packs have immutable Hub receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from axquant.cuda import convert_cuda_nvfp4, plan_cuda_nvfp4  # noqa: E402
from axquant.factory import FACTORY_MODELS, require_factory_host  # noqa: E402
from axquant.schema.cuda import CudaPackManifest, CudaQuantizationPlan  # noqa: E402
from axquant.serde import (  # noqa: E402
    file_sha256,
    load_model,
    read_data,
    stable_sha256,
    write_data,
)
from scripts import run_nemotron3_mx_mtp as mlx_campaign  # noqa: E402

MTP_COUNTS = {"lightning": 270, "super": 1040}
KEEP_PATTERNS = [
    "mtp.*",
    "backbone.layers.*.mixer.in_proj.*",
    "backbone.layers.*.mixer.out_proj.*",
    "backbone.layers.*.mixer.q_proj.*",
    "backbone.layers.*.mixer.k_proj.*",
    "backbone.layers.*.mixer.v_proj.*",
    "backbone.layers.*.mixer.o_proj.*",
    "backbone.layers.*.mixer.shared_experts.*",
    "backbone.layers.*.mixer.fc1_latent_proj.*",
    "backbone.layers.*.mixer.fc2_latent_proj.*",
]
COLLECTIONS = {
    "cuda": "AutomatosX/cuda-6ac1b59d9199d121c6c7e2e3",
    "nvfp4": "AutomatosX/nvfp4-6ac1a489659e8a86ec4280fa",
    "mtp": mlx_campaign.COLLECTIONS["mtp"],
    "nemotron": mlx_campaign.COLLECTIONS["nemotron"],
}


def repo_id(family: str) -> str:
    return "AutomatosX/" + str(mlx_campaign.PACKS[f"{family}-mxfp4"]["repo"]).replace(
        "MLX-AXQ-MXFP4", "CUDA-AXQ-NVFP4"
    )


def output_path(family: str) -> Path:
    return Path(FACTORY_MODELS) / repo_id(family).split("/", 1)[1]


def verify_integrated_mtp(
    source: Path, output: Path, plan: CudaQuantizationPlan, expected_count: int
) -> list[dict[str, Any]]:
    import torch
    from safetensors import safe_open

    mtp = [item for item in plan.allocations if item.tensor_name.startswith("mtp.")]
    if len(mtp) != expected_count or any(item.method != "preserve" for item in mtp):
        raise SystemExit("NVFP4 MTP inventory or protection policy is invalid")
    source_config, output_config = (
        read_data(source / "config.json"),
        read_data(output / "config.json"),
    )
    for name in ("num_nextn_predict_layers", "mtp_hybrid_override_pattern"):
        if source_config.get(name) != output_config.get(name):
            raise SystemExit("NVFP4 conversion changed the source MTP configuration")
    index = read_data(output / "model.safetensors.index.json")["weight_map"]
    records: list[dict[str, Any]] = []
    by_file: dict[str, list[Any]] = {}
    for item in mtp:
        if index.get(item.tensor_name) != item.source_file:
            raise SystemExit("integrated MTP is missing from the CUDA checkpoint index")
        by_file.setdefault(item.source_file, []).append(item)
    for member, allocations in sorted(by_file.items()):
        with (
            safe_open(source / member, framework="pt", device="cpu") as original,
            safe_open(output / member, framework="pt", device="cpu") as converted,
        ):
            for item in sorted(allocations, key=lambda entry: entry.tensor_name):
                before = original.get_tensor(item.tensor_name)
                after = converted.get_tensor(item.tensor_name)
                original_bytes = before.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                converted_bytes = after.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                if (
                    before.dtype != after.dtype
                    or before.shape != after.shape
                    or original_bytes != converted_bytes
                ):
                    raise SystemExit("integrated MTP tensor payload changed during conversion")
                records.append(
                    {
                        "name": item.tensor_name,
                        "shape": list(item.shape),
                        "dtype": item.dtype,
                        "data_sha256": hashlib.sha256(original_bytes).hexdigest(),
                    }
                )
    return sorted(records, key=lambda record: record["name"])


def card(family: str, backend: str) -> str:
    source = mlx_campaign.SOURCES[family]
    name = repo_id(family).split("/", 1)[1]
    license_name = "openmdw-1.1" if family == "lightning" else "nvidia-nemotron-open-model-license"
    license_url = (
        "https://openmdw.ai/license/1-1/" if family == "lightning" else mlx_campaign.LICENSE_URL
    )
    return (
        "---\nlibrary_name: transformers\npipeline_tag: text-generation\n"
        f"base_model: {source['id']}\nlicense: other\nlicense_name: {license_name}\n"
        f"license_link: {license_url}\ntags:\n- axquant\n- cuda\n- nvfp4\n- mtp\n"
        f"---\n\n# {name}\n\n"
        "Native AXQuant RTN NVFP4A16 development checkpoint, exported from the NVIDIA BF16 "
        "source with no AWQ or third-party quantizer. Weight blocks contain 16 elements, "
        "E2M1 weights use E4M3FN block scales and FP32 inverse global scales. Activations "
        "remain BF16. The checkpoint uses the public compressed-tensors NVFP4 format.\n\n"
        f"Source: [{source['id']}](https://huggingface.co/{source['id']}) at "
        f"`{source['revision']}`. Serialization backend: `{backend}`. CPU serialization "
        "is format evidence only; it does not establish GPU execution or speed.\n\n"
        "## Precision and MTP\n\n"
        "Eligible text expert and MLP matrices use NVFP4. Embeddings, output head, norms, "
        "routers, Mamba state and attention projections, shared experts, latent projections, "
        "and all integrated `mtp.*` tensors keep source precision. The original config, "
        "tensor names and main-index MTP layout remain available to compatible CUDA runtimes. "
        "`axquant_nemotron_release.json` records byte-preservation checks for every MTP tensor. "
        "MTP runtime compatibility is unverified; the name means trained MTP weights are "
        "packaged. No runtime, quality, or throughput certificate is claimed.\n\n"
        "## Consumers\n\n"
        "A consumer must support Nemotron-H, compressed-tensors NVFP4A16, and the source "
        "integrated MTP layout to execute all components. Runtime selection, kernels and "
        "speculative-decoding settings are runtime responsibilities. The AXQuant plan, "
        "manifest and SHA256 inventory are included for reproducible artifact inspection.\n\n"
        "The source license and available notices are included.\n"
    )


def build(family: str, device: str) -> None:
    require_factory_host(socket.gethostname())
    source, output = mlx_campaign.source_path(family), output_path(family)
    spec = mlx_campaign.SOURCES[family]
    work = mlx_campaign.WORK / family / "nvfp4"
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    if plan_path.exists():
        plan = load_model(plan_path, CudaQuantizationPlan)
    else:
        plan = plan_cuda_nvfp4(
            source,
            model_id=spec["id"],
            revision=spec["revision"],
            keep_patterns=KEEP_PATTERNS,
            allow_unmeasured=True,
        )
        write_data(plan_path, plan)
    if plan.model_id != spec["id"] or plan.revision != spec["revision"]:
        raise SystemExit("NVFP4 source identity differs from the pinned campaign source")
    if (output / "axquant_cuda_manifest.json").exists():
        manifest = load_model(output / "axquant_cuda_manifest.json", CudaPackManifest)
    else:
        print(f"converting {family} to native NVFP4A16", flush=True)
        manifest = convert_cuda_nvfp4(source, plan, output, device=device, allow_unmeasured=True)
    if manifest.plan_sha256 != stable_sha256(plan):
        raise SystemExit("NVFP4 manifest differs from the bound campaign plan")
    for member in manifest.files:
        if file_sha256(output / member.path) != member.sha256:
            raise SystemExit("NVFP4 output differs from its manifest")
    mtp = verify_integrated_mtp(source, output, plan, MTP_COUNTS[family])
    for name in ("LICENSE", "NOTICE", "NOTICE.md", "LICENSE.txt"):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    if family == "super":
        shutil.copyfile(mlx_campaign.LICENSE_PDF, output / mlx_campaign.LICENSE_PDF.name)
    (output / "README.md").write_text(card(family, manifest.backend), encoding="utf-8")
    write_data(
        output / "axquant_nemotron_release.json",
        {
            "repo": repo_id(family),
            "source_model": spec["id"],
            "source_revision": spec["revision"],
            "format": "nvfp4",
            "activation_precision": 16,
            "backend": manifest.backend,
            "development_only": True,
            "runtime_verified": False,
            "quality_certified": False,
            "manifest_sha256": file_sha256(output / "axquant_cuda_manifest.json"),
            "mtp_packaging": "integrated-source-layout",
            "mtp_runtime_compatibility": "unverified",
            "mtp_tensor_count": len(mtp),
            "mtp_tensor_inventory_sha256": stable_sha256(mtp),
            "mtp_tensors": mtp,
        },
    )
    hashes = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        if path.suffix in {".json", ".md", ".py", ".txt"}:
            content = path.read_text(encoding="utf-8")
            if "/Volumes/" in content or "/Users/" in content:
                raise SystemExit("publication blocked: local paths in a CUDA artifact")
        hashes[path.relative_to(output).as_posix()] = file_sha256(path)
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8"
    )
    print(
        json.dumps({"pack": repo_id(family), "mtp_tensor_count": len(mtp), "status": "built"}),
        flush=True,
    )


def publish(family: str) -> None:
    require_factory_host(socket.gethostname())
    from huggingface_hub import HfApi, hf_hub_download

    for key in mlx_campaign.PACKS:
        spec = mlx_campaign.PACKS[key]
        receipt = mlx_campaign.WORK / spec["family"] / spec["precision"] / "publication.json"
        if not receipt.is_file() or not read_data(receipt).get("revision"):
            raise SystemExit("publish the four MLX campaign packs before CUDA NVFP4")
    output = output_path(family)
    release = read_data(output / "axquant_nemotron_release.json")
    if release.get("development_only") is not True or release.get("runtime_verified") is not False:
        raise SystemExit("NVFP4 publication requires explicit development evidence labels")
    manifest = load_model(output / "axquant_cuda_manifest.json", CudaPackManifest)
    if release["manifest_sha256"] != file_sha256(output / "axquant_cuda_manifest.json"):
        raise SystemExit("NVFP4 publication receipt is stale")
    hashes = {}
    for row in (output / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = row.split("  ", 1)
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output) or name in hashes:
            raise SystemExit("invalid NVFP4 checksum inventory")
        if file_sha256(path) != digest:
            raise SystemExit("NVFP4 checkpoint checksum changed")
        hashes[name] = digest
    hashes["SHA256SUMS.txt"] = file_sha256(output / "SHA256SUMS.txt")
    if set(hashes) != {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}:
        raise SystemExit("NVFP4 checksum inventory is incomplete")
    for member in manifest.files:
        if hashes.get(member.path) != member.sha256:
            raise SystemExit("NVFP4 checksum inventory differs from its manifest")
    api = HfApi()
    target = repo_id(family)
    api.create_repo(target, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=target,
        folder_path=str(output),
        commit_message="Publish native AXQuant NVFP4A16 development checkpoint with preserved MTP",
    )
    for slug in COLLECTIONS.values():
        api.add_collection_item(
            collection_slug=slug,
            item_id=target,
            item_type="model",
            note="Native AXQuant NVFP4A16 development pack; MTP preserved, execution unverified.",
            exists_ok=True,
        )
    info = HfApi(token=False).model_info(target, revision=commit.oid, files_metadata=True)
    if info.private or {item.rfilename for item in info.siblings} - {".gitattributes"} != set(
        hashes
    ):
        raise SystemExit("anonymous NVFP4 Hub inventory differs from the local pack")
    for item in info.siblings:
        if item.rfilename == ".gitattributes":
            continue
        if item.lfs is not None:
            if item.lfs.sha256 != hashes[item.rfilename]:
                raise SystemExit("Hub NVFP4 weight checksum mismatch")
        else:
            path = hf_hub_download(
                target,
                item.rfilename,
                revision=info.sha,
                token=False,
                cache_dir=str(mlx_campaign.HF_HOME / "hub"),
            )
            if file_sha256(path) != hashes[item.rfilename]:
                raise SystemExit("anonymous Hub NVFP4 file checksum mismatch")
    for slug in COLLECTIONS.values():
        if not any(item.item_id == target for item in api.get_collection(slug).items):
            raise SystemExit("published NVFP4 pack is missing from a campaign collection")
    write_data(
        mlx_campaign.WORK / family / "nvfp4/publication.json",
        {"repo_id": target, "revision": info.sha, "files_sha256": hashes},
    )
    print(f"published https://huggingface.co/{target} revision={info.sha}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", *mlx_campaign.SOURCES])
    parser.add_argument("--stage", choices=["build", "publish", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    for family in mlx_campaign.SOURCES if args.target == "all" else [args.target]:
        if args.stage in {"build", "all"}:
            build(family, args.device)
        if args.stage in {"publish", "all"}:
            publish(family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
