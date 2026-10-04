#!/usr/bin/env python3
"""Build and publish four development Nemotron MLX MXFP4/MXFP8 packs.

Run on df-macstudio-m2 after the pinned BF16 sources are downloaded:

  PYTHONPATH=src .venv/bin/python scripts/run_nemotron3_mx_mtp.py all --stage build
  PYTHONPATH=src .venv/bin/python scripts/run_nemotron3_mx_mtp.py all --stage load
  PYTHONPATH=src .venv/bin/python scripts/run_nemotron3_mx_mtp.py all --stage publish
  PYTHONPATH=src .venv/bin/python scripts/run_nemotron3_mx_mtp.py lightning-mxfp4

The converter selects bounded-memory streaming automatically when source bytes
exceed available RAM. Hub collection updates are additive and never remove items.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import FACTORY_MODELS, require_factory_host  # noqa: E402
from axquant.schema.artifacts import ArtifactManifest, NemotronMtpSidecarManifest  # noqa: E402
from axquant.schema.planning import QuantizationPlan  # noqa: E402
from axquant.serde import file_sha256, load_model, stable_sha256, write_data  # noqa: E402

BASE = Path(FACTORY_MODELS) / "nemotron-mxfp4-mxfp8-20261004"
SOURCE_ROOT = BASE / "sources"
WORK = BASE / "work"
LICENSE_PDF = BASE / "NVIDIA-Nemotron-Open-Model-License.pdf"
LICENSE_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/"
HF_HOME = Path(os.environ.get("HF_HOME", "/Volumes/Ext16TR0/huggingface"))
HUB_ROOT = "AutomatosX"

SOURCES: dict[str, dict[str, str]] = {
    "lightning": {
        "id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "revision": "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
        "directory": "lightning",
        "display": "Nemotron 3.5 Lightning 30B-A3B",
    },
    "super": {
        "id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "revision": "2dc98e2afe4face0e4ce40972a915c45368bd34a",
        "directory": "super",
        "display": "Nemotron 3 Super 120B-A12B",
    },
}
FORMATS = {
    "mxfp4": {"q_mode": "mxfp4", "recipe": "nemotron3-axq-mxfp4-mtp-v0.1.yaml"},
    "mxfp8": {"q_mode": "mxfp8", "recipe": "nemotron3-axq-mxfp8-mtp-v0.1.yaml"},
}
PACKS = {
    f"{family}-{precision}": {
        "family": family,
        "precision": precision,
        "repo": (
            f"AX-Nemotron-3.5-Lightning-30B-A3B-MLX-AXQ-{precision.upper()}-MTP"
            if family == "lightning"
            else f"AX-Nemotron-3-Super-120B-A12B-MLX-AXQ-{precision.upper()}-MTP"
        ),
    }
    for family in SOURCES
    for precision in FORMATS
}
COLLECTIONS = {
    "mxfp4": "AutomatosX/mxfp4-6ac15f9e6eda0bc49dee0a32",
    "mxfp8": "AutomatosX/mxfp8-6ac15fa0b73470ecf715faec",
    "mtp": "AutomatosX/mtp-6ac1a4e2daa11b99c2da9338",
    "nemotron": "AutomatosX/nemotron-6ac1b99e003f6638a7513f59",
    "mlx": "AutomatosX/mlx-6ab432f9724c4d8a1037df1c",
}


def command(args: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = {
        **os.environ,
        "HF_HOME": str(HF_HOME),
        "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": str(HF_HOME / "xet"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if value
        ),
    }
    merged_env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if env is not None:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"command={Path(args[0]).name}\nstatus=running\n")
        with subprocess.Popen(
            args,
            cwd=ROOT,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                for private_path in (str(ROOT), str(BASE), str(HF_HOME)):
                    line = line.replace(private_path, "<local-path>")
                stream.write(line)
                stream.flush()
            return_code = process.wait()
        status = "passed" if return_code == 0 else "failed"
        stream.write(f"status={status}\nexit_code={return_code}\n")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)


def python_bin() -> str:
    return str(ROOT / ".venv" / "bin" / "python")


def axquant_bin() -> str:
    return str(ROOT / ".venv" / "bin" / "axquant")


def source_path(family: str) -> Path:
    return SOURCE_ROOT / SOURCES[family]["directory"]


def pack_path(key: str) -> Path:
    return Path(FACTORY_MODELS) / str(PACKS[key]["repo"])


def _card(pack_key: str) -> str:
    spec = PACKS[pack_key]
    source = SOURCES[str(spec["family"])]
    precision = str(spec["precision"])
    license_frontmatter = (
        "license: other\nlicense_name: openmdw-1.1\nlicense_link: https://openmdw.ai/license/1-1/\n"
        if spec["family"] == "lightning"
        else "license: other\n"
        "license_name: nvidia-nemotron-open-model-license\n"
        f"license_link: {LICENSE_URL}\n"
    )
    return (
        "---\n"
        "library_name: mlx\n"
        f"base_model: {source['id']}\n"
        f"{license_frontmatter}"
        "pipeline_tag: text-generation\n"
        "tags:\n"
        "- axquant\n- mlx\n- mxfp\n- mtp\n"
        "---\n\n"
        f"# {spec['repo']}\n\n"
        "This repository contains an AXQuant development conversion of the pinned NVIDIA BF16 "
        "source. The main checkpoint uses standard MLX-LM config, tokenizer, index, "
        "and safetensors "
        "files. MXFP quantization is applied to eligible backbone weights; protected tensors keep "
        "their declared precision.\n\n"
        "## Source and format\n\n"
        f"- Source: [{source['id']}](https://huggingface.co/{source['id']})\n"
        f"- Immutable source revision: `{source['revision']}`\n"
        f"- AXQuant physical format: `{precision.upper()}`\n"
        "- AXQuant plan and file hashes are included in this repository.\n"
        "- This is a development artifact; no quality certificate is claimed.\n\n"
        "## MTP payload\n\n"
        "The source integrates Nemotron-H MTP tensors into its original checkpoint shards. AXQuant "
        "preserves those tensor payloads byte-for-byte in `mtp.safetensors` and records the source "
        "and payload digests in `ax_nemotron_mtp_manifest.json`. Runtime compatibility "
        "is unverified. "
        "This package does not claim MTP execution support in MLX-LM, AX Engine, MTPLX, "
        "or oMLX; each runtime owns that support.\n\n"
        "## Load the standard MLX backbone\n\n"
        "```python\nfrom mlx_lm import load, generate\n"
        f'model, tokenizer = load("{HUB_ROOT}/{spec["repo"]}")\n'
        'print(generate(model, tokenizer, prompt="Hello", max_tokens=32))\n```\n\n'
        "The example loads the text backbone. It does not load or activate the separate "
        "MTP payload. "
        "Super needs substantial memory; actual use depends on format, context, and runtime.\n\n"
        "## License and notices\n\n"
        + (
            "The source uses OpenMDW License 1.1; its license and notices are included.\n"
            if spec["family"] == "lightning"
            else f"The source uses the [NVIDIA Nemotron Open Model License]({LICENSE_URL}); "
            "included as `NVIDIA-Nemotron-Open-Model-License.pdf`.\n"
        )
    )


def build(pack_key: str) -> None:
    require_factory_host(socket.gethostname())
    spec = PACKS[pack_key]
    family, precision = str(spec["family"]), str(spec["precision"])
    source, output = source_path(family), pack_path(pack_key)
    source_spec, format_spec = SOURCES[family], FORMATS[precision]
    work = WORK / family / precision
    work.mkdir(parents=True, exist_ok=True)
    if not (source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"source snapshot is incomplete: {source}")
    inventory = work / "inventory.json"
    plan = work / "plan.json"
    if not inventory.is_file():
        command(
            [
                axquant_bin(),
                "inspect",
                "--model",
                str(source),
                "--model-id",
                source_spec["id"],
                "--revision",
                source_spec["revision"],
                "--output",
                str(inventory),
            ],
            work / "inspect.log",
        )
    if not plan.is_file():
        command(
            [
                axquant_bin(),
                "plan-manual",
                "--inventory",
                str(inventory),
                "--recipe",
                str(ROOT / "examples" / str(format_spec["recipe"])),
                "--output",
                str(plan),
            ],
            work / "plan.log",
        )
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    source_identity = plan_payload.get("source_model")
    if not isinstance(source_identity, dict):
        raise SystemExit(f"AXQuant plan source identity is missing in {plan}")
    source_identity["local_path"] = None
    write_data(plan, plan_payload)
    portable_plan = load_model(plan, QuantizationPlan)
    if portable_plan.source_model.local_path is not None:
        raise SystemExit(f"AXQuant plan contains a local source path: {plan.name}")
    if not (output / "axquant_manifest.json").is_file():
        if output.exists():
            raise SystemExit(f"incomplete output directory exists: {output}")
        command(
            [
                axquant_bin(),
                "convert",
                "--model",
                str(source),
                "--revision",
                source_spec["revision"],
                "--plan",
                str(plan),
                "--output",
                str(output),
                "--q-mode",
                str(format_spec["q_mode"]),
                "--allow-unmeasured",
                "--ax-engine-manifest",
                "skip",
            ],
            work / "convert.log",
            env={**os.environ, "AXQUANT_FORCE_CPU": "1"},
        )
    public_plan_path = output / "axquant_plan.json"
    write_data(public_plan_path, portable_plan)
    manifest_path = output / "axquant_manifest.json"
    mtp_manifest_path = output / "ax_nemotron_mtp_manifest.json"
    for path in (manifest_path, mtp_manifest_path):
        manifest_payload = json.loads(path.read_text(encoding="utf-8"))
        source_identity = manifest_payload.get("source_model")
        if not isinstance(source_identity, dict):
            raise SystemExit(f"AXQuant source identity is missing in {path.name}")
        source_identity["local_path"] = None
        write_data(path, manifest_payload)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["plan_sha256"] = stable_sha256(portable_plan)
    files = manifest_payload.get("files")
    if not isinstance(files, list):
        raise SystemExit(f"AXQuant file inventory is missing in {manifest_path.name}")
    for member in files:
        if isinstance(member, dict) and member.get("path") in {
            "axquant_plan.json",
            mtp_manifest_path.name,
        }:
            file_path = output / str(member["path"])
            member["size_bytes"] = file_path.stat().st_size
            member["sha256"] = file_sha256(file_path)
    write_data(manifest_path, manifest_payload)
    manifest = load_model(manifest_path, ArtifactManifest)
    mtp_manifest = load_model(mtp_manifest_path, NemotronMtpSidecarManifest)
    if (
        manifest.source_model.model_id != source_spec["id"]
        or manifest.source_model.revision != source_spec["revision"]
        or manifest.source_model.local_path is not None
        or mtp_manifest.source_model.model_id != source_spec["id"]
        or mtp_manifest.source_model.revision != source_spec["revision"]
        or mtp_manifest.source_model.local_path is not None
    ):
        raise SystemExit(f"AXQuant public manifest source identity is invalid in {output}")
    sidecar = output / "mtp.safetensors"
    mtp_manifest = output / "ax_nemotron_mtp_manifest.json"
    if not sidecar.is_file() or not mtp_manifest.is_file():
        raise SystemExit(f"Nemotron MTP payload or provenance is missing in {output}")
    manifest_data = json.loads(mtp_manifest.read_text(encoding="utf-8"))
    if manifest_data.get("payload_sha256") != file_sha256(sidecar):
        raise SystemExit(f"MTP payload digest mismatch in {output}")
    if manifest_data.get("runtime_compatibility") != "unverified":
        raise SystemExit(f"unexpected Nemotron MTP runtime status in {output}")
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".safetensors", ".pdf", ".model"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "/Volumes/" in content or str(ROOT) in content or str(HF_HOME) in content:
            raise SystemExit(f"public artifact contains a local filesystem path: {path.name}")
    if family == "lightning":
        shutil.copy2(source / "LICENSE", output / "LICENSE")
        for name in ("NOTICE", "NOTICE.txt"):
            if (source / name).is_file():
                shutil.copy2(source / name, output / name)
    else:
        if not LICENSE_PDF.is_file():
            raise SystemExit(f"official source license copy is missing: {LICENSE_PDF}")
        shutil.copy2(LICENSE_PDF, output / "NVIDIA-Nemotron-Open-Model-License.pdf")
        (output / "NOTICE").write_text(
            "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License.\n",
            encoding="utf-8",
        )
    (output / "README.md").write_text(_card(pack_key), encoding="utf-8")
    result = {
        "pack": pack_key,
        "repo": f"{HUB_ROOT}/{spec['repo']}",
        "source_model": source_spec["id"],
        "source_revision": source_spec["revision"],
        "format": precision,
        "mtp_tensor_count": manifest_data["mtp_tensor_count"],
        "mtp_sha256": file_sha256(sidecar),
        "file_count": sum(
            1
            for path in output.rglob("*")
            if path.is_file()
            and path.name not in {"axquant_nemotron_release.json", "SHA256SUMS.txt"}
        )
        + 2,
        "development_only": True,
    }
    (output / "axquant_nemotron_release.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hash_rows = []
    for path in sorted(file for file in output.rglob("*") if file.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        hash_rows.append(f"{file_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS.txt").write_text("\n".join(hash_rows) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def standard_load(pack_key: str) -> None:
    require_factory_host(socket.gethostname())
    output = pack_path(pack_key)
    test = (
        "import json, time\nfrom pathlib import Path\nfrom mlx_lm.utils import load\n"
        f"p=Path({str(output)!r}); start=time.time(); model, tokenizer=load(p, lazy=True)\n"
        "print(json.dumps({'status':'pass','model_class':type(model).__name__,'seconds':time.time()-start}))\n"
    )
    log_path = WORK / PACKS[pack_key]["family"] / PACKS[pack_key]["precision"] / "mlx-lm-load.log"
    command([python_bin(), "-c", test], log_path)
    write_data(
        log_path.with_suffix(".json"),
        {
            "status": "passed",
            "scope": "standard MLX-LM lazy backbone load; MTP execution unverified",
            "manifest_sha256": file_sha256(output / "axquant_manifest.json"),
        },
    )


def verify_publication_inputs(pack_key: str) -> dict[str, str]:
    output = pack_path(pack_key)
    manifest_path = output / "axquant_manifest.json"
    load_record = WORK / PACKS[pack_key]["family"] / PACKS[pack_key]["precision"]
    evidence = json.loads((load_record / "mlx-lm-load.json").read_text())
    if evidence.get("status") != "passed" or evidence.get("manifest_sha256") != file_sha256(
        manifest_path
    ):
        raise SystemExit("publication blocked: missing or stale backbone load evidence")
    hashes: dict[str, str] = {}
    for row in (output / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = row.split("  ", 1)
        path = output / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(output.resolve())
            or name in hashes
            or file_sha256(path) != digest
        ):
            raise SystemExit("publication blocked: checkpoint checksum mismatch")
        hashes[name] = digest
    members = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if members != set(hashes) | {"SHA256SUMS.txt"}:
        raise SystemExit("publication blocked: checksum inventory is incomplete")
    hashes["SHA256SUMS.txt"] = file_sha256(output / "SHA256SUMS.txt")
    return hashes


def standard_generate(pack_key: str) -> None:
    require_factory_host(socket.gethostname())
    output = pack_path(pack_key)
    test = (
        "import json, signal\nfrom pathlib import Path\n"
        "from mlx_lm import load, stream_generate\n"
        "from mlx_lm.sample_utils import make_sampler\n"
        "signal.alarm(600)\n"
        f"model, tokenizer=load(Path({str(output)!r}), lazy=True)\n"
        "prompt=tokenizer.apply_chat_template([{'role':'user','content':'What is 2 + 2?'}], "
        "tokenize=False, add_generation_prompt=True)\n"
        "tokens=[]\n"
        "for response in stream_generate(model, tokenizer, prompt, max_tokens=8, "
        "sampler=make_sampler(temp=0)):\n"
        "    tokens.append(response.token)\n"
        "if not tokens: raise RuntimeError('backbone did not generate any tokens')\n"
        "print(json.dumps({'status':'pass','generated_token_ids':tokens,"
        "'scope':'backbone generation smoke only; MTP execution unverified'}))\n"
    )
    record_dir = WORK / PACKS[pack_key]["family"] / PACKS[pack_key]["precision"]
    command([python_bin(), "-c", test], record_dir / "mlx-lm-generate.log")
    write_data(
        record_dir / "mlx-lm-generate.json",
        {
            "status": "passed",
            "scope": "standard MLX-LM backbone generation smoke; no quality or MTP claim",
            "manifest_sha256": file_sha256(output / "axquant_manifest.json"),
        },
    )


def publish(pack_key: str) -> None:
    require_factory_host(socket.gethostname())
    from huggingface_hub import HfApi, hf_hub_download

    spec = PACKS[pack_key]
    output = pack_path(pack_key)
    result = json.loads((output / "axquant_nemotron_release.json").read_text(encoding="utf-8"))
    if result.get("development_only") is not True:
        raise SystemExit("publication blocked: artifact is not marked development-only")
    hashes = verify_publication_inputs(pack_key)
    repo_id = f"{HUB_ROOT}/{spec['repo']}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output),
        commit_message=f"Publish {spec['repo']} development checkpoint",
        ignore_patterns=[".DS_Store", "*.lock"],
    )
    collection_names = (
        str(spec["precision"]),
        "mtp",
        "nemotron",
        "mlx",
    )
    note = (
        f"AXQ development {str(spec['precision']).upper()} standard MLX backbone; "
        "Nemotron MTP sidecar packaged, runtime compatibility unverified."
    )
    for collection_name in collection_names:
        api.add_collection_item(
            collection_slug=COLLECTIONS[collection_name],
            item_id=repo_id,
            item_type="model",
            note=note,
            exists_ok=True,
        )
    public_info = HfApi(token=False).model_info(repo_id, revision=commit.oid, files_metadata=True)
    if public_info.private:
        raise SystemExit(f"published model unexpectedly private: {repo_id}")
    siblings = {sibling.rfilename for sibling in public_info.siblings}
    required = {"config.json", "mtp.safetensors", "ax_nemotron_mtp_manifest.json", "README.md"}
    missing = required - siblings
    if missing:
        raise SystemExit(
            f"anonymous Hub metadata is missing files for {repo_id}: {sorted(missing)}"
        )
    if siblings - {".gitattributes"} != set(hashes):
        raise SystemExit(f"anonymous Hub file inventory differs from the local pack: {repo_id}")
    for sibling in public_info.siblings:
        if sibling.rfilename == ".gitattributes":
            continue
        if sibling.lfs is not None and sibling.lfs.sha256 != hashes[sibling.rfilename]:
            raise SystemExit(f"Hub weight checksum mismatch: {sibling.rfilename}")
        if sibling.lfs is None:
            downloaded = hf_hub_download(
                repo_id,
                sibling.rfilename,
                revision=public_info.sha,
                token=False,
                cache_dir=str(HF_HOME / "hub"),
            )
            if file_sha256(downloaded) != hashes[sibling.rfilename]:
                raise SystemExit(f"anonymous Hub file checksum mismatch: {sibling.rfilename}")
    for collection_name in collection_names:
        collection = api.get_collection(COLLECTIONS[collection_name])
        if not any(item.item_id == repo_id for item in collection.items):
            raise SystemExit(f"model is missing from {collection_name} collection: {repo_id}")
    write_data(
        WORK / spec["family"] / spec["precision"] / "publication.json",
        {"repo_id": repo_id, "revision": public_info.sha, "files_sha256": hashes},
    )
    print(f"published https://huggingface.co/{repo_id} revision={public_info.sha}", flush=True)


def run(pack_key: str, stage: str) -> None:
    if stage in {"build", "all"}:
        build(pack_key)
    if stage in {"load", "all"}:
        standard_load(pack_key)
    if stage == "generate":
        standard_generate(pack_key)
    if stage in {"publish", "all"}:
        publish(pack_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", *sorted(PACKS)])
    parser.add_argument(
        "--stage", choices=["build", "load", "generate", "publish", "all"], default="all"
    )
    args = parser.parse_args()
    if args.target == "all":
        for pack_key in sorted(PACKS):
            run(pack_key, args.stage)
    else:
        run(args.target, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
