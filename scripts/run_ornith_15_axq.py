#!/usr/bin/env python3
"""Factory: convert + publish Ornith-1.5 AXQ MXFP4 / 6-bit packs.

Development evidence. Not the Qwen 3.6 cert track. One pack at a time:

  PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 9b --pack mxfp4 all
  PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 35b --pack axq6 all
  PYTHONPATH=src .venv/bin/python scripts/run_ornith_15_axq.py --model 397b --pack mxfp4 all
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import FACTORY_HOST_ID, require_factory_host  # noqa: E402

DEFAULT_DISK = "/Volumes/Ext16TR0"
if not Path(DEFAULT_DISK).is_dir():
    DEFAULT_DISK = "/Volumes/Ext12T"

MODELS: dict[str, dict[str, Any]] = {
    "9b": {
        "source_id": "ornith-ai/Ornith-1.5-9B",
        "source_rev": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
        "adapter_id": "qwen35-dense-v1",
        "src_name": "src-ornith-15-9b",
        "work_name": "ornith-15-9b",
        "expected_shards": 4,
        "packs": {
            "mxfp4": {
                "hub_name": "AX-Ornith-1.5-9B-MLX-AXQ-MXFP4-MTP",
                "recipe": ROOT / "examples" / "ornith-15-9b-axq-mxfp4-v0.1.yaml",
                "q_mode": "mxfp4",
                "product_class": "MXFP4",
            },
            "axq6": {
                "hub_name": "AX-Ornith-1.5-9B-MLX-AXQ-6bit-MTP",
                "recipe": ROOT / "examples" / "ornith-15-9b-axq6-v0.1.yaml",
                "q_mode": "affine",
                "product_class": "6bit",
            },
        },
    },
    "35b": {
        "source_id": "ornith-ai/Ornith-1.5-35B-A3B",
        "source_rev": "10fbf86fed7ecee4a061f8b499a618f46001cac1",
        "adapter_id": "qwen35-moe-v1",
        "src_name": "src-ornith-15-35b",
        "work_name": "ornith-15-35b",
        "expected_shards": 16,
        "packs": {
            "mxfp4": {
                "hub_name": "AX-Ornith-1.5-35B-A3B-MLX-AXQ-MXFP4-MTP",
                "recipe": ROOT / "examples" / "ornith-15-35b-axq-mxfp4-v0.1.yaml",
                "q_mode": "mxfp4",
                "product_class": "MXFP4",
            },
            "axq6": {
                "hub_name": "AX-Ornith-1.5-35B-A3B-MLX-AXQ-6bit-MTP",
                "recipe": ROOT / "examples" / "ornith-15-35b-axq6-v0.1.yaml",
                "q_mode": "affine",
                "product_class": "6bit",
            },
        },
    },
    "397b": {
        "source_id": "ornith-ai/Ornith-1.5-397B",
        "source_rev": "8f6cc8a7aea505364523f84ccf37706e8aea0ee7",
        "adapter_id": "qwen35-moe-v1",
        "src_name": "src-ornith-15-397b",
        "work_name": "ornith-15-397b",
        "expected_shards": 122,
        "packs": {
            "mxfp4": {
                "hub_name": "AX-Ornith-1.5-397B-MLX-AXQ-MXFP4-MTP",
                "recipe": ROOT / "examples" / "ornith-15-397b-axq-mxfp4-v0.1.yaml",
                "q_mode": "mxfp4",
                "product_class": "MXFP4",
            },
            "axq6": {
                "hub_name": "AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP",
                "recipe": ROOT / "examples" / "ornith-15-397b-axq6-v0.1.yaml",
                "q_mode": "affine",
                "product_class": "6bit",
            },
        },
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def model_spec(name: str) -> dict[str, Any]:
    if name not in MODELS:
        raise SystemExit(f"unknown model {name}; choose {sorted(MODELS)}")
    return MODELS[name]


def pack_spec(model: str, pack: str) -> dict[str, Any]:
    packs = model_spec(model)["packs"]
    if pack not in packs:
        raise SystemExit(f"unknown pack {pack}; choose {sorted(packs)}")
    return packs[pack]


def disk_root() -> Path:
    return Path(os.environ.get("ORNITH15_DISK", DEFAULT_DISK))


def work_dir(model: str) -> Path:
    spec = model_spec(model)
    default = disk_root() / "axquant" / "work" / spec["work_name"]
    return Path(os.environ.get("ORNITH15_WORK", str(default)))


def source_dir(model: str) -> Path:
    override = os.environ.get("ORNITH15_SOURCE")
    if override:
        return Path(override)
    return work_dir(model) / str(model_spec(model)["src_name"])


def models_root() -> Path:
    return Path(os.environ.get("ORNITH15_MODELS", str(disk_root() / "models")))


def pack_dir(model: str, pack: str) -> Path:
    return models_root() / str(pack_spec(model, pack)["hub_name"])


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found")


def hf_home() -> str:
    return os.environ.get("HF_HOME", str(disk_root() / "huggingface"))


def hf_env() -> dict[str, str]:
    home = hf_home()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": home,
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", f"{home}/hub"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", f"{home}/hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{home}/xet"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    return env


def run(
    cmd: list[str],
    log_path: Path | None = None,
    *,
    force_cpu: bool = False,
    streaming_convert: bool = False,
) -> None:
    log("$ " + " ".join(cmd))
    env = hf_env()
    if force_cpu:
        env["AXQUANT_FORCE_CPU"] = "1"
    if streaming_convert:
        env["AXQUANT_STREAMING_CONVERT"] = "1"
    if log_path is None:
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): see {log_path}")


def source_complete(dest: Path) -> bool:
    if not (dest / "config.json").is_file():
        return False
    index = dest / "model.safetensors.index.json"
    if not index.is_file():
        return False
    payload = json.loads(index.read_text(encoding="utf-8"))
    files = {dest / name for name in payload.get("weight_map", {}).values()}
    extra = dest / "model-mtp.safetensors"
    if extra.is_file():
        files.add(extra)
    return bool(files) and all(path.is_file() and path.stat().st_size > 0 for path in files)


def cmd_download(model: str) -> None:
    require_factory_host(socket.gethostname())
    spec = model_spec(model)
    dest = source_dir(model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_complete(dest):
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(hf) if hf.is_file() else "hf",
            "download",
            spec["source_id"],
            "--revision",
            spec["source_rev"],
            "--local-dir",
            str(dest),
        ],
        work_dir(model) / "logs" / "download-src.log",
    )
    if not source_complete(dest):
        raise SystemExit(f"incomplete BF16 source {dest}")


def cmd_inspect(model: str, pack: str) -> None:
    require_factory_host(socket.gethostname())
    spec = model_spec(model)
    source = source_dir(model)
    if not source_complete(source):
        raise SystemExit(f"incomplete source {source}; run download first")
    work = work_dir(model)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    inventory = work / "inventory.json"
    if inventory.is_file():
        log(f"reuse inventory {inventory}")
        return
    run(
        [
            *axquant_cmd(),
            "inspect",
            "--model",
            str(source),
            "--model-id",
            spec["source_id"],
            "--revision",
            spec["source_rev"],
            "--output",
            str(inventory),
        ],
        work / "logs" / "inspect.log",
    )
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    profile = payload.get("architecture") or payload.get("architecture_profile") or {}
    adapter = profile.get("adapter_id")
    expected = spec["adapter_id"]
    if adapter != expected:
        raise SystemExit(f"inspect adapter {adapter!r} != {expected!r}")
    log(f"inspect ok adapter={adapter} pack={pack}")


def cmd_convert(model: str, pack: str) -> None:
    require_factory_host(socket.gethostname())
    item = pack_spec(model, pack)
    recipe = Path(item["recipe"])
    if not recipe.is_file():
        raise SystemExit(f"missing recipe {recipe}")
    source = source_dir(model)
    if not source_complete(source):
        raise SystemExit(f"incomplete source {source}; run download first")
    work = work_dir(model)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    inventory = work / "inventory.json"
    if not inventory.is_file():
        cmd_inspect(model, pack)
    plan = work / f"plan-{pack}.json"
    if not plan.is_file():
        run(
            [
                *axquant_cmd(),
                "plan-manual",
                "--inventory",
                str(inventory),
                "--recipe",
                str(recipe),
                "--output",
                str(plan),
            ],
            work / "logs" / f"plan-{pack}.log",
        )
    out = pack_dir(model, pack)
    if (out / "axquant_manifest.json").is_file():
        log(f"reuse pack {out}")
        return
    if out.exists():
        raise SystemExit(f"incomplete pack dir exists: {out}")
    run(
        [
            *axquant_cmd(),
            "convert",
            "--model",
            str(source),
            "--plan",
            str(plan),
            "--output",
            str(out),
            "--q-mode",
            str(item["q_mode"]),
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "skip",
        ],
        work / "logs" / f"convert-{pack}.log",
        force_cpu=True,
        streaming_convert=model == "397b",
    )
    if not (out / "axquant_manifest.json").is_file():
        raise SystemExit(f"convert produced no manifest in {out}")
    log(f"convert ok {out}")


def cmd_smoke(model: str, pack: str) -> None:
    require_factory_host(socket.gethostname())
    out = pack_dir(model, pack)
    if not (out / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {out}")
    work = work_dir(model)
    log_path = work / "logs" / f"smoke-{pack}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    py = ROOT / ".venv" / "bin" / "python"
    script = (
        "from mlx_lm import load, generate\n"
        f"model, tokenizer = load({str(out)!r})\n"
        "text = generate(model, tokenizer, prompt='Say exactly: AXQ OK', max_tokens=32)\n"
        "print(text)\n"
        "assert text.strip()\n"
    )
    env = hf_env()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(script + "\n")
        handle.flush()
        proc = subprocess.run(
            [str(py) if py.is_file() else "python", "-c", script],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
        )
    if proc.returncode != 0:
        if model == "397b":
            log(f"397B smoke failed (likely memory); see {log_path}")
            log("Continuing without a runtime claim. 192 GB cannot resident-load 397B.")
            return
        raise SystemExit(f"smoke failed: see {log_path}")
    log(f"smoke ok {pack}")


def cmd_publish(model: str, pack: str) -> None:
    require_factory_host(socket.gethostname())
    spec = model_spec(model)
    item = pack_spec(model, pack)
    out = pack_dir(model, pack)
    if not (out / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {out}")
    repo = f"AutomatosX/{item['hub_name']}"
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    work = work_dir(model)
    run(
        [
            str(py) if py.is_file() else "python",
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(out),
            "--repo-id",
            repo,
            "--product-class",
            item["product_class"],
            "--no-public-certification",
        ],
        work / "logs" / f"prepare-card-{pack}.log",
    )
    run([hf_bin, "repo", "create", repo, "--exist-ok"], work / "logs" / f"hf-repo-{pack}.log")
    run(
        [
            hf_bin,
            "upload",
            repo,
            str(out),
            "--repo-type",
            "model",
            "--commit-message",
            f"Publish {item['hub_name']} from {spec['source_id']}@{spec['source_rev']}",
        ],
        work / "logs" / f"hf-upload-{pack}.log",
    )
    script = f"from huggingface_hub import model_info\nprint(model_info({repo!r}).sha)\n"
    proc = subprocess.run(
        [str(py) if py.is_file() else "python", "-c", script],
        check=True,
        cwd=str(ROOT),
        env=hf_env(),
        capture_output=True,
        text=True,
    )
    commit = (proc.stdout or "").strip().splitlines()[-1].strip()
    (work / f"hub-commit-{pack}.txt").write_text(commit + "\n", encoding="utf-8")
    log(f"published {repo} commit={commit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--pack", required=True, choices=("mxfp4", "axq6"))
    parser.add_argument(
        "step",
        choices=("download", "inspect", "convert", "smoke", "publish", "all"),
    )
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    require_factory_host(socket.gethostname())
    log(f"host={FACTORY_HOST_ID} model={args.model} pack={args.pack} step={args.step}")
    if args.step in ("download", "all"):
        cmd_download(args.model)
    if args.step in ("inspect", "all"):
        cmd_inspect(args.model, args.pack)
    if args.step in ("convert", "all"):
        cmd_convert(args.model, args.pack)
    if args.step in ("smoke", "all") and not args.skip_smoke:
        cmd_smoke(args.model, args.pack)
    if args.step in ("publish", "all"):
        cmd_publish(args.model, args.pack)


if __name__ == "__main__":
    main()
