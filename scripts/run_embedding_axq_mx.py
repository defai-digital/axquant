#!/usr/bin/env python3
"""Factory: convert + publish Qwen3-Embedding / Nemotron-3-Embed AXQ MXFP4/MXFP8 dev packs.

Development packs only (no Tier 1 certification). Retrieval checkpoints —
do not claim generative or MTP metrics. Run on the factory host:

  PYTHONPATH=src .venv/bin/python scripts/run_embedding_axq_mx.py --pack qwen3-8b-mxfp4 all
  ... (10 packs: qwen3-{0.6b,4b,8b} + nemotron-{1b,8b}, mxfp4/mxfp8)
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_HF_HOME,
    FACTORY_MODELS,
    require_factory_host,
)

SOURCES: dict[str, dict[str, str]] = {
    "qwen3-0.6b": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "rev": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    },
    "qwen3-4b": {
        "id": "Qwen/Qwen3-Embedding-4B",
        "rev": "5cf2132abc99cad020ac570b19d031efec650f2b",
    },
    "qwen3-8b": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "rev": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    },
    "nemotron-1b": {
        "id": "nvidia/Nemotron-3-Embed-1B-BF16",
        "rev": "c0c9fea93ea424587517f2c59e20db9f1d6bf615",
    },
    "nemotron-8b": {
        "id": "nvidia/Nemotron-3-Embed-8B-BF16",
        "rev": "d1f2f25730bbd775b99b29185134bc86653bf2d1",
    },
}

ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)


def _pack(model: str, hub_stem: str, recipe_stem: str, q_mode: str) -> dict[str, Any]:
    cls = q_mode.upper()
    return {
        "model": model,
        "hub_name": f"{hub_stem}-MLX-AXQ-{cls}",
        "display_name": f"{hub_stem} MLX AXQ {cls}",
        "recipe": ROOT / "examples" / f"{recipe_stem}-axq-{q_mode}-v0.1.yaml",
        "q_mode": q_mode,
        "product_class": cls,
    }


PACKS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b-mxfp4": _pack(
        "qwen3-0.6b", "AX-Qwen3-Embedding-0.6B", "qwen3-embedding-0.6b", "mxfp4"
    ),
    "qwen3-0.6b-mxfp8": _pack(
        "qwen3-0.6b", "AX-Qwen3-Embedding-0.6B", "qwen3-embedding-0.6b", "mxfp8"
    ),
    "qwen3-4b-mxfp4": _pack("qwen3-4b", "AX-Qwen3-Embedding-4B", "qwen3-embedding-4b", "mxfp4"),
    "qwen3-4b-mxfp8": _pack("qwen3-4b", "AX-Qwen3-Embedding-4B", "qwen3-embedding-4b", "mxfp8"),
    "qwen3-8b-mxfp4": _pack("qwen3-8b", "AX-Qwen3-Embedding-8B", "qwen3-embedding-8b", "mxfp4"),
    "qwen3-8b-mxfp8": _pack("qwen3-8b", "AX-Qwen3-Embedding-8B", "qwen3-embedding-8b", "mxfp8"),
    "nemotron-1b-mxfp4": _pack(
        "nemotron-1b", "AX-Nemotron-3-Embed-1B", "nemotron-3-embed-1b", "mxfp4"
    ),
    "nemotron-1b-mxfp8": _pack(
        "nemotron-1b", "AX-Nemotron-3-Embed-1B", "nemotron-3-embed-1b", "mxfp8"
    ),
    "nemotron-8b-mxfp4": _pack(
        "nemotron-8b", "AX-Nemotron-3-Embed-8B", "nemotron-3-embed-8b", "mxfp4"
    ),
    "nemotron-8b-mxfp8": _pack(
        "nemotron-8b", "AX-Nemotron-3-Embed-8B", "nemotron-3-embed-8b", "mxfp8"
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    return PACKS[key]


def work_dir(model: str) -> Path:
    slug = model.replace(".", "p")
    return Path(
        os.environ.get(
            f"EMB_{slug.upper().replace('-', '_')}_WORK",
            f"{FACTORY_CERT_ROOT}/embedding-{slug}",
        )
    )


def source_dir(model: str) -> Path:
    return work_dir(model) / "src"


def pack_dir(key: str) -> Path:
    return Path(FACTORY_MODELS) / str(spec(key)["hub_name"])


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found")


def hf_env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": os.environ.get("HF_HOME", FACTORY_HF_HOME),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", f"{FACTORY_HF_HOME}/hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    return env


def run(cmd: list[str], log_path: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    env = hf_env()
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


def cmd_download(model: str) -> None:
    source = SOURCES[model]
    dest = source_dir(model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").is_file() and any(dest.glob("*.safetensors")):
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(hf) if hf.is_file() else "hf",
            "download",
            source["id"],
            "--revision",
            source["rev"],
            "--local-dir",
            str(dest),
        ],
        work_dir(model) / "logs" / "download-src.log",
    )


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    model = str(item["model"])
    source = SOURCES[model]
    recipe = item["recipe"]
    src = source_dir(model)
    if not (src / "config.json").is_file():
        raise SystemExit(f"missing source {src}; run download first")
    work = work_dir(model)
    pack = pack_dir(key)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    inventory = work / f"inventory-{key}.json"
    plan = work / f"plan-{key}.json"
    if not inventory.is_file():
        run(
            [
                *axquant_cmd(),
                "inspect",
                "--model",
                str(src),
                "--model-id",
                source["id"],
                "--revision",
                source["rev"],
                "--output",
                str(inventory),
            ],
            work / "logs" / f"inspect-{key}.log",
        )
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
            work / "logs" / f"plan-{key}.log",
        )
    if (pack / "axquant_manifest.json").is_file():
        log(f"reuse pack {pack}")
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    engine = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    run(
        [
            *axquant_cmd(),
            "convert",
            "--model",
            str(src),
            "--plan",
            str(plan),
            "--output",
            str(pack),
            "--q-mode",
            str(item["q_mode"]),
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "skip",
            "--ax-engine-bench",
            engine,
        ],
        work / "logs" / f"convert-{key}.log",
    )


def cmd_runtime(key: str) -> None:
    require_factory_host(socket.gethostname())
    model = str(spec(key)["model"])
    work = work_dir(model)
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    pack = pack_dir(key)
    smoke = rdir / f"{key}-mlx-lm.json"
    py = ROOT / ".venv" / "bin" / "python"
    script = (
        "import json, time\n"
        "from pathlib import Path\n"
        "from mlx_lm.utils import load\n"
        f"pack = Path({str(pack)!r})\n"
        "t0 = time.time()\n"
        "model, _ = load(pack, lazy=False)\n"
        "n = sum(1 for _ in model.named_modules())\n"
        "payload = {'status': 'pass', 'modules': n, 'seconds': time.time()-t0}\n"
        f"Path({str(smoke)!r}).write_text(json.dumps(payload, indent=2)+'\\n')\n"
        "print(payload)\n"
    )
    if not smoke.is_file():
        run(
            [str(py) if py.is_file() else "python", "-c", script],
            work / "logs" / f"runtime-{key}.log",
        )


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    model = str(item["model"])
    pack = pack_dir(key)
    repo = f"AutomatosX/{item['hub_name']}"
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [
            str(py) if py.is_file() else "python",
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(pack),
            "--repo-id",
            repo,
            "--product-class",
            item["product_class"],
        ],
        work_dir(model) / "logs" / f"prepare-card-{key}.log",
    )
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir(model) / "logs" / f"hf-repo-{key}.log",
    )
    run(
        [
            hf_bin,
            "upload",
            repo,
            str(pack),
            "--repo-type",
            "model",
            "--commit-message",
            f"Publish {item['hub_name']}",
        ],
        work_dir(model) / "logs" / f"hf-upload-{key}.log",
    )


def cmd_all(key: str) -> None:
    cmd_download(str(spec(key)["model"]))
    cmd_convert(key)
    cmd_runtime(key)
    cmd_publish(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=sorted(PACKS), required=True)
    parser.add_argument(
        "stage",
        choices=["download", "convert", "runtime", "publish", "all"],
    )
    args = parser.parse_args()
    stages = {
        "download": lambda: cmd_download(str(spec(args.pack)["model"])),
        "convert": lambda: cmd_convert(args.pack),
        "runtime": lambda: cmd_runtime(args.pack),
        "publish": lambda: cmd_publish(args.pack),
        "all": lambda: cmd_all(args.pack),
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
