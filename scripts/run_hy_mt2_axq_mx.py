#!/usr/bin/env python3
"""Factory: Hy-MT2-30B-A3B MLX AXQ MXFP4 and MXFP8 packs.

Source tencent/Hy-MT2-30B-A3B is Apache-2.0 hy_v3. Development evidence only.
No translation-quality claim.

  PYTHONPATH=src .venv/bin/python scripts/run_hy_mt2_axq_mx.py --pack all all
"""

from __future__ import annotations

import argparse
import json
import os
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

SOURCE = {
    "id": "tencent/Hy-MT2-30B-A3B",
    "rev": "d3ead4dba61c09aac60a261a96ad1df3e705febb",
}
REV = SOURCE["rev"]
COPY_IF_PRESENT = (
    "LICENSE",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)

PACKS: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "hub_name": "AX-Hy-MT2-30B-A3B-MLX-AXQ-MXFP4",
        "recipe": ROOT / "examples" / "hy-mt2-30b-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
    },
    "mxfp8": {
        "hub_name": "AX-Hy-MT2-30B-A3B-MLX-AXQ-MXFP8",
        "recipe": ROOT / "examples" / "hy-mt2-30b-axq-mxfp8-v0.1.yaml",
        "q_mode": "mxfp8",
        "product_class": "MXFP8",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def work_dir() -> Path:
    return Path(os.environ.get("HY_MT2_WORK", f"{FACTORY_CERT_ROOT}/hy-mt2-30b"))


def source_dir() -> Path:
    return work_dir() / "src"


def pack_dir(key: str) -> Path:
    return Path(FACTORY_MODELS) / PACKS[key]["hub_name"]


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    raise SystemExit("axquant not found")


def hf_env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": os.environ.get("HF_HOME", FACTORY_HF_HOME),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", f"{FACTORY_HF_HOME}/hub"),
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": os.environ.get(
            "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY", "8"
        ),
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": os.environ.get(
            "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY", "4"
        ),
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": os.environ.get(
            "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY", "2"
        ),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    env.pop("HF_HUB_DISABLE_XET", None)
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


def source_ready() -> bool:
    src = source_dir()
    index_path = src / "model.safetensors.index.json"
    if not (src / "config.json").is_file() or not index_path.is_file():
        return False
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shards = {src / name for name in weight_map.values()}
    return bool(shards) and all(path.is_file() and path.stat().st_size > 0 for path in shards)


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_ready():
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(hf) if hf.is_file() else "hf",
            "download",
            SOURCE["id"],
            "--revision",
            REV,
            "--local-dir",
            str(dest),
        ],
        work_dir() / "logs" / "download-src.log",
    )
    config = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "hy_v3":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    if config.get("architectures") != ["HYV3ForCausalLM"]:
        raise SystemExit(f"unexpected architectures {config.get('architectures')}")
    if not source_ready():
        raise SystemExit("download finished with an incomplete weight index")


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = PACKS[key]
    src = source_dir()
    if not source_ready():
        raise SystemExit(f"missing source {src}; run download first")
    work = work_dir()
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
                SOURCE["id"],
                "--revision",
                REV,
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
                str(item["recipe"]),
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
            "ax-engine-bench",
        ],
        work / "logs" / f"convert-{key}.log",
    )


def _reject_local_paths(pack: Path) -> None:
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")


def attach_sidecars(key: str) -> None:
    src = source_dir()
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    for name in COPY_IF_PRESENT:
        origin = src / name
        if origin.is_file() and not (pack / name).is_file():
            (pack / name).write_bytes(origin.read_bytes())
            log(f"copied {name}")


def cmd_smoke(key: str) -> None:
    require_factory_host(socket.gethostname())
    attach_sidecars(key)
    pack = pack_dir(key)
    work = work_dir()
    smoke = work / "runtime" / f"{key}-integrity.json"
    if smoke.is_file():
        log(f"reuse smoke {smoke}")
        return
    smoke.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pack / "axquant_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((pack / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "hy_v3":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if not quant:
        raise SystemExit("converted config has no quantization block")
    _reject_local_paths(pack)
    payload = {
        "status": "pass",
        "check": "pack-integrity",
        "measured_main_bpw": manifest.get("measured_main_bpw"),
        "model_type": config.get("model_type"),
        "translation_measured": False,
    }
    smoke.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(json.dumps(payload))


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    attach_sidecars(key)
    item = PACKS[key]
    pack = pack_dir(key)
    repo = f"AutomatosX/{item['hub_name']}"
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(py),
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(pack),
            "--repo-id",
            repo,
            "--product-class",
            item["product_class"],
            "--no-public-certification",
        ],
        work_dir() / "logs" / f"prepare-card-{key}.log",
    )
    _reject_local_paths(pack)
    readme = (pack / "README.md").read_text(encoding="utf-8")
    if "license: apache-2.0" not in readme.split("---", 2)[1]:
        raise SystemExit("model card front matter is not Apache-2.0")
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir() / "logs" / f"hf-repo-{key}.log",
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
        work_dir() / "logs" / f"hf-upload-{key}.log",
    )


def cmd_all(key: str) -> None:
    cmd_download()
    cmd_convert(key)
    cmd_smoke(key)
    cmd_publish(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=[*sorted(PACKS), "all"], required=True)
    parser.add_argument("stage", choices=["download", "convert", "smoke", "publish", "all"])
    args = parser.parse_args()
    keys = ["mxfp4", "mxfp8"] if args.pack == "all" else [args.pack]
    if args.stage == "download":
        cmd_download()
        return 0
    for key in keys:
        if args.stage == "convert":
            cmd_convert(key)
        elif args.stage == "smoke":
            cmd_smoke(key)
        elif args.stage == "publish":
            cmd_publish(key)
        else:
            cmd_all(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
