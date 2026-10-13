#!/usr/bin/env python3
"""Factory: Clef-Omni MLX AXQ MXFP4 and MXFP8 development packs.

Apache-2.0 source Cloudflare/clef-omni. Thinker and talker attention, MLP, and
experts are quantized. Vision, the audio encoder, code2wav, norms, LM heads,
and the joint schema head stay BF16. This is a decision model. Text generation
is not a quality check.

  PYTHONPATH=src .venv/bin/python scripts/run_clef_omni_axq_mx.py --pack mxfp4 convert
"""

from __future__ import annotations

import argparse
import hashlib
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

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_HF_HOME,
    FACTORY_MODELS,
    require_factory_host,
)

SOURCE = {
    "id": "Cloudflare/clef-omni",
    "rev": "0db1cd2607d76a7bdb2a382f659e7b313079f84b",
}
REV = SOURCE["rev"]
JOINT_HEAD = "joint_head.safetensors"
SIDECARS = (
    "joint_head_config.json",
    "joint_schema_model.py",
    "LICENSE",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

PACKS: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "hub_name": "AX-Clef-Omni-30B-A3B-MLX-AXQ-MXFP4",
        "recipe": ROOT / "examples" / "clef-omni-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
    },
    "mxfp8": {
        "hub_name": "AX-Clef-Omni-30B-A3B-MLX-AXQ-MXFP8",
        "recipe": ROOT / "examples" / "clef-omni-axq-mxfp8-v0.1.yaml",
        "q_mode": "mxfp8",
        "product_class": "MXFP8",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def work_dir() -> Path:
    return Path(os.environ.get("CLEF_OMNI_WORK", f"{FACTORY_CERT_ROOT}/clef-omni"))


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
    parked = work_dir() / "decision-sidecar" / JOINT_HEAD
    return (
        (src / "config.json").is_file()
        and ((src / JOINT_HEAD).is_file() or parked.is_file())
        and any(src.glob("model-*.safetensors"))
    )


def _park_joint_head() -> None:
    src = source_dir() / JOINT_HEAD
    parked = work_dir() / "decision-sidecar" / JOINT_HEAD
    if not src.is_file():
        return
    parked.parent.mkdir(parents=True, exist_ok=True)
    if parked.is_file():
        parked.unlink()
    shutil.move(src, parked)
    log(f"parked {JOINT_HEAD} outside the indexed snapshot")


def _restore_joint_head() -> None:
    src = source_dir() / JOINT_HEAD
    parked = work_dir() / "decision-sidecar" / JOINT_HEAD
    if src.is_file() or not parked.is_file():
        return
    shutil.move(parked, src)
    log(f"restored {JOINT_HEAD}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    _park_joint_head()
    try:
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
    finally:
        _restore_joint_head()


def attach_sidecars(key: str) -> None:
    src = source_dir()
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    head_src = src / JOINT_HEAD
    if not head_src.is_file():
        parked = work_dir() / "decision-sidecar" / JOINT_HEAD
        head_src = parked if parked.is_file() else head_src
    if not head_src.is_file():
        raise SystemExit(f"missing {JOINT_HEAD}")
    head_dst = pack / JOINT_HEAD
    if not head_dst.is_file():
        shutil.copy2(head_src, head_dst)
    if sha256(head_src) != sha256(head_dst):
        raise SystemExit("joint head bytes changed")
    for name in SIDECARS:
        origin = src / name
        if origin.is_file() and not (pack / name).is_file():
            shutil.copy2(origin, pack / name)
            log(f"copied {name}")
    if not (pack / "LICENSE").is_file():
        raise SystemExit("pack is missing LICENSE")
    mode = str(PACKS[key]["q_mode"]).upper()
    notice = f"""This directory is a quantized MLX derivative of {SOURCE['id']}.

Changes from the upstream BF16 release:
- Thinker and talker attention, MLP, and expert weights were quantized to {mode} by AXQuant.
- Token embeddings and routers are 8-bit affine.
- Vision, the audio encoder, code2wav, norms, and the LM heads stay BF16.
- joint_head.safetensors is the original BF16 joint schema head, copied unchanged.
- Speech output was not measured. Talker experts were quantized; code2wav was not.

The upstream work is Apache-2.0. See LICENSE.
"""
    (pack / "NOTICE").write_text(notice, encoding="utf-8")


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
    head_cfg = json.loads((pack / "joint_head_config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_omni_moe":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if not quant:
        raise SystemExit("converted config has no quantization block")
    if not (pack / JOINT_HEAD).is_file():
        raise SystemExit("joint head missing from pack")
    if not isinstance(head_cfg, dict) or "hidden_size" not in head_cfg:
        raise SystemExit("joint_head_config.json is missing hidden_size")
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")
    payload = {
        "status": "pass",
        "check": "pack-integrity",
        "measured_main_bpw": manifest.get("measured_main_bpw"),
        "model_type": config.get("model_type"),
        "joint_head_hidden_size": head_cfg.get("hidden_size"),
        "decision_forward": False,
        "note": (
            "Joint-head probabilities were not executed. "
            "mlx-lm generate is not a Clef-Omni check."
        ),
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
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")
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
        if args.stage in {"convert", "all"}:
            cmd_download()
            cmd_convert(key)
        if args.stage in {"smoke", "all"}:
            cmd_smoke(key)
        if args.stage in {"publish", "all"}:
            cmd_publish(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
