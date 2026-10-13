#!/usr/bin/env python3
"""Factory: Swift-1.5-Qwen3.8-27B MLX AXQ MXFP4 and MXFP8 packs.

Source ukisai/Swift-1.5-Qwen3.8-27b is a Qwen3.8-27B finetune (qwen3_5 dense
hybrid, integrated MTP). The Swift Open License v1.0 allows a quantized
derivative when LICENSE, LICENSE-APACHE-2.0, and NOTICE travel with the pack.
Vision, MTP, and norms stay BF16. Development evidence only.

  PYTHONPATH=src .venv/bin/python scripts/run_swift15_qwen38_axq_mx.py --pack all all
"""

from __future__ import annotations

import argparse
import hashlib
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
    "id": "ukisai/Swift-1.5-Qwen3.8-27b",
    "rev": "b4c84d42903a8646b25857eb2827288d92ed85a4",
}
REV = SOURCE["rev"]
LICENSE_FILES = ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE")
COPY_IF_MISSING = (
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)

PACKS: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "hub_name": "AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
        "recipe": ROOT / "examples" / "qwen38-27b-axq-mxfp4-mtp-v0.2.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
    },
    "mxfp8": {
        "hub_name": "AX-Swift-1.5-Qwen3.8-27B-MLX-AXQ-MXFP8-MTP",
        "recipe": ROOT / "examples" / "qwen38-27b-axq-mxfp8-mtp-v0.1.yaml",
        "q_mode": "mxfp8",
        "product_class": "MXFP8",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def work_dir() -> Path:
    return Path(os.environ.get("SWIFT15_WORK", f"{FACTORY_CERT_ROOT}/swift-1.5-qwen38-27b"))


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
    if not all((src / name).is_file() for name in LICENSE_FILES):
        return False
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shards = {src / name for name in weight_map.values()}
    return bool(shards) and all(path.is_file() and path.stat().st_size > 0 for path in shards)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_ready() and (dest / "model.safetensors.index.json").is_file():
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
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise SystemExit(f"unexpected architectures {config.get('architectures')}")
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    if text.get("mtp_num_hidden_layers") != 1:
        raise SystemExit("source is missing the integrated MTP layer")
    for name in LICENSE_FILES:
        if not (dest / name).is_file():
            raise SystemExit(f"source is missing {name}")
    license_text = (dest / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "Swift Open License v1.0" not in license_text:
        raise SystemExit("LICENSE is not the Swift Open License v1.0")


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


def _change_notice(key: str) -> str:
    mode = str(PACKS[key]["q_mode"]).upper()
    return f"""AXQuant changes

This directory is a quantized MLX derivative of {SOURCE["id"]} at {REV}.
AutomatosX changed the weight files:
- Language attention and MLP weights were quantized to {mode}.
- Token embeddings and the LM head are 8-bit affine.
- Vision, MTP, and norms stay BF16.
- MTP acceleration was not measured. Generation quality was not measured.

The Swift contribution remains under the Swift Open License v1.0 (LICENSE).
The Qwen3.8-27B base remains Apache License 2.0 (LICENSE-APACHE-2.0).
Commercial use by a legal entity with gross revenue of US$1,000,000 or more
in its most recent fiscal year is not licensed.
"""


def attach_sidecars(key: str) -> None:
    src = source_dir()
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    for name in LICENSE_FILES:
        origin = src / name
        if not origin.is_file():
            raise SystemExit(f"missing source {name}")
        dest = pack / name
        if name == "NOTICE":
            upstream = origin.read_text(encoding="utf-8", errors="replace").rstrip()
            marker = "AXQuant changes"
            body = upstream if marker in upstream else f"{upstream}\n\n{_change_notice(key)}"
            dest.write_text(body.rstrip() + "\n", encoding="utf-8")
            continue
        if not dest.is_file() or sha256(origin) != sha256(dest):
            dest.write_bytes(origin.read_bytes())
            log(f"copied {name}")
    for name in COPY_IF_MISSING:
        origin = src / name
        if origin.is_file() and not (pack / name).is_file():
            dest = pack / name
            dest.write_bytes(origin.read_bytes())
            log(f"copied {name}")
    benchmarks = pack / "benchmarks"
    if benchmarks.exists():
        raise SystemExit("pack contains upstream benchmark files")
    if not (pack / "LICENSE").is_file():
        raise SystemExit("pack is missing LICENSE")


def _reject_local_paths(pack: Path) -> None:
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")


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
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    if text_cfg.get("mtp_num_hidden_layers") != 1 and config.get("mtp_num_hidden_layers") != 1:
        raise SystemExit("MTP layer count was not preserved")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if not quant:
        raise SystemExit("converted config has no quantization block")
    if not (pack / "mtp.safetensors").is_file():
        raise SystemExit("integrated MTP sidecar is missing")
    if "Swift Open License v1.0" not in (pack / "LICENSE").read_text(encoding="utf-8"):
        raise SystemExit("LICENSE text is not the Swift Open License")
    if not (pack / "LICENSE-APACHE-2.0").is_file():
        raise SystemExit("Apache-2.0 base license is missing")
    if "AXQuant changes" not in (pack / "NOTICE").read_text(encoding="utf-8"):
        raise SystemExit("NOTICE is missing the change statement")
    _reject_local_paths(pack)
    payload = {
        "status": "pass",
        "check": "pack-integrity",
        "measured_main_bpw": manifest.get("measured_main_bpw"),
        "model_type": config.get("model_type"),
        "mtp_sidecar": True,
        "generation_measured": False,
        "mtp_acceleration_measured": False,
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
    if "license: other" not in readme or "swift-open-license-1.0" not in readme:
        raise SystemExit("model card does not carry the Swift Open License")
    if "license: apache-2.0" in readme.split("---", 2)[1]:
        raise SystemExit("model card front matter is Apache-2.0")
    if "US$1,000,000" not in readme:
        raise SystemExit("model card omits the commercial-use threshold")
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
    parser.add_argument(
        "stage",
        choices=["download", "convert", "smoke", "publish", "all"],
    )
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
