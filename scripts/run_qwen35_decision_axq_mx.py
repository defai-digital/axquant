#!/usr/bin/env python3
"""Factory: Qwen3.5-0.8B-decision MLX AXQ MXFP4 and MXFP8 packs.

Apache-2.0 source mlx-community/Qwen3.5-0.8B-decision. The published checkpoint
is a rank-16 LoRA on the BF16 backbone. Quantizing the backbone alone drops the
decision fine-tune, so this runner merges the LoRA first. Vision, MTP, norms,
and the joint schema head stay BF16. Text generation is not a quality check.

  PYTHONPATH=src .venv/bin/python scripts/run_qwen35_decision_axq_mx.py --pack mxfp4 convert
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

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_HF_HOME,
    FACTORY_MODELS,
    require_factory_host,
)

from qwen35_decision_lora import lora_delta, weight_key_for_lora  # noqa: E402  # isort: skip

SOURCE = {
    "id": "mlx-community/Qwen3.5-0.8B-decision",
    "rev": "36e166ca10c4d16471a231d66cfc22d80039fe34",
}
REV = SOURCE["rev"]
JOINT_HEAD = "joint_head.safetensors"
ADAPTERS = "adapters.safetensors"
RANK = 16
SCALE = 1.0
OPTIONAL_SIDECARS = (
    "joint_head_config.json",
    "decision_config.json",
    "joint_schema_model.py",
    "LICENSE",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)

PACKS: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "hub_name": "AX-Qwen3.5-0.8B-decision-MLX-AXQ-MXFP4-MTP",
        "recipe": ROOT / "examples" / "qwen35-0.8b-decision-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
    },
    "mxfp8": {
        "hub_name": "AX-Qwen3.5-0.8B-decision-MLX-AXQ-MXFP8-MTP",
        "recipe": ROOT / "examples" / "qwen35-0.8b-decision-axq-mxfp8-v0.1.yaml",
        "q_mode": "mxfp8",
        "product_class": "MXFP8",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def work_dir() -> Path:
    return Path(os.environ.get("QWEN35_DECISION_WORK", f"{FACTORY_CERT_ROOT}/qwen35-0.8b-decision"))


def source_dir() -> Path:
    return work_dir() / "src"


def merged_dir() -> Path:
    return work_dir() / "merged"


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
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep),
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
    return (
        (src / "config.json").is_file()
        and (src / ADAPTERS).is_file()
        and (src / "adapter_config.json").is_file()
        and (src / JOINT_HEAD).is_file()
        and (src / "model.safetensors.index.json").is_file()
    )


def _other_hf_download() -> bool:
    proc = subprocess.run(
        ["pgrep", "-fl", "bin/hf download"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if "bin/hf download" in line]
    return any(SOURCE["id"] not in line for line in lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_contract(src: Path) -> None:
    config = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("fine_tune_type") != "lora":
        raise SystemExit(f"expected a LoRA adapter, found {config.get('fine_tune_type')}")
    params = config.get("lora_parameters")
    if not isinstance(params, dict):
        params = config
    rank = params.get("rank", config.get("rank"))
    scale = params.get("scale", config.get("scale"))
    dropout = params.get("dropout", config.get("dropout"))
    if rank != RANK or scale != SCALE:
        raise SystemExit(f"expected rank {RANK} scale {SCALE}, found {config}")
    if dropout not in (0, 0.0):
        raise SystemExit(f"refusing dropout {dropout}")


def _pairs(adapter: dict[str, mx.array]) -> dict[str, tuple[mx.array, mx.array]]:
    grouped: dict[str, dict[str, mx.array]] = {}
    for key, value in adapter.items():
        if key == "__metadata__":
            continue
        stem = weight_key_for_lora(key)
        slot = "a" if key.endswith(".lora_a") else "b"
        grouped.setdefault(stem, {})[slot] = value
    pairs: dict[str, tuple[mx.array, mx.array]] = {}
    for stem, slots in grouped.items():
        if set(slots) != {"a", "b"}:
            raise SystemExit(f"LoRA factor pair is incomplete for {stem}")
        pairs[stem] = (slots["a"], slots["b"])
    if not pairs:
        raise SystemExit("adapter file has no LoRA factors")
    return pairs


def merge_lora(src: Path, dest: Path) -> None:
    """Write a BF16 snapshot with the LoRA folded into language Linear weights."""

    _adapter_contract(src)
    if (dest / "model.safetensors.index.json").is_file() and not (dest / ADAPTERS).exists():
        marker = dest / "lora_merge.json"
        if marker.is_file():
            log(f"reuse merged snapshot {dest}")
            return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    skip = {ADAPTERS, "adapter_config.json", JOINT_HEAD, "README.md"}
    for path in src.iterdir():
        if path.name in skip or path.name.startswith("."):
            continue
        target = dest / path.name
        if path.is_file():
            shutil.copy2(path, target)
    index_path = dest / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SystemExit("merged snapshot is missing weight_map")
    shards = sorted({name for name in weight_map.values() if isinstance(name, str)})
    pairs = _pairs(mx.load(str(src / ADAPTERS)))
    merged_names: list[str] = []
    for shard_name in shards:
        shard_path = dest / shard_name
        weights = mx.load(str(shard_path))
        changed = False
        for weight_name, (lora_a, lora_b) in pairs.items():
            if weight_name not in weights:
                continue
            base = weights[weight_name]
            if base.ndim != 2:
                raise SystemExit(f"{weight_name} is not a matrix")
            delta = lora_delta(
                np.array(lora_a.astype(mx.float32)),
                np.array(lora_b.astype(mx.float32)),
                scale=SCALE,
                rank=RANK,
            )
            if tuple(delta.shape) != tuple(base.shape):
                raise SystemExit(
                    f"{weight_name} shape {tuple(base.shape)} != LoRA delta {delta.shape}"
                )
            updated = base.astype(mx.float32) + mx.array(delta)
            weights[weight_name] = updated.astype(base.dtype)
            merged_names.append(weight_name)
            changed = True
        if changed:
            # The shard is memory-mapped. Saving over that path truncates it
            # before the unchanged tensors are copied out.
            mx.eval(*weights.values())
            temporary = shard_path.with_name("merged-shard.safetensors")
            mx.save_safetensors(str(temporary), weights)
            temporary.replace(shard_path)
    missing = sorted(set(pairs) - set(merged_names))
    if missing:
        raise SystemExit(f"LoRA targets missing from the backbone: {missing[:8]}")
    if len(merged_names) != len(pairs):
        raise SystemExit("LoRA targets were merged more than once")
    marker = {
        "source": SOURCE["id"],
        "revision": REV,
        "rank": RANK,
        "scale": SCALE,
        "merged_tensors": len(merged_names),
    }
    (dest / "lora_merge.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    log(f"merged {len(merged_names)} LoRA matrices into {dest}")


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_ready():
        log(f"reuse source {dest}")
        return
    if _other_hf_download():
        raise SystemExit("another hf download is running; refusing a second Xet client")
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


def cmd_merge() -> None:
    if not source_ready():
        raise SystemExit(f"missing source {source_dir()}; run download first")
    merge_lora(source_dir(), merged_dir())


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = PACKS[key]
    if not source_ready():
        raise SystemExit(f"missing source {source_dir()}; run download first")
    merge_lora(source_dir(), merged_dir())
    src = merged_dir()
    if (src / ADAPTERS).exists() or (src / JOINT_HEAD).exists():
        raise SystemExit("merged snapshot still contains the adapter or the joint head")
    work = work_dir()
    pack = pack_dir(key)
    (work / "logs").mkdir(parents=True, exist_ok=True)
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


def attach_sidecars(key: str) -> None:
    src = source_dir()
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    head_src = src / JOINT_HEAD
    if not head_src.is_file():
        raise SystemExit(f"missing {JOINT_HEAD}")
    head_dst = pack / JOINT_HEAD
    if not head_dst.is_file():
        shutil.copy2(head_src, head_dst)
    if sha256(head_src) != sha256(head_dst):
        raise SystemExit("joint head bytes changed")
    for name in OPTIONAL_SIDECARS:
        origin = src / name
        if origin.is_file() and not (pack / name).is_file():
            shutil.copy2(origin, pack / name)
            log(f"copied {name}")
    if not (pack / "LICENSE").is_file():
        raise SystemExit("pack is missing LICENSE")
    if not (pack / "decision_config.json").is_file():
        raise SystemExit("pack is missing decision_config.json")
    if (pack / ADAPTERS).exists() or (pack / "adapter_config.json").exists():
        raise SystemExit("pack still contains the unmerged adapter")
    readme = pack / "README.md"
    if readme.is_file() and "0.772" in readme.read_text(encoding="utf-8", errors="replace"):
        readme.unlink()
    mode = str(PACKS[key]["q_mode"]).upper()
    notice = f"""This directory is a quantized MLX derivative of {SOURCE["id"]}.

Changes from the upstream release:
- The published rank-16 LoRA (scale 1.0) was merged into the BF16 language weights
  before quantization. The adapter file is not in this pack.
- Language attention and MLP weights were quantized to {mode} by AXQuant.
- Token embeddings are 8-bit affine.
- Vision, MTP, and norms stay BF16.
- joint_head.safetensors is the original BF16 joint schema head, copied unchanged.
- decision_config.json is copied unchanged.

The upstream work is Apache-2.0. See LICENSE.
This pack does not repeat the upstream OptiQ decision scores.
"""
    (pack / "NOTICE").write_text(notice, encoding="utf-8")


def _reject_local_paths(pack: Path) -> None:
    for path in pack.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/Volumes/" in text or "/Users/" in text or "/home/devop/" in text:
            raise SystemExit(f"local path in {path.name}")
        if "0.772" in text:
            raise SystemExit(f"upstream score leaked into {path.name}")


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
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"unexpected model_type {config.get('model_type')}")
    if text_cfg.get("mtp_num_hidden_layers") != 1 and config.get("mtp_num_hidden_layers") != 1:
        raise SystemExit("MTP layer count was not preserved")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if not quant:
        raise SystemExit("converted config has no quantization block")
    if not (pack / JOINT_HEAD).is_file():
        raise SystemExit("joint head missing from pack")
    if not isinstance(head_cfg, dict) or head_cfg.get("hidden_size") != 1024:
        raise SystemExit("joint_head_config.json hidden_size is not 1024")
    _reject_local_paths(pack)
    payload = {
        "status": "pass",
        "check": "pack-integrity",
        "measured_main_bpw": manifest.get("measured_main_bpw"),
        "model_type": config.get("model_type"),
        "lora_merged": True,
        "joint_head_hidden_size": head_cfg.get("hidden_size"),
        "decision_forward": False,
        "note": (
            "Joint-head probabilities were not executed. "
            "mlx_lm.generate is not a decision-model check."
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
    _reject_local_paths(pack)
    readme = (pack / "README.md").read_text(encoding="utf-8")
    if "## Decision model" not in readme or "## Run with MLX-LM" in readme:
        raise SystemExit("model card does not describe the decision model")
    if "mlx_lm.generate \\\n" in readme:
        raise SystemExit("model card offers text generation as the product check")
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
    cmd_merge()
    cmd_convert(key)
    cmd_smoke(key)
    cmd_publish(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=[*sorted(PACKS), "all"], required=True)
    parser.add_argument(
        "stage",
        choices=["download", "merge", "convert", "smoke", "publish", "all"],
    )
    args = parser.parse_args()
    keys = ["mxfp4", "mxfp8"] if args.pack == "all" else [args.pack]
    if args.stage == "download":
        cmd_download()
        return 0
    if args.stage == "merge":
        cmd_merge()
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
