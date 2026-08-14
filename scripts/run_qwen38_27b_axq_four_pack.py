#!/usr/bin/env python3
"""Factory: Qwen3.8-27B AXQ 4/6-bit × {no-MTP, MTP} (four development packs).

Waits for a complete local BF16 snapshot, prepares a no-MTP source view (MTP
tensors live only in the last shard for this SKU), then runs four
architecture-prior simple converts.

Usage:
  /path/to/axquant/.venv/bin/python scripts/run_qwen38_27b_axq_four_pack.py

Environment overrides:
  QWEN38_BF16_DIR, QWEN38_WORK_DIR, QWEN38_OUT_DIR, QWEN38_REV,
  QWEN38_MODEL_ID, QWEN38_PROFILE, QWEN38_RUNTIME_SMOKE, QWEN38_SKIP_WAIT
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"
DEFAULT_BF16 = Path("/Volumes/Ext4T/models/Qwen3.8-27B-bf16")
DEFAULT_WORK = Path("/Volumes/Ext4T/axquant/work/qwen38-27b-axq")
DEFAULT_OUT = Path("/Volumes/Ext4T/models")

SHARD_COUNT = 18
EXPECTED_MTP_KEYS = 15


def log(msg: str) -> None:
    print(msg, flush=True)


def axquant_bin() -> Path:
    candidate = ROOT / ".venv" / "bin" / "axquant"
    if candidate.is_file():
        return candidate
    which = shutil.which("axquant")
    if which:
        return Path(which)
    raise SystemExit("axquant not found (expected .venv/bin/axquant)")


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def shard_name(i: int) -> str:
    return f"model-{i:05d}-of-{SHARD_COUNT:05d}.safetensors"


def source_complete(source: Path) -> tuple[bool, str]:
    if not source.is_dir():
        return False, f"missing directory {source}"
    required = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for name in required:
        if not (source / name).is_file():
            return False, f"missing {name}"
    missing = [shard_name(i) for i in range(1, SHARD_COUNT + 1) if not (source / shard_name(i)).is_file()]
    if missing:
        return False, f"missing {len(missing)} shards (e.g. {missing[0]})"
    # Reject incomplete downloads still writing *.incomplete under local-dir cache.
    incomplete = list((source / ".cache").rglob("*.incomplete")) if (source / ".cache").exists() else []
    if incomplete:
        return False, f"{len(incomplete)} incomplete download parts remain"
    return True, "ok"


def wait_for_source(source: Path, *, skip: bool) -> None:
    if skip:
        ok, reason = source_complete(source)
        if not ok:
            raise SystemExit(f"source not complete and QWEN38_SKIP_WAIT set: {reason}")
        return
    log(f"waiting for complete BF16 at {source}")
    while True:
        ok, reason = source_complete(source)
        if ok:
            log(f"source ready: {source}")
            return
        log(f"  not ready: {reason}; sleep 60s")
        time.sleep(60)


def mtp_keys_from_index(source: Path) -> dict[str, str]:
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    return {k: v for k, v in weight_map.items() if "mtp" in k.lower()}


def prepare_no_mtp_source(source: Path, dest: Path, work: Path) -> Path:
    """Build a no-MTP BF16 view: symlink non-MTP files, rewrite MTP-bearing shards.

    Uses MLX for BF16 safetensors I/O (numpy cannot materialize bfloat16).
    """
    import mlx.core as mx
    from safetensors import safe_open

    if dest.exists():
        marker = dest / ".axquant_no_mtp_ready"
        if marker.is_file():
            log(f"reusing no-MTP source {dest}")
            return dest
        log(f"removing incomplete no-MTP dir {dest}")
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)
    mtp_map = mtp_keys_from_index(source)
    if not mtp_map:
        raise SystemExit(f"no MTP tensors in index under {source}; cannot build no-MTP cut")
    mtp_shards = sorted(set(mtp_map.values()))
    log(f"MTP keys={len(mtp_map)} in shards={mtp_shards}")
    if len(mtp_map) < EXPECTED_MTP_KEYS:
        log(f"warning: expected ~{EXPECTED_MTP_KEYS} MTP keys, found {len(mtp_map)}")

    # Copy/symlink non-weight config files.
    skip_names = {
        "model.safetensors.index.json",
        ".cache",
        ".gitattributes",
    }
    for path in source.iterdir():
        if path.name in skip_names or path.name.startswith("."):
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        target = dest / path.name
        if path.is_dir():
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(path.resolve())

    # Link weight shards that have no MTP tensors; rewrite those that do.
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = dict(index["weight_map"])
    for i in range(1, SHARD_COUNT + 1):
        name = shard_name(i)
        src_shard = source / name
        dst_shard = dest / name
        if name not in mtp_shards:
            if dst_shard.exists() or dst_shard.is_symlink():
                dst_shard.unlink()
            dst_shard.symlink_to(src_shard.resolve())
            continue
        log(f"rewriting {name} without MTP tensors (MLX BF16 path)")
        # List keys without decoding BF16 via numpy.
        with safe_open(str(src_shard), framework="np") as handle:
            all_keys = list(handle.keys())
        non_mtp_keys = [key for key in all_keys if "mtp" not in key.lower()]
        if not non_mtp_keys:
            log(f"  {name} was MTP-only; omitting file")
            for key, shard in list(weight_map.items()):
                if shard == name:
                    del weight_map[key]
            continue
        loaded = mx.load(str(src_shard))
        keep = {key: loaded[key] for key in non_mtp_keys}
        mx.save_safetensors(str(dst_shard), keep)
        del loaded, keep
        for key in list(weight_map):
            if weight_map[key] == name and "mtp" in key.lower():
                del weight_map[key]

    # Drop any remaining MTP map entries.
    for key in list(weight_map):
        if "mtp" in key.lower():
            del weight_map[key]

    # Recompute metadata total_size if present (best-effort).
    metadata = dict(index.get("metadata") or {})
    if "total_size" in metadata:
        total = 0
        for shard in sorted(set(weight_map.values())):
            total += (dest / shard).stat().st_size
        metadata["total_size"] = total
    new_index = {"metadata": metadata, "weight_map": weight_map}
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2) + "\n", encoding="utf-8"
    )

    # Clear MTP declaration in config so inventory is non-MTP.
    config_path = dest / "config.json"
    # config.json is a symlink to source; replace with a rewritten copy.
    if config_path.is_symlink() or config_path.exists():
        config_path.unlink()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config")
    if isinstance(text, dict):
        text = dict(text)
        text["mtp_num_hidden_layers"] = 0
        # Keep other mtp_* keys explicit when present.
        if "mtp_use_dedicated_embeddings" in text:
            text["mtp_use_dedicated_embeddings"] = False
        config["text_config"] = text
    config["_name_or_path"] = DEFAULT_MODEL_ID
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    note = {
        "kind": "qwen38-27b-no-mtp-bf16-view",
        "source": str(source),
        "mtp_keys_removed": sorted(mtp_map),
        "mtp_shards_rewritten": mtp_shards,
        "notes": [
            "Development prepare: MTP tensors and mtp_num_hidden_layers cleared.",
            "Non-MTP AXQ packs convert from this tree only.",
        ],
    }
    (dest / "axquant_no_mtp_prepare.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    (dest / ".axquant_no_mtp_ready").write_text("ok\n", encoding="utf-8")
    log(f"no-MTP source ready at {dest}")
    return dest


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}")
    log(f"  log -> {log_path}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(cmd)}\n\n")
        handle.flush()
        # Large BF16 re-packs often trip Metal command-buffer errors; CPU path
        # matches GPT-OSS / other 27B+ factory converts (AXQUANT_FORCE_CPU=1).
        env = {
            **os.environ,
            "HF_HOME": os.environ.get("HF_HOME", "/Volumes/Ext4T/huggingface"),
            "HF_HUB_CACHE": os.environ.get(
                "HF_HUB_CACHE", "/Volumes/Ext4T/huggingface/hub"
            ),
            "AXQUANT_FORCE_CPU": os.environ.get("AXQUANT_FORCE_CPU", "1"),
        }
        proc = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=str(ROOT),
            env=env,
        )
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)} (see {log_path})")


def convert_pack(
    *,
    axquant: Path,
    model_dir: Path,
    model_id: str,
    revision: str,
    target_bpw: float,
    output: Path,
    profile: str,
    runtime_smoke: str,
    log_path: Path,
) -> None:
    marker = output / "axquant_manifest.json"
    if output.is_dir() and marker.is_file():
        log(f"skip convert (exists): {output}")
        return
    if output.exists():
        log(f"removing incomplete output {output}")
        shutil.rmtree(output)
    cmd = [
        str(axquant),
        "quantize",
        str(model_dir),
        "--model-id",
        model_id,
        "--revision",
        revision,
        "--target-bpw",
        f"{target_bpw:.1f}",
        "--ladder",
        "prior",
        "--profile",
        profile,
        "--runtime-smoke",
        runtime_smoke,
        "--output",
        str(output),
        "--json",
        str(log_path.with_suffix(".summary.json")),
    ]
    run_cmd(cmd, log_path)


def inspect_model(
    axquant: Path,
    model_dir: Path,
    model_id: str,
    revision: str,
    out_json: Path,
) -> None:
    run_cmd(
        [
            str(axquant),
            "inspect",
            "--model",
            str(model_dir),
            "--model-id",
            model_id,
            "--revision",
            revision,
            "--output",
            str(out_json),
        ],
        out_json.with_suffix(".log"),
    )


def main() -> int:
    revision = os.environ.get("QWEN38_REV", DEFAULT_REV)
    model_id = os.environ.get("QWEN38_MODEL_ID", DEFAULT_MODEL_ID)
    source = env_path("QWEN38_BF16_DIR", DEFAULT_BF16)
    work = env_path("QWEN38_WORK_DIR", DEFAULT_WORK)
    out_root = env_path("QWEN38_OUT_DIR", DEFAULT_OUT)
    profile = os.environ.get("QWEN38_PROFILE", "agent-coding")
    runtime_smoke = os.environ.get("QWEN38_RUNTIME_SMOKE", "none")
    skip_wait = os.environ.get("QWEN38_SKIP_WAIT", "").strip() in {"1", "true", "yes"}

    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = work / "status.json"
    axquant = axquant_bin()

    status: dict[str, object] = {
        "model_id": model_id,
        "revision": revision,
        "source": str(source),
        "work": str(work),
        "out_root": str(out_root),
        "profile": profile,
        "stage": "start",
        "packs": {},
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    wait_for_source(source, skip=skip_wait)
    status["stage"] = "source_ready"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    mtp_map = mtp_keys_from_index(source)
    log(f"source MTP keys: {len(mtp_map)}")
    if not mtp_map:
        raise SystemExit("expected MTP tensors on Qwen3.8-27B source")

    no_mtp = prepare_no_mtp_source(source, work / "Qwen3.8-27B-bf16-no-mtp", work)
    status["no_mtp_source"] = str(no_mtp)
    status["stage"] = "no_mtp_prepared"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    inspect_model(axquant, source, model_id, revision, work / "inventory-full-mtp.json")
    inspect_model(axquant, no_mtp, model_id, revision, work / "inventory-no-mtp.json")

    packs = [
        # (label, model_dir, bpw, out_name)
        ("axq4-mtp", source, 4.0, "AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP"),
        ("axq6-mtp", source, 6.0, "AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP"),
        ("axq4", no_mtp, 4.0, "AX-Qwen3.8-27B-MLX-AXQ-4bit"),
        ("axq6", no_mtp, 6.0, "AX-Qwen3.8-27B-MLX-AXQ-6bit"),
    ]

    pack_status: dict[str, object] = {}
    for label, model_dir, bpw, name in packs:
        output = out_root / name
        log_path = logs / f"convert-{label}.log"
        status["stage"] = f"convert:{label}"
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        try:
            convert_pack(
                axquant=axquant,
                model_dir=model_dir,
                model_id=model_id,
                revision=revision,
                target_bpw=bpw,
                output=output,
                profile=profile,
                runtime_smoke=runtime_smoke,
                log_path=log_path,
            )
            pack_status[label] = {
                "output": str(output),
                "ok": (output / "axquant_manifest.json").is_file(),
                "log": str(log_path),
            }
        except SystemExit as exc:
            pack_status[label] = {
                "output": str(output),
                "ok": False,
                "error": str(exc),
                "log": str(log_path),
            }
            status["packs"] = pack_status
            status["stage"] = f"failed:{label}"
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            raise

    status["packs"] = pack_status
    status["stage"] = "done"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    log("all four packs finished")
    for label, info in pack_status.items():
        log(f"  {label}: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
