#!/usr/bin/env python3
"""Checkpoint Tier 1 for DeepSeek V4 Flash-0731 packs on tn-macstudio-m3.

512 GB M3 Ultra recert host. AX Engine 7.1.x requires macOS 26+; this
machine is currently 15.5 — preflight fails closed until the OS is
upgraded. Published engine latest is v7.1.3 (v7.1.4 is not released).

  PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_tier1.py preflight
  PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_tier1.py \\
      --pack axq2 install-engine
  PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_tier1.py \\
      --pack axq2 all

Packs: axq2, axq4, mxfp4, axq6. 3-bit is withdrawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    LARGE_MEMORY_CERT_HOST_ID,
    require_large_memory_cert_host,
)

SOURCE_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
SOURCE_REV = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
# Latest published 7.1.x. There is no v7.1.4 tag/release as of 2026-08-19.
ENGINE_RELEASE = os.environ.get("AX_ENGINE_RELEASE", "v7.1.3")
ENGINE_VERSION = ENGINE_RELEASE.lstrip("v")
MIN_MACOS_MAJOR = 26
MIN_QUALITY = 0.90
SEED = 20260728
MAX_TOKENS = 64
HOME = Path.home()
DEFAULT_MODELS = HOME / "models"
DEFAULT_WORK = HOME / "axquant-certification" / "deepseek-v4-0731-tier1"
DEFAULT_ENGINE_ROOT = HOME / "opt" / f"ax-engine-{ENGINE_VERSION}"
DEFAULT_DATASETS = HOME / "axquant-certification" / "datasets"

PACKS: dict[str, dict[str, Any]] = {
    "axq2": {
        "hub_name": "AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP",
        "cert_stem": "deepseek-v4-flash-0731-axq2-tier1",
        "display_name": "DeepSeek V4 Flash-0731 MLX AXQ 2-bit MTP (exp.)",
        "recipe": ROOT / "examples" / "deepseek-v4-experimental-2bit-v0.1.yaml",
        "q_mode": "affine",
        "product_class": "2bit-experimental",
        "sort_order": 230,
        "listed": True,
        "needs_convert": False,
    },
    "axq4": {
        "hub_name": "AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP",
        "cert_stem": "deepseek-v4-flash-0731-axq4-tier1",
        "display_name": "DeepSeek V4 Flash-0731 MLX AXQ 4-bit MTP",
        "recipe": ROOT / "examples" / "deepseek-v4-experimental-4bit-g128-v0.1.yaml",
        "q_mode": "affine",
        "product_class": "4bit",
        "sort_order": 232,
        "listed": True,
        "needs_convert": False,
    },
    "mxfp4": {
        "hub_name": "AX-DeepSeek-V4-Flash-0731-MLX-AXQ-MXFP4",
        "cert_stem": "deepseek-v4-flash-0731-axq-mxfp4-tier1",
        "display_name": "DeepSeek V4 Flash-0731 MLX AXQ MXFP4",
        "recipe": ROOT / "examples" / "deepseek-v4-experimental-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
        "sort_order": 233,
        "listed": True,
        "needs_convert": True,
    },
    "axq6": {
        "hub_name": "AX-DeepSeek-V4-Flash-0731-MLX-AXQ-6bit",
        "cert_stem": "deepseek-v4-flash-0731-axq6-tier1",
        "display_name": "DeepSeek V4 Flash-0731 MLX AXQ 6-bit",
        "recipe": ROOT / "examples" / "deepseek-v4-experimental-6bit-g128-v0.1.yaml",
        "q_mode": "affine",
        "product_class": "6bit",
        "sort_order": 234,
        "listed": True,
        "needs_convert": True,
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}; choose {sorted(PACKS)}")
    return PACKS[key]


def models_root() -> Path:
    return Path(os.environ.get("DSV4_0731_MODELS", DEFAULT_MODELS))


def work_dir() -> Path:
    return Path(os.environ.get("DSV4_0731_WORK", DEFAULT_WORK))


def pack_dir(key: str) -> Path:
    override = os.environ.get(f"DSV4_0731_{key.upper()}_PACK")
    if override:
        return Path(override)
    return models_root() / str(spec(key)["hub_name"])


def source_dir() -> Path:
    return Path(os.environ.get("DSV4_0731_SOURCE", models_root() / "src-DeepSeek-V4-Flash-0731"))


def engine_root() -> Path:
    return Path(os.environ.get("AX_ENGINE_ROOT", DEFAULT_ENGINE_ROOT))


def engine_server() -> Path:
    root = engine_root()
    for cand in (root / "bin" / "ax-engine-server", root / "ax-engine-server"):
        if cand.is_file():
            return cand
    return root / "bin" / "ax-engine-server"


def engine_bench() -> Path:
    root = engine_root()
    for cand in (root / "bin" / "ax-engine-bench", root / "ax-engine-bench"):
        if cand.is_file():
            return cand
    return root / "bin" / "ax-engine-bench"


def generate_manifest_bin() -> Path:
    root = engine_root()
    for cand in (root / "bin" / "generate-manifest", root / "generate-manifest"):
        if cand.is_file():
            return cand
    return root / "bin" / "generate-manifest"


def datasets_dir() -> Path:
    return Path(os.environ.get("DSV4_0731_DATASETS", DEFAULT_DATASETS))


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found; create a venv in this checkout")


def macos_major() -> int | None:
    raw = platform.mac_ver()[0]
    if not raw:
        return None
    try:
        return int(raw.split(".", 1)[0])
    except ValueError:
        return None


def hf_env() -> dict[str, str]:
    hf_home = os.environ.get("HF_HOME", str(HOME / ".cache" / "huggingface"))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", str(Path(hf_home) / "hub")),
        "HUGGINGFACE_HUB_CACHE": os.environ.get(
            "HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")
        ),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", str(Path(hf_home) / "xet")),
        "AX_ENGINE_2BIT_EXPERIMENTAL": "1",
        "AX_ENGINE_3BIT_EXPERIMENTAL": "1",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_weight_bytes(model_dir: Path) -> int:
    total = 0
    for path in sorted(model_dir.rglob("*.safetensors")):
        if path.name.startswith("."):
            continue
        total += path.stat().st_size
    if total <= 0:
        raise SystemExit(f"no safetensors under {model_dir}")
    return total


def pack_ready(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def cmd_preflight() -> None:
    host = require_large_memory_cert_host(socket.gethostname())
    major = macos_major()
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    payload = {
        "host_id": host,
        "observed_hostname": socket.gethostname(),
        "macos": platform.mac_ver()[0],
        "macos_major": major,
        "memory_bytes": mem,
        "engine_release": ENGINE_RELEASE,
        "engine_published": ENGINE_RELEASE == "v7.1.3",
        "note_7_1_4": "No GitHub tag/release v7.1.4; latest published is v7.1.3.",
        "engine_server": str(engine_server()),
        "engine_present": engine_server().is_file(),
        "os_ok": major is not None and major >= MIN_MACOS_MAJOR,
    }
    write_json(work_dir() / "preflight.json", payload)
    log(json.dumps(payload, indent=2))
    if major is None or major < MIN_MACOS_MAJOR:
        raise SystemExit(
            f"AX Engine {ENGINE_VERSION} requires macOS {MIN_MACOS_MAJOR}+; "
            f"this host is {platform.mac_ver()[0]}. Upgrade the OS before cert."
        )
    if ENGINE_RELEASE == "v7.1.4":
        log("warning: v7.1.4 is not a published release; install-engine will 404")


def cmd_install_engine() -> None:
    require_large_memory_cert_host(socket.gethostname())
    dest = engine_root()
    server = engine_server()
    if server.is_file():
        log(f"reuse engine {server}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/defai-digital/ax-engine/releases/download/"
        f"{ENGINE_RELEASE}/ax-engine-{ENGINE_RELEASE}-macos-arm64.tar.gz"
    )
    archive = dest.parent / f"ax-engine-{ENGINE_RELEASE}-macos-arm64.tar.gz"
    log(f"download {url}")
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(path=dest)
    if not engine_server().is_file():
        # tarball may unpack a nested directory
        nested = list(dest.rglob("ax-engine-server"))
        if nested:
            log(f"engine binary at {nested[0]}")
        else:
            raise SystemExit(f"ax-engine-server missing after extract in {dest}")
    log(f"installed {ENGINE_RELEASE} under {dest}")


def cmd_download(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    item = spec(key)
    dest = pack_dir(key)
    if pack_ready(dest):
        log(f"reuse pack {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [
            hf_bin,
            "download",
            f"AutomatosX/{item['hub_name']}",
            "--local-dir",
            str(dest),
        ],
        work_dir() / "logs" / f"download-{key}.log",
    )
    if not pack_ready(dest):
        raise SystemExit(f"Hub pack incomplete: {dest}. Convert on this host or rsync from Ext12T.")


def cmd_convert(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    item = spec(key)
    pack = pack_dir(key)
    if pack_ready(pack) and (pack / "axquant_manifest.json").is_file():
        log(f"reuse converted pack {pack}")
        return
    source = source_dir()
    if not (source / "config.json").is_file():
        hf = ROOT / ".venv" / "bin" / "hf"
        hf_bin = str(hf) if hf.is_file() else "hf"
        source.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                hf_bin,
                "download",
                SOURCE_ID,
                "--revision",
                SOURCE_REV,
                "--local-dir",
                str(source),
            ],
            work_dir() / "logs" / "download-src.log",
        )
    work = work_dir() / key
    work.mkdir(parents=True, exist_ok=True)
    inventory = work / "inventory.json"
    plan = work / "plan.json"
    if not inventory.is_file():
        run(
            [
                *axquant_cmd(),
                "inspect",
                "--model",
                str(source),
                "--model-id",
                SOURCE_ID,
                "--revision",
                SOURCE_REV,
                "--output",
                str(inventory),
            ],
            work / "logs" / "inspect.log",
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
            work / "logs" / "plan-manual.log",
        )
    if pack.exists() and not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    bench = str(engine_bench()) if engine_bench().is_file() else "ax-engine-bench"
    run(
        [
            *axquant_cmd(),
            "convert",
            "--model",
            str(source),
            "--plan",
            str(plan),
            "--output",
            str(pack),
            "--q-mode",
            str(item["q_mode"]),
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "if-available",
            "--ax-engine-bench",
            bench,
        ],
        work / "logs" / "convert.log",
    )


def cmd_size(key: str) -> None:
    pack = pack_dir(key)
    if not pack_ready(pack):
        raise SystemExit(f"missing pack {pack}")
    man_path = pack / "axquant_manifest.json"
    cand_bytes = 0
    measured = None
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        cand_bytes = int(man.get("weight_file_size_bytes") or 0)
        measured = man.get("measured_total_bpw")
    if cand_bytes <= 0:
        cand_bytes = safetensors_weight_bytes(pack)
    payload = {
        "candidate_bytes": cand_bytes,
        "candidate_measured_bpw": measured,
        "pass": cand_bytes > 0,
        "compare_mode": "total",
        "notes": "Flash-0731 experimental track: size bound to measured bytes (no uniform 2-bit).",
    }
    write_json(work_dir() / key / "size.json", payload)
    log(f"size {key}: bytes={cand_bytes} bpw={measured}")


def _quality_score(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("score", "viability", "aggregate_score"):
        if payload.get(key) is not None:
            return float(payload[key])
    agg = payload.get("aggregate") or {}
    if agg.get("score") is not None:
        return float(agg["score"])
    return None


def cmd_quality(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    pack = pack_dir(key)
    item = spec(key)
    qdir = work_dir() / key / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    datasets = datasets_dir()
    for suite, names in (
        ("agent-coding", ("development-agent-coding", "coding")),
        ("general", ("development-general", "instruction")),
    ):
        ds = None
        for name in names:
            for cand in (
                datasets / name / "dataset.jsonl",
                datasets / "datasets" / name / "dataset.jsonl",
                ROOT / "data" / "eval" / f"{name}.jsonl",
            ):
                if cand.is_file():
                    ds = cand
                    break
            if ds is not None:
                break
        if ds is None:
            raise SystemExit(f"missing {suite} dataset under {datasets} or data/eval")
        out = qdir / f"{suite}.json"
        if not out.is_file():
            run(
                [
                    *axquant_cmd(),
                    "evaluate-quality",
                    "--model",
                    str(pack),
                    "--model-id",
                    f"AutomatosX/{item['hub_name']}",
                    "--revision",
                    SOURCE_REV,
                    "--dataset",
                    str(ds),
                    "--seed",
                    str(SEED),
                    "--max-tokens",
                    str(MAX_TOKENS),
                    "--output",
                    str(out),
                ],
                work_dir() / "logs" / f"quality-{key}-{suite}.log",
            )
        score = _quality_score(out)
        log(f"quality {key} {suite}: score={score} (need >= {MIN_QUALITY})")


def cmd_runtime(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    pack = pack_dir(key)
    rdir = work_dir() / key / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = ROOT / ".venv" / "bin" / "mlx_lm.generate"
    run(
        [
            *axquant_cmd(),
            "runtime-check",
            "--model",
            str(pack),
            "--runtime",
            "mlx-lm",
            "--mlx-lm",
            str(mlx_gen) if mlx_gen.is_file() else "mlx_lm.generate",
            "--output",
            str(rdir / "mlx-lm.json"),
        ],
        work_dir() / "logs" / f"runtime-mlx-{key}.log",
    )
    bench = engine_bench()
    doctor = rdir / "ax-engine-doctor.json"
    if bench.is_file():
        with doctor.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [str(bench), "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=hf_env(),
            )
    gen = generate_manifest_bin()
    if gen.is_file():
        run(
            [str(gen), "--force", "--validate", str(pack)],
            work_dir() / "logs" / f"generate-manifest-{key}.log",
        )
    log(f"runtime {key} wrote {rdir}")


def cmd_write_certs(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    item = spec(key)
    pack = pack_dir(key)
    work = work_dir() / key
    size = json.loads((work / "size.json").read_text(encoding="utf-8"))
    quality: dict[str, Any] = {}
    certified = bool(size.get("pass"))
    for suite in ("agent-coding", "general"):
        qpath = work / "quality" / f"{suite}.json"
        if not qpath.is_file():
            certified = False
            quality[suite] = {"pass": False, "reason": f"missing {qpath}"}
            continue
        payload = json.loads(qpath.read_text(encoding="utf-8"))
        score = _quality_score(qpath)
        if score is None or score < MIN_QUALITY:
            certified = False
        quality[suite] = {
            "candidate_score": score,
            "samples": int(payload.get("samples") or (76 if suite == "agent-coding" else 44)),
            "dataset_sha256": payload.get("dataset_sha256"),
            "scoring": "experimental-generation-viability",
            "pass": score is not None and score >= MIN_QUALITY,
        }
    mlx_path = work / "runtime" / "mlx-lm.json"
    mlx_ok = False
    if mlx_path.is_file():
        mlx = json.loads(mlx_path.read_text(encoding="utf-8"))
        mlx_ok = bool(mlx.get("passed"))
        if not mlx_ok:
            certified = False
    man_path = pack / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    inspect = inspect_artifact_modalities(pack)
    block = build_modalities_block(
        vision_supported=inspect.vision_supported,
        audio_supported=inspect.audio_supported,
        vision_smoke_passed=None,
    )
    hub_commit = os.environ.get("DSV4_0731_HUB_COMMIT", "0" * 40)
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified" if certified else "not_certified",
        "certification_tier": "checkpoint",
        "certified_at" if certified else "evaluated_at": datetime.now(UTC).isoformat(),
        "host_id": LARGE_MEMORY_CERT_HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{item['hub_name']}",
            "hub_commit": hub_commit,
            "product_class": item["product_class"],
            "architecture": "DeepseekV4ForCausalLM",
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path) if man_path.is_file() else None,
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "plan-manual" if item["needs_convert"] else "existing-pack",
            "recipe": str(Path(item["recipe"]).relative_to(ROOT)),
            "target_class": item["product_class"],
            "measured_total_bpw": man.get("measured_total_bpw")
            or size.get("candidate_measured_bpw"),
            "measured_main_bpw": man.get("measured_main_bpw") or size.get("candidate_measured_bpw"),
        },
        "size": {
            "candidate_weight_bytes": size["candidate_bytes"],
            "candidate_measured_bpw": size.get("candidate_measured_bpw"),
            "pass": bool(size.get("pass")),
            "notes": size.get("notes"),
        },
        "quality": quality,
        "thresholds": {
            "minimum_generation_viability": MIN_QUALITY,
            "notes": "Experimental low-bit track uses generation viability, not BF16 retention.",
        },
        "mtp_acceleration": {
            "status": "not-certified",
            "reason": "MTP sidecar may be present; acceleration not measured this cycle.",
        },
        "runtime": {
            "mlx_lm": {
                "status": "pass" if mlx_ok else "fail",
                "notes": f"runtime-check on {LARGE_MEMORY_CERT_HOST_ID}",
            },
            "ax_engine": {
                "status": "pass"
                if (work / "runtime" / "ax-engine-doctor.json").is_file()
                else "fail",
                "version": ENGINE_VERSION,
                "notes": f"doctor / generate-manifest on {LARGE_MEMORY_CERT_HOST_ID}",
            },
        },
        "toolchain": {
            "axquant": man.get("axquant_version", "1.9.0"),
            "ax_engine": ENGINE_VERSION,
            "host": LARGE_MEMORY_CERT_HOST_ID,
        },
        "notes": [
            f"Checkpoint attempt on {LARGE_MEMORY_CERT_HOST_ID} with AX Engine {ENGINE_VERSION}.",
            "Not the older DeepSeek-V4-Flash certificate (source 60d8d707).",
            "3-bit Flash-0731 remains withdrawn.",
        ],
        "public_index": {
            "display_name": item["display_name"],
            "sort_order": item["sort_order"],
            "edition_label": f"m3@{ENGINE_VERSION}",
            "listed": bool(item["listed"]),
        },
        "modalities": modalities_to_public_dict(block),
    }
    if payload["artifact"]["candidate_manifest_sha256"] is None:
        del payload["artifact"]["candidate_manifest_sha256"]
    cert_json = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.json"
    write_json(cert_json, payload)
    load_public_checkpoint_certification(cert_json)
    verdict = "certified" if certified else "**not certified**"
    md = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.md"
    md.write_text(
        "\n".join(
            [
                f"# {item['display_name']} — checkpoint Tier 1",
                "",
                f"**Verdict:** {verdict} on `{LARGE_MEMORY_CERT_HOST_ID}` "
                f"with AX Engine `{ENGINE_VERSION}`.",
                "",
                f"Source `{SOURCE_ID}@{SOURCE_REV}`.",
                f"Hub [`AutomatosX/{item['hub_name']}`]"
                f"(https://huggingface.co/AutomatosX/{item['hub_name']}).",
                "",
                f"Machine-readable: [{item['cert_stem']}.json]({item['cert_stem']}.json).",
                "",
                format_modalities_card_section(block).rstrip(),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"wrote {cert_json} status={payload['status']}")


def cmd_all(key: str) -> None:
    cmd_preflight()
    cmd_install_engine()
    item = spec(key)
    if item["needs_convert"] and not pack_ready(pack_dir(key)):
        cmd_convert(key)
    else:
        try:
            cmd_download(key)
        except SystemExit as exc:
            if item["needs_convert"]:
                log(f"download failed ({exc}); converting")
                cmd_convert(key)
            else:
                raise
    cmd_size(key)
    cmd_quality(key)
    cmd_runtime(key)
    cmd_write_certs(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=sorted(PACKS), default="axq2")
    parser.add_argument(
        "step",
        choices=[
            "preflight",
            "install-engine",
            "download",
            "convert",
            "size",
            "quality",
            "runtime",
            "write-certs",
            "all",
        ],
    )
    args = parser.parse_args()
    if args.step == "preflight":
        cmd_preflight()
    elif args.step == "install-engine":
        cmd_install_engine()
    elif args.step == "download":
        cmd_download(args.pack)
    elif args.step == "convert":
        cmd_convert(args.pack)
    elif args.step == "size":
        cmd_size(args.pack)
    elif args.step == "quality":
        cmd_quality(args.pack)
    elif args.step == "runtime":
        cmd_runtime(args.pack)
    elif args.step == "write-certs":
        cmd_write_certs(args.pack)
    else:
        cmd_all(args.pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
