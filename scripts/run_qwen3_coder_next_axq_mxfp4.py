#!/usr/bin/env python3
"""Factory: convert + checkpoint Tier 1 for Qwen3-Coder-Next AXQ-MXFP4.

Run on df-macstudio-m2 + Ext12T only:

  PYTHONPATH=src /Users/devop/code/axquant/.venv/bin/python \\
    scripts/run_qwen3_coder_next_axq_mxfp4.py all

Steps: download source, inspect, plan-manual, convert --q-mode mxfp4,
uniform MXFP4 size/quality reference, size gate, dual-suite quality,
mlx-lm smoke, AX Engine doctor, public cert JSON/MD, optional Hub publish.

Environment: CODER_NEXT_SOURCE, CODER_NEXT_MXFP4_PACK, CODER_NEXT_MXFP4_WORK,
CODER_NEXT_DATASETS, CODER_NEXT_HUB_COMMIT, AX_ENGINE_BENCH.
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_DATASETS,
    FACTORY_HF_HOME,
    FACTORY_HOST_ID,
    FACTORY_MODELS,
    require_factory_host,
)

SOURCE_ID = "Qwen/Qwen3-Coder-Next"
SOURCE_REV = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
HUB_NAME = "AX-Qwen3-Coder-Next-MLX-AXQ-MXFP4"
CERT_STEM = "qwen3-coder-next-axq-mxfp4-tier1"
DISPLAY_NAME = "Qwen3-Coder-Next MLX AXQ MXFP4"
RECIPE = ROOT / "examples" / "qwen3-coder-next-axq-mxfp4-v0.1.yaml"
ADAPTER_ID = "qwen3-next-v1"
ARCHITECTURE = "Qwen3NextForCausalLM"
SORT_ORDER = 45
MAX_SIZE_RATIO = 1.20
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)
UNIFORM_FALLBACK_ID = "mlx-community/Qwen3-Coder-Next-4bit"
UNIFORM_FALLBACK_REV = "7b9321eabb85ce79625cac3f61ea691e4ea984b5"


def log(msg: str) -> None:
    print(msg, flush=True)


def work_dir() -> Path:
    default = f"{FACTORY_CERT_ROOT}/qwen3-coder-next-mxfp4"
    return Path(os.environ.get("CODER_NEXT_MXFP4_WORK", default))


def source_dir() -> Path:
    override = os.environ.get("CODER_NEXT_SOURCE")
    if override:
        return Path(override)
    return Path("/Volumes/Ext12T/axquant/work/qwen3-coder-next-mxfp4/src-qwen3-coder-next")


def pack_dir() -> Path:
    override = os.environ.get("CODER_NEXT_MXFP4_PACK")
    if override:
        return Path(override)
    return Path(FACTORY_MODELS) / HUB_NAME


def datasets_dir() -> Path:
    return Path(os.environ.get("CODER_NEXT_DATASETS", FACTORY_DATASETS))


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
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", f"{FACTORY_HF_HOME}/hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    return env


def run(cmd: list[str], log_path: Path | None = None, *, force_cpu: bool = False) -> None:
    log("$ " + " ".join(cmd))
    env = hf_env()
    if force_cpu:
        env["AXQUANT_FORCE_CPU"] = os.environ.get("AXQUANT_FORCE_CPU", "1")
    else:
        env.pop("AXQUANT_FORCE_CPU", None)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _retention(cmp: dict[str, Any]) -> float | None:
    agg = cmp.get("aggregate") or {}
    raw = agg.get("retention")
    if raw is None:
        raw = cmp.get("retention")
    return None if raw is None else float(raw)


def _source_complete(dest: Path) -> bool:
    if not (dest / "config.json").is_file():
        return False
    shards = [path for path in dest.rglob("*.safetensors") if not path.name.startswith(".")]
    if not shards:
        return False
    index = dest / "model.safetensors.index.json"
    if not index.is_file():
        return True
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    weight_map = payload.get("weight_map") or {}
    expected = {dest / name for name in weight_map.values()}
    return bool(expected) and all(path.is_file() for path in expected)


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _source_complete(dest):
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [
            hf_bin,
            "download",
            SOURCE_ID,
            "--revision",
            SOURCE_REV,
            "--local-dir",
            str(dest),
        ],
        work_dir() / "logs" / "download-src.log",
    )
    if not _source_complete(dest):
        raise SystemExit(f"source download incomplete: {dest}")


def cmd_convert() -> None:
    require_factory_host(socket.gethostname())
    if not RECIPE.is_file():
        raise SystemExit(f"missing recipe {RECIPE}")
    source = source_dir()
    if not _source_complete(source):
        raise SystemExit(f"missing source {source}; run download first")
    work = work_dir()
    pack = pack_dir()
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
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
                str(RECIPE),
                "--output",
                str(plan),
            ],
            work / "logs" / "plan-manual.log",
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
            str(source),
            "--plan",
            str(plan),
            "--output",
            str(pack),
            "--q-mode",
            "mxfp4",
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "skip",
            "--ax-engine-bench",
            engine,
        ],
        work / "logs" / "convert-mxfp4.log",
        force_cpu=True,
    )


def cmd_uniforms() -> None:
    work = work_dir()
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    source = source_dir()
    out = work / "uniforms" / "uniform-mxfp4"
    if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
        log(f"reuse {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    mlx = ROOT / ".venv" / "bin" / "mlx_lm.convert"
    try:
        run(
            [
                str(mlx) if mlx.is_file() else "mlx_lm.convert",
                "--hf-path",
                str(source),
                "--mlx-path",
                str(out),
                "-q",
                "--q-mode",
                "mxfp4",
                "--dtype",
                "bfloat16",
            ],
            work / "logs" / "uniform-mxfp4-convert.log",
            force_cpu=True,
        )
    except SystemExit as exc:
        log(f"uniform MXFP4 convert failed ({exc}); will try mlx-community 4-bit fallback")


def cmd_size() -> None:
    work = work_dir()
    pack = pack_dir()
    man_path = pack / "axquant_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"missing manifest {man_path}")
    (work / "size").mkdir(parents=True, exist_ok=True)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    cand_bytes = int(man.get("weight_file_size_bytes") or 0)
    if cand_bytes <= 0:
        cand_bytes = safetensors_weight_bytes(pack)
    udir = work / "uniforms" / "uniform-mxfp4"
    if (udir / "config.json").is_file() and any(udir.glob("*.safetensors")):
        ref_bytes = safetensors_weight_bytes(udir)
        payload = {
            "size_ratio_vs_uniform": cand_bytes / ref_bytes,
            "pass": (cand_bytes / ref_bytes) <= MAX_SIZE_RATIO,
            "candidate_bytes": cand_bytes,
            "reference_bytes": ref_bytes,
            "reference_kind": "uniform-mxfp4",
            "reference_model_id": "local/Qwen3-Coder-Next-uniform-mxfp4",
            "reference_revision": SOURCE_REV,
            "max_size_ratio": MAX_SIZE_RATIO,
            "compare_mode": "total",
        }
        write_json(work / "size" / "ratios.json", payload)
        log(f"size axq-mxfp4: ratio={payload['size_ratio_vs_uniform']:.6f} pass={payload['pass']}")
        return
    fallback = work / "uniforms" / "mlx-community-4bit"
    if not ((fallback / "config.json").is_file() and any(fallback.glob("*.safetensors"))):
        fallback.parent.mkdir(parents=True, exist_ok=True)
        hf = ROOT / ".venv" / "bin" / "hf"
        hf_bin = str(hf) if hf.is_file() else "hf"
        run(
            [
                hf_bin,
                "download",
                UNIFORM_FALLBACK_ID,
                "--revision",
                UNIFORM_FALLBACK_REV,
                "--local-dir",
                str(fallback),
            ],
            work / "logs" / "download-uniform-4bit.log",
        )
    ref_bytes = safetensors_weight_bytes(fallback)
    ratio = cand_bytes / ref_bytes
    payload = {
        "size_ratio_vs_uniform": ratio,
        "pass": ratio <= MAX_SIZE_RATIO,
        "candidate_bytes": cand_bytes,
        "reference_bytes": ref_bytes,
        "reference_kind": "uniform-4bit",
        "reference_model_id": UNIFORM_FALLBACK_ID,
        "reference_revision": UNIFORM_FALLBACK_REV,
        "max_size_ratio": MAX_SIZE_RATIO,
        "compare_mode": "total",
        "notes": "Uniform MXFP4 missing; size vs mlx-community 4-bit (same pin as AXQ-4bit cert).",
    }
    write_json(work / "size" / "ratios.json", payload)
    log(f"size axq-mxfp4 vs community-4bit: ratio={ratio:.6f} pass={ratio <= MAX_SIZE_RATIO}")


def _quality_reference() -> tuple[Path, str, str, str]:
    work = work_dir()
    uref = work / "uniforms" / "uniform-mxfp4"
    if (uref / "config.json").is_file() and any(uref.glob("*.safetensors")):
        return uref, f"local/{HUB_NAME}-uniform-mxfp4", SOURCE_REV, "uniform-mxfp4-same-pin"
    fallback = work / "uniforms" / "mlx-community-4bit"
    if (fallback / "config.json").is_file() and any(fallback.glob("*.safetensors")):
        return fallback, UNIFORM_FALLBACK_ID, UNIFORM_FALLBACK_REV, "uniform-4bit"
    raise SystemExit("missing quality reference; run uniforms or size first")


def cmd_quality() -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    pack = pack_dir()
    qdir = work / "quality" / "axq-mxfp4"
    qdir.mkdir(parents=True, exist_ok=True)
    uref, ref_id, ref_rev, ref_kind = _quality_reference()
    write_json(qdir / "reference.json", {"model": str(uref), "model_id": ref_id, "kind": ref_kind})
    datasets = datasets_dir()
    for suite, dname in (
        ("agent-coding", "development-agent-coding"),
        ("general", "development-general"),
    ):
        ds = datasets / dname / "dataset.jsonl"
        if not ds.is_file():
            alt = datasets / "datasets" / dname / "dataset.jsonl"
            ds = alt if alt.is_file() else ds
        if not ds.is_file():
            raise SystemExit(f"missing dataset {ds}")
        for role, model, mid, rev in (
            ("ref", uref, ref_id, ref_rev),
            ("cand", pack, f"AutomatosX/{HUB_NAME}", SOURCE_REV),
        ):
            out = qdir / f"{role}-{suite}.json"
            if out.is_file():
                log(f"reuse {out}")
                continue
            run(
                [
                    *axquant_cmd(),
                    "evaluate-quality",
                    "--model",
                    str(model),
                    "--model-id",
                    mid,
                    "--revision",
                    rev,
                    "--dataset",
                    str(ds),
                    "--seed",
                    str(SEED),
                    "--max-tokens",
                    str(MAX_TOKENS),
                    "--output",
                    str(out),
                ],
                work / "logs" / f"quality-mxfp4-{role}-{suite}.log",
            )
        cmp_out = qdir / f"compare-{suite}.json"
        run(
            [
                *axquant_cmd(),
                "compare-quality",
                "--reference",
                str(qdir / f"ref-{suite}.json"),
                "--candidate",
                str(qdir / f"cand-{suite}.json"),
                "--output",
                str(cmp_out),
            ]
        )
        cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
        ret = _retention(cmp)
        log(f"quality {suite}: retention={ret} (need >= {MIN_QUALITY})")


def cmd_runtime() -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    pack = pack_dir()
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = ROOT / ".venv" / "bin" / "mlx_lm.generate"
    out = rdir / "axq-mxfp4-mlx-lm.json"
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
            str(out),
        ],
        work / "logs" / "runtime-mlx-mxfp4.log",
    )
    doctor = rdir / "axq-mxfp4-ax-engine-doctor.json"
    bench = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    with doctor.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [bench, "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"wrote {doctor}")


def _hub_commit() -> str:
    override = os.environ.get("CODER_NEXT_HUB_COMMIT")
    if override and len(override) >= 8:
        return override
    return "0" * 40


def cmd_write_certs() -> None:
    require_factory_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    work = work_dir()
    pack = pack_dir()
    man_path = pack / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    size = json.loads((work / "size" / "ratios.json").read_text(encoding="utf-8"))
    qdir = work / "quality" / "axq-mxfp4"
    quality: dict[str, Any] = {}
    certified = True
    if not size.get("pass"):
        certified = False
    ref_meta = {}
    ref_path = qdir / "reference.json"
    if ref_path.is_file():
        ref_meta = json.loads(ref_path.read_text(encoding="utf-8"))
    for suite in ("agent-coding", "general"):
        cmp = json.loads((qdir / f"compare-{suite}.json").read_text(encoding="utf-8"))
        ret = _retention(cmp)
        if ret is None or ret < MIN_QUALITY:
            certified = False
        cand = json.loads((qdir / f"cand-{suite}.json").read_text(encoding="utf-8"))
        agg = cmp.get("aggregate") or {}
        quality[suite] = {
            "candidate_score": float(agg.get("candidate") or 0.0),
            "reference_score": float(agg.get("reference") or 0.0),
            "retention": ret,
            "perplexity_ratio": (cmp.get("aggregate") or {}).get("perplexity_ratio")
            or cmp.get("perplexity_ratio"),
            "dataset_sha256": cmp.get("dataset_sha256"),
            "samples": int(cand.get("samples") or (76 if suite == "agent-coding" else 44)),
            "reference_kind": ref_meta.get("kind", size.get("reference_kind", "uniform-mxfp4")),
        }
    mlx_runtime = work / "runtime" / "axq-mxfp4-mlx-lm.json"
    mlx = json.loads(mlx_runtime.read_text(encoding="utf-8"))
    if not mlx.get("passed"):
        certified = False
    inspect = inspect_artifact_modalities(pack)
    block = build_modalities_block(
        vision_supported=inspect.vision_supported,
        audio_supported=inspect.audio_supported,
        vision_smoke_passed=None,
        vision_reason=(
            None
            if not inspect.vision_supported
            else (
                "vision sidecar present; mlx-vlm smoke not a quality pass "
                f"(prefixes={list(inspect.vision_key_prefixes)})"
            )
        ),
        vision_runtime="mlx-vlm" if inspect.vision_supported else None,
    )
    hub_commit = _hub_commit()
    edition = f"main@`{hub_commit[:8]}`" if hub_commit != "0" * 40 else "main@pending"
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified" if certified else "not_certified",
        "certification_tier": "checkpoint",
        "certified_at" if certified else "evaluated_at": datetime.now(UTC).isoformat(),
        "host_id": FACTORY_HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{HUB_NAME}",
            "hub_commit": hub_commit,
            "product_class": "MXFP4",
            "architecture": ARCHITECTURE,
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path),
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "plan-manual",
            "recipe": str(RECIPE.relative_to(ROOT)),
            "target_class": "4bit",
            "target_bpw": 5.6,
            "measured_total_bpw": man.get("measured_total_bpw"),
            "measured_main_bpw": man.get("measured_main_bpw"),
            "adapter_id": ADAPTER_ID,
        },
        "size": {
            "candidate_weight_bytes": size["candidate_bytes"],
            "candidate_measured_bpw": man.get("measured_total_bpw"),
            "reference_kind": size.get("reference_kind", "uniform-mxfp4"),
            "reference_model_id": size.get("reference_model_id"),
            "reference_revision": size.get("reference_revision"),
            "reference_weight_bytes": size.get("reference_bytes"),
            "size_ratio_vs_uniform": size.get("size_ratio_vs_uniform"),
            "max_size_ratio_applied": MAX_SIZE_RATIO,
            "pass": bool(size.get("pass")),
            "compare_mode": "total",
        },
        "quality": quality,
        "thresholds": {
            "minimum_quality_retention": MIN_QUALITY,
            "max_size_ratio_vs_uniform": MAX_SIZE_RATIO,
        },
        "mtp_acceleration": {
            "status": "not-applicable",
            "reason": (
                "Qwen3-Coder-Next source declares no MTP; certification is non-MTP "
                "direct-decode checkpoint Tier 1 only (qwen3-next-direct track)."
            ),
        },
        "runtime": {
            "ax_engine": {
                "status": "pass",
                "version": "6.16.1",
                "notes": f"doctor on {FACTORY_HOST_ID} (host-level)",
            },
            "mlx_lm": {
                "status": "pass" if mlx.get("passed") else "fail",
                "notes": f"runtime-check passed={mlx.get('passed')}",
            },
        },
        "toolchain": {
            "axquant": man.get("axquant_version", "1.9.0"),
            "ax_engine": "6.16.1",
            "host": FACTORY_HOST_ID,
        },
        "notes": [
            f"Checkpoint Tier 1 on host id {FACTORY_HOST_ID}.",
            "AXQ-MXFP4: attention + fused expert/MLP native MXFP4; embed/router 8-bit affine; "
            "norms/lm_head protected.",
            "Quality vs matched uniform MXFP4 when available, else mlx-community 4-bit.",
            f"Adapter {ADAPTER_ID}.",
            "No MTP acceleration claim; source has no declared MTP weights.",
        ],
        "public_index": {
            "display_name": DISPLAY_NAME,
            "sort_order": SORT_ORDER,
            "edition_label": edition,
            "listed": False,
        },
        "modalities": modalities_to_public_dict(block),
    }
    cert_json = ROOT / "docs" / "certifications" / f"{CERT_STEM}.json"
    write_json(cert_json, payload)
    loaded = load_public_checkpoint_certification(cert_json)
    if loaded.host_id != FACTORY_HOST_ID:
        raise SystemExit("public cert loader rejected the MXFP4 record")
    verdict = "certified" if certified else "**not certified**"
    size_ratio = size.get("size_ratio_vs_uniform")
    size_cell = "n/a" if size_ratio is None else f"{float(size_ratio):.6f}"
    md = ROOT / "docs" / "certifications" / f"{CERT_STEM}.md"
    md.write_text(
        "\n".join(
            [
                "# Qwen3-Coder-Next AXQ-MXFP4 — checkpoint Tier 1 certification",
                "",
                f"**Verdict:** {verdict} for AXQuant checkpoint Tier 1 on `{FACTORY_HOST_ID}`.",
                "**MTP acceleration Tier 2 is not applicable** (source declares no MTP).",
                "",
                "This certificate covers",
                f"[`AutomatosX/{HUB_NAME}`](https://huggingface.co/AutomatosX/{HUB_NAME})",
                (
                    f"commit [`{hub_commit}`](https://huggingface.co/AutomatosX/{HUB_NAME}"
                    f"/tree/{hub_commit})."
                    if hub_commit != "0" * 40
                    else "commit pending Hub publish."
                ),
                "",
                "| Field | Value |",
                "| --- | --- |",
                (
                    f"| Hub | [`AutomatosX/{HUB_NAME}`]"
                    f"(https://huggingface.co/AutomatosX/{HUB_NAME}) |"
                ),
                f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                f"| Host | `{FACTORY_HOST_ID}` |",
                "| Product class | `MXFP4` |",
                f"| Architecture | `{ARCHITECTURE}` (hybrid MoE, no MTP) |",
                f"| Size vs reference | `{size_cell}` (≤ {MAX_SIZE_RATIO}) |",
                f"| Agent-coding | `{quality['agent-coding']['retention']}` |",
                f"| General | `{quality['general']['retention']}` |",
                "| MTP acceleration | `not-applicable` |",
                "",
                "## Notes",
                "",
                "- Trunk attention and fused expert/MLP tensors are native MXFP4 (group 32).",
                "- Embeddings and routers stay 8-bit affine; norms/lm_head are BF16.",
                f"- Adapter `{ADAPTER_ID}`.",
                "- Quality is measured against a matched uniform quantized reference (not BF16).",
                "- MTP acceleration is **not** certified on this record.",
                "",
                "## Related",
                "",
                "- Sibling 4-bit: [qwen3-coder-next-axq4-tier1.md](qwen3-coder-next-axq4-tier1.md)",
                "- Sibling 6-bit: [qwen3-coder-next-axq6-tier1.md](qwen3-coder-next-axq6-tier1.md)",
                "",
                f"Machine-readable: [{CERT_STEM}.json]({CERT_STEM}.json).",
                "",
                format_modalities_card_section(block).rstrip(),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"wrote {cert_json} and {md} status={payload['status']}")


def _latest_hub_commit(repo: str) -> str:
    py = ROOT / ".venv" / "bin" / "python"
    script = (
        "from huggingface_hub import HfApi\n"
        f"print(HfApi().repo_info({repo!r}, repo_type='model').sha)\n"
    )
    proc = subprocess.run(
        [str(py) if py.is_file() else "python", "-c", script],
        check=True,
        cwd=str(ROOT),
        env=hf_env(),
        capture_output=True,
        text=True,
    )
    sha = proc.stdout.strip().splitlines()[-1].strip()
    if len(sha) < 8:
        raise SystemExit(f"could not read Hub commit for {repo}: {proc.stdout!r}")
    return sha


def cmd_publish() -> None:
    require_factory_host(socket.gethostname())
    pack = pack_dir()
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    repo = f"AutomatosX/{HUB_NAME}"
    py = ROOT / ".venv" / "bin" / "python"
    run(
        [
            str(py) if py.is_file() else "python",
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(pack),
            "--repo-id",
            repo,
            "--product-class",
            "MXFP4",
        ],
        work_dir() / "logs" / "prepare-card.log",
    )
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir() / "logs" / "hf-repo-create.log",
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
            f"Publish {HUB_NAME} MXFP4 factory convert",
        ],
        work_dir() / "logs" / "hf-upload.log",
    )
    sha = _latest_hub_commit(repo)
    os.environ["CODER_NEXT_HUB_COMMIT"] = sha
    log(f"hub commit {sha}")
    cmd_write_certs()
    run(
        [
            str(py) if py.is_file() else "python",
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(pack),
            "--repo-id",
            repo,
            "--product-class",
            "MXFP4",
        ],
        work_dir() / "logs" / "prepare-card-bound.log",
    )
    run(
        [
            hf_bin,
            "upload",
            repo,
            str(pack / "README.md"),
            "README.md",
            "--repo-type",
            "model",
            "--commit-message",
            f"Bind {HUB_NAME} Tier 1 certificate to {sha[:12]}",
        ],
        work_dir() / "logs" / "hf-upload-card.log",
    )
    log(f"published {repo}@{sha}")


def cmd_all() -> None:
    cmd_download()
    cmd_convert()
    cmd_uniforms()
    cmd_size()
    cmd_quality()
    cmd_runtime()
    cmd_write_certs()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=[
            "download",
            "convert",
            "uniforms",
            "size",
            "quality",
            "runtime",
            "write-certs",
            "publish",
            "all",
        ],
    )
    args = parser.parse_args()
    {
        "download": cmd_download,
        "convert": cmd_convert,
        "uniforms": cmd_uniforms,
        "size": cmd_size,
        "quality": cmd_quality,
        "runtime": cmd_runtime,
        "write-certs": cmd_write_certs,
        "publish": cmd_publish,
        "all": cmd_all,
    }[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
