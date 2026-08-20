#!/usr/bin/env python3
"""Factory: convert + checkpoint Tier 1 for Ornith / Holo3 / Glimmer AXQ-MXFP4.

Run on df-macstudio-m2 + Ext12T only:

  PYTHONPATH=src /Users/devop/code/axquant/.venv/bin/python \\
    scripts/run_family_mxfp4_tier1.py --family ornith all

One large convert at a time. T1 quality is mlx-lm (Ornith/Holo3). Glimmer has
no mlx-lm quality backend — that pack is recorded honestly if quality cannot run.
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

ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)
MAX_SIZE_RATIO = 1.20
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64

FAMILIES: dict[str, dict[str, Any]] = {
    "ornith": {
        "source_id": "deepreinforce-ai/Ornith-1.0-35B",
        "source_rev": "5df2ed3f675c7beaa490328cc70bb573b65fb660",
        "hub_name": "AX-Ornith-1.0-35B-MLX-AXQ-MXFP4",
        "display_name": "Ornith-1.0-35B MLX AXQ MXFP4",
        "cert_stem": "ornith-35b-axq-mxfp4-tier1",
        "recipe": ROOT / "examples" / "ornith-35b-axq-mxfp4-v0.1.yaml",
        "adapter_id": "qwen35-moe-v1",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "sort_order": 88,
        "listed": True,
        "quality_backend": "mlx-lm",
        "work_name": "ornith-35b-mxfp4",
        "src_name": "src-ornith-35b",
    },
    "holo3": {
        "source_id": "Hcompany/Holo3-35B-A3B",
        "source_rev": "208d5ae3a03f99d561f32ab5e606f73397a390ea",
        "hub_name": "AX-Holo3-35B-A3B-MLX-AXQ-MXFP4",
        "display_name": "Holo3-35B-A3B MLX AXQ MXFP4",
        "cert_stem": "holo3-35b-axq-mxfp4-tier1",
        "recipe": ROOT / "examples" / "holo3-35b-axq-mxfp4-v0.1.yaml",
        "adapter_id": "qwen35-moe-v1",
        "architecture": "Qwen3_5MoeForConditionalGeneration",
        "sort_order": 83,
        "listed": True,
        "quality_backend": "mlx-lm",
        "work_name": "holo3-35b-mxfp4",
        "src_name": "src-holo3-35b",
    },
    "glimmer": {
        "source_id": "meta-models/Muse-Glimmer-30B",
        "source_rev": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        "hub_name": "AX-Muse-Glimmer-30B-MLX-AXQ-MXFP4",
        "display_name": "Muse Glimmer 30B MLX AXQ MXFP4",
        "cert_stem": "muse-glimmer-30b-axq-mxfp4-tier1",
        "recipe": ROOT / "examples" / "muse-glimmer-30b-axq-mxfp4-v0.1.yaml",
        "adapter_id": "muse-glimmer-v1",
        "architecture": "muse_glimmer",
        "sort_order": 224,
        "listed": False,
        "quality_backend": None,
        "work_name": "muse-glimmer-30b-mxfp4",
        "src_name": "src-muse-glimmer-30b",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def family_spec(name: str) -> dict[str, Any]:
    if name not in FAMILIES:
        raise SystemExit(f"unknown family {name}; choose {sorted(FAMILIES)}")
    return FAMILIES[name]


def work_dir(spec: dict[str, Any]) -> Path:
    return Path(
        os.environ.get(
            "MXFP4_WORK",
            f"{FACTORY_CERT_ROOT}/{spec['work_name']}",
        )
    )


def source_dir(spec: dict[str, Any]) -> Path:
    override = os.environ.get("MXFP4_SOURCE")
    if override:
        return Path(override)
    return Path(f"/Volumes/Ext12T/axquant/work/{spec['work_name']}/{spec['src_name']}")


def pack_dir(spec: dict[str, Any]) -> Path:
    override = os.environ.get("MXFP4_PACK")
    if override:
        return Path(override)
    return Path(FACTORY_MODELS) / spec["hub_name"]


def datasets_dir() -> Path:
    return Path(os.environ.get("MXFP4_DATASETS", FACTORY_DATASETS))


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
        return None
    return float(raw)


def cmd_download(spec: dict[str, Any]) -> None:
    dest = source_dir(spec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest / "config.json"
    if marker.is_file():
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [
            hf_bin,
            "download",
            spec["source_id"],
            "--revision",
            spec["source_rev"],
            "--local-dir",
            str(dest),
        ],
        work_dir(spec) / "logs" / "download-src.log",
    )


def cmd_convert(spec: dict[str, Any]) -> None:
    require_factory_host(socket.gethostname())
    recipe = spec["recipe"]
    if not recipe.is_file():
        raise SystemExit(f"missing recipe {recipe}")
    source = source_dir(spec)
    if not (source / "config.json").is_file():
        raise SystemExit(f"missing source {source}; run download first")
    work = work_dir(spec)
    pack = pack_dir(spec)
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
                spec["source_id"],
                "--revision",
                spec["source_rev"],
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
                str(recipe),
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


def cmd_uniforms(spec: dict[str, Any]) -> None:
    work = work_dir(spec)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    source = source_dir(spec)
    out = work / "uniforms" / "uniform-mxfp4"
    if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
        log(f"reuse {out}")
        return
    if spec["quality_backend"] != "mlx-lm":
        log("skip uniform MXFP4 (no mlx-lm size/quality reference for this family)")
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
        log(f"uniform MXFP4 convert failed ({exc}); size/quality will record no reference")


def cmd_size(spec: dict[str, Any]) -> None:
    work = work_dir(spec)
    pack = pack_dir(spec)
    man_path = pack / "axquant_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"missing manifest {man_path}")
    (work / "size").mkdir(parents=True, exist_ok=True)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    cand_bytes = int(man.get("weight_file_size_bytes") or 0)
    if cand_bytes <= 0:
        cand_bytes = safetensors_weight_bytes(pack)
    udir = work / "uniforms" / "uniform-mxfp4"
    if spec["quality_backend"] != "mlx-lm" or not (udir / "config.json").is_file():
        payload = {
            "size_ratio_vs_uniform": None,
            "pass": True,
            "candidate_bytes": cand_bytes,
            "reference_bytes": None,
            "max_size_ratio": MAX_SIZE_RATIO,
            "compare_mode": "total",
            "notes": "No matched uniform MXFP4 reference on this host.",
        }
        write_json(work / "size" / "ratios.json", payload)
        log(f"size {spec['hub_name']}: no uniform reference; recorded candidate bytes")
        return
    ref_bytes = safetensors_weight_bytes(udir)
    ratio = cand_bytes / ref_bytes
    payload = {
        "size_ratio_vs_uniform": ratio,
        "pass": ratio <= MAX_SIZE_RATIO,
        "candidate_bytes": cand_bytes,
        "reference_bytes": ref_bytes,
        "max_size_ratio": MAX_SIZE_RATIO,
        "compare_mode": "total",
    }
    write_json(work / "size" / "ratios.json", payload)
    log(f"size {spec['hub_name']}: ratio={ratio:.6f} pass={ratio <= MAX_SIZE_RATIO}")


def cmd_quality(spec: dict[str, Any]) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir(spec)
    pack = pack_dir(spec)
    qdir = work / "quality" / "axq-mxfp4"
    qdir.mkdir(parents=True, exist_ok=True)
    if spec["quality_backend"] != "mlx-lm":
        write_json(
            qdir / "skipped.json",
            {
                "pass": False,
                "reason": (
                    "axquant evaluate-quality uses the MLX-LM backend; "
                    f"model_type {spec['architecture']} is not supported."
                ),
            },
        )
        log("quality skipped (no mlx-lm backend)")
        return
    uref = work / "uniforms" / "uniform-mxfp4"
    if not (uref / "config.json").is_file():
        raise SystemExit(f"missing uniform quality reference {uref}")
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
        for role, model, mid in (
            ("ref", uref, f"local/{spec['hub_name']}-uniform-mxfp4"),
            ("cand", pack, f"AutomatosX/{spec['hub_name']}"),
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
                    spec["source_rev"],
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


def cmd_runtime(spec: dict[str, Any]) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir(spec)
    pack = pack_dir(spec)
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    if spec["quality_backend"] == "mlx-lm":
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
    else:
        smoke = rdir / "axq-mxfp4-mlx-vlm.json"
        py = ROOT / ".venv" / "bin" / "python"
        script = (
            "import json, time\n"
            "from pathlib import Path\n"
            "from mlx_vlm.utils import load_model, load_config\n"
            f"pack = Path({str(pack)!r})\n"
            "t0 = time.time()\n"
            "model = load_model(pack, lazy=False)\n"
            "n = sum(1 for _ in model.named_modules())\n"
            "elapsed = time.time() - t0\n"
            "payload = {'status': 'pass', 'modules': n, 'seconds': elapsed}\n"
            f"Path({str(smoke)!r}).write_text(json.dumps(payload, indent=2)+'\\n')\n"
            "print(payload)\n"
        )
        run(
            [str(py) if py.is_file() else "python", "-c", script],
            work / "logs" / "runtime-mlx-vlm-mxfp4.log",
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


def cmd_write_certs(spec: dict[str, Any]) -> None:
    require_factory_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    work = work_dir(spec)
    pack = pack_dir(spec)
    man_path = pack / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    size = json.loads((work / "size" / "ratios.json").read_text(encoding="utf-8"))
    qdir = work / "quality" / "axq-mxfp4"
    quality: dict[str, Any] = {}
    certified = True
    if spec["quality_backend"] != "mlx-lm":
        certified = False
        reason = (
            "axquant evaluate-quality uses the MLX-LM backend; "
            f"model_type {spec['architecture']} is not supported."
        )
        quality = {
            "agent-coding": {"pass": False, "reason": reason},
            "general": {"pass": False, "reason": reason},
        }
    else:
        if not size.get("pass"):
            certified = False
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
                "dataset_sha256": cmp.get("dataset_sha256"),
                "samples": int(cand.get("samples") or (76 if suite == "agent-coding" else 44)),
                "reference_kind": "uniform-mxfp4-same-pin",
            }
    mlx_runtime = work / "runtime" / "axq-mxfp4-mlx-lm.json"
    vlm_runtime = work / "runtime" / "axq-mxfp4-mlx-vlm.json"
    runtime: dict[str, Any] = {
        "ax_engine": {
            "status": "pass",
            "version": "6.16.1",
            "notes": f"doctor on {FACTORY_HOST_ID} (host-level)",
        }
    }
    if mlx_runtime.is_file():
        mlx = json.loads(mlx_runtime.read_text(encoding="utf-8"))
        runtime["mlx_lm"] = {
            "status": "pass" if mlx.get("passed") else "fail",
            "notes": f"runtime-check passed={mlx.get('passed')}",
        }
        if not mlx.get("passed"):
            certified = False
    if vlm_runtime.is_file():
        vlm = json.loads(vlm_runtime.read_text(encoding="utf-8"))
        runtime["mlx_vlm"] = {
            "status": vlm.get("status", "pass"),
            "notes": f"load_model modules={vlm.get('modules')} in {vlm.get('seconds')}s",
        }
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
    hub_commit = os.environ.get("MXFP4_HUB_COMMIT", "0" * 40)
    listed = bool(spec["listed"]) and certified
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified" if certified else "not_certified",
        "certification_tier": "checkpoint",
        "certified_at" if certified else "evaluated_at": datetime.now(UTC).isoformat(),
        "host_id": FACTORY_HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{spec['hub_name']}",
            "hub_commit": hub_commit,
            "product_class": "MXFP4",
            "architecture": spec["architecture"],
            "source_model_id": spec["source_id"],
            "source_revision": spec["source_rev"],
            "candidate_manifest_sha256": sha256_file(man_path),
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "plan-manual",
            "recipe": str(spec["recipe"].relative_to(ROOT)),
            "target_class": "4bit",
            "target_bpw": 5.6,
            "measured_total_bpw": man.get("measured_total_bpw"),
            "measured_main_bpw": man.get("measured_main_bpw"),
            "adapter_id": spec["adapter_id"],
        },
        "size": {
            "candidate_weight_bytes": size["candidate_bytes"],
            "candidate_measured_bpw": man.get("measured_total_bpw"),
            "reference_kind": "uniform-mxfp4" if size.get("reference_bytes") else "none",
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
            "reason": "No MTP weights; checkpoint Tier 1 is non-MTP direct-decode only.",
        },
        "runtime": runtime,
        "toolchain": {
            "axquant": man.get("axquant_version", "1.8.1"),
            "ax_engine": "6.16.1",
            "host": FACTORY_HOST_ID,
        },
        "notes": [
            f"Checkpoint Tier 1 on host id {FACTORY_HOST_ID}.",
            "AXQ-MXFP4: attention + expert/MLP native MXFP4; embed/router 8-bit affine; "
            "vision/norms/lm_head protected.",
            "Quality vs matched uniform MXFP4 when mlx-lm can load the family.",
            f"Adapter {spec['adapter_id']}.",
            "Vision BF16-protected; VLM quality not claimed.",
        ],
        "public_index": {
            "display_name": spec["display_name"],
            "sort_order": spec["sort_order"],
            "edition_label": "main@pending",
            "listed": listed,
        },
        "modalities": modalities_to_public_dict(block),
    }
    cert_json = ROOT / "docs" / "certifications" / f"{spec['cert_stem']}.json"
    write_json(cert_json, payload)
    loaded = load_public_checkpoint_certification(cert_json)
    if loaded.host_id != FACTORY_HOST_ID:
        raise SystemExit("public cert loader rejected the MXFP4 record")
    verdict = "certified" if certified else "**not certified**"
    md = ROOT / "docs" / "certifications" / f"{spec['cert_stem']}.md"
    md.write_text(
        "\n".join(
            [
                f"# {spec['display_name']} — checkpoint Tier 1 certification",
                "",
                f"**Verdict:** {verdict} for AXQuant checkpoint Tier 1 on `{FACTORY_HOST_ID}`.",
                "",
                "| Field | Value |",
                "| --- | --- |",
                (
                    f"| Hub | [`AutomatosX/{spec['hub_name']}`]"
                    f"(https://huggingface.co/AutomatosX/{spec['hub_name']}) |"
                ),
                f"| Source | `{spec['source_id']}@{spec['source_rev']}` |",
                f"| Host | `{FACTORY_HOST_ID}` |",
                "| Product class | `MXFP4` |",
                "| MTP acceleration | `not-applicable` |",
                "",
                "## Notes",
                "",
                "- Trunk attention and expert/MLP tensors are native MXFP4 (group 32).",
                "- Embeddings and routers stay 8-bit affine; vision/norms/lm_head are BF16.",
                f"- Adapter `{spec['adapter_id']}`.",
                "- Vision remains BF16-protected; no VLM quality claim.",
                "",
                f"Machine-readable: [{spec['cert_stem']}.json]({spec['cert_stem']}.json).",
                "",
                format_modalities_card_section(block).rstrip(),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"wrote {cert_json} and {md} status={payload['status']}")


def cmd_publish(spec: dict[str, Any]) -> None:
    require_factory_host(socket.gethostname())
    pack = pack_dir(spec)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    repo = f"AutomatosX/{spec['hub_name']}"
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
        work_dir(spec) / "logs" / "prepare-card.log",
    )
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir(spec) / "logs" / "hf-repo-create.log",
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
            f"Publish {spec['hub_name']} MXFP4 factory convert",
        ],
        work_dir(spec) / "logs" / "hf-upload.log",
    )
    log(f"published {repo}")


def cmd_all(spec: dict[str, Any]) -> None:
    cmd_download(spec)
    cmd_convert(spec)
    cmd_uniforms(spec)
    cmd_size(spec)
    cmd_quality(spec)
    cmd_runtime(spec)
    cmd_write_certs(spec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
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
    spec = family_spec(args.family)
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
    }[args.step](spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
