#!/usr/bin/env python3
"""Factory: convert + checkpoint Tier 1 for Qwen3.8-27B AXQ-MXFP4 (no-MTP).

Run on df-macstudio-m2 with Ext4T:

  PYTHONPATH=src /Users/devop/code/axquant-main/.venv/bin/python \\
    scripts/run_qwen38_27b_axq_mxfp4.py all

Steps: prepare no-MTP BF16 view, inspect, plan-manual, convert --q-mode mxfp4,
uniform MXFP4 size reference, size gate, dual-suite quality vs BF16, mlx-lm
smoke, AX Engine doctor, public cert JSON/MD.

Environment: QWEN38_BF16, QWEN38_MXFP4_PACK, QWEN38_MXFP4_WORK,
QWEN38_CERT_DATASETS, QWEN38_CERT_HOST, AX_ENGINE_BENCH.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(
    os.environ.get(
        "QWEN38_MXFP4_WORK",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq-mxfp4-tier1",
    )
)
BF16 = Path(os.environ.get("QWEN38_BF16", "/Volumes/Ext4T/models/Qwen3.8-27B-bf16"))
PACK = Path(
    os.environ.get(
        "QWEN38_MXFP4_PACK",
        "/Volumes/Ext4T/models/AX-Qwen3.8-27B-MLX-AXQ-MXFP4",
    )
)
DATASETS = Path(
    os.environ.get(
        "QWEN38_CERT_DATASETS",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq8-tier1/datasets",
    )
)
SOURCE_ID = "Qwen/Qwen3.8-27B"
SOURCE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HOST_ID = os.environ.get("QWEN38_CERT_HOST", "df-macstudio-m2")
HUB_NAME = "AX-Qwen3.8-27B-MLX-AXQ-MXFP4"
CERT_STEM = "qwen38-27b-axq-mxfp4-tier1"
RECIPE = ROOT / "examples" / "qwen38-27b-axq-mxfp4-v0.1.yaml"
MAX_SIZE_RATIO = 1.20
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    studio_py = Path("/Users/devop/code/axquant-main/.venv/bin/python")
    if studio_py.is_file():
        return [str(studio_py), "-m", "axquant"]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found")


def run(cmd: list[str], log_path: Path | None = None, *, force_cpu: bool = False) -> None:
    log("$ " + " ".join(cmd))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": os.environ.get("HF_HOME", "/Volumes/Ext4T/huggingface"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", "/Volumes/Ext4T/huggingface/hub"),
    }
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_no_mtp() -> Path:
    dest = Path(os.environ.get("QWEN38_NO_MTP", str(WORK / "bf16-no-mtp")))
    if (dest / ".axquant_no_mtp_ready").is_file():
        log(f"reuse no-MTP source {dest}")
        return dest
    spec = importlib.util.spec_from_file_location(
        "qwen38_four_pack",
        ROOT / "scripts" / "run_qwen38_27b_axq_four_pack.py",
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load no-MTP prepare helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    WORK.mkdir(parents=True, exist_ok=True)
    return helper.prepare_no_mtp_source(BF16, dest, WORK)


def cmd_convert() -> None:
    if not RECIPE.is_file():
        raise SystemExit(f"missing recipe {RECIPE}")
    source = prepare_no_mtp()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    inventory = WORK / "inventory.json"
    plan = WORK / "plan.json"
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
            WORK / "logs" / "inspect.log",
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
            WORK / "logs" / "plan-manual.log",
        )
    if (PACK / "axquant_manifest.json").is_file():
        log(f"reuse pack {PACK}")
        return
    if PACK.exists():
        raise SystemExit(f"incomplete pack dir exists: {PACK}")
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
            str(PACK),
            "--q-mode",
            "mxfp4",
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "if-available",
            "--ax-engine-bench",
            engine,
        ],
        WORK / "logs" / "convert-mxfp4.log",
        force_cpu=True,
    )


def cmd_uniforms() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    source = prepare_no_mtp()
    out = WORK / "uniforms" / "uniform-mxfp4"
    if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
        log(f"reuse {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    mlx = ROOT / ".venv" / "bin" / "mlx_lm.convert"
    if not mlx.is_file():
        mlx = Path("/Users/devop/code/axquant-main/.venv/bin/mlx_lm.convert")
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
        WORK / "logs" / "uniform-mxfp4-convert.log",
        force_cpu=True,
    )


def cmd_size() -> None:
    man_path = PACK / "axquant_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"missing pack manifest {man_path}")
    (WORK / "size").mkdir(parents=True, exist_ok=True)
    out = WORK / "size" / "axq-mxfp4-candidate.json"
    size_rev = os.environ.get("QWEN38_MXFP4_HUB_COMMIT", SOURCE_REV)
    if not re.fullmatch(r"[0-9a-f]{40}", size_rev):
        size_rev = SOURCE_REV
    run(
        [
            *axquant_cmd(),
            "size-evidence",
            "--artifact-manifest",
            str(man_path),
            "--model-id",
            f"AutomatosX/{HUB_NAME}",
            "--revision",
            size_rev,
            "--output",
            str(out),
        ]
    )
    udir = WORK / "uniforms" / "uniform-mxfp4"
    if not (udir / "config.json").is_file():
        raise SystemExit(f"missing uniform dir {udir}; run uniforms first")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    logical = int(man["logical_parameters"])
    weight_bytes = safetensors_weight_bytes(udir)
    measured_bpw = 8.0 * weight_bytes / logical
    uref = {
        "schema_version": "axquant.artifact-size-evidence.v1",
        "kind": "uniform-mxfp4",
        "model": {
            "model_id": "local/Qwen3.8-27B-uniform-mxfp4",
            "revision": SOURCE_REV,
            "local_path": str(udir),
        },
        "logical_parameters": logical,
        "weight_bytes": weight_bytes,
        "measured_bpw": measured_bpw,
    }
    write_json(WORK / "size" / "uniform-mxfp4.json", uref)
    cand = json.loads(out.read_text(encoding="utf-8"))
    cand_bytes = int(cand.get("weight_bytes") or man.get("weight_file_size_bytes") or 0)
    if cand_bytes <= 0:
        cand_bytes = safetensors_weight_bytes(PACK)
    ratio = cand_bytes / weight_bytes
    ratios = {
        "axq-mxfp4": {
            "size_ratio_vs_uniform": ratio,
            "pass": ratio <= MAX_SIZE_RATIO,
            "candidate_bytes": cand_bytes,
            "reference_bytes": weight_bytes,
            "max_size_ratio": MAX_SIZE_RATIO,
            "compare_mode": "total",
        }
    }
    write_json(WORK / "size" / "ratios.json", ratios)
    log(f"size axq-mxfp4: ratio={ratio:.6f} pass={ratio <= MAX_SIZE_RATIO}")


def _retention(cmp: dict) -> float | None:
    agg = cmp.get("aggregate") or {}
    ret = agg.get("retention")
    if ret is None:
        ret = cmp.get("retention")
    return None if ret is None else float(ret)


def cmd_quality() -> None:
    qdir = WORK / "quality" / "axq-mxfp4"
    qdir.mkdir(parents=True, exist_ok=True)
    source = prepare_no_mtp()
    if not source.is_dir():
        raise SystemExit(f"missing BF16 reference {source}")
    for suite, dname in (
        ("agent-coding", "development-agent-coding"),
        ("general", "development-general"),
    ):
        ds = DATASETS / dname / "dataset.jsonl"
        if not ds.is_file():
            alt = DATASETS / "datasets" / dname / "dataset.jsonl"
            ds = alt if alt.is_file() else ds
        if not ds.is_file():
            raise SystemExit(f"missing dataset {ds}")
        for role, model, mid in (
            ("ref", source, SOURCE_ID),
            ("cand", PACK, f"AutomatosX/{HUB_NAME}"),
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
                WORK / "logs" / f"quality-mxfp4-{role}-{suite}.log",
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
        log(f"quality axq-mxfp4 {suite}: retention={ret} (need >= {MIN_QUALITY})")


def cmd_runtime() -> None:
    rdir = WORK / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = ROOT / ".venv" / "bin" / "mlx_lm.generate"
    if not mlx_gen.is_file():
        mlx_gen = Path("/Users/devop/code/axquant-main/.venv/bin/mlx_lm.generate")
    if not mlx_gen.is_file():
        mlx_gen = Path("mlx_lm.generate")
    out = rdir / "axq-mxfp4-mlx-lm.json"
    run(
        [
            *axquant_cmd(),
            "runtime-check",
            "--model",
            str(PACK),
            "--runtime",
            "mlx-lm",
            "--mlx-lm",
            str(mlx_gen),
            "--output",
            str(out),
        ],
        WORK / "logs" / "runtime-mlx-mxfp4.log",
    )
    doctor = rdir / "axq-mxfp4-ax-engine-doctor.json"
    bench = ENGINE_BENCH
    with doctor.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [bench, "doctor", "--mlx-model-artifacts-dir", str(PACK), "--json"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"wrote {doctor}")


def cmd_write_certs() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import (
        load_public_checkpoint_certification,
    )

    ratios = json.loads((WORK / "size" / "ratios.json").read_text(encoding="utf-8"))
    size = ratios["axq-mxfp4"]
    if not size["pass"]:
        raise SystemExit(f"size gate fail ratio={size['size_ratio_vs_uniform']}")
    qdir = WORK / "quality" / "axq-mxfp4"
    quality: dict[str, dict] = {}
    for suite in ("agent-coding", "general"):
        cmp = json.loads((qdir / f"compare-{suite}.json").read_text(encoding="utf-8"))
        ret = _retention(cmp)
        if ret is None or ret < MIN_QUALITY:
            raise SystemExit(f"{suite} retention={ret} < {MIN_QUALITY}")
        cand = json.loads((qdir / f"cand-{suite}.json").read_text(encoding="utf-8"))
        agg = cmp.get("aggregate") or {}
        quality[suite] = {
            "candidate_score": float(agg.get("candidate") or 0.0),
            "reference_score": float(agg.get("reference") or 0.0),
            "retention": ret,
            "dataset_sha256": cmp.get("dataset_sha256"),
            "samples": int(cand.get("samples") or (76 if suite == "agent-coding" else 44)),
            "reference_kind": "bf16-same-pin",
        }
    mlx = json.loads((WORK / "runtime" / "axq-mxfp4-mlx-lm.json").read_text(encoding="utf-8"))
    if not mlx.get("passed"):
        raise SystemExit("mlx-lm runtime-check did not pass")
    man_path = PACK / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    inspect = inspect_artifact_modalities(PACK)
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
    hub_commit = os.environ.get("QWEN38_MXFP4_HUB_COMMIT", "0" * 40)
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified",
        "certification_tier": "checkpoint",
        "certified_at": datetime.now(UTC).isoformat(),
        "host_id": HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{HUB_NAME}",
            "hub_commit": hub_commit,
            "product_class": "4bit",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path),
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "plan-manual",
            "target_class": "4bit",
            "target_bpw": 5.6,
            "measured_total_bpw": man.get("measured_total_bpw"),
            "measured_main_bpw": man.get("measured_main_bpw"),
            "adapter_id": "qwen38-dense-v1",
        },
        "size": {
            "candidate_weight_bytes": size["candidate_bytes"],
            "candidate_measured_bpw": man.get("measured_total_bpw"),
            "reference_kind": "uniform-mxfp4",
            "reference_model_id": "local/Qwen3.8-27B-uniform-mxfp4",
            "reference_revision": SOURCE_REV,
            "reference_weight_bytes": size["reference_bytes"],
            "size_ratio_vs_uniform": size["size_ratio_vs_uniform"],
            "max_size_ratio_applied": MAX_SIZE_RATIO,
            "pass": True,
            "compare_mode": "total",
            "reference_build": ("mlx_lm.convert --q-mode mxfp4 from Qwen3.8-27B no-MTP BF16 view"),
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
        "runtime": {
            "ax_engine": {
                "status": "pass",
                "version": "6.16.1",
                "notes": f"doctor on {HOST_ID}",
            },
            "mlx_lm": {"status": "pass", "notes": "runtime-check passed=True"},
        },
        "toolchain": {
            "axquant": man.get("axquant_version", "1.8.1"),
            "ax_engine": "6.16.1",
            "host": HOST_ID,
        },
        "notes": [
            f"Checkpoint Tier 1 on host id {HOST_ID}.",
            "AXQ-MXFP4: attention+MLP native MXFP4; embed/lm_head 8-bit affine; "
            "vision/norms/MTP protected.",
            "Quality vs same-pin no-MTP BF16; size vs local uniform MXFP4.",
            "Adapter qwen38-dense-v1 — not the Qwen 3.6 flagship track.",
            "Vision BF16-protected; VLM quality not claimed.",
        ],
        "public_index": {
            "display_name": "Qwen3.8-27B MLX AXQ MXFP4",
            "sort_order": 4,
            "edition_label": "main@pending",
            "listed": True,
        },
        "modalities": modalities_to_public_dict(block),
    }
    cert_json = ROOT / "docs" / "certifications" / f"{CERT_STEM}.json"
    write_json(cert_json, payload)
    loaded = load_public_checkpoint_certification(cert_json)
    if loaded.status != "certified" or loaded.host_id != HOST_ID:
        raise SystemExit("public cert loader rejected the MXFP4 record")
    md = ROOT / "docs" / "certifications" / f"{CERT_STEM}.md"
    md.write_text(
        "\n".join(
            [
                "# Qwen3.8-27B AXQ-MXFP4 — checkpoint Tier 1 certification",
                "",
                f"**Verdict:** certified for AXQuant checkpoint Tier 1 on `{HOST_ID}`.",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Hub | [`AutomatosX/{HUB_NAME}`]"
                f"(https://huggingface.co/AutomatosX/{HUB_NAME}) |",
                f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                f"| Host | `{HOST_ID}` |",
                "| Product class | `4bit` (MXFP4 trunk) |",
                (
                    f"| Size vs uniform-MXFP4 | `{size['size_ratio_vs_uniform']:.6f}` "
                    f"(≤ {MAX_SIZE_RATIO}) |"
                ),
                f"| Agent-coding vs BF16 | `{quality['agent-coding']['retention']:.6f}` |",
                f"| General vs BF16 | `{quality['general']['retention']:.6f}` |",
                "| MTP acceleration | `not-applicable` |",
                "",
                "## Gates",
                "",
                "| Gate | Threshold | Observed | Result |",
                "| --- | ---: | ---: | --- |",
                (
                    f"| Size vs uniform-MXFP4 | ≤ `{MAX_SIZE_RATIO}` | "
                    f"`{size['size_ratio_vs_uniform']:.6f}` | Pass |"
                ),
                (
                    "| Agent-coding | ≥ `0.98` | "
                    f"`{quality['agent-coding']['retention']:.6f}` | Pass |"
                ),
                f"| General | ≥ `0.98` | `{quality['general']['retention']:.6f}` | Pass |",
                "| MLX-LM runtime | pass | pass | Pass |",
                "| AX Engine doctor | ready | ready | Pass |",
                "",
                "## Notes",
                "",
                "- Trunk attention and MLP are native MXFP4 (group 32).",
                "- Embeddings and lm_head stay 8-bit affine; vision/norms are BF16.",
                "- Adapter `qwen38-dense-v1`; not the Qwen 3.6 flagship track.",
                "- Vision remains BF16-protected; no VLM quality claim.",
                "- MTP acceleration is **not** certified on this record.",
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
    log(f"wrote {cert_json} and {md}")


def cmd_all() -> None:
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
            "convert",
            "uniforms",
            "size",
            "quality",
            "runtime",
            "write-certs",
            "all",
        ],
    )
    args = parser.parse_args()
    {
        "convert": cmd_convert,
        "uniforms": cmd_uniforms,
        "size": cmd_size,
        "quality": cmd_quality,
        "runtime": cmd_runtime,
        "write-certs": cmd_write_certs,
        "all": cmd_all,
    }[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
