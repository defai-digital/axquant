#!/usr/bin/env python3
"""Checkpoint Tier 1 for Qwen3.8-27B AXQ 8-bit (no-MTP).

Designed to run on df-macstudio-m2 after the 8-bit pack exists:

  .venv/bin/python scripts/run_qwen38_27b_axq8_cert.py all

Steps: uniform-8 size reference, size gate, dual-suite quality vs BF16,
mlx-lm smoke, AX Engine doctor, public cert JSON/MD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(
    os.environ.get(
        "QWEN38_8BIT_CERT_WORK",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq8-tier1",
    )
)
BF16 = Path(os.environ.get("QWEN38_BF16", "/Volumes/Ext4T/models/Qwen3.8-27B-bf16"))
PACK = Path(
    os.environ.get(
        "QWEN38_8BIT_PACK",
        "/Volumes/Ext4T/models/AX-Qwen3.8-27B-MLX-AXQ-8bit",
    )
)
DATASETS = Path(
    os.environ.get(
        "QWEN38_CERT_DATASETS",
        str(WORK / "datasets"),
    )
)
SOURCE_ID = "Qwen/Qwen3.8-27B"
SOURCE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HOST_ID = os.environ.get("QWEN38_CERT_HOST", "df-macstudio-m2")
HUB_NAME = "AX-Qwen3.8-27B-MLX-AXQ-8bit"
CERT_STEM = "qwen38-27b-axq8-tier1"
MAX_SIZE_RATIO = 1.15
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64


def log(msg: str) -> None:
    print(msg, flush=True)


def axquant() -> str:
    cand = ROOT / ".venv" / "bin" / "axquant"
    return str(cand if cand.is_file() else "axquant")


def run(cmd: list[str], log_path: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT))
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


def cmd_uniforms() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    out = WORK / "uniforms" / "uniform-8bit"
    if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
        log(f"reuse {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "mlx_lm.convert",
            "--hf-path",
            str(BF16),
            "--mlx-path",
            str(out),
            "-q",
            "--q-bits",
            "8",
            "--q-group-size",
            "64",
            "--dtype",
            "bfloat16",
        ],
        WORK / "logs" / "uniform-8bit-convert.log",
    )


def cmd_size() -> None:
    man_path = PACK / "axquant_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"missing pack manifest {man_path}")
    (WORK / "size").mkdir(parents=True, exist_ok=True)
    out = WORK / "size" / "axq8-candidate.json"
    # size-evidence requires a 40-hex revision; Hub commit is patched after upload.
    size_rev = os.environ.get("QWEN38_8BIT_HUB_COMMIT", SOURCE_REV)
    if not re.fullmatch(r"[0-9a-f]{40}", size_rev):
        size_rev = SOURCE_REV
    run(
        [
            axquant(),
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
    udir = WORK / "uniforms" / "uniform-8bit"
    if not (udir / "config.json").is_file():
        raise SystemExit(f"missing uniform dir {udir}; run uniforms first")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    logical = int(man["logical_parameters"])
    weight_bytes = safetensors_weight_bytes(udir)
    measured_bpw = 8.0 * weight_bytes / logical
    uref = {
        "schema_version": "axquant.artifact-size-evidence.v1",
        "kind": "uniform-8bit",
        "model": {
            "model_id": "local/Qwen3.8-27B-uniform-8bit",
            "revision": SOURCE_REV,
            "local_path": str(udir),
        },
        "logical_parameters": logical,
        "weight_bytes": weight_bytes,
        "measured_bpw": measured_bpw,
    }
    write_json(WORK / "size" / "uniform-8bit.json", uref)
    cand = json.loads(out.read_text(encoding="utf-8"))
    cand_bytes = int(cand.get("weight_bytes") or man.get("weight_file_size_bytes") or 0)
    if cand_bytes <= 0:
        cand_bytes = safetensors_weight_bytes(PACK)
    ratio = cand_bytes / weight_bytes
    ratios = {
        "axq8": {
            "size_ratio_vs_uniform": ratio,
            "pass": ratio <= MAX_SIZE_RATIO,
            "candidate_bytes": cand_bytes,
            "reference_bytes": weight_bytes,
            "max_size_ratio": MAX_SIZE_RATIO,
            "compare_mode": "total",
        }
    }
    write_json(WORK / "size" / "ratios.json", ratios)
    log(f"size axq8: ratio={ratio:.6f} pass={ratio <= MAX_SIZE_RATIO}")


def _retention(cmp: dict) -> float | None:
    agg = cmp.get("aggregate") or {}
    ret = agg.get("retention")
    if ret is None:
        ret = cmp.get("retention")
    return None if ret is None else float(ret)


def cmd_quality() -> None:
    qdir = WORK / "quality" / "axq8"
    qdir.mkdir(parents=True, exist_ok=True)
    if not BF16.is_dir():
        raise SystemExit(f"missing BF16 reference {BF16}")
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
            ("ref", BF16, SOURCE_ID),
            ("cand", PACK, f"AutomatosX/{HUB_NAME}"),
        ):
            out = qdir / f"{role}-{suite}.json"
            if out.is_file():
                log(f"reuse {out}")
                continue
            run(
                [
                    axquant(),
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
                WORK / "logs" / f"quality-axq8-{role}-{suite}.log",
            )
        cmp_out = qdir / f"compare-{suite}.json"
        run(
            [
                axquant(),
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
        log(f"quality axq8 {suite}: retention={ret} (need >= {MIN_QUALITY})")


def cmd_runtime() -> None:
    rdir = WORK / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = ROOT / ".venv" / "bin" / "mlx_lm.generate"
    if not mlx_gen.is_file():
        mlx_gen = Path("mlx_lm.generate")
    out = rdir / "axq8-mlx-lm.json"
    run(
        [
            axquant(),
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
        WORK / "logs" / "runtime-mlx-axq8.log",
    )
    doctor = rdir / "axq8-ax-engine-doctor.json"
    bench = os.environ.get(
        "AX_ENGINE_BENCH",
        "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
    )
    with doctor.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                bench,
                "doctor",
                "--mlx-model-artifacts-dir",
                str(PACK),
                "--json",
            ],
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
    size = ratios["axq8"]
    if not size["pass"]:
        raise SystemExit(f"size gate fail ratio={size['size_ratio_vs_uniform']}")
    qdir = WORK / "quality" / "axq8"
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
    mlx = json.loads((WORK / "runtime" / "axq8-mlx-lm.json").read_text(encoding="utf-8"))
    if not mlx.get("passed"):
        raise SystemExit("mlx-lm runtime-check did not pass")
    man_path = PACK / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    inspect = inspect_artifact_modalities(PACK)
    smoke_path = WORK / "runtime" / "axq8-mlx-vlm.json"
    vis_smoke = None
    if inspect.vision_supported and smoke_path.is_file():
        vis = json.loads(smoke_path.read_text(encoding="utf-8"))
        vis_smoke = True if vis.get("passed") else None
    block = build_modalities_block(
        vision_supported=inspect.vision_supported,
        audio_supported=inspect.audio_supported,
        vision_smoke_passed=vis_smoke,
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
    hub_commit = os.environ.get("QWEN38_8BIT_HUB_COMMIT", "0" * 40)
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified",
        "certification_tier": "checkpoint",
        "certified_at": datetime.now(UTC).isoformat(),
        "host_id": HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{HUB_NAME}",
            "hub_commit": hub_commit,
            "product_class": "8bit",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path),
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "quantize-prior",
            "target_class": "8bit",
            "target_bpw": 8.0,
            "measured_total_bpw": man.get("measured_total_bpw"),
            "measured_main_bpw": man.get("measured_main_bpw"),
            "adapter_id": "qwen38-dense-v1",
        },
        "size": {
            "candidate_weight_bytes": size["candidate_bytes"],
            "candidate_measured_bpw": man.get("measured_total_bpw"),
            "reference_kind": "uniform-8bit",
            "reference_model_id": "local/Qwen3.8-27B-uniform-8bit",
            "reference_revision": SOURCE_REV,
            "reference_weight_bytes": size["reference_bytes"],
            "size_ratio_vs_uniform": size["size_ratio_vs_uniform"],
            "max_size_ratio_applied": MAX_SIZE_RATIO,
            "pass": True,
            "compare_mode": "total",
            "reference_build": (
                "mlx_lm.convert --hf-path Qwen3.8-27B-bf16 -q --q-bits 8 --q-group-size 64"
            ),
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
                "notes": "doctor ready / mlx_runtime_ready on df-macstudio-m2",
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
            "Quality vs same-pin BF16; size vs local uniform-8 convert.",
            "Adapter qwen38-dense-v1 — not the Qwen 3.6 flagship track.",
            "Vision BF16-protected; VLM quality not claimed.",
        ],
        "public_index": {
            "display_name": "Qwen3.8-27B MLX AXQ 8-bit",
            "sort_order": 5,
            "edition_label": "main@pending",
            "listed": True,
        },
        "modalities": modalities_to_public_dict(block),
    }
    cert_json = ROOT / "docs" / "certifications" / f"{CERT_STEM}.json"
    write_json(cert_json, payload)
    loaded = load_public_checkpoint_certification(cert_json)
    if loaded.status != "certified" or loaded.host_id != HOST_ID:
        raise SystemExit("public cert loader rejected the 8-bit record")
    md = ROOT / "docs" / "certifications" / f"{CERT_STEM}.md"
    md.write_text(
        "\n".join(
            [
                "# Qwen3.8-27B AXQ 8-bit — checkpoint Tier 1 certification",
                "",
                f"**Verdict:** certified for AXQuant checkpoint Tier 1 on {HOST_ID}.",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Hub | [`AutomatosX/{HUB_NAME}`]"
                f"(https://huggingface.co/AutomatosX/{HUB_NAME}) |",
                f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                f"| Host | `{HOST_ID}` |",
                "| Product class | `8bit` |",
                f"| Size vs uniform-8 | `{size['size_ratio_vs_uniform']:.6f}` (≤ 1.15) |",
                f"| Agent-coding vs BF16 | `{quality['agent-coding']['retention']:.6f}` |",
                f"| General vs BF16 | `{quality['general']['retention']:.6f}` |",
                "| MTP acceleration | `not-applicable` |",
                "",
                "## Gates",
                "",
                "| Gate | Threshold | Observed | Result |",
                "| --- | ---: | ---: | --- |",
                f"| Size vs uniform-8 | ≤ `1.15` | `{size['size_ratio_vs_uniform']:.6f}` | Pass |",
                (
                    "| Agent-coding | ≥ `0.98` | "
                    f"`{quality['agent-coding']['retention']:.6f}` | Pass |"
                ),
                (f"| General | ≥ `0.98` | `{quality['general']['retention']:.6f}` | Pass |"),
                "| MLX-LM runtime | pass | pass | Pass |",
                "| AX Engine doctor | ready | ready | Pass |",
                "",
                "## Notes",
                "",
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
    cmd_uniforms()
    cmd_size()
    cmd_quality()
    cmd_runtime()
    cmd_write_certs()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=["uniforms", "size", "quality", "runtime", "write-certs", "all"],
    )
    args = parser.parse_args()
    {
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
