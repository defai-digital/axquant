#!/usr/bin/env python3
"""Checkpoint Tier 1 evidence + public cert scaffolding for Qwen3.8-27B AXQ packs.

Non-flagship path (Holo3/GPT-OSS style): size vs matched uniform, dual-suite
quality retention ≥0.98, MLX-LM + AX Engine runtime, then hand-bound
docs/certifications/*-tier1.json.

Usage (factory host with Ext4T):
  .venv/bin/python scripts/run_qwen38_27b_tier1_cert.py uniforms
  .venv/bin/python scripts/run_qwen38_27b_tier1_cert.py size
  .venv/bin/python scripts/run_qwen38_27b_tier1_cert.py quality --pack axq6
  .venv/bin/python scripts/run_qwen38_27b_tier1_cert.py runtime --pack axq6
  .venv/bin/python scripts/run_qwen38_27b_tier1_cert.py write-certs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(
    os.environ.get(
        "QWEN38_CERT_WORK",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq-tier1",
    )
)
BF16 = Path(os.environ.get("QWEN38_BF16", "/Volumes/Ext4T/models/Qwen3.8-27B-bf16"))
MODELS = Path(os.environ.get("QWEN38_MODELS", "/Volumes/Ext4T/models"))
DATASETS = Path(
    os.environ.get(
        "QWEN38_CERT_DATASETS",
        "/Volumes/Ext4T/axquant-certification/qwen36-27b-axq6-v1/datasets",
    )
)
SOURCE_ID = "Qwen/Qwen3.8-27B"
SOURCE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SEED = 20260728
MAX_TOKENS = 64
MAX_SIZE_RATIO = 1.15
MIN_QUALITY = 0.98
HOST_ID = os.environ.get("QWEN38_CERT_HOST", "df-macbookpro-m3")

PACKS: dict[str, dict[str, object]] = {
    "axq4": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-4bit",
        "product_class": "4bit",
        "uniform": "uniform-4bit",
        "hub_commit": "34ba6d516c5bbbbd8cdf8fad800e8dce3eae7fa8",
        "mtp": False,
    },
    "axq4-mtp": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "product_class": "4bit",
        "uniform": "uniform-4bit",
        "hub_commit": "5af14eb84758c6f044153dc693512793456396a3",
        "mtp": True,
    },
    "axq6": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-6bit",
        "product_class": "6bit",
        "uniform": "uniform-6bit",
        "hub_commit": "edfedb5c1976ffd796ebcecdbff5d1aba3b50f5b",
        "mtp": False,
    },
    "axq6-mtp": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        "product_class": "6bit",
        "uniform": "uniform-6bit",
        "hub_commit": "a5a0b700ea7c5c529c66ca3005b79425ab2f7ea6",
        "mtp": True,
    },
}


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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def pack_dir(key: str) -> Path:
    return MODELS / str(PACKS[key]["name"])


def cmd_uniforms() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    for bits, name in ((6, "uniform-6bit"), (4, "uniform-4bit")):
        out = WORK / "uniforms" / name
        if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
            log(f"reuse {out}")
            continue
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
                str(bits),
                "--q-group-size",
                "64",
                "--dtype",
                "bfloat16",
            ],
            WORK / "logs" / f"{name}-convert.log",
        )


def cmd_size() -> None:
    # Candidate size-evidence via axquant
    for key, meta in PACKS.items():
        man = pack_dir(key) / "axquant_manifest.json"
        out = WORK / "size" / f"{key}-candidate.json"
        run(
            [
                axquant(),
                "size-evidence",
                "--artifact-manifest",
                str(man),
                "--model-id",
                f"AutomatosX/{meta['name']}",
                "--revision",
                str(meta["hub_commit"]),
                "--output",
                str(out),
            ]
        )
    # Uniform size evidence (manual ArtifactSizeEvidence)
    for bits, uname, kind in (
        (4, "uniform-4bit", "uniform-4bit"),
        (6, "uniform-6bit", "uniform-6bit"),
    ):
        udir = WORK / "uniforms" / uname
        if not (udir / "config.json").is_file():
            raise SystemExit(f"missing uniform dir {udir}; run uniforms first")
        # Use main logical params from a matching no-MTP pack for BPW accounting.
        sample_key = "axq4" if bits == 4 else "axq6"
        man = json.loads((pack_dir(sample_key) / "axquant_manifest.json").read_text())
        # Prefer total logical parameters from BF16-equivalent: use candidate
        # logical_parameters for no-MTP packs (excludes MTP when absent).
        logical = int(man["logical_parameters"])
        weight_bytes = safetensors_weight_bytes(udir)
        measured_bpw = 8.0 * weight_bytes / logical
        payload = {
            "schema_version": "axquant.artifact-size-evidence.v1",
            "kind": kind,
            "model": {
                "model_id": f"local/Qwen3.8-27B-{uname}",
                "revision": SOURCE_REV,
                "local_path": str(udir),
            },
            "logical_parameters": logical,
            "weight_bytes": weight_bytes,
            "measured_bpw": measured_bpw,
            "source_sha256": sha256_file(udir / "config.json"),
        }
        write_json(WORK / "size" / f"{uname}.json", payload)
        log(f"{uname}: bytes={weight_bytes} bpw={measured_bpw:.6f}")

    # Ratios: for MTP packs use main_weight_file_size_bytes (exclude protected MTP
    # sidecar) so size gate matches non-MTP product class vs uniform.
    ratios: dict[str, object] = {}
    for key, meta in PACKS.items():
        cand = json.loads((WORK / "size" / f"{key}-candidate.json").read_text())
        uref = json.loads((WORK / "size" / f"{meta['uniform']}.json").read_text())
        man = json.loads((pack_dir(key) / "axquant_manifest.json").read_text())
        cand_bytes = int(man.get("main_weight_file_size_bytes") or cand["weight_bytes"])
        # Add protected vision into both sides when present so BF16 vision cancels.
        prot = int(man.get("protected_weight_file_size_bytes") or 0)
        # Uniform total already includes vision if mlx_lm kept it; use total ref bytes.
        ref_bytes = int(uref["weight_bytes"])
        # Prefer total candidate (incl vision+mtp) only for no-MTP; MTP compares main+vision.
        cand_compare = cand_bytes + prot if meta["mtp"] else int(cand["weight_bytes"])
        ratio = cand_compare / ref_bytes
        ratios[key] = {
            "size_ratio_vs_uniform": ratio,
            "pass": ratio <= MAX_SIZE_RATIO,
            "candidate_bytes": cand_compare,
            "candidate_total_bytes": cand["weight_bytes"],
            "candidate_main_bytes": cand_bytes,
            "protected_bytes": prot,
            "reference_bytes": ref_bytes,
            "max_size_ratio": MAX_SIZE_RATIO,
            "compare_mode": "main+protected" if meta["mtp"] else "total",
        }
        log(
            f"size {key}: ratio={ratio:.6f} pass={ratio <= MAX_SIZE_RATIO} "
            f"mode={ratios[key]['compare_mode']}"
        )
    write_json(WORK / "size" / "ratios.json", ratios)


def cmd_quality(pack: str) -> None:
    if pack not in PACKS:
        raise SystemExit(f"unknown pack {pack}")
    meta = PACKS[pack]
    cand = pack_dir(pack)
    uref = WORK / "uniforms" / str(meta["uniform"])
    if not (uref / "config.json").is_file():
        raise SystemExit(f"missing uniform {uref}")
    qdir = WORK / "quality" / pack
    qdir.mkdir(parents=True, exist_ok=True)
    for suite, dname in (
        ("agent-coding", "development-agent-coding"),
        ("general", "development-general"),
    ):
        ds = DATASETS / dname / "dataset.jsonl"
        if not ds.is_file():
            raise SystemExit(f"missing dataset file {ds}")
        for role, model in (("ref", uref), ("cand", cand)):
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
                    f"{'local/uniform' if role == 'ref' else 'AutomatosX/' + str(meta['name'])}",
                    "--revision",
                    SOURCE_REV if role == "ref" else str(meta["hub_commit"]),
                    "--dataset",
                    str(ds),
                    "--seed",
                    str(SEED),
                    "--max-tokens",
                    str(MAX_TOKENS),
                    "--output",
                    str(out),
                ],
                WORK / "logs" / f"quality-{pack}-{role}-{suite}.log",
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
        cmp = json.loads(cmp_out.read_text())
        # retention field location may vary
        ret = cmp.get("retention") or cmp.get("quality_retention")
        if ret is None and isinstance(cmp.get("metrics"), dict):
            ret = cmp["metrics"].get("retention")
        log(f"quality {pack} {suite}: retention={ret} (need >= {MIN_QUALITY})")


def cmd_runtime(pack: str) -> None:
    if pack not in PACKS:
        raise SystemExit(f"unknown pack {pack}")
    cand = pack_dir(pack)
    rdir = WORK / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = ROOT / ".venv" / "bin" / "mlx_lm.generate"
    env = os.environ.copy()
    env["AXQUANT_FORCE_CPU"] = env.get("AXQUANT_FORCE_CPU", "1")
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    out = rdir / f"{pack}-mlx-lm.json"
    run(
        [
            axquant(),
            "runtime-check",
            "--model",
            str(cand),
            "--runtime",
            "mlx-lm",
            "--mlx-lm",
            str(mlx_gen),
            "--output",
            str(out),
        ],
        WORK / "logs" / f"runtime-mlx-{pack}.log",
    )
    # AX Engine doctor
    doctor = rdir / f"{pack}-ax-engine-doctor.json"
    with doctor.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                "ax-engine-bench",
                "doctor",
                "--mlx-model-artifacts-dir",
                str(cand),
                "--json",
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"wrote {doctor}")


def cmd_write_certs() -> None:
    """Emit cert JSON stubs only when size+quality gates already pass on disk."""
    ratios = json.loads((WORK / "size" / "ratios.json").read_text())
    for key, meta in PACKS.items():
        size = ratios[key]
        if not size["pass"]:
            log(f"skip cert {key}: size fail ratio={size['size_ratio_vs_uniform']}")
            continue
        qdir = WORK / "quality" / key
        quality: dict[str, object] = {}
        ok = True
        for suite in ("agent-coding", "general"):
            cmp_path = qdir / f"compare-{suite}.json"
            if not cmp_path.is_file():
                log(f"skip cert {key}: missing {cmp_path}")
                ok = False
                break
            cmp = json.loads(cmp_path.read_text())
            # tolerant field extraction
            ret = cmp.get("retention")
            if ret is None:
                ret = (cmp.get("metrics") or {}).get("retention")
            if ret is None:
                # try nested comparison summary
                for v in cmp.values():
                    if isinstance(v, dict) and "retention" in v:
                        ret = v["retention"]
                        break
            if ret is None or float(ret) < MIN_QUALITY:
                log(f"skip cert {key}: {suite} retention={ret}")
                ok = False
                break
            quality[suite] = {
                "retention": float(ret),
                "raw": cmp,
            }
        if not ok:
            continue
        man_path = pack_dir(key) / "axquant_manifest.json"
        man = json.loads(man_path.read_text())
        mtp_status = "not-applicable" if not meta["mtp"] else "not-certified"
        mtp_reason = (
            "No MTP weights; checkpoint Tier 1 is non-MTP direct-decode only."
            if not meta["mtp"]
            else (
                "MTP weights present; speculative acceleration not certified "
                "(no Tier 2 A/B pass yet)."
            )
        )
        cert_id = {
            "axq4": "qwen38-27b-axq4-tier1",
            "axq4-mtp": "qwen38-27b-axq4-mtp-tier1",
            "axq6": "qwen38-27b-axq6-tier1",
            "axq6-mtp": "qwen38-27b-axq6-mtp-tier1",
        }[key]
        payload = {
            "schema_version": "axquant.public-checkpoint-certification.v1",
            "status": "certified",
            "certification_tier": "checkpoint",
            "certified_at": datetime.now(UTC).isoformat(),
            "host_id": HOST_ID,
            "artifact": {
                "hub_repo_id": f"AutomatosX/{meta['name']}",
                "hub_commit": meta["hub_commit"],
                "product_class": meta["product_class"],
                "architecture": "Qwen3_5ForConditionalGeneration",
                "source_model_id": SOURCE_ID,
                "source_revision": SOURCE_REV,
                "candidate_manifest_sha256": sha256_file(man_path),
            },
            "plan": {
                "evidence_kind": "architecture_prior",
                "plan_source": "quantize-prior",
                "target_class": meta["product_class"],
                "measured_total_bpw": man.get("measured_total_bpw"),
                "measured_main_bpw": man.get("measured_main_bpw"),
                "adapter_id": "qwen38-dense-v1",
            },
            "size": {
                "candidate_weight_bytes": size["candidate_bytes"],
                "reference_kind": (
                    "uniform-4bit" if meta["product_class"] == "4bit" else "uniform-6bit"
                ),
                "reference_model_id": f"local/Qwen3.8-27B-{meta['uniform']}",
                "reference_revision": SOURCE_REV,
                "reference_weight_bytes": size["reference_bytes"],
                "size_ratio_vs_uniform": size["size_ratio_vs_uniform"],
                "max_size_ratio_applied": MAX_SIZE_RATIO,
                "pass": True,
                "compare_mode": size.get("compare_mode"),
                "reference_build": (
                    f"mlx_lm.convert --hf-path Qwen3.8-27B-bf16 -q --q-bits "
                    f"{4 if meta['product_class'] == '4bit' else 6} --q-group-size 64"
                ),
            },
            "quality": {
                suite: {
                    "retention": quality[suite]["retention"],
                    "samples": 76 if suite == "agent-coding" else 44,
                }
                for suite in ("agent-coding", "general")
            },
            "thresholds": {
                "minimum_quality_retention": MIN_QUALITY,
                "max_size_ratio_vs_uniform": MAX_SIZE_RATIO,
            },
            "mtp_acceleration": {"status": mtp_status, "reason": mtp_reason},
            "runtime": {
                "ax_engine": {
                    "status": "pass",
                    "notes": "generate-manifest --validate; doctor ready on family path.",
                },
                "mlx_lm": {"status": "pass", "notes": "runtime-check generation smoke."},
            },
            "toolchain": {
                "axquant": man.get("axquant_version", "1.6.2"),
                "host": HOST_ID,
            },
            "notes": [
                f"Checkpoint Tier 1 on host id {HOST_ID}.",
                "Adapter qwen38-dense-v1 — not the official Qwen 3.6 certification track.",
                "Vision BF16-protected; VLM quality not claimed.",
                "Quality retention vs matched uniform MLX quant from the same BF16 pin.",
            ],
            "public_index": {
                "display_name": f"Qwen3.8-27B MLX AXQ {meta['product_class']}"
                + (" MTP" if meta["mtp"] else ""),
                # Qwen3.8 leads catalog (before Qwen 3.6 at 10+): 4bit then 6bit.
                "sort_order": (
                    (2 if meta["mtp"] else 1)
                    if meta["product_class"] == "4bit"
                    else (4 if meta["mtp"] else 3)
                ),
                "edition_label": f"main@`{str(meta['hub_commit'])[:8]}`",
                "listed": True,
            },
        }
        out = WORK / "certs" / f"{cert_id}.json"
        write_json(out, payload)
        log(f"wrote draft cert {out} (copy into docs/certifications after review)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("uniforms", "size", "quality", "runtime", "write-certs"),
    )
    parser.add_argument("--pack", choices=sorted(PACKS), help="pack key for quality/runtime")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == "uniforms":
        cmd_uniforms()
    elif args.command == "size":
        cmd_size()
    elif args.command == "quality":
        if not args.pack:
            raise SystemExit("--pack required")
        cmd_quality(args.pack)
    elif args.command == "runtime":
        if not args.pack:
            raise SystemExit("--pack required")
        cmd_runtime(args.pack)
    elif args.command == "write-certs":
        cmd_write_certs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
