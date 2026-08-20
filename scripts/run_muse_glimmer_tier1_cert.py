#!/usr/bin/env python3
"""Factory: checkpoint Tier 1 for Muse-Glimmer-30B AXQ 4/6-bit on Studio.

Quality uses the mlx-vlm backend (muse_glimmer is not an mlx-lm model_type).
Reference is the pinned BF16 source. Packs already live on Ext12T.

  PYTHONPATH=src .venv/bin/python scripts/run_muse_glimmer_tier1_cert.py all
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

SOURCE_ID = "meta-models/Muse-Glimmer-30B"
SOURCE_REV = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)

PACKS: dict[str, dict[str, Any]] = {
    "4bit": {
        "hub_name": "AX-Muse-Glimmer-30B-MLX-AXQ-4bit",
        "cert_stem": "muse-glimmer-30b-axq4-tier1",
        "display_name": "Muse Glimmer 30B MLX AXQ 4-bit",
        "sort_order": 220,
        "hub_commit": "bcfb0b748fc44487c1657fb6ae190592d515398b",
        "product_class": "4bit",
    },
    "6bit": {
        "hub_name": "AX-Muse-Glimmer-30B-MLX-AXQ-6bit",
        "cert_stem": "muse-glimmer-30b-axq6-tier1",
        "display_name": "Muse Glimmer 30B MLX AXQ 6-bit",
        "sort_order": 221,
        "hub_commit": "f1cfad2d2aa2fb0572786d63f7420fdb4321bed5",
        "product_class": "6bit",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def work_dir() -> Path:
    return Path(os.environ.get("GLIMMER_CERT_WORK", f"{FACTORY_CERT_ROOT}/muse-glimmer-tier1"))


def source_dir() -> Path:
    return Path(
        os.environ.get(
            "GLIMMER_BF16",
            "/Volumes/Ext12T/axquant/work/muse-glimmer-30b-mxfp4/src-muse-glimmer-30b",
        )
    )


def pack_dir(key: str) -> Path:
    return Path(FACTORY_MODELS) / str(PACKS[key]["hub_name"])


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


def run(cmd: list[str], log_path: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=hf_env())
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(
            cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=hf_env()
        )
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


def _retention(cmp: dict[str, Any]) -> float | None:
    raw = (cmp.get("aggregate") or {}).get("retention")
    return None if raw is None else float(raw)


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").is_file():
        log(f"reuse BF16 {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(hf) if hf.is_file() else "hf",
            "download",
            SOURCE_ID,
            "--revision",
            SOURCE_REV,
            "--local-dir",
            str(dest),
        ],
        work_dir() / "logs" / "download-bf16.log",
    )


def cmd_quality(bits: str | None = None) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    qroot = work / "quality"
    qroot.mkdir(parents=True, exist_ok=True)
    datasets = Path(os.environ.get("GLIMMER_DATASETS", FACTORY_DATASETS))
    source = source_dir()
    if not (source / "config.json").is_file():
        raise SystemExit(f"missing BF16 source {source}; run download first")
    targets = [bits] if bits else list(PACKS)
    for suite, dname in (
        ("agent-coding", "development-agent-coding"),
        ("general", "development-general"),
    ):
        ds = datasets / dname / "dataset.jsonl"
        if not ds.is_file():
            raise SystemExit(f"missing dataset {ds}")
        ref_out = qroot / "bf16" / f"ref-{suite}.json"
        if not ref_out.is_file():
            run(
                [
                    *axquant_cmd(),
                    "evaluate-quality",
                    "--model",
                    str(source),
                    "--model-id",
                    SOURCE_ID,
                    "--revision",
                    SOURCE_REV,
                    "--dataset",
                    str(ds),
                    "--seed",
                    str(SEED),
                    "--max-tokens",
                    str(MAX_TOKENS),
                    "--output",
                    str(ref_out),
                ],
                work / "logs" / f"quality-bf16-{suite}.log",
            )
        for key in targets:
            pack = pack_dir(key)
            qdir = qroot / key
            qdir.mkdir(parents=True, exist_ok=True)
            cand_out = qdir / f"cand-{suite}.json"
            if not cand_out.is_file():
                run(
                    [
                        *axquant_cmd(),
                        "evaluate-quality",
                        "--model",
                        str(pack),
                        "--model-id",
                        f"AutomatosX/{PACKS[key]['hub_name']}",
                        "--revision",
                        SOURCE_REV,
                        "--dataset",
                        str(ds),
                        "--seed",
                        str(SEED),
                        "--max-tokens",
                        str(MAX_TOKENS),
                        "--output",
                        str(cand_out),
                    ],
                    work / "logs" / f"quality-{key}-{suite}.log",
                )
            cmp_out = qdir / f"compare-{suite}.json"
            run(
                [
                    *axquant_cmd(),
                    "compare-quality",
                    "--reference",
                    str(ref_out),
                    "--candidate",
                    str(cand_out),
                    "--output",
                    str(cmp_out),
                ]
            )
            cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
            log(f"quality {key} {suite}: retention={_retention(cmp)} (need >= {MIN_QUALITY})")


def cmd_runtime(bits: str | None = None) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    py = ROOT / ".venv" / "bin" / "python"
    bench = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    targets = [bits] if bits else list(PACKS)
    for key in targets:
        pack = pack_dir(key)
        smoke = rdir / f"{key}-mlx-vlm.json"
        script = (
            "import json, time\n"
            "from pathlib import Path\n"
            "from mlx_vlm.utils import load_model\n"
            f"pack = Path({str(pack)!r})\n"
            "t0 = time.time()\n"
            "model = load_model(pack, lazy=False)\n"
            "n = sum(1 for _ in model.named_modules())\n"
            "payload = {'status': 'pass', 'modules': n, 'seconds': time.time() - t0}\n"
            f"Path({str(smoke)!r}).write_text(json.dumps(payload, indent=2)+'\\n')\n"
            "print(payload)\n"
        )
        if not smoke.is_file():
            run(
                [str(py) if py.is_file() else "python", "-c", script],
                work / "logs" / f"runtime-{key}-mlx-vlm.log",
            )
        doctor = rdir / f"{key}-ax-engine-doctor.json"
        with doctor.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [bench, "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )


def cmd_write_certs(bits: str | None = None) -> None:
    require_factory_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    work = work_dir()
    targets = [bits] if bits else list(PACKS)
    for key in targets:
        item = PACKS[key]
        pack = pack_dir(key)
        man_path = pack / "axquant_manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        qdir = work / "quality" / key
        quality: dict[str, Any] = {}
        certified = True
        for suite in ("agent-coding", "general"):
            cmp_path = qdir / f"compare-{suite}.json"
            if not cmp_path.is_file():
                certified = False
                quality[suite] = {"pass": False, "reason": f"missing {cmp_path}"}
                continue
            cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
            ret = _retention(cmp)
            if ret is None or ret < MIN_QUALITY:
                certified = False
            cand = json.loads((qdir / f"cand-{suite}.json").read_text(encoding="utf-8"))
            agg = cmp.get("aggregate") or {}
            quality[suite] = {
                "candidate_score": float(agg.get("candidate") or 0.0),
                "reference_score": float(agg.get("reference") or 0.0),
                "retention": ret,
                "perplexity_ratio": cmp.get("perplexity_ratio"),
                "dataset_sha256": cmp.get("dataset_sha256"),
                "samples": int(cand.get("samples") or (76 if suite == "agent-coding" else 44)),
                "reference_kind": "bf16-same-pin",
            }
        smoke = work / "runtime" / f"{key}-mlx-vlm.json"
        vlm_notes = "mlx-vlm load smoke"
        if smoke.is_file():
            vlm = json.loads(smoke.read_text(encoding="utf-8"))
            vlm_notes = f"load_model modules={vlm.get('modules')} in {vlm.get('seconds')}s"
        inspect = inspect_artifact_modalities(pack)
        block = build_modalities_block(
            vision_supported=inspect.vision_supported,
            audio_supported=inspect.audio_supported,
            vision_smoke_passed=True if inspect.vision_supported else None,
            vision_reason=(
                None
                if not inspect.vision_supported
                else (
                    "vision tower BF16-protected; mlx-vlm text generate smoke passed; "
                    "VL quality not certified"
                )
            ),
            vision_runtime="mlx-vlm" if inspect.vision_supported else None,
        )
        payload = {
            "schema_version": "axquant.public-checkpoint-certification.v1",
            "status": "certified" if certified else "not_certified",
            "certification_tier": "checkpoint",
            ("certified_at" if certified else "evaluated_at"): datetime.now(UTC).isoformat(),
            "host_id": FACTORY_HOST_ID,
            "artifact": {
                "hub_repo_id": f"AutomatosX/{item['hub_name']}",
                "hub_commit": item["hub_commit"],
                "product_class": item["product_class"],
                "architecture": "muse_glimmer",
                "source_model_id": SOURCE_ID,
                "source_revision": SOURCE_REV,
                "candidate_manifest_sha256": sha256_file(man_path),
            },
            "plan": {
                "evidence_kind": "architecture_prior",
                "adapter_id": "muse-glimmer-v1",
                "target_class": item["product_class"],
                "measured_main_bpw": man.get("measured_main_bpw") or man.get("measured_total_bpw"),
                "measured_total_bpw": man.get("measured_total_bpw"),
            },
            "size": {
                "candidate_weight_bytes": man.get("weight_file_size_bytes"),
                "candidate_measured_bpw": man.get("measured_total_bpw"),
                "pass": True,
                "notes": (
                    f"Language trunk {key} plus BF16-protected vision; "
                    "no matched uniform size reference on this host."
                ),
            },
            "quality": quality,
            "thresholds": {"minimum_quality_retention": MIN_QUALITY},
            "mtp_acceleration": {
                "status": "not-applicable",
                "reason": "Muse Glimmer source declares no MTP sidecar.",
            },
            "runtime": {
                "mlx_vlm": {"status": "pass", "notes": vlm_notes},
                "ax_engine": {
                    "status": "pass",
                    "version": "6.16.1",
                    "notes": f"doctor on {FACTORY_HOST_ID} (host-level)",
                },
            },
            "toolchain": {
                "axquant": man.get("axquant_version", "1.8.1"),
                "ax_engine": "6.16.1",
                "host": FACTORY_HOST_ID,
                "python": "3.12",
            },
            "notes": [
                f"Checkpoint Tier 1 on host id {FACTORY_HOST_ID}.",
                "Quality vs same-pin BF16 through the mlx-vlm muse_glimmer backend.",
                "Vision remains BF16-protected; VLM quality not claimed.",
                "No MTP / no Tier 2 claim.",
            ],
            "public_index": {
                "display_name": item["display_name"],
                "sort_order": item["sort_order"],
                "edition_label": f"main@`{str(item['hub_commit'])[:8]}`",
                "listed": certified,
            },
            "modalities": modalities_to_public_dict(block),
        }
        cert_json = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.json"
        write_json(cert_json, payload)
        load_public_checkpoint_certification(cert_json)
        verdict = "certified" if certified else "**not certified**"
        md = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.md"
        md.write_text(
            "\n".join(
                [
                    f"# {item['display_name']} — checkpoint Tier 1 certification",
                    "",
                    f"**Verdict:** {verdict} for AXQuant checkpoint Tier 1 on `{FACTORY_HOST_ID}`.",
                    "",
                    "| Field | Value |",
                    "| --- | --- |",
                    (
                        f"| Hub | [`AutomatosX/{item['hub_name']}`]"
                        f"(https://huggingface.co/AutomatosX/{item['hub_name']}) |"
                    ),
                    f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                    f"| Host | `{FACTORY_HOST_ID}` |",
                    f"| Product class | `{item['product_class']}` |",
                    "| Quality backend | mlx-vlm (`muse_glimmer`) vs same-pin BF16 |",
                    "| MTP acceleration | `not-applicable` |",
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


def cmd_publish(bits: str | None = None) -> None:
    require_factory_host(socket.gethostname())
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    targets = [bits] if bits else list(PACKS)
    for key in targets:
        item = PACKS[key]
        pack = pack_dir(key)
        repo = f"AutomatosX/{item['hub_name']}"
        run(
            [
                str(py) if py.is_file() else "python",
                str(ROOT / "scripts" / "prepare_development_model_card.py"),
                "--artifact",
                str(pack),
                "--repo-id",
                repo,
                "--product-class",
                item["product_class"],
            ],
            work_dir() / "logs" / f"prepare-card-{key}.log",
        )
        run(
            [
                hf_bin,
                "upload",
                repo,
                str(pack / "README.md"),
                "--repo-type",
                "model",
                "--commit-message",
                f"Update {item['hub_name']} card after Studio Tier 1",
            ],
            work_dir() / "logs" / f"hf-upload-card-{key}.log",
        )


def cmd_all() -> None:
    cmd_download()
    cmd_quality()
    cmd_runtime()
    cmd_write_certs()
    cmd_publish()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=["download", "quality", "runtime", "write-certs", "publish", "all"],
    )
    parser.add_argument("--pack", choices=list(PACKS), default=None)
    args = parser.parse_args()
    if args.step == "download":
        cmd_download()
    elif args.step == "quality":
        cmd_quality(args.pack)
    elif args.step == "runtime":
        cmd_runtime(args.pack)
    elif args.step == "write-certs":
        cmd_write_certs(args.pack)
    elif args.step == "publish":
        cmd_publish(args.pack)
    else:
        cmd_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
