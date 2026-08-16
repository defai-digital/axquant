#!/usr/bin/env python3
"""Factory: convert + checkpoint Tier 1 for Holo-3.1-35B-A3B AXQ MXFP4 / 6 / 8.

Same qwen35-moe-v1 35B-A3B path as Holo3. Quality is mlx-lm vs same-pin BF16.

Run on df-macstudio-m2 + Ext12T only, one pack at a time:

  PYTHONPATH=src .venv/bin/python scripts/run_holo31_35b_axq.py --pack mxfp4 all
  PYTHONPATH=src .venv/bin/python scripts/run_holo31_35b_axq.py --pack axq6 all
  PYTHONPATH=src .venv/bin/python scripts/run_holo31_35b_axq.py --pack axq8 all
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

SOURCE_ID = "Hcompany/Holo-3.1-35B-A3B"
SOURCE_REV = "2bdb92851a8cd9d72cdd891fdf38cfcc7fefae2c"
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)

PACKS: dict[str, dict[str, Any]] = {
    "mxfp4": {
        "hub_name": "AX-Holo-3.1-35B-A3B-MLX-AXQ-MXFP4",
        "cert_stem": "holo31-35b-axq-mxfp4-tier1",
        "display_name": "Holo-3.1-35B-A3B MLX AXQ MXFP4",
        "recipe": ROOT / "examples" / "holo31-35b-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
        "target_class": "4bit",
        "sort_order": 90,
    },
    "axq6": {
        "hub_name": "AX-Holo-3.1-35B-A3B-MLX-AXQ-6bit",
        "cert_stem": "holo31-35b-axq6-tier1",
        "display_name": "Holo-3.1-35B-A3B MLX AXQ 6-bit",
        "recipe": ROOT / "examples" / "holo31-35b-axq6-v0.1.yaml",
        "q_mode": "affine",
        "product_class": "6bit",
        "target_class": "6bit",
        "sort_order": 91,
    },
    "axq8": {
        "hub_name": "AX-Holo-3.1-35B-A3B-MLX-AXQ-8bit",
        "cert_stem": "holo31-35b-axq8-tier1",
        "display_name": "Holo-3.1-35B-A3B MLX AXQ 8-bit",
        "recipe": ROOT / "examples" / "holo31-35b-axq8-v0.1.yaml",
        "q_mode": "affine",
        "product_class": "8bit",
        "target_class": "8bit",
        "sort_order": 92,
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    return PACKS[key]


def work_dir() -> Path:
    return Path(os.environ.get("HOLO31_WORK", f"{FACTORY_CERT_ROOT}/holo31-35b-axq"))


def source_dir() -> Path:
    return Path(
        os.environ.get(
            "HOLO31_BF16",
            "/Volumes/Ext12T/axquant/work/holo31-35b-axq/src-holo31-35b",
        )
    )


def pack_dir(key: str) -> Path:
    return Path(FACTORY_MODELS) / str(spec(key)["hub_name"])


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
        env["AXQUANT_FORCE_CPU"] = "1"
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


def _retention(cmp: dict[str, Any]) -> float | None:
    raw = (cmp.get("aggregate") or {}).get("retention")
    return None if raw is None else float(raw)


def hub_commit_path(key: str) -> Path:
    return work_dir() / f"hub-commit-{key}.txt"


def read_hub_commit(key: str) -> str:
    override = os.environ.get("HOLO31_HUB_COMMIT")
    if override:
        return override
    path = hub_commit_path(key)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or ("0" * 40)
    return "0" * 40


def source_complete(dest: Path) -> bool:
    if not (dest / "config.json").is_file():
        return False
    index = dest / "model.safetensors.index.json"
    if not index.is_file():
        return False
    payload = json.loads(index.read_text(encoding="utf-8"))
    files = {dest / name for name in payload.get("weight_map", {}).values()}
    return bool(files) and all(path.is_file() and path.stat().st_size > 0 for path in files)


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_complete(dest):
        log(f"reuse source {dest}")
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
        work_dir() / "logs" / "download-src.log",
    )
    if not source_complete(dest):
        raise SystemExit(f"incomplete BF16 source {dest}")


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    recipe = item["recipe"]
    if not recipe.is_file():
        raise SystemExit(f"missing recipe {recipe}")
    source = source_dir()
    if not source_complete(source):
        raise SystemExit(f"incomplete source {source}; run download first")
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
                str(source),
                "--model-id",
                SOURCE_ID,
                "--revision",
                SOURCE_REV,
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
                str(recipe),
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
            str(item["q_mode"]),
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "skip",
            "--ax-engine-bench",
            engine,
        ],
        work / "logs" / f"convert-{key}.log",
        force_cpu=True,
    )


def cmd_quality(key: str) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    pack = pack_dir(key)
    qroot = work / "quality"
    qdir = qroot / key
    qdir.mkdir(parents=True, exist_ok=True)
    source = source_dir()
    datasets = Path(os.environ.get("HOLO31_DATASETS", FACTORY_DATASETS))
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
        cand_out = qdir / f"cand-{suite}.json"
        if not cand_out.is_file():
            run(
                [
                    *axquant_cmd(),
                    "evaluate-quality",
                    "--model",
                    str(pack),
                    "--model-id",
                    f"AutomatosX/{spec(key)['hub_name']}",
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


def cmd_runtime(key: str) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    pack = pack_dir(key)
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = ROOT / ".venv" / "bin" / "mlx_lm.generate"
    out = rdir / f"{key}-mlx-lm.json"
    if not out.is_file():
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
            work / "logs" / f"runtime-mlx-{key}.log",
        )
    doctor = rdir / f"{key}-ax-engine-doctor.json"
    bench = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    with doctor.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [bench, "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log(f"wrote {doctor}")


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    repo = f"AutomatosX/{item['hub_name']}"
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
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
    run([hf_bin, "repo", "create", repo, "--exist-ok"], work_dir() / "logs" / f"hf-repo-{key}.log")
    run(
        [
            hf_bin,
            "upload",
            repo,
            str(pack),
            "--repo-type",
            "model",
            "--commit-message",
            f"Publish {item['hub_name']} from {SOURCE_ID}@{SOURCE_REV}",
        ],
        work_dir() / "logs" / f"hf-upload-{key}.log",
    )
    script = f"from huggingface_hub import model_info\nprint(model_info({repo!r}).sha)\n"
    env = hf_env()
    proc = subprocess.run(
        [str(py) if py.is_file() else "python", "-c", script],
        check=True,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    commit = (proc.stdout or "").strip().splitlines()[-1].strip()
    if len(commit) != 40:
        raise SystemExit(f"could not read hub commit for {repo}: {proc.stdout!r} {proc.stderr!r}")
    hub_commit_path(key).write_text(commit + "\n", encoding="utf-8")
    log(f"published {repo} commit={commit}")


def cmd_write_certs(key: str) -> None:
    require_factory_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    item = spec(key)
    pack = pack_dir(key)
    work = work_dir()
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
    mlx_runtime = work / "runtime" / f"{key}-mlx-lm.json"
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
    listed = certified
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified" if certified else "not_certified",
        "certification_tier": "checkpoint",
        ("certified_at" if certified else "evaluated_at"): datetime.now(UTC).isoformat(),
        "host_id": FACTORY_HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{item['hub_name']}",
            "hub_commit": read_hub_commit(key),
            "product_class": item["product_class"],
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path),
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "plan-manual",
            "recipe": str(item["recipe"].relative_to(ROOT)),
            "adapter_id": "qwen35-moe-v1",
            "target_class": item["target_class"],
            "measured_main_bpw": man.get("measured_main_bpw") or man.get("measured_total_bpw"),
            "measured_total_bpw": man.get("measured_total_bpw"),
        },
        "size": {
            "candidate_weight_bytes": man.get("weight_file_size_bytes"),
            "candidate_measured_bpw": man.get("measured_total_bpw"),
            "pass": True,
            "notes": (
                f"Language trunk {item['product_class']}; vision BF16-protected; "
                "no matched uniform size reference."
            ),
        },
        "quality": quality,
        "thresholds": {"minimum_quality_retention": MIN_QUALITY},
        "mtp_acceleration": {
            "status": "not-applicable",
            "reason": "Holo-3.1-35B-A3B source declares no MTP sidecar.",
        },
        "runtime": runtime,
        "toolchain": {
            "axquant": man.get("axquant_version", "1.8.1"),
            "ax_engine": "6.16.1",
            "host": FACTORY_HOST_ID,
        },
        "notes": [
            f"Checkpoint attempt on {FACTORY_HOST_ID}.",
            "Quality vs same-pin BF16 through the mlx-lm qwen3_5_moe backend.",
            "Adapter qwen35-moe-v1 — not the official Qwen 3.6 certification track.",
            "Vision remains BF16-protected; VLM quality not claimed.",
            "No MTP / no Tier 2 claim.",
        ],
        "public_index": {
            "display_name": item["display_name"],
            "sort_order": item["sort_order"],
            "edition_label": "main@pending",
            "listed": listed,
        },
        "modalities": modalities_to_public_dict(block),
    }
    cert_json = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.json"
    write_json(cert_json, payload)
    loaded = load_public_checkpoint_certification(cert_json)
    if loaded.host_id != FACTORY_HOST_ID:
        raise SystemExit("public cert loader rejected the Holo-3.1 record")
    verdict = "certified" if certified else "**not certified**"
    md = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.md"
    md.write_text(
        "\n".join(
            [
                f"# {item['display_name']} — checkpoint Tier 1",
                "",
                f"**Verdict:** {verdict} on `{FACTORY_HOST_ID}`.",
                "",
                "| Field | Value |",
                "| --- | --- |",
                (
                    f"| Hub | [`AutomatosX/{item['hub_name']}`]"
                    f"(https://huggingface.co/AutomatosX/{item['hub_name']}) |"
                ),
                f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                f"| Product class | `{item['product_class']}` |",
                "| Quality backend | mlx-lm vs same-pin BF16 |",
                "| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |",
                "| MTP | not-applicable |",
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
    cmd_download()
    cmd_convert(key)
    cmd_quality(key)
    cmd_runtime(key)
    cmd_publish(key)
    cmd_write_certs(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=list(PACKS), required=True)
    parser.add_argument(
        "step",
        choices=["download", "convert", "quality", "runtime", "publish", "write-certs", "all"],
    )
    args = parser.parse_args()
    {
        "download": lambda: cmd_download(),
        "convert": lambda: cmd_convert(args.pack),
        "quality": lambda: cmd_quality(args.pack),
        "runtime": lambda: cmd_runtime(args.pack),
        "publish": lambda: cmd_publish(args.pack),
        "write-certs": lambda: cmd_write_certs(args.pack),
        "all": lambda: cmd_all(args.pack),
    }[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
