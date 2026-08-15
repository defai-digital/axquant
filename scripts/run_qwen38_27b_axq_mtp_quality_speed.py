#!/usr/bin/env python3
"""Four-way QA quality vs MTP speed for Qwen3.8-27B AXQ MTP packs.

Run on df-macstudio-m2:

  export PATH="/Users/devop/opt/ax-engine-6.16.1/bin:$PATH"
  PYTHONPATH=src python scripts/run_qwen38_27b_axq_mtp_quality_speed.py all

Compares MXFP4 / 4-bit / 6-bit / 8-bit MTP siblings on one host. Quality is
dual-suite retention vs the same-pin BF16 source. Speed is AX Engine MTP
off/on A/B under the Qwen3.8 exact-async profile.

This is a measured comparison, not a certification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.benchmark import (  # noqa: E402
    QWEN38_EXACT_MTP_PROFILE_ENV,
    compare_mtp_ab_results,
    run_benchmark,
)
from axquant.schema import BenchmarkConfig, ModelIdentity  # noqa: E402
from axquant.serde import file_sha256, write_data  # noqa: E402

HOST_ID = os.environ.get("QWEN38_BENCH_HOST", "df-macstudio-m2")
SEED = 20260728
MAX_TOKENS_QUALITY = 64
WORK = Path(
    os.environ.get(
        "QWEN38_QS_WORK",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq-mtp-quality-speed",
    )
)
MODELS = Path(os.environ.get("QWEN38_MODELS", "/Volumes/Ext4T/models"))
BF16 = Path(os.environ.get("QWEN38_BF16", "/Volumes/Ext4T/models/Qwen3.8-27B-bf16"))
DATASETS = Path(
    os.environ.get(
        "QWEN38_CERT_DATASETS",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq8-tier1/datasets",
    )
)
SOURCE_ID = "Qwen/Qwen3.8-27B"
SOURCE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)

PACKS: dict[str, dict[str, str]] = {
    "mxfp4": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
        "label": "MXFP4",
        "hub": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
    },
    "axq4": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "label": "4-bit",
        "hub": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
    },
    "axq6": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        "label": "6-bit",
        "hub": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
    },
    "axq8": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP",
        "label": "8-bit",
        "hub": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    studio = Path("/Users/devop/code/axquant-main/.venv/bin/python")
    if studio.is_file():
        return [str(studio), "-m", "axquant"]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found")


def run(cmd: list[str], log_path: Path) -> None:
    log("$ " + " ".join(cmd))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "PATH": f"/Users/devop/opt/ax-engine-6.16.1/bin:{os.environ.get('PATH', '')}",
    }
    env.pop("AXQUANT_FORCE_CPU", None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): see {log_path}")


def _retention(cmp: dict) -> float | None:
    agg = cmp.get("aggregate") or {}
    ret = agg.get("retention")
    if ret is None:
        ret = cmp.get("retention")
    return None if ret is None else float(ret)


def cmd_quality() -> None:
    qdir = WORK / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    # Prefer a completed BF16 ref from the MXFP4-MTP T1 run.
    reused_root = Path(
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq-mxfp4-mtp-tier1/quality/axq-mxfp4"
    )
    for suite, dname in (
        ("agent-coding", "development-agent-coding"),
        ("general", "development-general"),
    ):
        ds = DATASETS / dname / "dataset.jsonl"
        if not ds.is_file():
            raise SystemExit(f"missing dataset {ds}")
        ref = qdir / f"ref-{suite}.json"
        reused = reused_root / f"ref-{suite}.json"
        if not ref.is_file() and reused.is_file():
            shutil.copy2(reused, ref)
            log(f"reuse BF16 quality {reused}")
        if not ref.is_file():
            run(
                [
                    *axquant_cmd(),
                    "evaluate-quality",
                    "--model",
                    str(BF16),
                    "--model-id",
                    SOURCE_ID,
                    "--revision",
                    SOURCE_REV,
                    "--dataset",
                    str(ds),
                    "--seed",
                    str(SEED),
                    "--max-tokens",
                    str(MAX_TOKENS_QUALITY),
                    "--output",
                    str(ref),
                ],
                WORK / "logs" / f"quality-ref-{suite}.log",
            )
        for key, meta in PACKS.items():
            pack = MODELS / meta["name"]
            if not (pack / "axquant_manifest.json").is_file():
                raise SystemExit(f"missing pack {pack}")
            cand = qdir / f"{key}-{suite}.json"
            if cand.is_file():
                log(f"reuse {cand}")
            else:
                run(
                    [
                        *axquant_cmd(),
                        "evaluate-quality",
                        "--model",
                        str(pack),
                        "--model-id",
                        meta["hub"],
                        "--revision",
                        SOURCE_REV,
                        "--dataset",
                        str(ds),
                        "--seed",
                        str(SEED),
                        "--max-tokens",
                        str(MAX_TOKENS_QUALITY),
                        "--output",
                        str(cand),
                    ],
                    WORK / "logs" / f"quality-{key}-{suite}.log",
                )
            cmp_out = qdir / f"compare-{key}-{suite}.json"
            run(
                [
                    *axquant_cmd(),
                    "compare-quality",
                    "--reference",
                    str(ref),
                    "--candidate",
                    str(cand),
                    "--output",
                    str(cmp_out),
                ],
                WORK / "logs" / f"compare-{key}-{suite}.log",
            )
            cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
            log(f"quality {key} {suite}: retention={_retention(cmp)}")


def _subset(source: Path, dest: Path, count: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                kept.append(line if line.endswith("\n") else line + "\n")
            if len(kept) >= count:
                break
    if not kept:
        raise SystemExit(f"no prompts in {source}")
    dest.write_text("".join(kept), encoding="utf-8")
    return dest


def _speed_config(
    *,
    model_dir: Path,
    hub: str,
    workload: str,
    dataset_sha256: str,
    prompt_count: int,
    mtp_enabled: bool,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        model=ModelIdentity(
            model_id=hub,
            revision="local",
            format="mlx",
            local_path=str(model_dir),
        ),
        mtp_enabled=mtp_enabled,
        baseline_kind="axquant-mtp-on" if mtp_enabled else "axquant-mtp-off",
        workload=workload,
        dataset_sha256=dataset_sha256,
        prompt_count=prompt_count,
        warmup_trials=1,
        measured_trials=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=64,
        ignore_eos=True,
        draft_depth=1,
        random_seed=SEED,
        timeout_seconds=600.0,
        runtime_env=dict(QWEN38_EXACT_MTP_PROFILE_ENV),
    )


def cmd_speed() -> None:
    sdir = WORK / "speed"
    sdir.mkdir(parents=True, exist_ok=True)
    executable = Path(ENGINE_BENCH)
    if not executable.is_file():
        raise SystemExit(f"missing ax-engine-bench {executable}")
    os.environ["PATH"] = f"{executable.parent}:{os.environ.get('PATH', '')}"
    agent_src = DATASETS / "development-agent-coding" / "dataset.jsonl"
    gen_src = DATASETS / "development-general" / "dataset.jsonl"
    if not agent_src.is_file():
        raise SystemExit(f"missing {agent_src}")
    summaries: dict[str, object] = {}
    for key, meta in PACKS.items():
        pack = MODELS / meta["name"]
        if not (pack / "mtp.safetensors").is_file():
            raise SystemExit(f"pack missing mtp.safetensors: {pack}")
        subprocess.run(
            [str(executable), "generate-manifest", "--validate", "--", str(pack)],
            check=False,
            cwd=str(ROOT),
        )
        out = sdir / key
        out.mkdir(parents=True, exist_ok=True)
        agent_ds = _subset(agent_src, out / "datasets" / "agent-coding.jsonl", 16)
        gen_ds = _subset(gen_src, out / "datasets" / "general.jsonl", 12)
        pack_row: dict[str, object] = {"label": meta["label"], "path": str(pack)}
        for workload, ds, count in (
            ("agent-coding", agent_ds, 16),
            ("general", gen_ds, 12),
        ):
            ds_sha = file_sha256(ds)
            off_dir = out / workload / "mtp-off"
            on_dir = out / workload / "mtp-on"
            off_dir.mkdir(parents=True, exist_ok=True)
            on_dir.mkdir(parents=True, exist_ok=True)
            log(f"=== speed {key} {workload} mtp-off ===")
            off = run_benchmark(
                _speed_config(
                    model_dir=pack,
                    hub=meta["hub"],
                    workload=workload,
                    dataset_sha256=ds_sha,
                    prompt_count=count,
                    mtp_enabled=False,
                ),
                dataset_path=ds,
                executable=str(executable),
                output_dir=off_dir,
            )
            log(f"=== speed {key} {workload} mtp-on ===")
            on = run_benchmark(
                _speed_config(
                    model_dir=pack,
                    hub=meta["hub"],
                    workload=workload,
                    dataset_sha256=ds_sha,
                    prompt_count=count,
                    mtp_enabled=True,
                ),
                dataset_path=ds,
                executable=str(executable),
                output_dir=on_dir,
            )
            comparison = compare_mtp_ab_results(
                off,
                on,
                profile_name="quality-speed-ab",
                minimum_speedup=1.20,
                speedup_metric="token-weighted-decode-tps",
                minimum_prompt_median_speedup=1.10,
            )
            cmp_path = out / workload / "mtp_ab_comparison.json"
            write_data(cmp_path, comparison)
            cmp = (
                comparison.model_dump(mode="json")
                if hasattr(comparison, "model_dump")
                else json.loads(json.dumps(comparison, default=str))
            )
            metrics = cmp.get("metrics") if isinstance(cmp.get("metrics"), dict) else {}
            weighted = float(
                cmp.get("token_weighted_decode_speedup")
                or metrics.get("token_weighted_decode_speedup")
                or 0.0
            )
            prompt_med = float(
                cmp.get("prompt_median_speedup") or metrics.get("prompt_median_speedup") or 0.0
            )
            off_tps = float(cmp.get("direct_token_weighted_decode_tps") or 0.0)
            on_tps = float(cmp.get("mtp_token_weighted_decode_tps") or 0.0)
            pack_row[workload] = {
                "exactness_pass": bool(cmp.get("exactness_pass")),
                "divergent_trial_count": cmp.get("divergent_trial_count", 0),
                "mtp_off_decode_tps": off_tps,
                "mtp_on_decode_tps": on_tps,
                "token_weighted_decode_speedup": weighted,
                "prompt_median_speedup": prompt_med,
                "comparison": str(cmp_path),
            }
            log(
                f"speed {key} {workload}: exact={cmp.get('exactness_pass')} "
                f"off={off_tps:.2f} on={on_tps:.2f} speedup={weighted:.3f}"
            )
        summaries[key] = pack_row
    write_data(sdir / "summary.json", summaries)


def cmd_summarize() -> None:
    qdir = WORK / "quality"
    sdir = WORK / "speed"
    rows: list[dict[str, object]] = []
    for key, meta in PACKS.items():
        pack = MODELS / meta["name"]
        man = {}
        if (pack / "axquant_manifest.json").is_file():
            man = json.loads((pack / "axquant_manifest.json").read_text(encoding="utf-8"))
        quality: dict[str, object] = {}
        for suite in ("agent-coding", "general"):
            cmp_path = qdir / f"compare-{key}-{suite}.json"
            if cmp_path.is_file():
                cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
                agg = cmp.get("aggregate") or {}
                quality[suite] = {
                    "retention": _retention(cmp),
                    "candidate": agg.get("candidate"),
                    "reference": agg.get("reference"),
                }
        speed = {}
        if (sdir / "summary.json").is_file():
            speed = json.loads((sdir / "summary.json").read_text(encoding="utf-8")).get(key, {})
        rows.append(
            {
                "key": key,
                "label": meta["label"],
                "hub": meta["hub"],
                "path": str(pack),
                "measured_main_bpw": man.get("measured_main_bpw"),
                "measured_total_bpw": man.get("measured_total_bpw"),
                "weight_file_size_bytes": man.get("weight_file_size_bytes"),
                "quality": quality,
                "speed": speed,
            }
        )
    payload = {
        "schema_version": "axquant.qwen38-axq-mtp-quality-speed.v1",
        "status": "measured-comparison",
        "not_a_certification": True,
        "created_at": datetime.now(UTC).isoformat(),
        "host_id": HOST_ID,
        "runtime": "ax-engine 6.16.1",
        "mtp_profile": "QWEN38_EXACT_MTP_PROFILE_ENV",
        "source": f"{SOURCE_ID}@{SOURCE_REV}",
        "seed": SEED,
        "quality_max_tokens": MAX_TOKENS_QUALITY,
        "packs": rows,
    }
    out = WORK / "summary.json"
    write_data(out, payload)
    repo_raw = ROOT / "docs" / "eval" / "qwen38-27b-axq-mtp-quality-speed-macstudio-m2"
    repo_raw.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, repo_raw / "summary.json")
    log(f"wrote {out} and {repo_raw / 'summary.json'}")


def cmd_all() -> None:
    cmd_quality()
    cmd_speed()
    cmd_summarize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["quality", "speed", "summarize", "all"])
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {
        "quality": cmd_quality,
        "speed": cmd_speed,
        "summarize": cmd_summarize,
        "all": cmd_all,
    }[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
