#!/usr/bin/env python3
"""MTP Tier 2 A/B for Qwen3.8-27B AXQ *-MTP (dense hybrid, model_type=qwen3_5).

Uses the same dense exact-MTP runtime contract as certified Qwen 3.6 27B
(``QWEN38_EXACT_MTP_PROFILE_ENV``): AX Engine treats Qwen3.5/3.6/3.8 dense
linear-attention packs under the shared ``qwen3_5`` family.

Usage (factory host with Ext4T + ax-engine-bench)::

  export PATH="/opt/homebrew/bin:$PATH"
  .venv/bin/python scripts/run_qwen38_27b_tier2_mtp_ab.py --pack axq6
  .venv/bin/python scripts/run_qwen38_27b_tier2_mtp_ab.py --pack axq4
  .venv/bin/python scripts/run_qwen38_27b_tier2_mtp_ab.py --pack both --full
"""

from __future__ import annotations

import argparse
import hashlib
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

SEED = 20260728
WEIGHTED_MIN = 1.20
PROMPT_MEDIAN_MIN = 1.10
HOST_ID = os.environ.get("QWEN38_CERT_HOST", "df-macbookpro-m3")

PACKS: dict[str, dict[str, object]] = {
    "axq4": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "hub_repo": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "product_class": "4bit",
        "tier1_cert": "qwen38-27b-axq4-mtp-tier1",
        "tier2_cert": "qwen38-27b-axq4-mtp-tier2",
        "sort_order": 2,
    },
    "axq6": {
        "name": "AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        "hub_repo": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        "product_class": "6bit",
        "tier1_cert": "qwen38-27b-axq6-mtp-tier1",
        "tier2_cert": "qwen38-27b-axq6-mtp-tier2",
        "sort_order": 4,
    },
}

DEFAULT_MODELS = Path(os.environ.get("QWEN38_MODELS", "/Volumes/Ext4T/models"))
DEFAULT_DATASETS = Path(
    os.environ.get(
        "QWEN38_TIER2_DATASETS",
        "/Volumes/Ext4T/axquant/flagship/qwen36-mtp-v2-c1/datasets",
    )
)
DEFAULT_WORK = Path(
    os.environ.get(
        "QWEN38_TIER2_WORK",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq-tier2",
    )
)


def log(msg: str) -> None:
    print(msg, flush=True)


def engine_binary() -> Path:
    which = shutil.which("ax-engine-bench")
    if which:
        return Path(which)
    raise SystemExit("ax-engine-bench not found on PATH")


def engine_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def subset_dataset(source: Path, dest: Path, count: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                kept.append(line if line.endswith("\n") else line + "\n")
            if len(kept) >= count:
                break
    if not kept:
        raise SystemExit(f"no prompts in {source}")
    dest.write_text("".join(kept), encoding="utf-8")
    return dest


def hub_commit_for(local: Path, fallback: str) -> str:
    # Prefer axquant_runtime / published cert tip; else fallback.
    cert = ROOT / "docs" / "certifications"
    for stem in PACKS.values():
        if stem["name"] == local.name:
            t1 = cert / f"{stem['tier1_cert']}.json"
            if t1.is_file():
                data = json.loads(t1.read_text(encoding="utf-8"))
                return str(data["artifact"]["hub_commit"])
    return fallback


def base_config(
    *,
    model_dir: Path,
    hub_repo: str,
    hub_commit: str,
    workload: str,
    dataset_sha256: str,
    prompt_count: int,
    mtp_enabled: bool,
    max_tokens: int,
    timeout_seconds: float,
    warmup: int,
    measured: int,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        model=ModelIdentity(
            model_id=hub_repo,
            revision=hub_commit,
            format="mlx",
            local_path=str(model_dir),
        ),
        mtp_enabled=mtp_enabled,
        baseline_kind="axquant-mtp-on" if mtp_enabled else "axquant-mtp-off",
        workload=workload,
        dataset_sha256=dataset_sha256,
        prompt_count=prompt_count,
        warmup_trials=warmup,
        measured_trials=measured,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=max_tokens,
        # Fixed-token budgets for fair decode-heavy A/B (short general prompts
        # otherwise stop after a handful of tokens and understate MTP speedup).
        ignore_eos=True,
        draft_depth=1,
        random_seed=SEED,
        timeout_seconds=timeout_seconds,
        runtime_env=dict(QWEN38_EXACT_MTP_PROFILE_ENV),
    )


def run_pack(
    *,
    pack_key: str,
    models_root: Path,
    datasets: Path,
    work: Path,
    full: bool,
    executable: Path,
) -> dict[str, object]:
    meta = PACKS[pack_key]
    name = str(meta["name"])
    model_dir = models_root / name
    if not (model_dir / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {model_dir}")
    if not (model_dir / "mtp.safetensors").is_file():
        raise SystemExit(f"pack missing mtp.safetensors: {model_dir}")

    # Ensure native manifest is validated
    subprocess.run(
        ["ax-engine-bench", "generate-manifest", "--validate", "--", str(model_dir)],
        check=False,
        cwd=str(ROOT),
    )

    out = work / pack_key
    out.mkdir(parents=True, exist_ok=True)
    hub_repo = str(meta["hub_repo"])
    # Prefer cert-bound commit when present
    t1_path = ROOT / "docs" / "certifications" / f"{meta['tier1_cert']}.json"
    hub_commit = "main"
    if t1_path.is_file():
        hub_commit = json.loads(t1_path.read_text(encoding="utf-8"))["artifact"]["hub_commit"]

    agent_src = datasets / "formal-agent-coding" / "dataset.jsonl"
    gen_src = datasets / "formal-general" / "dataset.jsonl"
    if not agent_src.is_file() or not gen_src.is_file():
        # Fall back to development suites if formal not mounted
        agent_src = datasets / "development-agent-coding" / "dataset.jsonl"
        gen_src = datasets / "development-general" / "dataset.jsonl"
        if not agent_src.is_file():
            agent_src = Path(
                "/Volumes/Ext4T/axquant-certification/qwen36-27b-axq6-v1/datasets/"
                "development-agent-coding/dataset.jsonl"
            )
            gen_src = Path(
                "/Volumes/Ext4T/axquant-certification/qwen36-27b-axq6-v1/datasets/"
                "development-general/dataset.jsonl"
            )

    if full:
        agent_n, gen_n = 24, 16
        max_tokens, warmup, measured = 64, 1, 2
        timeout = 600.0
    else:
        agent_n, gen_n = 8, 6
        max_tokens, warmup, measured = 48, 0, 1
        timeout = 300.0

    agent_ds = subset_dataset(agent_src, out / "datasets" / "agent-coding.jsonl", agent_n)
    gen_ds = subset_dataset(gen_src, out / "datasets" / "general-long.jsonl", gen_n)

    profiles: dict[str, object] = {}
    all_pass = True
    for workload, ds, count in (
        ("agent-coding", agent_ds, agent_n),
        ("general-long", gen_ds, gen_n),
    ):
        ds_sha = file_sha256(ds)
        log(f"=== {pack_key} {workload} mtp-off ===")
        off_cfg = base_config(
            model_dir=model_dir,
            hub_repo=hub_repo,
            hub_commit=hub_commit,
            workload=workload,
            dataset_sha256=ds_sha,
            prompt_count=count,
            mtp_enabled=False,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            warmup=warmup,
            measured=measured,
        )
        off_dir = out / workload / "mtp-off"
        off_dir.mkdir(parents=True, exist_ok=True)
        off_result = run_benchmark(
            off_cfg,
            dataset_path=ds,
            executable=str(executable),
            output_dir=off_dir,
        )
        log(f"=== {pack_key} {workload} mtp-on ===")
        on_cfg = base_config(
            model_dir=model_dir,
            hub_repo=hub_repo,
            hub_commit=hub_commit,
            workload=workload,
            dataset_sha256=ds_sha,
            prompt_count=count,
            mtp_enabled=True,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            warmup=warmup,
            measured=measured,
        )
        on_dir = out / workload / "mtp-on"
        on_dir.mkdir(parents=True, exist_ok=True)
        on_result = run_benchmark(
            on_cfg,
            dataset_path=ds,
            executable=str(executable),
            output_dir=on_dir,
        )
        comparison = compare_mtp_ab_results(
            off_result,
            on_result,
            profile_name="benchmark-ab",
            minimum_speedup=WEIGHTED_MIN,
            speedup_metric="token-weighted-decode-tps",
            minimum_prompt_median_speedup=PROMPT_MEDIAN_MIN,
        )
        cmp_path = out / workload / "mtp_ab_comparison.json"
        write_data(cmp_path, comparison)
        # comparison may be a model or dict
        if hasattr(comparison, "model_dump"):
            cmp = comparison.model_dump(mode="json")
        elif hasattr(comparison, "dict"):
            cmp = comparison.dict()
        else:
            cmp = (
                comparison
                if isinstance(comparison, dict)
                else json.loads(json.dumps(comparison, default=str))
            )

        exact = bool(cmp.get("exactness_pass"))
        weighted = float(cmp.get("token_weighted_decode_speedup") or cmp.get("speedup") or 0.0)
        prompt_med = float(cmp.get("prompt_median_speedup") or 0.0)
        release = bool(cmp.get("release_ready"))
        # Some schemas nest metrics
        if weighted == 0.0 and isinstance(cmp.get("metrics"), dict):
            weighted = float(cmp["metrics"].get("token_weighted_decode_speedup") or 0.0)
            prompt_med = float(cmp["metrics"].get("prompt_median_speedup") or prompt_med)

        profile_pass = exact and weighted >= WEIGHTED_MIN and prompt_med >= PROMPT_MEDIAN_MIN
        all_pass = all_pass and profile_pass and release
        profiles[workload] = {
            "exactness_pass": exact,
            "divergent_trial_count": cmp.get("divergent_trial_count", 0),
            "token_weighted_decode_speedup": weighted,
            "prompt_median_speedup": prompt_med,
            "release_ready": release,
            "dataset_sha256": ds_sha,
            "comparison_sha256": file_sha256(cmp_path),
            "gate_pass": profile_pass,
        }
        log(
            f"{pack_key} {workload}: exact={exact} weighted={weighted:.4f} "
            f"prompt_med={prompt_med:.4f} release_ready={release} pass={profile_pass}"
        )

    summary = {
        "schema_version": "axquant.qwen38-dense-mtp-tier2-probe.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "host_id": HOST_ID,
        "pack_key": pack_key,
        "hub_repo_id": hub_repo,
        "hub_commit": hub_commit,
        "local_path": str(model_dir),
        "product_class": meta["product_class"],
        "full": full,
        "thresholds": {
            "exactness_required": 1.0,
            "token_weighted_decode_speedup_min": WEIGHTED_MIN,
            "prompt_median_speedup_min": PROMPT_MEDIAN_MIN,
        },
        "runtime_env": dict(QWEN38_EXACT_MTP_PROFILE_ENV),
        "formal_route": (
            "AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE=1 with qwen36 dense exact profile"
        ),
        "ax_engine_executable": str(executable),
        "engine_binary_sha256": engine_sha(executable),
        "profiles": profiles,
        "technical_tier2_pass": all_pass
        and all(
            bool(p.get("gate_pass"))
            for p in profiles.values()  # type: ignore[union-attr]
        ),
    }
    write_data(out / "TIER2_TECHNICAL_SUMMARY.json", summary)
    # also copy profile comparisons to flat names for evidence package
    for workload in ("agent-coding", "general-long"):
        src = out / workload / "mtp_ab_comparison.json"
        if src.is_file():
            shutil.copy2(src, out / f"{workload}-mtp_ab_comparison.json")
    log(f"{pack_key} technical_tier2_pass={summary['technical_tier2_pass']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=["axq4", "axq6", "both"], default="both")
    parser.add_argument("--full", action="store_true", help="larger formal-like probe")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()

    executable = engine_binary()
    log(f"engine={executable} host={HOST_ID} full={args.full}")
    keys = ["axq4", "axq6"] if args.pack == "both" else [args.pack]
    results = []
    for key in keys:
        results.append(
            run_pack(
                pack_key=key,
                models_root=args.models_root,
                datasets=args.datasets,
                work=args.work,
                full=args.full,
                executable=executable,
            )
        )
    write_data(args.work / "all_packs_summary.json", {"results": results})
    ok = all(bool(r.get("technical_tier2_pass")) for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
