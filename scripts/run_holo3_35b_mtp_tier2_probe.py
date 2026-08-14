#!/usr/bin/env python3
"""Formal-style MTP Tier 2 probe for Holo3-35B-A3B AXQ *-MTP (grafted parent MTP).

Uses the Qwen 3.6 **MoE exact** runtime contract (same env family as certified
Qwen 35B-A3B Tier 2) but binds Holo3 Hub identity and records accept/speedup
gates honestly.

Default is a **medium probe** (subset prompts, fewer trials) so operators can
decide whether a full authorizing scoreboard is worth running. Pass
``--full`` for formal-like size (still on the host you choose; formal Qwen
certs used ``df-macbookpro-m5``).

Usage (factory)::

    export PATH=/Users/devop/code/ax-engine-v6150-bin:$PATH
    export PYTHONPATH=/Users/devop/code/axquant/src  # tree with grafted tooling
    python scripts/run_holo3_35b_mtp_tier2_probe.py \\
      --model-dir /Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP \\
      --output-root /Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/tier2-probe-6bit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from axquant.benchmark import (
    QWEN36_MOE_EXACT_MTP_PROFILE_ENV,
    compare_mtp_ab_results,
    run_benchmark,
)
from axquant.schema import BenchmarkConfig, ModelIdentity
from axquant.serde import file_sha256, write_data

HUB_REPO_6 = "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP"
HUB_COMMIT_6 = "f474549461817cafb73909847af43af2431d4a0d"
HUB_REPO_4 = "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP"
HUB_COMMIT_4 = "c048f577843225ac0545be5674b4d68b9a51dcf0"
SEED = 20260728
WEIGHTED_MIN = 1.20
PROMPT_MEDIAN_MIN = 1.10

DEFAULT_FORMAL_AGENT = Path(
    "/Volumes/Ext4T/axquant/flagship/qwen36-mtp-v2-c1/datasets/"
    "formal-agent-coding/dataset.jsonl"
)
DEFAULT_FORMAL_GENERAL = Path(
    "/Volumes/Ext4T/axquant/flagship/qwen36-mtp-v2-c1/datasets/"
    "formal-general/dataset.jsonl"
)


def _engine_binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _subset_dataset(source: Path, dest: Path, count: int, seed: int) -> Path:
    """Write first ``count`` non-empty JSONL records (stable order)."""
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
    # seed reserved for future shuffle parity with formal harness
    _ = seed
    return dest


def _base_config(
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
        draft_depth=1,
        random_seed=SEED,
        timeout_seconds=timeout_seconds,
        runtime_env=dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
    )


def _phase_accept_summary(comparison: object) -> dict:
    """Best-effort accept stats from comparison / nested phase timing."""
    payload = comparison.model_dump() if hasattr(comparison, "model_dump") else {}
    phase = payload.get("phase_timing") or {}
    if not phase and isinstance(payload.get("profiles"), list):
        return {}
    proposed = phase.get("proposed_tokens")
    accepted = phase.get("accepted_tokens")
    rate = None
    if isinstance(proposed, int) and proposed > 0 and isinstance(accepted, int):
        rate = accepted / proposed
    return {
        "accepted_tokens": accepted,
        "proposed_tokens": proposed,
        "accept_rate": rate,
        "direct_fallback_steps": phase.get("direct_fallback_steps"),
        "mtp_decode_steps": phase.get("mtp_decode_steps"),
    }


def run_profile(
    *,
    name: str,
    dataset: Path,
    model_dir: Path,
    hub_repo: str,
    hub_commit: str,
    output_dir: Path,
    executable: str,
    prompt_count: int,
    max_tokens: int,
    timeout_seconds: float,
    warmup: int,
    measured: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_sha = file_sha256(dataset)
    n_lines = sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip())
    use_count = min(prompt_count, n_lines)

    direct_cfg = _base_config(
        model_dir=model_dir,
        hub_repo=hub_repo,
        hub_commit=hub_commit,
        workload=name,
        dataset_sha256=ds_sha,
        prompt_count=use_count,
        mtp_enabled=False,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        warmup=warmup,
        measured=measured,
    )
    mtp_cfg = _base_config(
        model_dir=model_dir,
        hub_repo=hub_repo,
        hub_commit=hub_commit,
        workload=name,
        dataset_sha256=ds_sha,
        prompt_count=use_count,
        mtp_enabled=True,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        warmup=warmup,
        measured=measured,
    )

    print(f"==== {name} MTP-off start {datetime.now(timezone.utc).isoformat()}", flush=True)
    direct = run_benchmark(
        direct_cfg,
        dataset_path=dataset,
        executable=executable,
        output_dir=output_dir / "mtp-off",
    )
    print(
        f"==== {name} MTP-off done measured={direct.measured_count} failed={direct.failed_count}",
        flush=True,
    )

    print(f"==== {name} MTP-on start {datetime.now(timezone.utc).isoformat()}", flush=True)
    mtp = run_benchmark(
        mtp_cfg,
        dataset_path=dataset,
        executable=executable,
        output_dir=output_dir / "mtp-on",
    )
    print(
        f"==== {name} MTP-on done measured={mtp.measured_count} failed={mtp.failed_count}",
        flush=True,
    )

    comparison = compare_mtp_ab_results(
        direct,
        mtp,
        profile_name="benchmark-ab",
        minimum_speedup=WEIGHTED_MIN,
        speedup_metric="token-weighted-decode-tps",
        minimum_prompt_median_speedup=PROMPT_MEDIAN_MIN,
    )
    write_data(output_dir / "mtp_ab_comparison.json", comparison)

    accept = _phase_accept_summary(comparison)
    # Also scrape raw MTP logs if present for accept counts.
    raw_on = output_dir / "mtp-on" / "benchmark_raw_log.json"
    raw_accept: dict = {}
    if raw_on.is_file():
        try:
            raw = json.loads(raw_on.read_text(encoding="utf-8"))
            # best-effort aggregate
            totals = {"accepted": 0, "proposed": 0, "fallback": 0}
            for trial in raw if isinstance(raw, list) else raw.get("trials", []) or []:
                if not isinstance(trial, dict):
                    continue
                pt = trial.get("phase_timing") or trial.get("mtp_phase_timing") or {}
                totals["accepted"] += int(pt.get("accepted_tokens") or 0)
                totals["proposed"] += int(pt.get("proposed_tokens") or 0)
                totals["fallback"] += int(pt.get("direct_fallback_steps") or 0)
            if totals["proposed"] or totals["accepted"]:
                raw_accept = {
                    **totals,
                    "accept_rate": (
                        totals["accepted"] / totals["proposed"]
                        if totals["proposed"]
                        else None
                    ),
                }
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            raw_accept = {}

    summary = {
        "profile": name,
        "dataset_path": str(dataset),
        "dataset_sha256": ds_sha,
        "prompt_count": use_count,
        "hub_repo": hub_repo,
        "hub_commit": hub_commit,
        "exactness_pass": comparison.exactness_pass,
        "divergent_trial_count": comparison.divergent_trial_count,
        "token_weighted_decode_speedup": comparison.token_weighted_decode_speedup,
        "prompt_median_speedup": comparison.prompt_median_speedup,
        "speedup_pass": comparison.speedup_pass,
        "prompt_median_speedup_pass": comparison.prompt_median_speedup_pass,
        "release_ready": comparison.release_ready,
        "issues": list(comparison.issues),
        "comparison_sha256": file_sha256(output_dir / "mtp_ab_comparison.json"),
        "direct_token_weighted_decode_tps": comparison.direct_token_weighted_decode_tps,
        "mtp_token_weighted_decode_tps": comparison.mtp_token_weighted_decode_tps,
        "phase_accept": accept,
        "raw_log_accept": raw_accept,
        "runtime_env": dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
        "gates": {
            "exactness_required": 1.0,
            "token_weighted_decode_speedup_min": WEIGHTED_MIN,
            "prompt_median_speedup_min": PROMPT_MEDIAN_MIN,
        },
    }
    write_data(output_dir / "profile_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/"
            "AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP"
        ),
    )
    parser.add_argument(
        "--bits",
        choices=("4", "6"),
        default="6",
        help="Select Hub identity binding for 4-bit or 6-bit MTP pack",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/tier2-probe-6bit"),
    )
    parser.add_argument(
        "--executable",
        default="/Users/devop/code/ax-engine-v6150-bin/ax-engine-bench",
    )
    parser.add_argument("--agent-coding-dataset", type=Path, default=DEFAULT_FORMAL_AGENT)
    parser.add_argument("--general-dataset", type=Path, default=DEFAULT_FORMAL_GENERAL)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Formal-like size: 5 measured trials, max 512 tokens, more prompts",
    )
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--measured", type=int, default=0)
    parser.add_argument("--prompt-count", type=int, default=0)
    parser.add_argument(
        "--profiles",
        default="agent-coding",
        help="Comma-separated: agent-coding,general-long",
    )
    args = parser.parse_args(argv)

    if args.bits == "6":
        hub_repo, hub_commit = HUB_REPO_6, HUB_COMMIT_6
    else:
        hub_repo, hub_commit = HUB_REPO_4, HUB_COMMIT_4

    if args.full:
        max_tokens = args.max_tokens or 512
        warmup = args.warmup or 2
        measured = args.measured or 5
        prompt_count = args.prompt_count or 20
    else:
        max_tokens = args.max_tokens or 128
        warmup = args.warmup or 1
        measured = args.measured or 2
        prompt_count = args.prompt_count or 8

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"model dir missing: {model_dir}", file=sys.stderr)
        return 2
    if not (model_dir / "mtp.safetensors").is_file():
        print(f"missing mtp.safetensors under {model_dir}", file=sys.stderr)
        return 2

    executable = args.executable
    exe_path = Path(shutil.which(executable) or executable)
    if not exe_path.is_file():
        print(f"ax-engine-bench not found: {executable}", file=sys.stderr)
        return 2

    out_root = args.output_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ds_dir = out_root / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)

    host = subprocess.check_output(["hostname"], text=True).strip()
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model_dir": str(model_dir),
        "hub_repo": hub_repo,
        "hub_commit": hub_commit,
        "full": bool(args.full),
        "max_tokens": max_tokens,
        "warmup": warmup,
        "measured": measured,
        "prompt_count": prompt_count,
        "executable": str(exe_path),
        "engine_binary_sha256": _engine_binary_sha256(exe_path),
        "runtime_env": dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
        "seed": SEED,
        "note": (
            "Holo3 MTP is grafted from Qwen3.5 parent; this probe uses MoE exact "
            "env to test whether formal contract can recover accept/speedup."
        ),
    }
    write_data(out_root / "probe_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)

    profile_map = {
        "agent-coding": args.agent_coding_dataset,
        "general-long": args.general_dataset,
    }
    summaries: list[dict] = []
    for name in [p.strip() for p in args.profiles.split(",") if p.strip()]:
        if name not in profile_map:
            print(f"unknown profile {name}", file=sys.stderr)
            return 2
        src = profile_map[name]
        if not src.is_file():
            print(f"dataset missing: {src}", file=sys.stderr)
            return 2
        subset = _subset_dataset(src, ds_dir / f"{name}.jsonl", prompt_count, SEED)
        summary = run_profile(
            name=name,
            dataset=subset,
            model_dir=model_dir,
            hub_repo=hub_repo,
            hub_commit=hub_commit,
            output_dir=out_root / name,
            executable=str(exe_path),
            prompt_count=prompt_count,
            max_tokens=max_tokens,
            timeout_seconds=args.timeout_seconds,
            warmup=warmup,
            measured=measured,
        )
        summaries.append(summary)

    any_release = any(bool(s.get("release_ready")) for s in summaries)
    any_exact = any(bool(s.get("exactness_pass")) for s in summaries)
    all_exact = all(bool(s.get("exactness_pass")) for s in summaries) if summaries else False
    decision = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "any_release_ready": any_release,
        "all_exactness_pass": all_exact,
        "any_exactness_pass": any_exact,
        "profiles": summaries,
        "recommendation": (
            "candidate_for_full_tier2_scoreboard"
            if any_release
            else (
                "exactness_ok_speedup_fail_graft_limit_likely"
                if all_exact
                else "exactness_and_speedup_fail_no_tier2"
            )
        ),
    }
    write_data(out_root / "probe_decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)
    return 0 if any_release else 1


if __name__ == "__main__":
    raise SystemExit(main())
