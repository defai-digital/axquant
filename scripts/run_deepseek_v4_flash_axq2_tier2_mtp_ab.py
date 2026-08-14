#!/usr/bin/env python3
"""Formal MTP Tier 2 A/B for DeepSeek V4 Flash AXQ 2-bit on df-macstudio-m2.

Runs the same three gates as Qwen flagship Tier 2:
  - greedy exactness 100%
  - token-weighted decode speedup >= 1.20x
  - prompt-median speedup >= 1.10x

DeepSeek uses nextn MTP (not Qwen linear MTP). Profile pins generic exactness
controls only; Qwen/Gemma-specific flags are omitted.

Usage (on formal host)::

    export PATH=/Users/devop/code/ax-engine-v6150-bin:$PATH
    export AX_ENGINE_2BIT_EXPERIMENTAL=1
    export AX_ENGINE_3BIT_EXPERIMENTAL=1
    export PYTHONPATH=/Users/devop/code/axquant-tier2-src/src:$PYTHONPATH
    export DYLD_FALLBACK_LIBRARY_PATH=.../mlx/lib
    python scripts/run_deepseek_v4_flash_axq2_tier2_mtp_ab.py \\
      --model-dir /Volumes/Ext4T/models/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit \\
      --output-root /Volumes/Ext4T/axquant-certification/.../tier2-mtp-2bit \\
      --executable /Users/devop/code/ax-engine-v6150-bin/ax-engine-bench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from axquant.benchmark import (
    DEEPSEEK_V4_EXACT_MTP_PROFILE_ENV,
    compare_mtp_ab_results,
    run_benchmark,
)
from axquant.errors import BenchmarkError, InvariantViolationError
from axquant.schema import BenchmarkConfig, ModelIdentity
from axquant.serde import file_sha256, write_data

HUB_REPO = "AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit"
HUB_COMMIT = "e22b117aa812b29943b160bb0fbf0b962d0d3819"
SEED = 20260728
WEIGHTED_MIN = 1.20
PROMPT_MEDIAN_MIN = 1.10


def _sha256_file(path: Path) -> str:
    return file_sha256(path)


def _engine_binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_config(
    *,
    model_dir: Path,
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
            model_id=HUB_REPO,
            revision=HUB_COMMIT,
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
        runtime_env=dict(DEEPSEEK_V4_EXACT_MTP_PROFILE_ENV),
    )


def run_profile(
    *,
    name: str,
    dataset: Path,
    model_dir: Path,
    output_dir: Path,
    executable: str,
    prompt_count: int,
    max_tokens: int,
    timeout_seconds: float,
    warmup: int,
    measured: int,
    enforce_exactness: bool,
    enforce_speedup: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_sha = _sha256_file(dataset)
    direct_cfg = _base_config(
        model_dir=model_dir,
        workload=name,
        dataset_sha256=ds_sha,
        prompt_count=prompt_count,
        mtp_enabled=False,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        warmup=warmup,
        measured=measured,
    )
    mtp_cfg = _base_config(
        model_dir=model_dir,
        workload=name,
        dataset_sha256=ds_sha,
        prompt_count=prompt_count,
        mtp_enabled=True,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        warmup=warmup,
        measured=measured,
    )

    print(f"==== {name} MTP-off start {datetime.now(UTC).isoformat()}", flush=True)
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

    print(f"==== {name} MTP-on start {datetime.now(UTC).isoformat()}", flush=True)
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

    issues = list(comparison.issues)
    if direct.failed_count or mtp.failed_count:
        issues.append(f"failed trials: direct={direct.failed_count} mtp={mtp.failed_count}")
    if enforce_exactness and not comparison.exactness_pass:
        issues.append("exactness gate failed")
    if enforce_speedup and not comparison.speedup_pass:
        issues.append("speedup gate failed")

    summary = {
        "profile": name,
        "dataset_path": str(dataset),
        "dataset_sha256": ds_sha,
        "exactness_pass": comparison.exactness_pass,
        "divergent_trial_count": comparison.divergent_trial_count,
        "token_weighted_decode_speedup": comparison.token_weighted_decode_speedup,
        "prompt_median_speedup": comparison.prompt_median_speedup,
        "speedup_pass": comparison.speedup_pass,
        "prompt_median_speedup_pass": comparison.prompt_median_speedup_pass,
        "release_ready": comparison.release_ready,
        "issues": issues,
        "comparison_sha256": _sha256_file(output_dir / "mtp_ab_comparison.json"),
        "direct_token_weighted_decode_tps": comparison.direct_token_weighted_decode_tps,
        "mtp_token_weighted_decode_tps": comparison.mtp_token_weighted_decode_tps,
    }
    write_data(output_dir / "profile_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/Volumes/Ext4T/models/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/Ext4T/axquant-certification/"
            "deepseek-v4-flash-axq-axengine-v6150/tier2-mtp-2bit"
        ),
    )
    parser.add_argument(
        "--executable",
        default="/Users/devop/code/ax-engine-v6150-bin/ax-engine-bench",
    )
    parser.add_argument(
        "--agent-coding-dataset",
        type=Path,
        default=None,
        help="JSONL with prompt field; default <output-root>/datasets/agent-coding-5.jsonl",
    )
    parser.add_argument(
        "--general-long-dataset",
        type=Path,
        default=None,
        help="JSONL; default <output-root>/datasets/general-long.jsonl",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument(
        "--profiles",
        default="agent-coding,general-long",
        help="Comma-separated profile names to run",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help="Record gate failures without aborting (still writes comparison)",
    )
    args = parser.parse_args()

    # Bit-width experimental gates must be in the process env (not BenchmarkConfig).
    for key in ("AX_ENGINE_2BIT_EXPERIMENTAL", "AX_ENGINE_3BIT_EXPERIMENTAL"):
        if os.environ.get(key) != "1":
            print(f"ERROR: export {key}=1 before running", file=sys.stderr)
            return 2

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"ERROR: model dir missing: {model_dir}", file=sys.stderr)
        return 2
    if not (model_dir / "mtp.safetensors").is_file():
        print(f"ERROR: mtp.safetensors missing under {model_dir}", file=sys.stderr)
        return 2

    executable = str(Path(args.executable).expanduser())
    if not Path(executable).is_file():
        print(f"ERROR: executable missing: {executable}", file=sys.stderr)
        return 2

    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    agent_ds = (
        args.agent_coding_dataset.expanduser().resolve()
        if args.agent_coding_dataset
        else out / "datasets" / "agent-coding-5.jsonl"
    )
    general_ds = (
        args.general_long_dataset.expanduser().resolve()
        if args.general_long_dataset
        else out / "datasets" / "general-long.jsonl"
    )

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    profile_map = {
        "agent-coding": agent_ds,
        "general-long": general_ds,
    }

    results: list[dict] = []
    for name in profiles:
        if name not in profile_map:
            print(f"ERROR: unknown profile {name}", file=sys.stderr)
            return 2
        ds = profile_map[name]
        if not ds.is_file():
            print(f"ERROR: dataset missing for {name}: {ds}", file=sys.stderr)
            return 2
        # Use min(prompt_count, lines in dataset)
        n_lines = sum(1 for line in ds.read_text().splitlines() if line.strip())
        prompt_count = min(args.prompt_count, n_lines)
        try:
            summary = run_profile(
                name=name,
                dataset=ds,
                model_dir=model_dir,
                output_dir=out / name,
                executable=executable,
                prompt_count=prompt_count,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                warmup=args.warmup,
                measured=args.measured,
                enforce_exactness=not args.soft,
                enforce_speedup=not args.soft,
            )
        except (BenchmarkError, InvariantViolationError) as exc:
            print(f"PROFILE_FAILED {name}: {exc}", file=sys.stderr)
            if not args.soft:
                return 1
            summary = {"profile": name, "error": str(exc), "release_ready": False}
        results.append(summary)

    tech = {
        "schema_version": "ax-engine.deepseek-v4-nextn-mtp-tier2-evidence.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "host_id": "df-macstudio-m2",
        "host_hardware": "Apple M2 Ultra 192 GB",
        "host_uname": platform.platform(),
        "hub_repo_id": HUB_REPO,
        "hub_commit": HUB_COMMIT,
        "model_dir": str(model_dir),
        "engine_binary": executable,
        "engine_binary_sha256": _engine_binary_sha256(Path(executable)),
        "ax_engine_version": "6.15.0",
        "ax_engine_commit": "28dbcd252331f8a0eca9829609f2975a1b4be6a8",
        "gates": {
            "weighted_min": WEIGHTED_MIN,
            "prompt_median_min": PROMPT_MEDIAN_MIN,
            "exactness": 1.0,
        },
        "runtime_env_profile": DEEPSEEK_V4_EXACT_MTP_PROFILE_ENV,
        "process_env_required": {
            "AX_ENGINE_2BIT_EXPERIMENTAL": "1",
            "AX_ENGINE_3BIT_EXPERIMENTAL": "1",
        },
        "profiles": {item.get("profile", f"p{i}"): item for i, item in enumerate(results)},
        "technical_tier2_pass": all(
            item.get("exactness_pass")
            and item.get("speedup_pass")
            and item.get("prompt_median_speedup_pass")
            and item.get("release_ready")
            for item in results
            if "error" not in item
        )
        and all("error" not in item for item in results),
        "claim_scope": {
            "authorizing_workloads": profiles,
            "product_default_mtp": "off-direct-fallback",
            "formal_route": "deepseek_v4 nextn MTP with exactness pins",
            "excluded": ["short-answer-chat", "3bit sibling", "vision"],
        },
    }
    write_data(out / "TIER2_TECHNICAL_SUMMARY.json", tech)
    print("==== TIER2 SUMMARY ====", flush=True)
    print(json.dumps(tech, indent=2), flush=True)
    return 0 if tech["technical_tier2_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
