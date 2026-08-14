#!/usr/bin/env python3
"""Holo3 MTP align campaign: measure → decide → recommend next stage.

Best practices:
  * KPI order: accept_rate → speedup → formal Tier 2
  * Do not env-tune when accept≈0
  * Prefer freeze-trunk stage-1 fc/norm adapt over co-train first

Example (factory)::

    export PYTHONPATH=/Users/devop/code/axquant/src
    export PATH=/Users/devop/code/ax-engine-v6150-bin:$PATH
    python scripts/run_holo3_mtp_align_campaign.py \\
      --model-dir .../AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP \\
      --probe-report .../tier2-probe-6bit/probe_decision.json \\
      --output-root .../align-campaign
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from axquant.mtp_align.evaluate import load_report_metrics
from axquant.mtp_align.gates import evaluate_ladder
from axquant.serde import write_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-report",
        type=Path,
        required=True,
        help="Existing online probe_decision or mtp_ab_comparison JSON",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        help="Optional prompts JSONL for offline teacher-force baseline",
    )
    parser.add_argument("--run-teacher-force", action="store_true")
    parser.add_argument("--max-positions", type=int, default=32)
    parser.add_argument("--max-prompts", type=int, default=4)
    args = parser.parse_args(argv)

    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_report_metrics(args.probe_report)
    decision = evaluate_ladder(metrics)
    write_data(out / "align_decision_online.json", decision.as_dict())

    teacher_report = None
    if args.run_teacher_force:
        if args.prompts is None:
            print("--run-teacher-force requires --prompts", file=sys.stderr)
            return 2
        from axquant.mtp_align.teacher_force import run_teacher_force

        mtp = args.model_dir / "mtp.safetensors"
        teacher_report = run_teacher_force(
            args.model_dir,
            mtp,
            args.prompts,
            max_positions=args.max_positions,
            max_prompts=args.max_prompts,
        )
        write_data(out / "teacher_force.json", teacher_report.as_dict())
        # Re-evaluate with offline metric filled in
        from axquant.mtp_align.gates import AlignMetrics

        combined = AlignMetrics(
            online_accept_rate=metrics.online_accept_rate,
            offline_top1=teacher_report.top1,
            token_weighted_decode_speedup=metrics.token_weighted_decode_speedup,
            prompt_median_speedup=metrics.prompt_median_speedup,
            exactness_pass=metrics.exactness_pass,
            source="online+offline",
        )
        decision = evaluate_ladder(combined)
        write_data(out / "align_decision_combined.json", decision.as_dict())

    summary = {
        "schema_version": "axquant.mtp-align-campaign.v1",
        "finished_at": datetime.now(UTC).isoformat(),
        "model_dir": str(args.model_dir),
        "probe_report": str(args.probe_report),
        "recommendation": decision.recommendation.value,
        "stage_reached": decision.stage_reached,
        "reasons": list(decision.reasons),
        "online_accept_rate": metrics.online_accept_rate,
        "offline_top1": None if teacher_report is None else teacher_report.top1,
        "token_weighted_decode_speedup": metrics.token_weighted_decode_speedup,
        "next_commands": _next_commands(decision.recommendation.value, args.model_dir, out),
    }
    write_data(out / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _next_commands(recommendation: str, model_dir: Path, out: Path) -> list[str]:
    model = str(model_dir)
    if recommendation == "adapt_fc_norms":
        return [
            f"axquant mtp-align-prepare-data --model {model} --prompts PROMPTS.jsonl "
            f"--output {out}/data.jsonl --max-samples 256",
            f"axquant mtp-align-adapt-fc --model {model} --data {out}/data.jsonl "
            f"--init-mtp {model}/mtp.safetensors --output {out}/mtp-adapted-fc "
            f"--steps 200",
            f"axquant compose-grafted-mtp --model-dir CERTIFIED_TRUNK "
            f"--mtp-dir {out}/mtp-adapted-fc --output {out}/pack-adapted",
            "Re-run scripts/run_holo3_35b_mtp_tier2_probe.py on pack-adapted",
        ]
    if recommendation == "stop_env_tuning":
        return [
            "Do not spend more on env flags alone",
            "If offline top-1 high but online accept 0: debug offline vs engine wiring",
            "Otherwise start stage-1 adapt-fc",
        ]
    if recommendation == "online_speedup_sweep":
        return [
            "Run MoE-exact medium/full probe; tune draft depth only if accept stays high",
        ]
    if recommendation == "ready_for_formal_tier2":
        return ["Run formal authorizing Tier 2 scoreboard on declared host"]
    return ["See campaign_summary reasons"]


if __name__ == "__main__":
    raise SystemExit(main())
