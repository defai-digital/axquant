#!/usr/bin/env python3
"""Multi-hour factory campaign: prepare labels → stage-1 adapt → measure.

Intended host: df-macstudio-m2 with Ext4T Holo3 6-bit-MTP pack.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from axquant.grafted_mtp import compose_grafted_mtp_onto_pack
from axquant.mtp_align.adapt_fc import adapt_fc_norms
from axquant.mtp_align.dataset import prepare_self_distill_dataset
from axquant.mtp_align.evaluate import load_report_metrics
from axquant.mtp_align.gates import evaluate_ladder
from axquant.mtp_align.teacher_force import run_teacher_force
from axquant.serde import file_sha256, write_data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pack",
        type=Path,
        default=Path(
            "/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP"
        ),
    )
    p.add_argument(
        "--trunk",
        type=Path,
        default=Path("/Volumes/Ext4T/axquant/work/holo3-35b-axq-dev/AX-Holo3-35B-A3B-MLX-AXQ-6bit"),
        help="Certified non-MTP trunk for compose (main digests preserved)",
    )
    p.add_argument(
        "--work",
        type=Path,
        default=Path("/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/align-campaign-v2"),
    )
    p.add_argument("--prompts", type=Path, default=None)
    p.add_argument("--max-prompts", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--max-samples", type=int, default=384)
    p.add_argument("--max-seq-len", type=int, default=96)
    p.add_argument("--adapt-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tf-positions", type=int, default=32)
    p.add_argument("--tf-prompts", type=int, default=4)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-adapt", action="store_true")
    p.add_argument("--skip-online", action="store_true")
    p.add_argument(
        "--online-probe-script",
        type=Path,
        default=Path(
            "/Users/devop/code/axquant-ornith-ship/scripts/run_holo3_35b_mtp_tier2_probe.py"
        ),
    )
    args = p.parse_args(argv)

    work = args.work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = args.pack.expanduser().resolve()
    trunk = args.trunk.expanduser().resolve()
    prompts = args.prompts or (work / "prompts16.jsonl")
    if not prompts.is_file():
        raise SystemExit(f"prompts missing: {prompts}")

    summary: dict = {
        "schema_version": "axquant.mtp-adapt-campaign.v1",
        "started_at": datetime.now(UTC).isoformat(),
        "pack": str(pack),
        "trunk": str(trunk),
        "work": str(work),
    }

    # Baseline TF if present
    before_path = work / "teacher_force_before.json"
    if before_path.is_file():
        summary["teacher_force_before"] = json.loads(before_path.read_text())
    else:
        before = run_teacher_force(
            pack,
            pack / "mtp.safetensors",
            prompts,
            max_positions=args.tf_positions,
            max_prompts=args.tf_prompts,
        )
        write_data(before_path, before.as_dict())
        summary["teacher_force_before"] = before.as_dict()

    data_path = work / "data.jsonl"
    if not args.skip_prepare:
        print("==== prepare-data", flush=True)
        prep = prepare_self_distill_dataset(
            pack,
            prompts,
            data_path,
            max_prompts=args.max_prompts,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            max_seq_len=args.max_seq_len,
            write_features=True,
        )
        write_data(work / "prepare_summary.json", prep)
        summary["prepare"] = prep
        print(json.dumps(prep, indent=2), flush=True)
    else:
        summary["prepare"] = {"output": str(data_path), "skipped": True}

    adapted_dir = work / "mtp-adapted-fc"
    if not args.skip_adapt:
        if adapted_dir.exists():
            # allow resume only if empty-ish re-run requires wipe
            import shutil

            shutil.rmtree(adapted_dir)
        print("==== adapt-fc", flush=True)
        adapt = adapt_fc_norms(
            pack,
            data_path,
            pack / "mtp.safetensors",
            adapted_dir,
            steps=args.adapt_steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            trunk_model_id="Hcompany/Holo3-35B-A3B",
            trunk_revision="208d5ae3a03f99d561f32ab5e606f73397a390ea",
            donor_model_id="Qwen/Qwen3.5-35B-A3B",
            donor_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        )
        write_data(work / "adapt_summary.json", adapt)
        summary["adapt"] = adapt
        print(json.dumps({k: adapt[k] for k in adapt if k != "train"}, indent=2), flush=True)
        train = adapt.get("train") or {}
        summary["adapt_loss_start"] = train.get("loss_start")
        summary["adapt_loss_end"] = train.get("loss_end")
        summary["adapt_steps"] = train.get("steps")
    else:
        summary["adapt"] = {"output_dir": str(adapted_dir), "skipped": True}

    # Compose adapted head onto certified trunk
    composed = work / "AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP-adapted-fc"
    if composed.exists():
        import shutil

        shutil.rmtree(composed)
    print("==== compose", flush=True)
    # main digest check on trunk model shards
    trunk_mains = sorted(trunk.glob("model-*.safetensors"))
    before_shas = {f.name: file_sha256(f) for f in trunk_mains}
    compose_grafted_mtp_onto_pack(trunk, adapted_dir, output_dir=composed)
    after_shas = {f.name: file_sha256(composed / f.name) for f in trunk_mains}
    digest_ok = before_shas == after_shas
    summary["compose"] = {
        "output": str(composed),
        "main_digests_unchanged": digest_ok,
        "main_files": len(before_shas),
    }
    if not digest_ok:
        summary["compose"]["mismatch"] = {
            k: {"before": before_shas[k], "after": after_shas.get(k)}
            for k in before_shas
            if before_shas[k] != after_shas.get(k)
        }
    print(json.dumps(summary["compose"], indent=2), flush=True)

    print("==== teacher-force after", flush=True)
    after = run_teacher_force(
        composed,
        composed / "mtp.safetensors",
        prompts,
        max_positions=args.tf_positions,
        max_prompts=args.tf_prompts,
    )
    write_data(work / "teacher_force_after.json", after.as_dict())
    summary["teacher_force_after"] = after.as_dict()
    before_top1 = float((summary.get("teacher_force_before") or {}).get("top1") or 0.0)
    after_top1 = float(after.top1)
    summary["top1_delta"] = after_top1 - before_top1
    summary["stage1_moved_top1"] = after_top1 > before_top1 + 1e-12

    if not args.skip_online and args.online_probe_script.is_file():
        print("==== online probe", flush=True)
        probe_out = work / "online-probe-after"
        cmd = [
            sys.executable,
            str(args.online_probe_script),
            "--model-dir",
            str(composed),
            "--bits",
            "6",
            "--output-root",
            str(probe_out),
            "--profiles",
            "agent-coding",
            "--prompt-count",
            "8",
            "--max-tokens",
            "128",
            "--warmup",
            "1",
            "--measured",
            "2",
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        (work / "online_probe_after.log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8"
        )
        decision_path = probe_out / "probe_decision.json"
        if decision_path.is_file():
            metrics = load_report_metrics(decision_path)
            decision = evaluate_ladder(metrics)
            write_data(work / "online_probe_after.json", json.loads(decision_path.read_text()))
            write_data(work / "align_decision_after.json", decision.as_dict())
            summary["online_after"] = {
                "accept_rate": metrics.online_accept_rate,
                "speedup": metrics.token_weighted_decode_speedup,
                "recommendation": decision.recommendation.value,
            }
        else:
            summary["online_after"] = {
                "error": "probe_decision missing",
                "returncode": proc.returncode,
            }
    else:
        summary["online_after"] = {"skipped": True}

    # Recommendation
    if summary.get("stage1_moved_top1"):
        summary["campaign_verdict"] = "stage1_improved_offline_top1"
        summary["next"] = "grow data / more steps / online accept sweep"
    else:
        summary["campaign_verdict"] = "stage1_fail_closed_no_top1_gain"
        summary["next"] = "escalate stage-2 full-layer unfreeze or stop"
    summary["finished_at"] = datetime.now(UTC).isoformat()
    write_data(work / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
