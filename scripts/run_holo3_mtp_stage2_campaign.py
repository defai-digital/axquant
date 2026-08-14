#!/usr/bin/env python3
"""Stage-2 full-layer MTP adapt campaign for Holo3 (factory)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from axquant.mtp_align.adapt_fc import adapt_full_layer, compose_adapted_onto_pack
from axquant.mtp_align.evaluate import load_report_metrics
from axquant.mtp_align.gates import evaluate_ladder
from axquant.mtp_align.teacher_force import run_teacher_force
from axquant.serde import file_sha256, write_data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work",
        type=Path,
        default=Path("/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/align-campaign-v2"),
    )
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
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=384)
    p.add_argument(
        "--probe-script",
        type=Path,
        default=Path(
            "/Users/devop/code/axquant-ornith-ship/scripts/run_holo3_35b_mtp_tier2_probe.py"
        ),
    )
    args = p.parse_args(argv)

    work = args.work.expanduser().resolve()
    out = work / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    pack = args.pack.expanduser().resolve()
    trunk = args.trunk.expanduser().resolve()
    prompts = work / "prompts16.jsonl"
    init = work / "mtp-adapted-fc" / "mtp.safetensors"
    data = work / "data.jsonl"
    feat = work / "data.features.safetensors"
    for path in (pack, trunk, prompts, init, data, feat):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    adapted = out / "mtp-adapted-full"
    pack_out = out / "pack-adapted-full"
    if adapted.exists():
        import shutil

        shutil.rmtree(adapted)
    if pack_out.exists():
        import shutil

        shutil.rmtree(pack_out)

    print("==== stage2 adapt-full", flush=True)
    adapt = adapt_full_layer(
        pack,
        data,
        init,
        adapted,
        steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        features_path=feat,
        trunk_model_id="Hcompany/Holo3-35B-A3B",
        trunk_revision="208d5ae3a03f99d561f32ab5e606f73397a390ea",
        donor_model_id="Qwen/Qwen3.5-35B-A3B",
        donor_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    )
    train = dict(adapt["train"])
    hist = train.pop("loss_history", []) or []
    train["loss_history_len"] = len(hist)
    write_data(
        out / "adapt_full_summary.json",
        {k: v for k, v in adapt.items() if k != "train"} | {"train": train},
    )
    print(
        json.dumps(
            {
                "loss_start": train.get("loss_start"),
                "loss_end": train.get("loss_end"),
                "steps": train.get("steps"),
                "trainable_count": train.get("trainable_count"),
            },
            indent=2,
        ),
        flush=True,
    )

    print("==== compose", flush=True)
    mains = sorted(trunk.glob("model-*.safetensors"))
    before = {f.name: file_sha256(f) for f in mains}
    compose_adapted_onto_pack(trunk, adapted, output_dir=pack_out)
    after = {f.name: file_sha256(pack_out / f.name) for f in mains}
    digest_ok = before == after
    print("main_digests_unchanged", digest_ok, flush=True)

    print("==== teacher-force", flush=True)
    s1_before = json.loads((work / "teacher_force_before.json").read_text(encoding="utf-8"))
    s1_after = json.loads((work / "teacher_force_after.json").read_text(encoding="utf-8"))
    tf = run_teacher_force(
        pack_out,
        pack_out / "mtp.safetensors",
        prompts,
        max_positions=32,
        max_prompts=4,
    )
    write_data(out / "teacher_force_stage2.json", tf.as_dict())
    print(
        json.dumps(
            {
                "baseline": s1_before["top1"],
                "stage1": s1_after["top1"],
                "stage2": tf.top1,
                "correct": tf.correct,
                "positions": tf.positions,
            },
            indent=2,
        ),
        flush=True,
    )

    online: dict = {"skipped": True}
    if args.probe_script.is_file():
        print("==== online probe", flush=True)
        probe_out = out / "online-probe"
        cmd = [
            sys.executable,
            str(args.probe_script),
            "--model-dir",
            str(pack_out),
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
        (out / "online_probe.log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8"
        )
        decision_path = probe_out / "probe_decision.json"
        if decision_path.is_file():
            metrics = load_report_metrics(decision_path)
            decision = evaluate_ladder(metrics)
            write_data(out / "online_probe_stage2.json", json.loads(decision_path.read_text()))
            write_data(out / "align_decision_stage2.json", decision.as_dict())
            online = {
                "accept_rate": metrics.online_accept_rate,
                "speedup": metrics.token_weighted_decode_speedup,
                "recommendation": decision.recommendation.value,
            }
        else:
            online = {"error": "probe_decision missing", "returncode": proc.returncode}
        print(json.dumps(online, indent=2), flush=True)

    summary = {
        "schema_version": "axquant.mtp-adapt-stage2.v1",
        "baseline_top1": s1_before["top1"],
        "stage1_top1": s1_after["top1"],
        "stage2_top1": tf.top1,
        "stage2_correct": tf.correct,
        "stage2_positions": tf.positions,
        "main_digests_unchanged": digest_ok,
        "online": online,
        "adapt_steps": args.steps,
        "init_mtp": str(init),
        "tier2_status": "not_certified",
        "stage2_improved_vs_stage1": tf.top1 > float(s1_after["top1"]),
        "stage2_improved_vs_baseline": tf.top1 > float(s1_before["top1"]),
    }
    write_data(out / "stage2_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("STAGE2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
