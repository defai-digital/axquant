#!/usr/bin/env python3
"""Holo3 MTP adapt v3: grow labels + longer stage-1 from best stage-1 init.

Factory path on Ext4T. Measures offline top-1 and online MoE-exact accept.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from axquant.mtp_align.adapt_fc import adapt_fc_norms, compose_adapted_onto_pack
from axquant.mtp_align.dataset import prepare_self_distill_dataset
from axquant.mtp_align.evaluate import load_report_metrics
from axquant.mtp_align.gates import evaluate_ladder
from axquant.mtp_align.teacher_force import run_teacher_force
from axquant.serde import file_sha256, write_data


def _merge_prompts(sources: list[Path], dest: Path, limit: int) -> int:
    seen: set[str] = set()
    lines: list[str] = []
    for src in sources:
        if not src.is_file():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line in seen:
                continue
            seen.add(line)
            lines.append(line if line.endswith("\n") else line + "\n")
            if len(lines) >= limit:
                break
        if len(lines) >= limit:
            break
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines), encoding="utf-8")
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work",
        type=Path,
        default=Path("/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/align-campaign-v3"),
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
        default=Path(
            "/Volumes/Ext4T/axquant/work/holo3-35b-axq-dev/AX-Holo3-35B-A3B-MLX-AXQ-6bit"
        ),
    )
    p.add_argument(
        "--init-mtp",
        type=Path,
        default=Path(
            "/Volumes/Ext4T/axquant/work/holo3-35b-mtp-axq/align-campaign-v2/"
            "mtp-adapted-fc/mtp.safetensors"
        ),
        help="Best stage-1 adapted head (preferred over raw graft)",
    )
    p.add_argument("--max-prompts", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--adapt-steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tf-positions", type=int, default=48)
    p.add_argument("--tf-prompts", type=int, default=6)
    p.add_argument(
        "--probe-script",
        type=Path,
        default=Path(
            "/Users/devop/code/axquant-ornith-ship/scripts/run_holo3_35b_mtp_tier2_probe.py"
        ),
    )
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-online", action="store_true")
    args = p.parse_args(argv)

    work = args.work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = args.pack.expanduser().resolve()
    trunk = args.trunk.expanduser().resolve()
    init_mtp = args.init_mtp.expanduser().resolve()
    if not init_mtp.is_file():
        init_mtp = pack / "mtp.safetensors"

    datasets_root = Path("/Volumes/Ext4T/axquant/flagship/qwen36-mtp-v2-c1/datasets")
    prompt_sources = [
        datasets_root / "formal-agent-coding" / "dataset.jsonl",
        datasets_root / "development-agent-coding" / "dataset.jsonl",
        datasets_root / "formal-general" / "dataset.jsonl",
        datasets_root / "development-general" / "dataset.jsonl",
    ]
    prompts = work / "prompts_v3.jsonl"
    n_prompts = _merge_prompts(prompt_sources, prompts, args.max_prompts)
    print(f"prompts={n_prompts} -> {prompts}", flush=True)

    summary: dict = {
        "schema_version": "axquant.mtp-adapt-v3.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pack": str(pack),
        "trunk": str(trunk),
        "init_mtp": str(init_mtp),
        "work": str(work),
        "prompt_count": n_prompts,
    }

    # Baseline TF: trunk pack + init mtp path (mlx_lm loads pack; head from init).
    print("==== teacher-force before (init)", flush=True)
    before = run_teacher_force(
        pack,
        init_mtp,
        prompts,
        max_positions=args.tf_positions,
        max_prompts=args.tf_prompts,
    )
    write_data(work / "teacher_force_before.json", before.as_dict())
    summary["teacher_force_before"] = before.as_dict()
    print(json.dumps(before.as_dict(), indent=2), flush=True)

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

    adapted = work / "mtp-adapted-fc-v3"
    if adapted.exists():
        shutil.rmtree(adapted)
    print("==== adapt-fc v3", flush=True)
    adapt = adapt_fc_norms(
        pack,
        data_path,
        init_mtp,
        adapted,
        steps=args.adapt_steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        trunk_model_id="Hcompany/Holo3-35B-A3B",
        trunk_revision="208d5ae3a03f99d561f32ab5e606f73397a390ea",
        donor_model_id="Qwen/Qwen3.5-35B-A3B",
        donor_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    )
    train = dict(adapt["train"])
    hist = train.pop("loss_history", []) or []
    train["loss_history_len"] = len(hist)
    write_data(
        work / "adapt_summary.json",
        {k: v for k, v in adapt.items() if k != "train"} | {"train": train},
    )
    summary["adapt"] = {
        "output_dir": adapt["output_dir"],
        "output_mtp_sha256": adapt["output_mtp_sha256"],
        "steps": train.get("steps"),
        "samples": train.get("samples"),
        "loss_start": train.get("loss_start"),
        "loss_end": train.get("loss_end"),
    }
    print(json.dumps(summary["adapt"], indent=2), flush=True)

    pack_out = work / "AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP-adapted-v3"
    if pack_out.exists():
        shutil.rmtree(pack_out)
    print("==== compose", flush=True)
    mains = sorted(trunk.glob("model-*.safetensors"))
    before_shas = {f.name: file_sha256(f) for f in mains}
    compose_adapted_onto_pack(trunk, adapted, output_dir=pack_out)
    after_shas = {f.name: file_sha256(pack_out / f.name) for f in mains}
    summary["compose"] = {
        "output": str(pack_out),
        "main_digests_unchanged": before_shas == after_shas,
        "main_files": len(before_shas),
    }
    print(json.dumps(summary["compose"], indent=2), flush=True)

    print("==== teacher-force after", flush=True)
    after = run_teacher_force(
        pack_out,
        pack_out / "mtp.safetensors",
        prompts,
        max_positions=args.tf_positions,
        max_prompts=args.tf_prompts,
    )
    write_data(work / "teacher_force_after.json", after.as_dict())
    summary["teacher_force_after"] = after.as_dict()
    summary["top1_delta"] = float(after.top1) - float(before.top1)
    summary["improved"] = after.top1 > before.top1 + 1e-12
    print(
        json.dumps(
            {
                "before": before.top1,
                "after": after.top1,
                "delta": summary["top1_delta"],
                "correct": after.correct,
                "positions": after.positions,
            },
            indent=2,
        ),
        flush=True,
    )

    online: dict = {"skipped": True}
    if not args.skip_online and args.probe_script.is_file():
        print("==== online probe", flush=True)
        probe_out = work / "online-probe"
        if probe_out.exists():
            shutil.rmtree(probe_out)
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
        (work / "online_probe.log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8"
        )
        decision_path = probe_out / "probe_decision.json"
        if decision_path.is_file():
            metrics = load_report_metrics(decision_path)
            decision = evaluate_ladder(metrics)
            write_data(work / "online_probe_after.json", json.loads(decision_path.read_text()))
            write_data(work / "align_decision_after.json", decision.as_dict())
            online = {
                "accept_rate": metrics.online_accept_rate,
                "speedup": metrics.token_weighted_decode_speedup,
                "recommendation": decision.recommendation.value,
            }
        else:
            online = {"error": "probe_decision missing", "returncode": proc.returncode}
        print(json.dumps(online, indent=2), flush=True)
    summary["online"] = online
    summary["tier2_status"] = "not_certified"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_data(work / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("V3_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
