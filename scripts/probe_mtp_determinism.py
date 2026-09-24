#!/usr/bin/env python3
"""Greedy determinism probe for MTP packs: same prompt, repeated generations.

Discriminates the two hypotheses behind MTP-vs-direct divergence:

* **H1 decode is nondeterministic** — the direct path itself returns different
  tokens across identical repeats (engine-level jitter: async scheduling,
  atomics, split-K reductions). Detected when one arm's repeats disagree.
* **H2 verify path differs from direct** — each arm is internally deterministic
  but the two arms disagree. Detected when both arms repeat cleanly yet their
  outputs differ.

Method: for each pack, suite, and the first ``--prompts`` prompts of the suite,
run ``--repeats`` generations of the direct arm (``AX_NO_SPEC=1``) and of the
MTP arm under the Qwen 3.6 MoE exact profile, all greedy, fixed 64-token
budget, ignore_eos. The benchmark harness rotates trials over the dataset, so a
single-prompt dataset file yields ``repeats`` identical-request generations per
arm. Output token-id hashes are compared within and across arms.

Usage (comparison host, development evidence)::

    export PATH="$HOME/tiel-tier2/engine-754/bin:$PATH"
    export TIEL_TIER2_HOST=df-macbookpro-m5
    ~/tiel-tier2/venv/bin/python scripts/probe_mtp_determinism.py \
      --packs-json ~/tiel-tier2/control-packs.json \
      --models-root ~/tiel-tier2/models \
      --datasets ~/tiel-tier2/datasets \
      --work ~/tiel-tier2/probe
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.benchmark import (  # noqa: E402
    QWEN36_MOE_EXACT_MTP_PROFILE_ENV,
    run_benchmark,
)
from axquant.factory import FACTORY_HOST_ID  # noqa: E402
from axquant.schema import BenchmarkConfig, ModelIdentity  # noqa: E402
from axquant.serde import file_sha256, write_data  # noqa: E402

SEED = 20260728
HOST_ID = os.environ.get("TIEL_TIER2_HOST", FACTORY_HOST_ID)
HOST_OVERRIDE = "TIEL_TIER2_HOST" in os.environ

SUITES = ("agent-coding", "general-long")
SUITE_DATASET_DIR = {
    "agent-coding": "development-agent-coding",
    "general-long": "development-general",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def first_diff_index(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def probe_prompt(
    *,
    pack_key: str,
    suite: str,
    prompt_index: int,
    prompt_line: str,
    model_dir: Path,
    hub_repo: str,
    hub_commit: str,
    work: Path,
    executable: Path,
    repeats: int,
    max_tokens: int,
) -> dict[str, object]:
    ds_dir = work / pack_key / suite / f"prompt-{prompt_index}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds_path = ds_dir / "dataset.jsonl"
    text = prompt_line if prompt_line.endswith("\n") else prompt_line + "\n"
    ds_path.write_text(text, encoding="utf-8")
    ds_sha = file_sha256(ds_path)

    def cfg(mtp_enabled: bool) -> BenchmarkConfig:
        return BenchmarkConfig(
            model=ModelIdentity(
                model_id=hub_repo,
                revision=hub_commit,
                format="mlx",
                local_path=str(model_dir),
            ),
            mtp_enabled=mtp_enabled,
            baseline_kind="axquant-mtp-on" if mtp_enabled else "axquant-mtp-off",
            workload=suite,
            dataset_sha256=ds_sha,
            prompt_count=1,
            warmup_trials=0,
            measured_trials=repeats,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=max_tokens,
            ignore_eos=True,
            draft_depth=1,
            random_seed=SEED,
            timeout_seconds=600.0,
            runtime_env=dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
        )

    arms: dict[str, dict[str, object]] = {}
    for label, enabled in (("direct", False), ("mtp", True)):
        out_dir = ds_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_benchmark(
            cfg(enabled),
            dataset_path=ds_path,
            executable=str(executable),
            output_dir=out_dir,
        )
        trials = [t for t in result.trials if t.success]
        shas = [t.output_sha256 for t in trials]
        ids = [t.output_token_ids or [] for t in trials]
        arms[label] = {
            "successful_repeats": len(trials),
            "unique_outputs": len(set(shas)),
            "all_repeats_identical": len(set(shas)) <= 1,
            "output_shas": shas,
            "tokens_per_second": [t.tokens_per_second for t in trials],
            "mtp_active": [t.mtp_active for t in trials],
            "accepted_over_proposed": [
                [t.mtp_accepted_tokens, t.mtp_proposed_tokens] for t in trials
            ],
            "_ids": ids,
        }

    direct_ids = arms["direct"]["_ids"]
    mtp_ids = arms["mtp"]["_ids"]
    cross = [first_diff_index(d, m) for d, m in zip(direct_ids, mtp_ids, strict=False)]
    for arm in arms.values():
        arm.pop("_ids")
    return {
        "pack": pack_key,
        "suite": suite,
        "prompt_index": prompt_index,
        "dataset_sha256": ds_sha,
        "repeats": repeats,
        "direct": {k: v for k, v in arms["direct"].items() if k != "_ids"},
        "mtp": {k: v for k, v in arms["mtp"].items() if k != "_ids"},
        "cross_arm_first_diff": cross,
        "verdict": (
            "decode-nondeterministic"
            if not arms["direct"]["all_repeats_identical"]
            or not arms["mtp"]["all_repeats_identical"]
            else (
                "verify-differs-from-direct"
                if any(i is not None for i in cross)
                else "exact-and-deterministic"
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs-json", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=2, help="prompts per suite")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--executable", default="ax-engine-bench")
    args = parser.parse_args()

    if HOST_OVERRIDE:
        log(
            f"WARNING: non-factory host {HOST_ID} ({socket.gethostname()}); "
            "development evidence only"
        )
    else:
        log(f"host={HOST_ID} {socket.gethostname()}")

    import shutil

    resolved_exe = shutil.which(args.executable)
    if resolved_exe is None:
        raise SystemExit(f"ax-engine-bench not found: {args.executable}")

    packs = json.loads(args.packs_json.read_text(encoding="utf-8"))
    args.work.mkdir(parents=True, exist_ok=True)
    results = []
    for pack_key in sorted(packs):
        meta = packs[pack_key]
        model_dir = args.models_root / str(meta["local_dir_name"])
        if not (model_dir / "axquant_manifest.json").is_file():
            raise SystemExit(f"missing pack {model_dir}")
        hub_repo = str(meta["hub_repo"])
        hub_commit = str(meta.get("hub_commit") or meta.get("fallback_hub_commit") or "main")
        for suite in SUITES:
            suite_path = args.datasets / SUITE_DATASET_DIR[suite] / "dataset.jsonl"
            lines = [ln for ln in suite_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for prompt_index in range(min(args.prompts, len(lines))):
                log(f"=== {pack_key} {suite} prompt {prompt_index} ===")
                results.append(
                    probe_prompt(
                        pack_key=pack_key,
                        suite=suite,
                        prompt_index=prompt_index,
                        prompt_line=lines[prompt_index],
                        model_dir=model_dir,
                        hub_repo=hub_repo,
                        hub_commit=hub_commit,
                        work=args.work,
                        executable=Path(resolved_exe),
                        repeats=args.repeats,
                        max_tokens=args.max_tokens,
                    )
                )
                r = results[-1]
                log(
                    f"{pack_key} {suite} p{prompt_index}: "
                    f"direct_unique={r['direct']['unique_outputs']} "
                    f"mtp_unique={r['mtp']['unique_outputs']} "
                    f"verdict={r['verdict']}"
                )
    report = {
        "schema_version": "axquant.mtp-determinism-probe.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "host_id": HOST_ID,
        "hostname": socket.gethostname(),
        "engine_executable": resolved_exe,
        "seed": SEED,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "runtime_env": dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
        "results": results,
    }
    write_data(args.work / "probe_report.json", report)
    n_exact = sum(1 for r in results if r["verdict"] == "exact-and-deterministic")
    n_verify = sum(1 for r in results if r["verdict"] == "verify-differs-from-direct")
    n_nondet = sum(1 for r in results if r["verdict"] == "decode-nondeterministic")
    log(f"exact={n_exact} verify-differs={n_verify} nondeterministic={n_nondet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
