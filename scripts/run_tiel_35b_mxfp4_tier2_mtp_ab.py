#!/usr/bin/env python3
"""Formal MTP Tier 2 A/B for the Tiel / Cyber-Tiel 35B-A3B MXFP4 -MTP packs.

Both packs are ``Qwen3_5MoeForConditionalGeneration`` (adapter ``qwen35-moe-v1``),
the same sparse-expert family as the certified Qwen 3.6 35B-A3B Tier 2 records,
so the harness pins the Qwen 3.6 **MoE exact** runtime contract
(``QWEN36_MOE_EXACT_MTP_PROFILE_ENV``, AXQ-041): greedy exactness 1.0,
token-weighted decode speedup >= 1.20, prompt-median speedup >= 1.10.

The factory formal 24/16-prompt suites were retired with the old
``axquant-certification`` tree; this run binds the repo development suites
(15 agent-coding + 15 general prompts, the same ``data/eval`` files bound by
the 2026-09-23 checkpoint Tier 1 records) and records their SHA-256 in every
profile. The certification spec binds the recorded datasets; it does not
mandate a prompt population.

Usage (factory host ``df-macstudio-m2``, run from the campaign repo copy)::

    export PATH="/opt/homebrew/bin:$PATH"
    .venv/bin/python scripts/run_tiel_35b_mxfp4_tier2_mtp_ab.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.benchmark import (  # noqa: E402
    QWEN36_MOE_EXACT_MTP_PROFILE_ENV,
    compare_mtp_ab_results,
    run_benchmark,
)
from axquant.factory import (  # noqa: E402
    FACTORY_HOST_ID,
    FACTORY_MODELS,
    require_factory_host,
)
from axquant.schema import BenchmarkConfig, ModelIdentity  # noqa: E402
from axquant.serde import file_sha256, write_data  # noqa: E402

SEED = 20260728
WEIGHTED_MIN = 1.20
PROMPT_MEDIAN_MIN = 1.10

HOST_ID = os.environ.get("TIEL_TIER2_HOST", FACTORY_HOST_ID)
# Explicit TIEL_TIER2_HOST selects a comparison host (diagnostic run). Only the
# default factory identity carries certification authority; a non-default host
# is recorded as development evidence and must not be published as a factory
# Tier 2 certificate.
HOST_OVERRIDE = "TIEL_TIER2_HOST" in os.environ

PACKS: dict[str, dict[str, object]] = {
    "tiel": {
        "name": "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP",
        "local_dir_name": "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP-redo-20260923",
        "hub_repo": "AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP",
        "product_class": "MXFP4",
        "tier1_cert": "tiel-coder-35b-axq-mxfp4-mtp-redo-tier1",
        "fallback_hub_commit": "607a7ba019a0b7478f49505d62d69fa1f4164ff5",
    },
    "cyber-tiel": {
        "name": "AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP",
        "local_dir_name": "AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP-redo-20260923",
        "hub_repo": "AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP",
        "product_class": "MXFP4",
        "tier1_cert": "cyber-tiel-coder-35b-axq-mxfp4-mtp-redo-tier1",
        "fallback_hub_commit": "90b785d81d68d022945289a2bb16febe500fd314",
    },
}

DEFAULT_MODELS = Path(os.environ.get("TIEL_TIER2_MODELS", FACTORY_MODELS))
DEFAULT_DATASETS = Path(
    os.environ.get(
        "TIEL_TIER2_DATASETS",
        "/Volumes/Ext16TR0/axquant/work/tiel-redo-20260923/datasets",
    )
)
DEFAULT_WORK = Path(
    os.environ.get(
        "TIEL_TIER2_WORK",
        "/Volumes/Ext16TR0/axquant/work/tiel-mxfp4-tier2-20260924",
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


def hub_commit_for(meta: dict[str, object]) -> str:
    t1 = ROOT / "docs" / "certifications" / f"{meta['tier1_cert']}.json"
    if t1.is_file():
        data = json.loads(t1.read_text(encoding="utf-8"))
        return str(data["artifact"]["hub_commit"])
    return str(meta["fallback_hub_commit"])


def base_config(
    *,
    model_dir: Path,
    hub_repo: str,
    hub_commit: str,
    workload: str,
    dataset_sha256: str,
    prompt_count: int,
    mtp_enabled: bool,
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
        warmup_trials=1,
        measured_trials=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=64,
        # Fixed-token budgets for fair decode-heavy A/B (short general prompts
        # otherwise stop early and understate MTP speedup).
        ignore_eos=True,
        draft_depth=1,
        random_seed=SEED,
        timeout_seconds=600.0,
        runtime_env=dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
    )


def run_pack(
    *,
    pack_key: str,
    packs: dict[str, dict[str, object]],
    models_root: Path,
    datasets: Path,
    work: Path,
    executable: Path,
) -> dict[str, object]:
    meta = packs[pack_key]
    model_dir = models_root / str(meta["local_dir_name"])
    if not (model_dir / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {model_dir}")
    if not (model_dir / "mtp.safetensors").is_file():
        raise SystemExit(f"pack missing mtp.safetensors: {model_dir}")

    subprocess.run(
        ["ax-engine-bench", "generate-manifest", "--validate", "--", str(model_dir)],
        check=False,
        cwd=str(ROOT),
    )

    out = work / pack_key
    out.mkdir(parents=True, exist_ok=True)
    hub_repo = str(meta["hub_repo"])
    hub_commit = hub_commit_for(meta)

    agent_src = datasets / "development-agent-coding" / "dataset.jsonl"
    gen_src = datasets / "development-general" / "dataset.jsonl"
    if not agent_src.is_file() or not gen_src.is_file():
        raise SystemExit(f"missing development suites under {datasets}")

    agent_ds = subset_dataset(agent_src, out / "datasets" / "agent-coding.jsonl", 15)
    gen_ds = subset_dataset(gen_src, out / "datasets" / "general-long.jsonl", 15)

    profiles: dict[str, object] = {}
    all_pass = True
    for workload, ds in (
        ("agent-coding", agent_ds),
        ("general-long", gen_ds),
    ):
        count = 15
        ds_sha = file_sha256(ds)
        log(f"=== {pack_key} {workload} mtp-off ===")
        off_dir = out / workload / "mtp-off"
        off_dir.mkdir(parents=True, exist_ok=True)
        off_result = run_benchmark(
            base_config(
                model_dir=model_dir,
                hub_repo=hub_repo,
                hub_commit=hub_commit,
                workload=workload,
                dataset_sha256=ds_sha,
                prompt_count=count,
                mtp_enabled=False,
            ),
            dataset_path=ds,
            executable=str(executable),
            output_dir=off_dir,
        )
        log(f"=== {pack_key} {workload} mtp-on ===")
        on_dir = out / workload / "mtp-on"
        on_dir.mkdir(parents=True, exist_ok=True)
        on_result = run_benchmark(
            base_config(
                model_dir=model_dir,
                hub_repo=hub_repo,
                hub_commit=hub_commit,
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
            off_result,
            on_result,
            profile_name="benchmark-ab",
            minimum_speedup=WEIGHTED_MIN,
            speedup_metric="token-weighted-decode-tps",
            minimum_prompt_median_speedup=PROMPT_MEDIAN_MIN,
        )
        cmp_path = out / workload / "mtp_ab_comparison.json"
        write_data(cmp_path, comparison)
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
            "measured_trial_count": 2,
            "gate_pass": profile_pass,
        }
        log(
            f"{pack_key} {workload}: exact={exact} weighted={weighted:.4f} "
            f"prompt_med={prompt_med:.4f} release_ready={release} pass={profile_pass}"
        )

    summary = {
        "schema_version": "axquant.tiel-moe-mtp-tier2-ab.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "host_id": HOST_ID,
        "hostname": socket.gethostname(),
        "pack_key": pack_key,
        "hub_repo_id": hub_repo,
        "hub_commit": hub_commit,
        "local_path": str(model_dir),
        "product_class": meta["product_class"],
        "thresholds": {
            "exactness_required": 1.0,
            "token_weighted_decode_speedup_min": WEIGHTED_MIN,
            "prompt_median_speedup_min": PROMPT_MEDIAN_MIN,
        },
        "runtime_env": dict(QWEN36_MOE_EXACT_MTP_PROFILE_ENV),
        "formal_route": "qwen36-moe-exact-profile-certification-candidate",
        "ax_engine_executable": str(executable),
        "engine_binary_sha256": engine_sha(executable),
        "harness": {
            "temperature": 0.0,
            "seed": SEED,
            "draft_depth": 1,
            "warmup_trials": 1,
            "measured_trials": 2,
            "max_tokens": 64,
            "ignore_eos": True,
            "prompt_count_agent_coding": 15,
            "prompt_count_general_long": 15,
            "speedup_metric": "token-weighted-decode-tps",
        },
        "profiles": profiles,
        "technical_tier2_pass": all_pass
        and all(bool(p.get("gate_pass")) for p in profiles.values()),
    }
    write_data(out / "TIER2_TECHNICAL_SUMMARY.json", summary)
    for workload in ("agent-coding", "general-long"):
        src = out / workload / "mtp_ab_comparison.json"
        if src.is_file():
            shutil.copy2(src, out / f"{workload}-mtp_ab_comparison.json")
    log(f"{pack_key} technical_tier2_pass={summary['technical_tier2_pass']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        choices=["tiel", "cyber-tiel", "both", "all-json"],
        default="both",
    )
    parser.add_argument(
        "--packs-json",
        type=Path,
        default=None,
        help="JSON object mapping pack key to a PACKS-style entry; overrides --pack",
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()

    packs: dict[str, dict[str, object]] = PACKS
    if args.packs_json is not None:
        packs = json.loads(args.packs_json.read_text(encoding="utf-8"))
        missing = {
            key
            for key, entry in packs.items()
            for field in ("local_dir_name", "hub_repo", "fallback_hub_commit")
            if field not in entry
        }
        if missing:
            raise SystemExit(f"packs-json entries missing fields: {sorted(missing)}")

    if HOST_OVERRIDE:
        log(
            f"WARNING: non-factory comparison host {HOST_ID} ({socket.gethostname()}); "
            "development evidence only, not certification authority"
        )
    else:
        require_factory_host(socket.gethostname())
    executable = engine_binary()
    log(f"engine={executable} host={HOST_ID} hostname={socket.gethostname()}")
    if args.packs_json is not None:
        keys = sorted(packs)
    else:
        keys = ["tiel", "cyber-tiel"] if args.pack == "both" else [args.pack]
    results = []
    for key in keys:
        results.append(
            run_pack(
                pack_key=key,
                packs=packs,
                models_root=args.models_root,
                datasets=args.datasets,
                work=args.work,
                executable=executable,
            )
        )
    write_data(args.work / "all_packs_summary.json", {"results": results})
    ok = all(bool(r.get("technical_tier2_pass")) for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
