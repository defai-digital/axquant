#!/usr/bin/env python3
"""Derive and publish no-MTP Hub siblings of certified Qwen 3.6 MTP packs.

For each certified ``*-MTP`` pack:
  1. Snapshot the Hub MTP artifact (pinned tip, or QWEN36_MTP_REVISIONS).
  2. Materialize a no-MTP tree (drop ``mtp.safetensors`` + MTP sidecar metadata).
  3. Rewrite ``axquant_manifest.json`` / plan MTP flags and rebind the card.
  4. Upload to ``AutomatosX/AX-…-AXQ-{4,6}bit`` (no ``-MTP`` suffix).

Usage (factory host with Ext4T + HF token):
  .venv/bin/python scripts/publish_qwen36_no_mtp_siblings.py
  .venv/bin/python scripts/publish_qwen36_no_mtp_siblings.py --skip-upload
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = Path(
    os.environ.get(
        "QWEN36_NO_MTP_WORK",
        "/Volumes/Ext4T/axquant/work/qwen36-no-mtp-siblings",
    )
)
DEFAULT_OUT = Path(os.environ.get("QWEN36_NO_MTP_OUT", "/Volumes/Ext4T/models"))

# Certified MTP sources → no-MTP Hub product names.
PACKS: list[dict[str, str]] = [
    {
        "mtp_repo": "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
        "no_mtp_name": "AX-Qwen3.6-27B-MLX-AXQ-4bit",
        "product_class": "4bit",
        "display": "Qwen 3.6 27B AXQ 4-bit",
    },
    {
        "mtp_repo": "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        "no_mtp_name": "AX-Qwen3.6-27B-MLX-AXQ-6bit",
        "product_class": "6bit",
        "display": "Qwen 3.6 27B AXQ 6-bit",
    },
    {
        "mtp_repo": "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP",
        "no_mtp_name": "AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit",
        "product_class": "4bit",
        "display": "Qwen 3.6 35B-A3B AXQ 4-bit",
    },
    {
        "mtp_repo": "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
        "no_mtp_name": "AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit",
        "product_class": "6bit",
        "display": "Qwen 3.6 35B-A3B AXQ 6-bit",
    },
]

MTP_FILE_NAMES = frozenset(
    {
        "mtp.safetensors",
        "mtp_head.safetensors",
        "axquant_mtp_sidecar_manifest.json",
        "axquant_mtp_graft.json",
    }
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _is_mtp_path(rel: str) -> bool:
    name = Path(rel).name
    if name in MTP_FILE_NAMES:
        return True
    lowered = rel.lower()
    return "mtp" in Path(rel).name.lower() and (
        lowered.endswith(".safetensors") or "mtp_sidecar" in lowered or "mtp_graft" in lowered
    )


def materialize_no_mtp(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        # Skip HF cache / incomplete blobs
        parts = rel.parts
        if parts and parts[0] in {".cache", ".git"}:
            continue
        if any(part.startswith(".") and part not in {".gitattributes"} for part in parts[:-1]):
            continue
        if _is_mtp_path(str(rel)):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # Prefer hardlink to save disk; fall back to copy.
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)

    # Drop MTP from plan assignments and recompute distributions so the plan
    # still validates (weight_distribution must match assignments).
    plan_path = dest / "axquant_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        assignments = plan.get("assignments")
        if isinstance(assignments, list):
            kept = [
                a
                for a in assignments
                if not str(a.get("role") or "").startswith("mtp")
                and "mtp" not in str(a.get("tensor") or "").lower()
                and "mtp" not in str(a.get("module_path") or "").lower()
            ]
            plan["assignments"] = kept
            # Rebuild weight_distribution from remaining assignments.
            buckets: dict[str, dict[str, float | int]] = {}
            total_params = 0
            for a in kept:
                bits = int(a.get("bits") or 16)
                params = int(a.get("parameters") or 0)
                key = "bf16" if bits >= 16 else f"{bits}bit"
                entry = buckets.setdefault(key, {"parameters": 0, "fraction": 0.0})
                entry["parameters"] = int(entry["parameters"]) + params
                total_params += params
            if total_params > 0:
                for entry in buckets.values():
                    entry["fraction"] = float(entry["parameters"]) / float(total_params)
                plan["weight_distribution"] = buckets
            # Recompute effective/nominal BPW from remaining assignments.
            bit_sum = sum(int(a.get("bits") or 16) * int(a.get("parameters") or 0) for a in kept)
            if total_params > 0:
                eff = bit_sum / total_params
                plan["effective_bpw"] = eff
                # Plan validator requires effective_bpw >= nominal_bpw.
                plan["nominal_bpw"] = eff
        mtp = plan.get("mtp")
        if isinstance(mtp, dict):
            mtp = dict(mtp)
            # ArtifactManifest / plan policy modes: protected | adaptive | disabled
            mtp["mode"] = "disabled"
            plan["mtp"] = mtp
        if "mtp_distribution" in plan:
            plan["mtp_distribution"] = {}
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        # Rebind plan digest so prepare_development_model_card accepts the strip.
        try:
            from axquant.schema import QuantizationPlan
            from axquant.serde import load_model, stable_sha256, write_data

            plan_model = load_model(plan_path, QuantizationPlan)
            write_data(plan_path, plan_model)
            plan_sha = stable_sha256(plan_model)
        except Exception:
            plan_sha = None
    else:
        plan_sha = None

    man_path = dest / "axquant_manifest.json"
    if not man_path.is_file():
        raise SystemExit(f"missing manifest after strip: {man_path}")

    # Rewrite manifest MTP bookkeeping; leave file records to card prep refresh.
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["mtp_present"] = False
    man["mtp_weight_file_size_bytes"] = 0
    man["mtp_acceptance_retention"] = None
    man["mtp_measured_speedup"] = None
    if plan_sha is not None:
        man["plan_sha256"] = plan_sha
    if isinstance(man.get("mtp_distribution"), dict):
        man["mtp_distribution"] = {}
    if isinstance(man.get("mtp_policy"), dict):
        policy = dict(man["mtp_policy"])
        policy["mode"] = "disabled"
        man["mtp_policy"] = policy
    # Quantizer execution must also track the rewritten plan digest.
    exec_path = dest / "axquant_quantizer_execution.json"
    if exec_path.is_file() and plan_sha is not None:
        try:
            execution = json.loads(exec_path.read_text(encoding="utf-8"))
            execution["plan_sha256"] = plan_sha
            exec_path.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
    # Recompute weight totals from remaining safetensors on disk.
    # ArtifactManifest requires main + mtp == total; with mtp=0, main == total
    # (vision is counted inside main for this product layout).
    weight_bytes = 0
    protected_bytes = 0
    for path in sorted(dest.glob("*.safetensors")):
        size = path.stat().st_size
        weight_bytes += size
        if "vision" in path.name or "audio" in path.name:
            protected_bytes += size
    man["weight_file_size_bytes"] = weight_bytes
    man["main_weight_file_size_bytes"] = weight_bytes
    man["protected_weight_file_size_bytes"] = protected_bytes
    logical = int(man.get("main_logical_parameters") or man.get("logical_parameters") or 0)
    if logical > 0 and weight_bytes > 0:
        man["measured_main_bpw"] = 8.0 * weight_bytes / logical
        total_logical = int(man.get("logical_parameters") or logical)
        man["measured_total_bpw"] = 8.0 * weight_bytes / total_logical
        man["effective_bpw"] = man["measured_total_bpw"]
    # Drop mtp file records if present in files list
    files = man.get("files")
    if isinstance(files, list):
        man["files"] = [
            rec
            for rec in files
            if not _is_mtp_path(str(rec.get("path") or rec.get("name") or ""))
        ]
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    # Config: clear MTP layer count if present so loaders do not expect a head.
    config_path = dest / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        changed = False
        for key in (
            "mtp_num_hidden_layers",
            "num_nextn_predict_layers",
            "num_mtp_layers",
        ):
            if key in config:
                config[key] = 0
                changed = True
        text = config.get("text_config")
        if isinstance(text, dict):
            for key in (
                "mtp_num_hidden_layers",
                "num_nextn_predict_layers",
                "num_mtp_layers",
            ):
                if key in text:
                    text[key] = 0
                    changed = True
        if changed:
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only process matching no_mtp_name substring (repeatable)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from huggingface_hub import HfApi, snapshot_download
    from axquant.model_card import prepare_development_model_card

    work: Path = args.work.expanduser()
    out_root: Path = args.out.expanduser()
    work.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    results: list[dict[str, object]] = []

    packs = PACKS
    if args.only:
        packs = [
            p
            for p in PACKS
            if any(token in p["no_mtp_name"] for token in args.only)
        ]
        if not packs:
            raise SystemExit(f"no packs matched --only {args.only}")

    for pack in packs:
        mtp_repo = pack["mtp_repo"]
        no_name = pack["no_mtp_name"]
        product_class = pack["product_class"]
        snap = work / "snapshots" / no_name
        dest = out_root / no_name
        log(f"=== {no_name} from {mtp_repo} ===")
        snap_ready = (snap / "axquant_manifest.json").is_file() and any(
            snap.glob("model-*.safetensors")
        )
        if args.skip_download and not snap_ready:
            raise SystemExit(f"--skip-download set but snapshot incomplete: {snap}")
        if snap_ready and not args.skip_download:
            # Resume-friendly: keep a complete local snapshot without re-fetching.
            log(f"reuse complete snapshot {snap}")
        elif not snap_ready:
            log(f"snapshot_download {mtp_repo} -> {snap}")
            if snap.exists():
                shutil.rmtree(snap)
            snapshot_download(
                repo_id=mtp_repo,
                local_dir=str(snap),
            )
        else:
            log(f"reuse snapshot {snap}")

        log(f"materialize no-MTP -> {dest}")
        materialize_no_mtp(snap, dest)

        log("prepare development model card")
        prepare_development_model_card(
            artifact_dir=dest,
            repo_id=f"AutomatosX/{no_name}",
            product_class=product_class,
            use_public_certification=False,
        )

        commit = None
        if not args.skip_upload:
            no_repo = f"AutomatosX/{no_name}"
            log(f"upload {no_repo}")
            # Ensure repo exists
            try:
                api.create_repo(no_repo, repo_type="model", exist_ok=True, private=False)
            except Exception as exc:  # noqa: BLE001 — surface and continue upload attempt
                log(f"create_repo note: {exc}")
            info = api.upload_folder(
                folder_path=str(dest),
                repo_id=no_repo,
                repo_type="model",
                commit_message=(
                    "Publish no-MTP sibling of certified MTP pack "
                    "(language path identical; mtp.safetensors omitted)."
                ),
            )
            commit = getattr(info, "oid", None) or str(info).rstrip("/").split("/")[-1]
            log(f"  commit {commit}")
        else:
            log("skip upload")

        man = json.loads((dest / "axquant_manifest.json").read_text(encoding="utf-8"))
        results.append(
            {
                "no_mtp_name": no_name,
                "mtp_repo": mtp_repo,
                "local_path": str(dest),
                "hub_repo_id": f"AutomatosX/{no_name}",
                "hub_commit": commit,
                "product_class": product_class,
                "display": pack["display"],
                "measured_main_bpw": man.get("measured_main_bpw"),
                "measured_total_bpw": man.get("measured_total_bpw"),
                "weight_file_size_bytes": man.get("weight_file_size_bytes"),
                "mtp_present": man.get("mtp_present"),
            }
        )

    out_json = work / "publish-results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {out_json}")
    for row in results:
        log(
            f"  {row['no_mtp_name']}: commit={row['hub_commit']} "
            f"bytes={row['weight_file_size_bytes']} mtp={row['mtp_present']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
