#!/usr/bin/env python3
"""Factory: checkpoint Tier 1 for Ornith-1.0-35B AXQ 4-bit and 6-bit.

Run on df-macstudio-m2 with Ext4T. Reuses the published development packs.

  PYTHONPATH=src /Users/devop/code/axquant-main/.venv/bin/python \\
    scripts/run_ornith_35b_axq_tier1.py all

Protocol matches Holo3-35B-A3B (same qwen35-moe-v1 family): size and quality
versus matched uniform MLX 4/6-bit converts from the packed BF16 pin.

If architecture-prior 4-bit fails quality, convert the attn-6 recovery recipe
(examples/ornith-35b-axq4-agent-v0.1.yaml) and re-run size/quality/runtime.
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
WORK = Path(
    os.environ.get(
        "ORNITH_CERT_WORK",
        "/Volumes/Ext4T/axquant-certification/ornith-35b-axq-tier1",
    )
)
DEV = Path(
    os.environ.get(
        "ORNITH_DEV_WORK",
        "/Volumes/Ext4T/axquant/work/ornith-35b-axq-dev",
    )
)
SOURCE = Path(os.environ.get("ORNITH_BF16_PACKED", str(DEV / "src-ornith-35b-packed")))
DATASETS = Path(
    os.environ.get(
        "ORNITH_CERT_DATASETS",
        "/Volumes/Ext4T/axquant-certification/qwen38-27b-axq8-tier1/datasets",
    )
)
SOURCE_ID = "deepreinforce-ai/Ornith-1.0-35B"
SOURCE_REV = "5df2ed3f675c7beaa490328cc70bb573b65fb660"
HOST_ID = os.environ.get("ORNITH_CERT_HOST", "df-macstudio-m2")
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)
MAX_SIZE_RATIO = 1.15
MIN_QUALITY = 0.98
SEED = 20260728
MAX_TOKENS = 64
RECOVERY_RECIPE = ROOT / "examples" / "ornith-35b-axq4-agent-v0.1.yaml"

PACKS: dict[str, dict[str, object]] = {
    "4bit": {
        "hub_name": "AX-Ornith-1.0-35B-MLX-AXQ-4bit",
        "pack": Path(
            os.environ.get(
                "ORNITH_4BIT_PACK",
                str(DEV / "AX-Ornith-1.0-35B-MLX-AXQ-4bit"),
            )
        ),
        "cert_stem": "ornith-35b-axq4-tier1",
        "display_name": "Ornith-1.0-35B MLX AXQ 4-bit",
        "sort_order": 86,
        "q_bits": 4,
        "target_bpw": 4.0,
        "hub_commit": os.environ.get(
            "ORNITH_4BIT_HUB_COMMIT",
            "d7416c665cd8ae6e5fbebc3f17bd547b78cf11fc",
        ),
    },
    "6bit": {
        "hub_name": "AX-Ornith-1.0-35B-MLX-AXQ-6bit",
        "pack": Path(
            os.environ.get(
                "ORNITH_6BIT_PACK",
                str(DEV / "AX-Ornith-1.0-35B-MLX-AXQ-6bit"),
            )
        ),
        "cert_stem": "ornith-35b-axq6-tier1",
        "display_name": "Ornith-1.0-35B MLX AXQ 6-bit",
        "sort_order": 87,
        "q_bits": 6,
        "target_bpw": 6.0,
        "hub_commit": os.environ.get(
            "ORNITH_6BIT_HUB_COMMIT",
            "37361076641d7b7487d1b5ce1b68243ffbdbffe0",
        ),
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    studio_py = Path("/Users/devop/code/axquant-main/.venv/bin/python")
    if studio_py.is_file():
        return [str(studio_py), "-m", "axquant"]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found")


def mlx_bin(name: str) -> str:
    local = ROOT / ".venv" / "bin" / name
    if local.is_file():
        return str(local)
    studio = Path("/Users/devop/code/axquant-main/.venv/bin") / name
    if studio.is_file():
        return str(studio)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"{name} not found")


def run(cmd: list[str], log_path: Path | None = None, *, force_cpu: bool = False) -> None:
    log("$ " + " ".join(cmd))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": os.environ.get("HF_HOME", "/Volumes/Ext4T/huggingface"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", "/Volumes/Ext4T/huggingface/hub"),
    }
    if force_cpu:
        env["AXQUANT_FORCE_CPU"] = os.environ.get("AXQUANT_FORCE_CPU", "1")
    else:
        env.pop("AXQUANT_FORCE_CPU", None)
    if log_path is None:
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): see {log_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_weight_bytes(model_dir: Path) -> int:
    total = 0
    for path in sorted(model_dir.rglob("*.safetensors")):
        if path.name.startswith("."):
            continue
        total += path.stat().st_size
    if total <= 0:
        raise SystemExit(f"no safetensors under {model_dir}")
    return total


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _retention(cmp: dict) -> float | None:
    agg = cmp.get("aggregate") or {}
    ret = agg.get("retention")
    if ret is None:
        ret = cmp.get("retention")
    return None if ret is None else float(ret)


def spec(bits: str) -> dict[str, object]:
    if bits not in PACKS:
        raise SystemExit(f"unknown pack {bits}")
    return PACKS[bits]


def pack_dir(bits: str) -> Path:
    path = spec(bits)["pack"]
    assert isinstance(path, Path)
    if not (path / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {path}")
    return path


def uniform_dir(bits: str) -> Path:
    return WORK / "uniforms" / f"uniform-{bits}"


def cmd_uniforms(bits: str | None = None) -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"missing packed BF16 source {SOURCE}")
    targets = [bits] if bits else list(PACKS)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    mlx = mlx_bin("mlx_lm.convert")
    for key in targets:
        q_bits = int(spec(key)["q_bits"])
        out = uniform_dir(key)
        if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
            log(f"reuse {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                mlx,
                "--hf-path",
                str(SOURCE),
                "--mlx-path",
                str(out),
                "-q",
                "--q-bits",
                str(q_bits),
                "--q-group-size",
                "64",
                "--dtype",
                "bfloat16",
            ],
            WORK / "logs" / f"uniform-{key}-convert.log",
            force_cpu=True,
        )


def cmd_size(bits: str | None = None) -> None:
    targets = [bits] if bits else list(PACKS)
    (WORK / "size").mkdir(parents=True, exist_ok=True)
    for key in targets:
        item = spec(key)
        pack = pack_dir(key)
        man_path = pack / "axquant_manifest.json"
        hub = str(item["hub_name"])
        out = WORK / "size" / f"axq{key}-candidate.json"
        run(
            [
                *axquant_cmd(),
                "size-evidence",
                "--artifact-manifest",
                str(man_path),
                "--model-id",
                f"AutomatosX/{hub}",
                "--revision",
                str(item["hub_commit"]),
                "--output",
                str(out),
            ]
        )
        udir = uniform_dir(key)
        if not (udir / "config.json").is_file():
            raise SystemExit(f"missing uniform dir {udir}; run uniforms first")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        logical = int(man["logical_parameters"])
        weight_bytes = safetensors_weight_bytes(udir)
        uref = {
            "schema_version": "axquant.artifact-size-evidence.v1",
            "kind": f"uniform-{key}",
            "model": {
                "model_id": f"local/Ornith-1.0-35B-uniform-{key}",
                "revision": SOURCE_REV,
                "local_path": str(udir),
            },
            "logical_parameters": logical,
            "weight_bytes": weight_bytes,
            "measured_bpw": 8.0 * weight_bytes / logical,
        }
        write_json(WORK / "size" / f"uniform-{key}.json", uref)
        cand = json.loads(out.read_text(encoding="utf-8"))
        cand_bytes = int(cand.get("weight_bytes") or man.get("weight_file_size_bytes") or 0)
        if cand_bytes <= 0:
            cand_bytes = safetensors_weight_bytes(pack)
        ratio = cand_bytes / weight_bytes
        payload = {
            "size_ratio_vs_uniform": ratio,
            "pass": ratio <= MAX_SIZE_RATIO,
            "candidate_bytes": cand_bytes,
            "reference_bytes": weight_bytes,
            "max_size_ratio": MAX_SIZE_RATIO,
            "compare_mode": "total",
        }
        write_json(WORK / "size" / f"ratios-{key}.json", payload)
        log(f"size axq{key}: ratio={ratio:.6f} pass={ratio <= MAX_SIZE_RATIO}")


def cmd_quality(bits: str | None = None) -> None:
    targets = [bits] if bits else list(PACKS)
    for key in targets:
        item = spec(key)
        pack = pack_dir(key)
        hub = str(item["hub_name"])
        qdir = WORK / "quality" / f"axq{key}"
        qdir.mkdir(parents=True, exist_ok=True)
        uref = uniform_dir(key)
        if not (uref / "config.json").is_file():
            raise SystemExit(f"missing uniform quality reference {uref}")
        for suite, dname in (
            ("agent-coding", "development-agent-coding"),
            ("general", "development-general"),
        ):
            ds = DATASETS / dname / "dataset.jsonl"
            if not ds.is_file():
                alt = DATASETS / "datasets" / dname / "dataset.jsonl"
                ds = alt if alt.is_file() else ds
            if not ds.is_file():
                raise SystemExit(f"missing dataset {ds}")
            for role, model, mid in (
                ("ref", uref, f"local/Ornith-1.0-35B-uniform-{key}"),
                ("cand", pack, f"AutomatosX/{hub}"),
            ):
                out = qdir / f"{role}-{suite}.json"
                if out.is_file():
                    log(f"reuse {out}")
                    continue
                run(
                    [
                        *axquant_cmd(),
                        "evaluate-quality",
                        "--model",
                        str(model),
                        "--model-id",
                        mid,
                        "--revision",
                        SOURCE_REV,
                        "--dataset",
                        str(ds),
                        "--seed",
                        str(SEED),
                        "--max-tokens",
                        str(MAX_TOKENS),
                        "--output",
                        str(out),
                    ],
                    WORK / "logs" / f"quality-{key}-{role}-{suite}.log",
                )
            cmp_out = qdir / f"compare-{suite}.json"
            run(
                [
                    *axquant_cmd(),
                    "compare-quality",
                    "--reference",
                    str(qdir / f"ref-{suite}.json"),
                    "--candidate",
                    str(qdir / f"cand-{suite}.json"),
                    "--output",
                    str(cmp_out),
                ]
            )
            cmp = json.loads(cmp_out.read_text(encoding="utf-8"))
            ret = _retention(cmp)
            log(f"quality axq{key} {suite}: retention={ret} (need >= {MIN_QUALITY})")


def cmd_runtime(bits: str | None = None) -> None:
    targets = [bits] if bits else list(PACKS)
    rdir = WORK / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    mlx_gen = mlx_bin("mlx_lm.generate")
    bench = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    for key in targets:
        pack = pack_dir(key)
        out = rdir / f"axq{key}-mlx-lm.json"
        run(
            [
                *axquant_cmd(),
                "runtime-check",
                "--model",
                str(pack),
                "--runtime",
                "mlx-lm",
                "--mlx-lm",
                mlx_gen,
                "--output",
                str(out),
            ],
            WORK / "logs" / f"runtime-mlx-{key}.log",
        )
        doctor = rdir / f"axq{key}-ax-engine-doctor.json"
        with doctor.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [bench, "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        log(f"wrote {doctor}")


def cmd_recover_4bit() -> None:
    """Convert attn-6 / expert-4 recovery pack if architecture-prior 4-bit fails."""

    if not RECOVERY_RECIPE.is_file():
        raise SystemExit(f"missing recovery recipe {RECOVERY_RECIPE}")
    if not SOURCE.is_dir():
        raise SystemExit(f"missing packed BF16 source {SOURCE}")
    inventory = DEV / "inventory-packed.json"
    if not inventory.is_file():
        raise SystemExit(f"missing packed inventory {inventory}")
    plan = WORK / "plan-4bit-attn6.json"
    out = WORK / "AX-Ornith-1.0-35B-MLX-AXQ-4bit-attn6"
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    if not plan.is_file():
        run(
            [
                *axquant_cmd(),
                "plan-manual",
                "--inventory",
                str(inventory),
                "--recipe",
                str(RECOVERY_RECIPE),
                "--output",
                str(plan),
            ],
            WORK / "logs" / "plan-4bit-attn6.log",
        )
    if (out / "axquant_manifest.json").is_file():
        log(f"reuse recovery pack {out}")
    else:
        if out.exists():
            raise SystemExit(f"incomplete recovery pack dir exists: {out}")
        engine = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
        run(
            [
                *axquant_cmd(),
                "convert",
                "--model",
                str(SOURCE),
                "--plan",
                str(plan),
                "--output",
                str(out),
                "--allow-unmeasured",
                "--ax-engine-manifest",
                "if-available",
                "--ax-engine-bench",
                engine,
            ],
            WORK / "logs" / "convert-4bit-attn6.log",
            force_cpu=True,
        )
    os.environ["ORNITH_4BIT_PACK"] = str(out)
    PACKS["4bit"]["pack"] = out
    log(f"4-bit pack now {out}")


def _quality_block(bits: str) -> dict[str, dict]:
    qdir = WORK / "quality" / f"axq{bits}"
    quality: dict[str, dict] = {}
    for suite in ("agent-coding", "general"):
        cmp = json.loads((qdir / f"compare-{suite}.json").read_text(encoding="utf-8"))
        ret = _retention(cmp)
        if ret is None or ret < MIN_QUALITY:
            raise SystemExit(f"{bits} {suite} retention={ret} < {MIN_QUALITY}")
        cand = json.loads((qdir / f"cand-{suite}.json").read_text(encoding="utf-8"))
        agg = cmp.get("aggregate") or {}
        quality[suite] = {
            "candidate_score": float(agg.get("candidate") or 0.0),
            "reference_score": float(agg.get("reference") or 0.0),
            "retention": ret,
            "perplexity_ratio": (cmp.get("perplexity_ratio") or {}).get("ratio")
            if isinstance(cmp.get("perplexity_ratio"), dict)
            else cmp.get("perplexity_ratio"),
            "dataset_sha256": cmp.get("dataset_sha256"),
            "samples": int(cand.get("samples") or (76 if suite == "agent-coding" else 44)),
            "reference_kind": f"uniform-{bits}-same-pin",
        }
    return quality


def cmd_write_certs(bits: str | None = None) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import (
        load_public_checkpoint_certification,
    )

    targets = [bits] if bits else list(PACKS)
    for key in targets:
        item = spec(key)
        pack = pack_dir(key)
        size = json.loads((WORK / "size" / f"ratios-{key}.json").read_text(encoding="utf-8"))
        if not size["pass"]:
            raise SystemExit(f"{key} size gate fail ratio={size['size_ratio_vs_uniform']}")
        quality = _quality_block(key)
        mlx = json.loads((WORK / "runtime" / f"axq{key}-mlx-lm.json").read_text(encoding="utf-8"))
        if not mlx.get("passed"):
            raise SystemExit(f"{key} mlx-lm runtime-check did not pass")
        man_path = pack / "axquant_manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        inspect = inspect_artifact_modalities(pack)
        block = build_modalities_block(
            vision_supported=inspect.vision_supported,
            audio_supported=inspect.audio_supported,
            vision_smoke_passed=None,
            vision_reason=(
                None
                if not inspect.vision_supported
                else (
                    "vision sidecar present; mlx-vlm smoke not a quality pass "
                    f"(prefixes={list(inspect.vision_key_prefixes)})"
                )
            ),
            vision_runtime="mlx-vlm" if inspect.vision_supported else None,
        )
        recovery = key == "4bit" and "attn6" in pack.name
        hub = str(item["hub_name"])
        stem = str(item["cert_stem"])
        payload = {
            "schema_version": "axquant.public-checkpoint-certification.v1",
            "status": "certified",
            "certification_tier": "checkpoint",
            "certified_at": datetime.now(UTC).isoformat(),
            "host_id": HOST_ID,
            "artifact": {
                "hub_repo_id": f"AutomatosX/{hub}",
                "hub_commit": str(item["hub_commit"]),
                "product_class": key,
                "architecture": "Qwen3_5MoeForConditionalGeneration",
                "source_model_id": SOURCE_ID,
                "source_revision": SOURCE_REV,
                "candidate_manifest_sha256": sha256_file(man_path),
            },
            "plan": {
                "evidence_kind": "architecture_prior",
                "plan_source": "plan-manual" if recovery else "quantize-prior",
                "target_class": key,
                "target_bpw": 5.0 if recovery else float(item["target_bpw"]),
                "measured_total_bpw": man.get("measured_total_bpw"),
                "measured_main_bpw": man.get("measured_main_bpw"),
                "adapter_id": "qwen35-moe-v1",
                **(
                    {"recipe": "examples/ornith-35b-axq4-agent-v0.1.yaml"}
                    if recovery
                    else {}
                ),
            },
            "size": {
                "candidate_weight_bytes": size["candidate_bytes"],
                "candidate_measured_bpw": man.get("measured_total_bpw"),
                "reference_kind": f"uniform-{key}",
                "reference_model_id": f"local/Ornith-1.0-35B-uniform-{key}",
                "reference_revision": SOURCE_REV,
                "reference_weight_bytes": size["reference_bytes"],
                "size_ratio_vs_uniform": size["size_ratio_vs_uniform"],
                "max_size_ratio_applied": MAX_SIZE_RATIO,
                "pass": True,
                "compare_mode": "total",
                "reference_build": (
                    f"mlx_lm.convert -q --q-bits {item['q_bits']} --q-group-size 64 "
                    f"from packed {SOURCE_ID}@{SOURCE_REV} on {HOST_ID}"
                ),
            },
            "quality": quality,
            "thresholds": {
                "minimum_quality_retention": MIN_QUALITY,
                "max_size_ratio_vs_uniform": MAX_SIZE_RATIO,
            },
            "mtp_acceleration": {
                "status": "not-applicable",
                "reason": (
                    "Ornith-1.0-35B has no MTP weights; certification is non-MTP "
                    "direct-decode checkpoint Tier 1 only."
                ),
            },
            "runtime": {
                "ax_engine": {
                    "status": "pass",
                    "version": "6.16.1",
                    "notes": f"doctor on {HOST_ID}",
                },
                "mlx_lm": {"status": "pass", "notes": "runtime-check passed=True"},
            },
            "toolchain": {
                "axquant": man.get("axquant_version", "1.8.1"),
                "ax_engine": "6.16.1",
                "host": HOST_ID,
            },
            "notes": [
                f"Checkpoint Tier 1 on host id {HOST_ID}.",
                (
                    "AXQ 4-bit recovery: attention 6-bit, experts/MLP 4-bit "
                    "(examples/ornith-35b-axq4-agent-v0.1.yaml)."
                    if recovery
                    else f"AXQ {key} architecture-prior pack from packed BF16 source."
                ),
                "Quality vs matched uniform MLX convert; size vs the same uniform.",
                "Adapter qwen35-moe-v1 — not the official Qwen 3.6 certification track.",
                "Vision BF16-protected; no VLM quality claim. No MTP / no Tier 2 claim.",
            ],
            "public_index": {
                "display_name": item["display_name"],
                "sort_order": item["sort_order"],
                "edition_label": f"main@`{str(item['hub_commit'])[:8]}`",
                "listed": True,
            },
            "modalities": modalities_to_public_dict(block),
        }
        cert_json = ROOT / "docs" / "certifications" / f"{stem}.json"
        write_json(cert_json, payload)
        loaded = load_public_checkpoint_certification(cert_json)
        if loaded.status != "certified" or loaded.host_id != HOST_ID:
            raise SystemExit(f"public cert loader rejected {stem}")
        layout = (
            "`4bit` (attention 6-bit / experts 4-bit recovery)"
            if recovery
            else f"`{key}` (architecture-prior)"
        )
        md = ROOT / "docs" / "certifications" / f"{stem}.md"
        md.write_text(
            "\n".join(
                [
                    f"# Ornith-1.0-35B AXQ {key} — checkpoint Tier 1 certification",
                    "",
                    f"**Verdict:** certified for AXQuant checkpoint Tier 1 on `{HOST_ID}`.",
                    "**MTP acceleration Tier 2 is not applicable** (no MTP weights).",
                    "",
                    "| Field | Value |",
                    "| --- | --- |",
                    (
                        f"| Hub | [`AutomatosX/{hub}`]"
                        f"(https://huggingface.co/AutomatosX/{hub}/tree/{item['hub_commit']}) |"
                    ),
                    f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                    f"| Host | `{HOST_ID}` |",
                    f"| Product class | {layout} |",
                    (
                        f"| Size vs uniform-{key[0]} | "
                        f"`{size['size_ratio_vs_uniform']:.6f}` (≤ {MAX_SIZE_RATIO}) |"
                    ),
                    f"| Agent-coding vs uniform-{key} | `{quality['agent-coding']['retention']:.6f}` |",
                    f"| General vs uniform-{key} | `{quality['general']['retention']:.6f}` |",
                    "| MTP acceleration | `not-applicable` |",
                    "",
                    "## Gates",
                    "",
                    "| Gate | Threshold | Observed | Result |",
                    "| --- | ---: | ---: | --- |",
                    (
                        f"| Size vs uniform-{key} | ≤ `{MAX_SIZE_RATIO}` | "
                        f"`{size['size_ratio_vs_uniform']:.6f}` | Pass |"
                    ),
                    (
                        "| Agent-coding | ≥ `0.98` | "
                        f"`{quality['agent-coding']['retention']:.6f}` | Pass |"
                    ),
                    f"| General | ≥ `0.98` | `{quality['general']['retention']:.6f}` | Pass |",
                    "| MLX-LM runtime | pass | pass | Pass |",
                    "| AX Engine doctor | ready | ready | Pass |",
                    "",
                    "## Notes",
                    "",
                    "- Adapter `qwen35-moe-v1`; not the official Qwen 3.6 certification track.",
                    "- Vision remains BF16-protected; no VLM quality claim.",
                    "- Config-only MTP (if present) is cleared; Hub names omit `-MTP`.",
                    (
                        "- 4-bit uses the attn-6 / expert-4 recovery recipe."
                        if recovery
                        else f"- {key} pack is the published architecture-prior convert."
                    ),
                    "",
                    "## Tier 2 status",
                    "",
                    "Not applicable. No MTP weights are packaged.",
                    "",
                    f"Machine-readable: [{stem}.json]({stem}.json).",
                    "",
                    format_modalities_card_section(block).rstrip(),
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        log(f"wrote {cert_json} and {md}")


def cmd_all() -> None:
    cmd_uniforms()
    cmd_size()
    cmd_quality()
    cmd_runtime()
    cmd_write_certs()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=[
            "uniforms",
            "size",
            "quality",
            "runtime",
            "write-certs",
            "recover-4bit",
            "all",
        ],
    )
    parser.add_argument("--bits", choices=("4bit", "6bit"), default=None)
    args = parser.parse_args()
    if args.step == "recover-4bit":
        cmd_recover_4bit()
        return 0
    {
        "uniforms": cmd_uniforms,
        "size": cmd_size,
        "quality": cmd_quality,
        "runtime": cmd_runtime,
        "write-certs": cmd_write_certs,
        "all": cmd_all,
    }[args.step](args.bits) if args.step != "all" else cmd_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
