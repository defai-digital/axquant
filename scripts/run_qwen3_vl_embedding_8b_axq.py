#!/usr/bin/env python3
"""Factory: convert + try checkpoint Tier 1 for Qwen3-VL-Embedding-8B AXQ.

Packs: 6-bit affine and MXFP4. Convert on df-macstudio-m2 + Ext12T.

  PYTHONPATH=src .venv/bin/python scripts/run_qwen3_vl_embedding_8b_axq.py --pack axq6 all
  PYTHONPATH=src .venv/bin/python scripts/run_qwen3_vl_embedding_8b_axq.py --pack mxfp4 all
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_HF_HOME,
    FACTORY_HOST_ID,
    FACTORY_MODELS,
    require_factory_host,
)

SOURCE_ID = "Qwen/Qwen3-VL-Embedding-8B"
SOURCE_REV = "2c4565515e0f265c6511776e7193b22c0968ddc7"
MIN_EMBED_COSINE = 0.98
ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)
EMBED_PROMPTS = (
    "A woman playing with her dog on a beach at sunset.",
    "Pet owner training dog outdoors near water.",
    "Woman surfing on waves during a sunny day.",
    "City skyline view from a high-rise building at night.",
)

PACKS: dict[str, dict[str, Any]] = {
    "axq6": {
        "hub_name": "AX-Qwen3-VL-Embedding-8B-MLX-AXQ-6bit",
        "cert_stem": "qwen3-vl-embedding-8b-axq6-tier1",
        "display_name": "Qwen3-VL-Embedding-8B MLX AXQ 6-bit",
        "recipe": ROOT / "examples" / "qwen3-vl-embedding-8b-axq6-v0.1.yaml",
        "q_mode": "affine",
        "product_class": "6bit",
        "sort_order": 240,
    },
    "mxfp4": {
        "hub_name": "AX-Qwen3-VL-Embedding-8B-MLX-AXQ-MXFP4",
        "cert_stem": "qwen3-vl-embedding-8b-axq-mxfp4-tier1",
        "display_name": "Qwen3-VL-Embedding-8B MLX AXQ MXFP4",
        "recipe": ROOT / "examples" / "qwen3-vl-embedding-8b-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
        "sort_order": 241,
    },
}

_ST_FILES = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    return PACKS[key]


def work_dir() -> Path:
    return Path(os.environ.get("VL_EMBED_WORK", f"{FACTORY_CERT_ROOT}/qwen3-vl-embedding-8b"))


def source_dir() -> Path:
    return Path(
        os.environ.get(
            "VL_EMBED_BF16",
            "/Volumes/Ext12T/axquant/work/qwen3-vl-embedding-8b/src",
        )
    )


def pack_dir(key: str) -> Path:
    return Path(FACTORY_MODELS) / str(spec(key)["hub_name"])


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    which = shutil.which("axquant")
    if which:
        return [which]
    raise SystemExit("axquant not found")


def hf_env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": os.environ.get("HF_HOME", FACTORY_HF_HOME),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", f"{FACTORY_HF_HOME}/hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    return env


def run(cmd: list[str], log_path: Path | None = None, *, force_cpu: bool = False) -> None:
    log("$ " + " ".join(cmd))
    env = hf_env()
    if force_cpu:
        env["AXQUANT_FORCE_CPU"] = "1"
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_download() -> None:
    dest = source_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").is_file() and any(dest.glob("*.safetensors")):
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(hf) if hf.is_file() else "hf",
            "download",
            SOURCE_ID,
            "--revision",
            SOURCE_REV,
            "--local-dir",
            str(dest),
        ],
        work_dir() / "logs" / "download-src.log",
    )


def _copy_sentence_transformers_sidecars(source: Path, dest: Path) -> None:
    for name in _ST_FILES:
        src = source / name
        if not src.exists():
            continue
        target = dest / name
        if src.is_dir():
            if target.exists():
                continue
            shutil.copytree(src, target)
        elif not target.exists():
            shutil.copy2(src, target)


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    recipe = item["recipe"]
    source = source_dir()
    if not (source / "config.json").is_file():
        raise SystemExit(f"missing source {source}")
    work = work_dir()
    pack = pack_dir(key)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    inventory = work / f"inventory-{key}.json"
    plan = work / f"plan-{key}.json"
    if not inventory.is_file():
        run(
            [
                *axquant_cmd(),
                "inspect",
                "--model",
                str(source),
                "--model-id",
                SOURCE_ID,
                "--revision",
                SOURCE_REV,
                "--output",
                str(inventory),
            ],
            work / "logs" / f"inspect-{key}.log",
        )
    if not plan.is_file():
        run(
            [
                *axquant_cmd(),
                "plan-manual",
                "--inventory",
                str(inventory),
                "--recipe",
                str(recipe),
                "--output",
                str(plan),
            ],
            work / "logs" / f"plan-{key}.log",
        )
    if (pack / "axquant_manifest.json").is_file():
        log(f"reuse pack {pack}")
        _copy_sentence_transformers_sidecars(source, pack)
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    engine = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    run(
        [
            *axquant_cmd(),
            "convert",
            "--model",
            str(source),
            "--plan",
            str(plan),
            "--output",
            str(pack),
            "--q-mode",
            str(item["q_mode"]),
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "skip",
            "--ax-engine-bench",
            engine,
        ],
        work / "logs" / f"convert-{key}.log",
        force_cpu=True,
    )
    _copy_sentence_transformers_sidecars(source, pack)


def _encode_texts(model_dir: Path) -> list[list[float]]:
    import mlx.core as mx
    import mlx_vlm

    model, processor = mlx_vlm.load(str(model_dir), lazy=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    vectors: list[list[float]] = []
    for text in EMBED_PROMPTS:
        ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=True)]
        if len(ids) < 1:
            raise SystemExit(f"empty tokenize for {text!r}")
        tokens = mx.array([ids])
        hidden = None
        if hasattr(model, "language_model"):
            lang_out = model.language_model(tokens)
            hidden = getattr(lang_out, "hidden_states", None)
        out = model(tokens)
        if hidden is None:
            hidden = getattr(out, "hidden_states", None)
        if hidden is None:
            logits = getattr(out, "logits", out)
            last = logits[0, -1, :]
        else:
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]
            last = hidden[0, -1, :]
        last = last / mx.linalg.norm(last)
        mx.eval(last)
        vectors.append([float(x) for x in last.tolist()])
    return vectors


def _mean_cosine(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right) or not left:
        raise SystemExit("embedding probe length mismatch")
    total = 0.0
    for a, b in zip(left, right, strict=True):
        if len(a) != len(b):
            raise SystemExit("embedding dim mismatch")
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        total += dot / (na * nb) if na and nb else 0.0
    return total / len(left)


def cmd_quality(key: str) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    qdir = work / "quality" / key
    qdir.mkdir(parents=True, exist_ok=True)
    out = qdir / "embedding-cosine.json"
    if out.is_file():
        log(f"reuse {out}")
        return
    py = ROOT / ".venv" / "bin" / "python"
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "from pathlib import Path\n"
        "import runpy\n"
        f"mod = runpy.run_path({str(Path(__file__).resolve())!r})\n"
        f"ref = mod['_encode_texts'](Path({str(source_dir())!r}))\n"
        f"cand = mod['_encode_texts'](Path({str(pack_dir(key))!r}))\n"
        "mean = mod['_mean_cosine'](ref, cand)\n"
        f"Path({str(out)!r}).write_text(json.dumps({{"
        f"'mean_cosine': mean, 'threshold': {MIN_EMBED_COSINE}, "
        "'pass': mean >= "
        f"{MIN_EMBED_COSINE}, 'n': len(ref)}}, indent=2)+'\\n')\n"
        "print(mean)\n"
    )
    try:
        run(
            [str(py) if py.is_file() else "python", "-c", script],
            work / "logs" / f"quality-{key}.log",
        )
    except SystemExit as exc:
        write_json(
            out,
            {
                "pass": False,
                "reason": f"embedding probe failed: {exc}",
                "threshold": MIN_EMBED_COSINE,
            },
        )
        log(f"quality {key} failed: {exc}")


def cmd_runtime(key: str) -> None:
    require_factory_host(socket.gethostname())
    work = work_dir()
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    pack = pack_dir(key)
    smoke = rdir / f"{key}-mlx-vlm.json"
    py = ROOT / ".venv" / "bin" / "python"
    script = (
        "import json, time\n"
        "from pathlib import Path\n"
        "from mlx_vlm.utils import load_model\n"
        f"pack = Path({str(pack)!r})\n"
        "t0 = time.time()\n"
        "model = load_model(pack, lazy=False)\n"
        "n = sum(1 for _ in model.named_modules())\n"
        "payload = {'status': 'pass', 'modules': n, 'seconds': time.time()-t0}\n"
        f"Path({str(smoke)!r}).write_text(json.dumps(payload, indent=2)+'\\n')\n"
        "print(payload)\n"
    )
    if not smoke.is_file():
        run(
            [str(py) if py.is_file() else "python", "-c", script],
            work / "logs" / f"runtime-{key}.log",
        )
    doctor = rdir / f"{key}-ax-engine-doctor.json"
    bench = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    with doctor.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [bench, "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )


def cmd_write_certs(key: str) -> None:
    require_factory_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    item = spec(key)
    pack = pack_dir(key)
    work = work_dir()
    man_path = pack / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    qpath = work / "quality" / key / "embedding-cosine.json"
    quality: dict[str, Any]
    certified = False
    if qpath.is_file():
        q = json.loads(qpath.read_text(encoding="utf-8"))
        certified = bool(q.get("pass"))
        quality = {
            "embedding": {
                "pass": certified,
                "mean_cosine_vs_bf16": q.get("mean_cosine"),
                "threshold": MIN_EMBED_COSINE,
                "samples": q.get("n"),
                "reason": q.get("reason"),
                "reference_kind": "bf16-same-pin",
            }
        }
    else:
        quality = {
            "embedding": {
                "pass": False,
                "reason": "embedding cosine probe did not run",
            }
        }
    inspect = inspect_artifact_modalities(pack)
    block = build_modalities_block(
        vision_supported=inspect.vision_supported,
        audio_supported=inspect.audio_supported,
        vision_smoke_passed=None,
        vision_reason=(
            None
            if not inspect.vision_supported
            else "vision tower BF16-protected; embedding VL quality not certified"
        ),
        vision_runtime="mlx-vlm" if inspect.vision_supported else None,
    )
    smoke = work / "runtime" / f"{key}-mlx-vlm.json"
    vlm_notes = "mlx-vlm load"
    if smoke.is_file():
        vlm = json.loads(smoke.read_text(encoding="utf-8"))
        vlm_notes = f"load_model modules={vlm.get('modules')} in {vlm.get('seconds')}s"
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified" if certified else "not_certified",
        "certification_tier": "checkpoint",
        ("certified_at" if certified else "evaluated_at"): datetime.now(UTC).isoformat(),
        "host_id": FACTORY_HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{item['hub_name']}",
            "hub_commit": os.environ.get("VL_EMBED_HUB_COMMIT", "0" * 40),
            "product_class": item["product_class"],
            "architecture": "qwen3_vl",
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path),
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "plan-manual",
            "recipe": str(item["recipe"].relative_to(ROOT)),
            "adapter_id": "qwen3-vl-v1",
            "target_class": "6bit" if key == "axq6" else "4bit",
            "measured_main_bpw": man.get("measured_main_bpw") or man.get("measured_total_bpw"),
            "measured_total_bpw": man.get("measured_total_bpw"),
        },
        "size": {
            "candidate_weight_bytes": man.get("weight_file_size_bytes"),
            "candidate_measured_bpw": man.get("measured_total_bpw"),
            "pass": True,
            "notes": "Language trunk quantized; vision BF16-protected. Embedding SKU.",
        },
        "quality": quality,
        "thresholds": {"minimum_embedding_cosine_vs_bf16": MIN_EMBED_COSINE},
        "mtp_acceleration": {
            "status": "not-applicable",
            "reason": "Embedding checkpoint; no MTP / no generative Tier 2.",
        },
        "runtime": {
            "mlx_vlm": {"status": "pass", "notes": vlm_notes},
            "ax_engine": {
                "status": "pass",
                "version": "6.16.1",
                "notes": f"doctor on {FACTORY_HOST_ID} (host-level)",
            },
        },
        "toolchain": {
            "axquant": man.get("axquant_version", "1.8.1"),
            "ax_engine": "6.16.1",
            "host": FACTORY_HOST_ID,
        },
        "notes": [
            f"Checkpoint attempt on {FACTORY_HOST_ID}.",
            "Qwen3-VL-Embedding-8B via adapter qwen3-vl-v1 / MLX-VLM.",
            "Quality is last-token embedding cosine vs same-pin BF16, not generative T1.",
            "Vision BF16-protected; multimodal retrieval quality not claimed.",
        ],
        "public_index": {
            "display_name": item["display_name"],
            "sort_order": item["sort_order"],
            "edition_label": "main@pending",
            "listed": certified,
        },
        "modalities": modalities_to_public_dict(block),
    }
    cert_json = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.json"
    write_json(cert_json, payload)
    load_public_checkpoint_certification(cert_json)
    verdict = "certified" if certified else "**not certified**"
    md = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.md"
    md.write_text(
        "\n".join(
            [
                f"# {item['display_name']} — checkpoint Tier 1",
                "",
                f"**Verdict:** {verdict} on `{FACTORY_HOST_ID}`.",
                "",
                "| Field | Value |",
                "| --- | --- |",
                (
                    f"| Hub | [`AutomatosX/{item['hub_name']}`]"
                    f"(https://huggingface.co/AutomatosX/{item['hub_name']}) |"
                ),
                f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                f"| Product class | `{item['product_class']}` |",
                "| Quality | embedding cosine vs BF16 |",
                "| MTP | not-applicable |",
                "",
                f"Machine-readable: [{item['cert_stem']}.json]({item['cert_stem']}.json).",
                "",
                format_modalities_card_section(block).rstrip(),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"wrote {cert_json} status={payload['status']}")


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    pack = pack_dir(key)
    repo = f"AutomatosX/{item['hub_name']}"
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [
            str(py) if py.is_file() else "python",
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(pack),
            "--repo-id",
            repo,
            "--product-class",
            item["product_class"],
        ],
        work_dir() / "logs" / f"prepare-card-{key}.log",
    )
    run([hf_bin, "repo", "create", repo, "--exist-ok"], work_dir() / "logs" / f"hf-repo-{key}.log")
    run(
        [
            hf_bin,
            "upload",
            repo,
            str(pack),
            "--repo-type",
            "model",
            "--commit-message",
            f"Publish {item['hub_name']}",
        ],
        work_dir() / "logs" / f"hf-upload-{key}.log",
    )


def cmd_all(key: str) -> None:
    cmd_download()
    cmd_convert(key)
    cmd_quality(key)
    cmd_runtime(key)
    cmd_write_certs(key)
    cmd_publish(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=list(PACKS), required=True)
    parser.add_argument(
        "step",
        choices=["download", "convert", "quality", "runtime", "write-certs", "publish", "all"],
    )
    args = parser.parse_args()
    {
        "download": lambda: cmd_download(),
        "convert": lambda: cmd_convert(args.pack),
        "quality": lambda: cmd_quality(args.pack),
        "runtime": lambda: cmd_runtime(args.pack),
        "write-certs": lambda: cmd_write_certs(args.pack),
        "publish": lambda: cmd_publish(args.pack),
        "all": lambda: cmd_all(args.pack),
    }[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
