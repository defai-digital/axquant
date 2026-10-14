#!/usr/bin/env python3
"""Factory: EmbeddingGemma 2 MLX AXQ MXFP4 and MXFP8 development packs.

Apache-2.0 source google/embeddinggemma-2. Language attention and MLP are
quantized. Vision, audio, norms, and the final embedding projection stay BF16.
This is a retrieval model: do not claim generation, MTP, or MTEB scores.

  PYTHONPATH=src .venv/bin/python scripts/run_embeddinggemma2_axq_mx.py --pack mxfp4 convert
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_HF_HOME,
    FACTORY_MODELS,
    require_factory_host,
)
from axquant.naming import model_name  # noqa: E402

SOURCE_ID = "google/embeddinggemma-2"
SOURCE_REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"
BASE = "EmbeddingGemma-2"
NOTICE = """This directory is a quantized MLX derivative of google/embeddinggemma-2.

Changes from the upstream BF16 release:
- Language attention and MLP weights were quantized by AXQuant.
- Token embeddings and per-layer embedder projections use the embedding floor.
- Vision, audio, modality projections, norms, and the final embedding projection stay BF16.
- Activations are not quantized. Do not cast this model to float16.

The upstream checkpoint is Apache-2.0, Copyright Google DeepMind.
Deployments must follow the Gemma Prohibited Use Policy.
"""


def log(message: str) -> None:
    print(message, flush=True)


def work_dir() -> Path:
    return Path(os.environ.get("EGEMMA_WORK", f"{FACTORY_CERT_ROOT}/embeddinggemma-2"))


def source_dir() -> Path:
    return Path(os.environ.get("EGEMMA_SRC", str(work_dir() / "src")))


def python_bin() -> str:
    override = os.environ.get("EGEMMA_PYTHON")
    if override:
        return override
    side = work_dir() / ".venv" / "bin" / "python"
    if side.is_file():
        return str(side)
    local = ROOT / ".venv" / "bin" / "python"
    return str(local) if local.is_file() else sys.executable


def pack_dir(q_mode: str) -> Path:
    name = model_name(BASE, target_class=q_mode.upper(), include_mlx=True)
    return Path(FACTORY_MODELS) / name


def hf_env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": os.environ.get("HF_HOME", FACTORY_HF_HOME),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", f"{FACTORY_HF_HOME}/hub"),
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": os.environ.get(
            "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY", "8"
        ),
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": os.environ.get(
            "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY", "4"
        ),
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": os.environ.get(
            "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY", "2"
        ),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    env.pop("HF_HUB_DISABLE_XET", None)
    return env


def run(cmd: list[str], log_path: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    env = hf_env()
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
        work_dir() / "logs" / "download-xet.log",
    )


def cmd_convert(q_mode: str) -> None:
    require_factory_host(socket.gethostname())
    src = source_dir()
    if not (src / "config.json").is_file():
        raise SystemExit(f"missing source {src}; run download first")
    work = work_dir()
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    inventory = work / f"inventory-{q_mode}.json"
    plan = work / f"plan-{q_mode}.json"
    pack = pack_dir(q_mode)
    py = python_bin()
    axquant = [py, "-m", "axquant"]
    if not inventory.is_file():
        run(
            [
                *axquant,
                "inspect",
                "--model",
                str(src),
                "--model-id",
                SOURCE_ID,
                "--revision",
                SOURCE_REV,
                "--output",
                str(inventory),
            ],
            work / "logs" / f"inspect-{q_mode}.log",
        )
    if not plan.is_file():
        run(
            [
                *axquant,
                "plan-manual",
                "--inventory",
                str(inventory),
                "--recipe",
                str(ROOT / "examples" / f"embeddinggemma-2-axq-{q_mode}-v0.1.yaml"),
                "--output",
                str(plan),
            ],
            work / "logs" / f"plan-{q_mode}.log",
        )
    if (pack / "axquant_manifest.json").is_file():
        log(f"reuse pack {pack}")
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    run(
        [
            *axquant,
            "convert",
            "--model",
            str(src),
            "--plan",
            str(plan),
            "--output",
            str(pack),
            "--q-mode",
            q_mode,
            "--allow-unmeasured",
            "--ax-engine-manifest",
            "skip",
        ],
        work / "logs" / f"convert-{q_mode}.log",
    )
    (pack / "NOTICE").write_text(NOTICE, encoding="utf-8")


def cmd_smoke(q_mode: str) -> None:
    require_factory_host(socket.gethostname())
    pack = pack_dir(q_mode)
    out = work_dir() / "runtime" / f"{q_mode}-smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    script = r"""
import json, sys
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
from mlx_vlm.embedding_loader import load_embedding_model

pack, source, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
texts = [
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Mars is known as the Red Planet.",
    "title: none | text: Venus is often called Earth's twin.",
]
tokenizer = AutoTokenizer.from_pretrained(pack)
inputs = tokenizer(texts, padding=True, return_tensors="np")
arrays = {key: mx.array(value) for key, value in inputs.items()}

def embed(path):
    model = load_embedding_model(path)
    vectors = model(**arrays).text_embeds
    mx.eval(vectors)
    return vectors

quant = embed(pack)
norms = mx.linalg.norm(quant, axis=-1)
finite = bool(mx.all(mx.isfinite(quant)).item())
unit = bool(mx.all(mx.abs(norms - 1) < 1e-3).item())
rank_ok = bool((quant[0] @ quant[1]) > (quant[0] @ quant[2]))
payload = {
    "status": "fail",
    "finite": finite,
    "unit_norm": unit,
    "shape": list(quant.shape),
    "mars_ranks_above_venus": rank_ok,
}
try:
    ref = embed(source)
    cosine = mx.sum(quant * ref, axis=-1)
    mx.eval(cosine)
    payload["min_cosine_vs_bf16"] = float(mx.min(cosine).item())
except Exception as exc:
    payload["bf16_reference_error"] = str(exc)
payload["status"] = (
    "pass"
    if finite and unit and quant.shape == (3, 768) and rank_ok
    and payload.get("min_cosine_vs_bf16", 0) >= 0.90
    else "fail"
)
dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload))
if payload["status"] != "pass":
    raise SystemExit(1)
"""
    smoke_path = work_dir() / "runtime" / f"smoke-{q_mode}.py"
    smoke_path.write_text(script, encoding="utf-8")
    run(
        [python_bin(), str(smoke_path), str(pack), str(source_dir()), str(out)],
        work_dir() / "logs" / f"smoke-{q_mode}.log",
    )


def cmd_publish(q_mode: str) -> None:
    require_factory_host(socket.gethostname())
    pack = pack_dir(q_mode)
    name = pack.name
    repo = f"AutomatosX/{name}"
    py = python_bin()
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    run(
        [
            py,
            str(ROOT / "scripts" / "prepare_development_model_card.py"),
            "--artifact",
            str(pack),
            "--repo-id",
            repo,
            "--product-class",
            q_mode.upper(),
            "--no-public-certification",
        ],
        work_dir() / "logs" / f"prepare-card-{q_mode}.log",
    )
    if not (pack / "NOTICE").is_file():
        (pack / "NOTICE").write_text(NOTICE, encoding="utf-8")
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir() / "logs" / f"hf-repo-{q_mode}.log",
    )
    run(
        [
            hf_bin,
            "upload",
            repo,
            str(pack),
            "--repo-type",
            "model",
            "--commit-message",
            f"Publish {name}",
        ],
        work_dir() / "logs" / f"hf-upload-{q_mode}.log",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=["mxfp4", "mxfp8", "all"], required=True)
    parser.add_argument("stage", choices=["download", "convert", "smoke", "publish", "all"])
    args = parser.parse_args()
    modes = ["mxfp4", "mxfp8"] if args.pack == "all" else [args.pack]
    if args.stage in {"download", "all"}:
        cmd_download()
    if args.stage == "download":
        return 0
    for mode in modes:
        if args.stage in {"convert", "all"}:
            cmd_convert(mode)
        if args.stage in {"smoke", "all"}:
            cmd_smoke(mode)
        if args.stage in {"publish", "all"}:
            cmd_publish(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
