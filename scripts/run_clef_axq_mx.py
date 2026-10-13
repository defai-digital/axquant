#!/usr/bin/env python3
"""Factory: Clef and Clef-Flash MLX AXQ MXFP4 and MXFP8 development packs.

Apache-2.0 sources (Cloudflare/clef, Cloudflare/clef-flash). The language
trunk is quantized. Vision, norms, the LM head, and the joint schema head stay
BF16. These are decision models: smoke them with clef_mlx.py, not text generation.

  PYTHONPATH=src .venv/bin/python scripts/run_clef_axq_mx.py --pack flash-mxfp4 all
  PYTHONPATH=src .venv/bin/python scripts/run_clef_axq_mx.py --pack all all
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    FACTORY_CERT_ROOT,
    FACTORY_HF_HOME,
    FACTORY_MODELS,
    require_factory_host,
)

SOURCES: dict[str, dict[str, str]] = {
    "clef": {
        "id": "Cloudflare/clef",
        "rev": "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
        "work": "clef",
    },
    "flash": {
        "id": "Cloudflare/clef-flash",
        "rev": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        "work": "clef-flash",
    },
}

ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)
LOADER = ROOT / "scripts" / "clef_mlx.py"

PACKS: dict[str, dict[str, Any]] = {
    "clef-mxfp4": {
        "size": "clef",
        "hub_name": "AX-Clef-27B-MLX-AXQ-MXFP4",
        "display_name": "Clef 27B MLX AXQ MXFP4",
        "recipe": ROOT / "examples" / "clef-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
    },
    "clef-mxfp8": {
        "size": "clef",
        "hub_name": "AX-Clef-27B-MLX-AXQ-MXFP8",
        "display_name": "Clef 27B MLX AXQ MXFP8",
        "recipe": ROOT / "examples" / "clef-axq-mxfp8-v0.1.yaml",
        "q_mode": "mxfp8",
        "product_class": "MXFP8",
    },
    "flash-mxfp4": {
        "size": "flash",
        "hub_name": "AX-Clef-Flash-9B-MLX-AXQ-MXFP4",
        "display_name": "Clef-Flash 9B MLX AXQ MXFP4",
        "recipe": ROOT / "examples" / "clef-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "product_class": "MXFP4",
    },
    "flash-mxfp8": {
        "size": "flash",
        "hub_name": "AX-Clef-Flash-9B-MLX-AXQ-MXFP8",
        "display_name": "Clef-Flash 9B MLX AXQ MXFP8",
        "recipe": ROOT / "examples" / "clef-axq-mxfp8-v0.1.yaml",
        "q_mode": "mxfp8",
        "product_class": "MXFP8",
    },
}

DECISION_MARKER = "## Decision model"
NOTICE = """This directory is a quantized MLX derivative of {source}.

Changes from the upstream BF16 release:
- Language attention and MLP weights were quantized to {q_mode} by AXQuant.
- Token embeddings are 8-bit affine. Norms, the vision tower, and the LM head stay BF16.
- joint_head.safetensors is the original BF16 joint schema head, copied unchanged.
- clef_mlx.py loads that head on the quantized backbone. It is an MLX port of the
  upstream joint_schema_model.py and does not require PyTorch.

The upstream work is Apache-2.0. See LICENSE (Copyright 2026 Alibaba Cloud).
"""
USAGE = """
## Decision model

Clef is not a chat model. `mlx_lm.generate` and `mlx_vlm.generate` load the backbone and
produce text that is not the model's output. A request is a state plus a schema of typed
questions (`noul`, `choice`, `score`). One prefill returns a probability for every allowed
option. The joint schema head in `joint_head.safetensors` is the original BF16 head.

```bash
pip install "mlx-vlm>=0.6.17" huggingface_hub pillow
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("{repo}")
sys.path.insert(0, path)
import clef_mlx

model = clef_mlx.load(path)
print(model.systemone({{
    "model": "{short}",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {{
        "department": {{
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {{"billing": "Payments or invoices", "technical": "Bugs or outages"}},
        }},
        "outage": {{"type": "noul", "instructions": "Is a service down?"}},
    }},
}}))
```

Images and videos use the same `images` / `videos` fields as the upstream model. Text generation
benchmarks do not measure this checkpoint. This package is development evidence: the factory smoke
checks that the joint head returns a probability for every option, and it does not claim Decision
Index parity with the BF16 release.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    return PACKS[key]


def work_dir(size: str) -> Path:
    name = SOURCES[size]["work"]
    return Path(os.environ.get(f"CLEF_{size.upper()}_WORK", f"{FACTORY_CERT_ROOT}/{name}"))


def source_dir(size: str) -> Path:
    return work_dir(size) / "src"


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
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
        # This link resets us.aws.cdn.hf.co when Xet ramps past a handful of
        # range gets. Keep Xet on, and keep the download concurrency cap.
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


JOINT_HEAD = "joint_head.safetensors"


def source_ready(size: str) -> bool:
    src = source_dir(size)
    parked = work_dir(size) / "decision-sidecar" / JOINT_HEAD
    return (
        (src / "config.json").is_file()
        and ((src / JOINT_HEAD).is_file() or parked.is_file())
        and any(src.glob("model-*.safetensors"))
    )


def _park_joint_head(size: str) -> None:
    """Move the joint head out of the snapshot while inspect and convert run.

    It is a BF16 sidecar, not a shard in model.safetensors.index.json. The
    inspector rejects Safetensors files the index does not name.
    """
    src = source_dir(size) / JOINT_HEAD
    parked = work_dir(size) / "decision-sidecar" / JOINT_HEAD
    if not src.is_file():
        return
    parked.parent.mkdir(parents=True, exist_ok=True)
    if parked.is_file():
        parked.unlink()
    shutil.move(src, parked)
    log(f"parked {JOINT_HEAD} outside the indexed snapshot")


def _restore_joint_head(size: str) -> None:
    src = source_dir(size) / JOINT_HEAD
    parked = work_dir(size) / "decision-sidecar" / JOINT_HEAD
    if src.is_file() or not parked.is_file():
        return
    shutil.move(parked, src)
    log(f"restored {JOINT_HEAD}")


def cmd_download(size: str) -> None:
    source = SOURCES[size]
    dest = source_dir(size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_ready(size):
        log(f"reuse source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    run(
        [
            str(hf) if hf.is_file() else "hf",
            "download",
            source["id"],
            "--revision",
            source["rev"],
            "--local-dir",
            str(dest),
        ],
        work_dir(size) / "logs" / "download-src.log",
    )


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    size = str(item["size"])
    source = SOURCES[size]
    src = source_dir(size)
    if not source_ready(size):
        raise SystemExit(f"missing source {src}; run download first")
    work = work_dir(size)
    pack = pack_dir(key)
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    _park_joint_head(size)
    try:
        _convert_indexed_snapshot(key, item, size, source, src, work, pack)
    finally:
        _restore_joint_head(size)


def _convert_indexed_snapshot(
    key: str,
    item: dict[str, Any],
    size: str,
    source: dict[str, str],
    src: Path,
    work: Path,
    pack: Path,
) -> None:
    inventory = work / f"inventory-{key}.json"
    plan = work / f"plan-{key}.json"
    if not inventory.is_file():
        run(
            [
                *axquant_cmd(),
                "inspect",
                "--model",
                str(src),
                "--model-id",
                source["id"],
                "--revision",
                source["rev"],
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
                str(item["recipe"]),
                "--output",
                str(plan),
            ],
            work / "logs" / f"plan-{key}.log",
        )
    if (pack / "axquant_manifest.json").is_file():
        log(f"reuse pack {pack}")
        return
    if pack.exists():
        raise SystemExit(f"incomplete pack dir exists: {pack}")
    engine = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    run(
        [
            *axquant_cmd(),
            "convert",
            "--model",
            str(src),
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
    )


def attach_decision_files(key: str) -> None:
    item = spec(key)
    size = str(item["size"])
    src = source_dir(size)
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    for name in ("joint_head.safetensors", "joint_head_config.json", "LICENSE"):
        target = pack / name
        if not target.is_file():
            shutil.copy2(src / name, target)
            log(f"copied {name} -> {target}")
    if not LOADER.is_file():
        raise SystemExit(f"missing loader {LOADER}")
    shutil.copy2(LOADER, pack / "clef_mlx.py")
    (pack / "NOTICE").write_text(
        NOTICE.format(source=SOURCES[size]["id"], q_mode=str(item["q_mode"]).upper()),
        encoding="utf-8",
    )


def cmd_smoke(key: str) -> None:
    require_factory_host(socket.gethostname())
    attach_decision_files(key)
    item = spec(key)
    size = str(item["size"])
    work = work_dir(size)
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    smoke = rdir / f"{key}-clef-mlx.json"
    if smoke.is_file():
        log(f"reuse smoke {smoke}")
        return
    pack = pack_dir(key)
    py = ROOT / ".venv" / "bin" / "python"
    script = r'''
import json, sys, time
from pathlib import Path
pack = Path(sys.argv[1])
out = Path(sys.argv[2])
sys.path.insert(0, str(pack))
import clef_mlx
t0 = time.time()
model = clef_mlx.load(pack)
response = model.systemone({
    "model": "clef",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
})
answers = response["answers"]
for key in ("department", "urgency", "outage"):
    if key not in answers:
        raise SystemExit(f"missing answer {key}: {response}")
dept = answers["department"]["probabilities"]
if abs(sum(dept.values()) - 1.0) > 0.02:
    raise SystemExit(f"choice probabilities do not sum to 1: {dept}")
if not 0.0 <= answers["outage"]["noul"] <= 1.0:
    raise SystemExit(f"noul out of range: {answers['outage']}")
payload = {"status": "pass", "seconds": time.time() - t0, "answers": answers}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload))
'''
    run(
        [str(py) if py.is_file() else "python", "-c", script, str(pack), str(smoke)],
        work / "logs" / f"smoke-{key}.log",
    )


def append_usage(key: str) -> None:
    item = spec(key)
    readme = pack_dir(key) / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if DECISION_MARKER in text:
        return
    short = "clef-flash" if item["size"] == "flash" else "clef"
    block = USAGE.format(repo=f"AutomatosX/{item['hub_name']}", short=short)
    readme.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    attach_decision_files(key)
    item = spec(key)
    size = str(item["size"])
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
            "--no-public-certification",
        ],
        work_dir(size) / "logs" / f"prepare-card-{key}.log",
    )
    append_usage(key)
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir(size) / "logs" / f"hf-repo-{key}.log",
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
            f"Publish {item['hub_name']}",
        ],
        work_dir(size) / "logs" / f"hf-upload-{key}.log",
    )


def cmd_all(key: str) -> None:
    cmd_download(str(spec(key)["size"]))
    cmd_convert(key)
    cmd_smoke(key)
    cmd_publish(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=[*sorted(PACKS), "all"], required=True)
    parser.add_argument(
        "stage",
        choices=["download", "convert", "smoke", "publish", "all"],
    )
    args = parser.parse_args()
    keys = list(PACKS) if args.pack == "all" else [args.pack]
    # Flash first: it is the smaller proof that the joint head survives quant.
    order = ["flash-mxfp4", "flash-mxfp8", "clef-mxfp4", "clef-mxfp8"]
    keys = [key for key in order if key in keys]
    stages = {
        "download": cmd_download,
        "convert": cmd_convert,
        "smoke": cmd_smoke,
        "publish": cmd_publish,
        "all": cmd_all,
    }
    for key in keys:
        if args.stage == "download":
            cmd_download(str(spec(key)["size"]))
        else:
            stages[args.stage](key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
