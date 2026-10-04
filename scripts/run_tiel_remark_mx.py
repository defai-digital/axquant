#!/usr/bin/env python3
"""Factory: remark Tiel-Coder-35B-A3B AXQ MXFP4+MTP and add MXFP8+MTP.

Development packs only (no Tier 1 certification). Run on the factory host:

  PYTHONPATH=src .venv/bin/python scripts/run_tiel_remark_mx.py --pack tiel-mxfp4 all
  ... (4 packs: tiel/cyber x {mxfp4, mxfp8})

MXFP4 packs overwrite the existing Hub repos with native-MXFP4 remarks (v0.2
recipes); the old Tier 1 wording is replaced by the development card until a
new certificate binds. MXFP8 packs are new repos.

Sources are the pinned already-quantized oQ6e repack checkpoints (no BF16
reference exists). They are small enough for the default Metal path; set
AXQUANT_FORCE_CPU=1 in the environment to force CPU instead.
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
    "tiel": {
        "id": "peculiar-ragdoll/Tiel-Coder-35B-A3B-MLX-oQ6e-MTP",
        "rev": "88625754ac91b542280a5602239ce6b2166366f0",
        "local": "",
    },
    "cyber": {
        "id": "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP",
        "rev": "a443d4e30fd5228942cb7695f916f4b14a88fae9",
        "local": "",
    },
}

ENGINE_BENCH = os.environ.get(
    "AX_ENGINE_BENCH",
    "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-bench",
)


def _pack(
    model: str,
    hub_name: str,
    recipe: str,
    q_mode: str,
    smoke: str = "mlx_vlm",
    legacy_4bit: bool = False,
    no_public_cert: bool = False,
) -> dict[str, Any]:
    return {
        "model": model,
        "hub_name": hub_name,
        "recipe": ROOT / "examples" / recipe,
        "q_mode": q_mode,
        "product_class": q_mode.upper(),
        "smoke": smoke,
        "legacy_4bit": legacy_4bit,
        "no_public_cert": no_public_cert,
    }


PACKS: dict[str, dict[str, Any]] = {
    # Remarks overwrite as development: never bind the stale in-tree Tier 1.
    "tiel-mxfp4": _pack(
        "tiel",
        "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP",
        "tiel-coder-35b-axq-mxfp4-mtp-v0.2.yaml",
        "mxfp4",
        no_public_cert=True,
    ),
    "cyber-mxfp4": _pack(
        "cyber",
        "AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP",
        "cyber-tiel-coder-35b-axq-mxfp4-mtp-v0.2.yaml",
        "mxfp4",
        no_public_cert=True,
    ),
    "tiel-mxfp8": _pack(
        "tiel",
        "AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP8-MTP",
        "tiel-coder-35b-axq-mxfp8-mtp-v0.1.yaml",
        "mxfp8",
    ),
    "cyber-mxfp8": _pack(
        "cyber",
        "AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP8-MTP",
        "cyber-tiel-coder-35b-axq-mxfp8-mtp-v0.1.yaml",
        "mxfp8",
    ),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    return PACKS[key]


def work_dir(model: str) -> Path:
    slug = {"tiel": "tiel-remark", "cyber": "cyber-remark"}[model]
    var = f"TIEL_{model.upper()}_WORK"
    return Path(os.environ.get(var, f"{FACTORY_CERT_ROOT}/{slug}"))


def source_dir(model: str) -> Path:
    local = SOURCES[model]["local"]
    if local:
        return Path(local)
    return work_dir(model) / "src"


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


def cmd_download(model: str) -> None:
    source = SOURCES[model]
    if source["local"]:
        dest = Path(source["local"])
        if not (dest / "config.json").is_file():
            raise SystemExit(f"missing verified local source {dest}")
        log(f"reuse verified local source {dest}")
        return
    dest = source_dir(model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").is_file() and any(dest.glob("*.safetensors")):
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
        work_dir(model) / "logs" / "download-src.log",
    )


def refuse_inflight_convert() -> None:
    """Refuse when another axquant convert holds factory memory (192GB shared)."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,args"], check=True, capture_output=True, text=True
        ).stdout
    except Exception as exc:
        log(f"warn: cannot scan for inflight converts: {exc}")
        return
    for line in out.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        if "axquant convert" in cmd and pid.strip() != str(os.getpid()):
            raise SystemExit(f"another axquant convert holds memory (pid {pid}): {cmd[:120]}")


def cmd_convert(key: str, *, force: bool = False, ignore_inflight: bool = False) -> None:
    require_factory_host(socket.gethostname())
    if not ignore_inflight:
        refuse_inflight_convert()
    item = spec(key)
    model = str(item["model"])
    source = SOURCES[model]
    recipe = item["recipe"]
    src = source_dir(model)
    if not (src / "config.json").is_file():
        raise SystemExit(f"missing source {src}; run download first")
    work = work_dir(model)
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
                str(src),
                "--model-id",
                source["id"],
                "--revision",
                source["rev"],
                # Pinned oQ6e repack sources are already quantized.
                "--allow-quantized",
                "--output",
                str(inventory),
            ],
            work / "logs" / f"inspect-{key}.log",
        )
    if not plan.is_file() or force:
        if force and plan.is_file():
            plan.unlink()
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
    if (pack / "axquant_manifest.json").is_file() and not force:
        log(f"reuse pack {pack}")
        return
    if pack.exists():
        if not force:
            raise SystemExit(f"incomplete pack dir exists: {pack}")
        backup = pack.with_name(pack.name + "-prev-remark")
        if backup.exists():
            raise SystemExit(f"backup dir already exists: {backup}")
        log(f"move previous pack {pack} -> {backup}")
        pack.rename(backup)
    engine = ENGINE_BENCH if Path(ENGINE_BENCH).is_file() else "ax-engine-bench"
    convert_cmd: list[str] = [
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
    ]
    if item["legacy_4bit"]:
        convert_cmd.append("--allow-legacy-4bit")
    convert_cmd += ["--ax-engine-manifest", "skip", "--ax-engine-bench", engine]
    run(convert_cmd, work / "logs" / f"convert-{key}.log")
    if not (pack / "mtp.safetensors").is_file():
        raise SystemExit(f"convert finished without mtp.safetensors: {pack}")


def cmd_runtime(key: str) -> None:
    require_factory_host(socket.gethostname())
    model = str(spec(key)["model"])
    work = work_dir(model)
    rdir = work / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    pack = pack_dir(key)
    smoke = rdir / f"{key}-mlx-vlm.json"
    py = ROOT / ".venv" / "bin" / "python"
    script = (
        "import json, time\n"
        "from pathlib import Path\n"
        "from mlx_vlm.utils import load\n"
        f"pack = Path({str(pack)!r})\n"
        "t0 = time.time()\n"
        "model, _ = load(pack, lazy=False)\n"
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


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    item = spec(key)
    model = str(item["model"])
    pack = pack_dir(key)
    repo = f"AutomatosX/{item['hub_name']}"
    py = ROOT / ".venv" / "bin" / "python"
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else "hf"
    card_cmd: list[str] = [
        str(py) if py.is_file() else "python",
        str(ROOT / "scripts" / "prepare_development_model_card.py"),
        "--artifact",
        str(pack),
        "--repo-id",
        repo,
        "--product-class",
        item["product_class"],
    ]
    if item["no_public_cert"]:
        card_cmd.append("--no-public-certification")
    run(card_cmd, work_dir(model) / "logs" / f"prepare-card-{key}.log")
    run(
        [hf_bin, "repo", "create", repo, "--exist-ok"],
        work_dir(model) / "logs" / f"hf-repo-{key}.log",
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
        work_dir(model) / "logs" / f"hf-upload-{key}.log",
    )


def cmd_all(key: str, *, force: bool = False, ignore_inflight: bool = False) -> None:
    cmd_download(str(spec(key)["model"]))
    cmd_convert(key, force=force, ignore_inflight=ignore_inflight)
    cmd_runtime(key)
    cmd_publish(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=sorted(PACKS), required=True)
    parser.add_argument(
        "--force", action="store_true", help="re-plan and reconvert over moves-aside"
    )
    parser.add_argument(
        "--ignore-inflight", action="store_true", help="convert alongside another axquant convert"
    )
    parser.add_argument(
        "stage",
        choices=["download", "convert", "runtime", "publish", "all"],
    )
    args = parser.parse_args()
    stages = {
        "download": lambda: cmd_download(str(spec(args.pack)["model"])),
        "convert": lambda: cmd_convert(
            args.pack, force=args.force, ignore_inflight=args.ignore_inflight
        ),
        "runtime": lambda: cmd_runtime(args.pack),
        "publish": lambda: cmd_publish(args.pack),
        "all": lambda: cmd_all(args.pack, force=args.force, ignore_inflight=args.ignore_inflight),
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
