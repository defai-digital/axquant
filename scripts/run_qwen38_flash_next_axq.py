#!/usr/bin/env python3
"""Factory convert + Hub publish for Qwen3.8-Flash-Next AXQ packs.

Development evidence only. Does not certify. Does not run quality eval.

Host: um-macstudio-m2 (hostname df-macstudio-m2) + Ext16TR0.
Hub download uses Hugging Face Xet high-performance (not hf_transfer).

Idempotent stages: bootstrap → download → inspect → convert → publish.

Usage (on the convert host, typically via the screen/nohup launcher):

  PYTHONPATH=src /path/to/venv/bin/python scripts/run_qwen38_flash_next_axq.py all
  PYTHONPATH=src .../python scripts/run_qwen38_flash_next_axq.py download
  PYTHONPATH=src .../python scripts/run_qwen38_flash_next_axq.py convert --pack axq4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import require_factory_host  # noqa: E402

SOURCE_ID = "Qwen/Qwen3.8-Flash-Next"
SOURCE_REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SHARD_COUNT = 131
HUB_OWNER = "AutomatosX"

EXT = Path(os.environ.get("FLASH_NEXT_EXT", "/Volumes/Ext16TR0"))
HF_HOME = Path(os.environ.get("HF_HOME", str(EXT / "huggingface")))
VENV = Path(os.environ.get("FLASH_NEXT_VENV", str(EXT / "axquant-venv")))
WORK = Path(os.environ.get("FLASH_NEXT_WORK", str(EXT / "axquant/work/qwen38-flash-next")))
SOURCE = Path(os.environ.get("FLASH_NEXT_SRC", str(EXT / "models/Qwen3.8-Flash-Next")))
MODELS = Path(os.environ.get("FLASH_NEXT_MODELS", str(EXT / "models")))

PACKS: dict[str, dict[str, Any]] = {
    "axq4": {
        "hub_name": "AX-Qwen3.8-Flash-Next-MLX-AXQ-4bit-MTP",
        "recipe": ROOT / "examples" / "qwen38-flash-next-axq4-v0.1.yaml",
        "q_mode": "affine",
        "label": "4-bit affine",
    },
    "mxfp4": {
        "hub_name": "AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP",
        "recipe": ROOT / "examples" / "qwen38-flash-next-axq-mxfp4-v0.1.yaml",
        "q_mode": "mxfp4",
        "label": "MXFP4",
    },
    "axq6": {
        "hub_name": "AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP",
        "recipe": ROOT / "examples" / "qwen38-flash-next-axq6-v0.1.yaml",
        "q_mode": "affine",
        "label": "6-bit affine",
    },
    "axq2": {
        "hub_name": "AX-Qwen3.8-Flash-Next-MLX-AXQ-2bit-MTP",
        "recipe": ROOT / "examples" / "qwen38-flash-next-axq2-v0.1.yaml",
        "q_mode": "affine",
        "label": "2-bit experimental",
        "experimental_2bit": True,
    },
}


class ConvertInterrupted(SystemExit):
    """Raised when a child convert is SIGTERM/SIGINT/SIGKILL'd.

    ``cmd_remaining`` must not treat this as a per-pack failure and start the
    next SKU — that is how a stopped MXFP4 job used to launch AXQ4.
    """


def log(msg: str) -> None:
    print(msg, flush=True)


def leftover_staging_dirs(pack: Path) -> list[Path]:
    """Temporary convert trees left behind when the process is SIGKILL'd.

    ``convert_model`` stages under ``.{pack.name}.<rand>/`` next to the output.
    ``finally: rmtree`` does not run after SIGKILL, so factory restarts must
    delete these before the next convert.
    """

    if pack.parent.is_dir():
        return sorted(path for path in pack.parent.glob(f".{pack.name}.*") if path.is_dir())
    return []


def remove_leftover_staging(pack: Path) -> None:
    for path in leftover_staging_dirs(pack):
        log(f"removing leftover convert staging {path}")
        shutil.rmtree(path, ignore_errors=True)


def convert_exit_was_signaled(returncode: int) -> bool:
    if returncode < 0:
        return True
    return returncode in {130, 137, 143}


def find_inflight_flash_next_converts(*, pid_self: int | None = None) -> list[tuple[int, str]]:
    """Return PIDs already converting this Flash-Next source or pack."""

    pid_self = os.getpid() if pid_self is None else pid_self
    try:
        raw = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    tokens = (
        "Qwen3.8-Flash-Next",
        "qwen38-flash-next",
        *(str(item["hub_name"]) for item in PACKS.values()),
    )
    found: list[tuple[int, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_s, _, command = stripped.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == pid_self:
            continue
        if "-m axquant convert" not in command and "axquant convert" not in command:
            continue
        if any(token in command for token in tokens):
            found.append((pid, command))
    return found


def refuse_inflight_convert() -> None:
    inflight = find_inflight_flash_next_converts()
    if inflight:
        pid, command = inflight[0]
        raise SystemExit(f"Flash-Next convert already running (pid {pid}): {command}")


def shard_name(index: int) -> str:
    return f"model-{index:05d}-of-{SHARD_COUNT:05d}.safetensors"


_SIDECARS = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def expected_source_files() -> list[str]:
    return [*_SIDECARS, *[shard_name(i) for i in range(1, SHARD_COUNT + 1)]]


def clear_incomplete(source: Path) -> None:
    cache = source / ".cache"
    if not cache.exists():
        return
    for path in cache.rglob("*.incomplete"):
        log(f"removing incomplete {path}")
        path.unlink(missing_ok=True)


def run_timed(cmd: list[str], log_path: Path, *, timeout_s: int) -> int:
    env = hf_env()
    require_xet_env(env)
    log("$ " + " ".join(cmd) + f"  [timeout {timeout_s}s]")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + f"  [timeout {timeout_s}s]\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return int(proc.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            log(f"timed out after {timeout_s}s; killing pid {proc.pid}")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return 124


def hf_env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep),
        "HF_HOME": str(HF_HOME),
        "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_HUB_CACHE": str(HF_HOME / "hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": str(HF_HOME / "xet"),
        "HF_XET_RECONSTRUCT_WRITE_ENABLED": os.environ.get(
            "HF_XET_RECONSTRUCT_WRITE_ENABLED", "1"
        ),
        # CAS reconstruction I/O errors are less frequent at lower fan-out.
        "HF_XET_NUM_CONCURRENT_RANGE_GETS": os.environ.get(
            "HF_XET_NUM_CONCURRENT_RANGE_GETS", "4"
        ),
        # 180B Flash-Next Metal quantize times out; factory converts on CPU.
        "AXQUANT_FORCE_CPU": os.environ.get("AXQUANT_FORCE_CPU", "1"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.pop("HF_HUB_DISABLE_XET", None)
    return env


def require_xet_env(env: dict[str, str]) -> None:
    if env.get("HF_XET_HIGH_PERFORMANCE") != "1":
        raise SystemExit("HF_XET_HIGH_PERFORMANCE must be 1 for Hub downloads")
    if env.get("HF_HUB_ENABLE_HF_TRANSFER"):
        raise SystemExit("HF_HUB_ENABLE_HF_TRANSFER must be unset (use Xet, not hf_transfer)")


def run(
    cmd: list[str],
    log_path: Path | None = None,
    *,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    env = hf_env()
    if extra_env:
        env.update(extra_env)
    require_xet_env(env)
    log("$ " + " ".join(cmd))
    if log_path is None:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        returncode = _wait_forwarding_stop_signals(proc)
        if convert_exit_was_signaled(returncode):
            raise ConvertInterrupted(f"interrupted ({returncode})")
        if check and returncode != 0:
            raise SystemExit(f"command failed ({returncode})")
        return int(returncode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env
        )
        returncode = _wait_forwarding_stop_signals(proc)
    if convert_exit_was_signaled(returncode):
        raise ConvertInterrupted(f"interrupted ({returncode}): see {log_path}")
    if check and returncode != 0:
        raise SystemExit(f"command failed ({returncode}): see {log_path}")
    return int(returncode)


def _wait_forwarding_stop_signals(proc: subprocess.Popen[Any]) -> int:
    """Wait for *proc*, forwarding SIGTERM/SIGINT so a stopped driver kills convert."""

    interrupted = False

    def _forward(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            return

    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    try:
        returncode = int(proc.wait())
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    if interrupted and not convert_exit_was_signaled(returncode):
        return -signal.SIGTERM
    return returncode


def venv_python() -> Path:
    return VENV / "bin" / "python"


def axquant_cmd() -> list[str]:
    axq = VENV / "bin" / "axquant"
    py = venv_python()
    if axq.is_file() and py.is_file():
        return [str(py), "-m", "axquant"]
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    raise SystemExit("axquant not found; run bootstrap first")


def hf_bin() -> str:
    candidate = VENV / "bin" / "hf"
    if candidate.is_file():
        return str(candidate)
    which = shutil.which("hf")
    if which:
        return which
    brew = Path("/opt/homebrew/bin/hf")
    if brew.is_file():
        return str(brew)
    raise SystemExit("hf CLI not found")


def uv_bin() -> str:
    which = shutil.which("uv")
    if which:
        return which
    local = Path.home() / ".local" / "bin" / "uv"
    if local.is_file():
        return str(local)
    raise SystemExit("uv not found")


def source_complete(source: Path) -> tuple[bool, str]:
    if not source.is_dir():
        return False, f"missing directory {source}"
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        if not (source / name).is_file():
            return False, f"missing {name}"
    missing = [
        shard_name(i) for i in range(1, SHARD_COUNT + 1) if not (source / shard_name(i)).is_file()
    ]
    if missing:
        return False, f"missing {len(missing)} shards (e.g. {missing[0]})"
    incomplete: list[Path] = []
    cache = source / ".cache"
    if cache.exists():
        incomplete = list(cache.rglob("*.incomplete"))
    if incomplete:
        return False, f"{len(incomplete)} incomplete download parts remain"
    return True, "ok"


def pack_dir(key: str) -> Path:
    return MODELS / str(PACKS[key]["hub_name"])


def write_readme(pack: Path, key: str, *, measured_bpw: str | None) -> None:
    item = PACKS[key]
    repo = f"{HUB_OWNER}/{item['hub_name']}"
    bpw_line = measured_bpw or "see axquant_manifest.json"
    text = f"""---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/{SOURCE_ID}/blob/main/LICENSE
library_name: mlx
base_model: {SOURCE_ID}
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- apple-silicon
- quantized
- mixed-precision
- axquant
- axq
- development
- qwen4-exp
- qwen3.8-flash-next
---

# {item["hub_name"]}

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted from
[`{SOURCE_ID}`](https://huggingface.co/{SOURCE_ID}/tree/{SOURCE_REV}) (`qwen4_exp`).

> **Development evidence — not a certified AXQuant release.** Conversion and
> artifact-integrity records only. No quality, long-context, or MTP-speed claim.

| Property | Value |
| --- | --- |
| Base model | `{SOURCE_ID}` |
| Source revision | `{SOURCE_REV}` |
| Product family | `qwen4-exp` |
| Adapter | `qwen4-exp-v1` |
| Recipe | `{item["recipe"].name}` |
| Quant lane | {item["label"]} |
| Measured BPW | {bpw_line} |
| Vision | BF16-protected |
| PLE / n-gram embeddings | embedding-floor (8-bit affine) |
| MTP | packaged `mtp.safetensors` (BF16, not a speed claim) |
| Runtime | MLX-VLM (`qwen4_exp`); AX Engine support is not claimed |

Load with a mlx-vlm build that includes `models.qwen4_exp`.
"""
    (pack / "README.md").write_text(text, encoding="utf-8")


def cmd_bootstrap() -> None:
    require_factory_host(socket.gethostname())
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "xet").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    uv = uv_bin()
    if not venv_python().is_file():
        log(f"creating venv {VENV}")
        run(
            [uv, "python", "install", "3.12"],
            WORK / "logs" / "bootstrap.log",
        )
        run(
            [uv, "venv", str(VENV), "--python", "3.12"],
            WORK / "logs" / "bootstrap.log",
        )
    py = str(venv_python())
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            py,
            "-e",
            f"{ROOT}[mlx]",
            "--upgrade",
            "hf-xet",
        ],
        WORK / "logs" / "bootstrap.log",
    )
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            py,
            "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git",
        ],
        WORK / "logs" / "bootstrap.log",
    )
    probe = (
        "import hf_xet, mlx_vlm, pathlib, sys; "
        "root = pathlib.Path(mlx_vlm.__file__).parent; "
        "q4 = root / 'models' / 'qwen4_exp'; "
        "assert q4.is_dir(), f'mlx-vlm missing qwen4_exp at {root}'; "
        "print('hf-xet', getattr(hf_xet, '__file__', 'ok')); "
        "print('mlx_vlm qwen4_exp', q4)"
    )
    run([py, "-c", probe], WORK / "logs" / "bootstrap.log")
    log("bootstrap ok")


def cmd_download() -> None:
    require_factory_host(socket.gethostname())
    env = hf_env()
    require_xet_env(env)
    SOURCE.mkdir(parents=True, exist_ok=True)
    ok, reason = source_complete(SOURCE)
    if ok:
        log(f"reuse complete source {SOURCE}")
        return
    log(f"Xet per-file download {SOURCE_ID}@{SOURCE_REV} -> {SOURCE} ({reason})")
    py = venv_python()
    if py.is_file():
        run(
            [
                str(py),
                "-c",
                "import hf_xet, os, sys; "
                "assert os.environ.get('HF_XET_HIGH_PERFORMANCE')=='1'; "
                "print('hf-xet', hf_xet.__file__)",
            ],
            WORK / "logs" / "download-xet-preflight.log",
        )
    file_attempts = int(os.environ.get("FLASH_NEXT_FILE_ATTEMPTS", "6"))
    sidecar_timeout = int(os.environ.get("FLASH_NEXT_SIDECAR_TIMEOUT", "180"))
    shard_timeout = int(os.environ.get("FLASH_NEXT_SHARD_TIMEOUT", "900"))
    missing = [name for name in expected_source_files() if not (SOURCE / name).is_file()]
    log(f"{len(missing)} files remaining")
    for name in missing:
        dest = SOURCE / name
        timeout_s = shard_timeout if name.startswith("model-") else sidecar_timeout
        for attempt in range(1, file_attempts + 1):
            if dest.is_file():
                break
            clear_incomplete(SOURCE)
            log(f"Xet get {name} attempt {attempt}/{file_attempts}")
            code = run_timed(
                [
                    hf_bin(),
                    "download",
                    SOURCE_ID,
                    name,
                    "--revision",
                    SOURCE_REV,
                    "--local-dir",
                    str(SOURCE),
                    "--max-workers",
                    "1",
                ],
                WORK / "logs" / "download-src.log",
                timeout_s=timeout_s,
            )
            if dest.is_file():
                log(f"got {name} ({dest.stat().st_size} bytes)")
                break
            log(f"missing {name} after attempt {attempt} (exit {code})")
            time.sleep(min(60, 10 * attempt))
        if not dest.is_file():
            raise SystemExit(f"failed to download {name} after {file_attempts} Xet attempts")
    ok, reason = source_complete(SOURCE)
    if not ok:
        raise SystemExit(f"source still incomplete: {reason}")
    log(f"source ready: {SOURCE}")


def cmd_inspect() -> None:
    require_factory_host(socket.gethostname())
    ok, reason = source_complete(SOURCE)
    if not ok:
        raise SystemExit(f"source not complete: {reason}")
    inventory = WORK / "inventory.json"
    if inventory.is_file():
        log(f"reuse inventory {inventory}")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    run(
        [
            *axquant_cmd(),
            "inspect",
            "--model",
            str(SOURCE),
            "--model-id",
            SOURCE_ID,
            "--revision",
            SOURCE_REV,
            "--output",
            str(inventory),
        ],
        WORK / "logs" / "inspect.log",
    )


def cmd_convert(key: str) -> None:
    require_factory_host(socket.gethostname())
    refuse_inflight_convert()
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    item = PACKS[key]
    recipe = Path(item["recipe"])
    if not recipe.is_file():
        raise SystemExit(f"missing recipe {recipe}")
    inventory = WORK / "inventory.json"
    if not inventory.is_file():
        raise SystemExit("missing inventory.json; run inspect first")
    pack = pack_dir(key)
    if (pack / "axquant_manifest.json").is_file():
        log(f"reuse pack {pack}")
        return
    if pack.exists():
        log(f"removing incomplete pack {pack}")
        shutil.rmtree(pack)
    remove_leftover_staging(pack)
    plan = WORK / f"plan-{key}.json"
    # Always rewrite the plan. Reusing yesterday's JSON after a recipe or
    # classifier change is how factory converts kept the wrong group size.
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
        WORK / "logs" / f"plan-{key}.log",
    )
    extra: dict[str, str] = {}
    if item.get("experimental_2bit"):
        extra["AX_ENGINE_2BIT_EXPERIMENTAL"] = "1"
    cmd = [
        *axquant_cmd(),
        "convert",
        "--model",
        str(SOURCE),
        "--revision",
        SOURCE_REV,
        "--plan",
        str(plan),
        "--output",
        str(pack),
        "--q-mode",
        str(item["q_mode"]),
        "--allow-unmeasured",
        "--ax-engine-manifest",
        "skip",
    ]
    run(cmd, WORK / "logs" / f"convert-{key}.log", extra_env=extra)
    log(f"convert ok {pack}")


def _measured_bpw(pack: Path) -> str | None:
    manifest = pack / "axquant_manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    value = payload.get("measured_total_bpw") or payload.get("effective_bpw")
    return None if value is None else str(value)


def cmd_publish(key: str) -> None:
    require_factory_host(socket.gethostname())
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}")
    item = PACKS[key]
    pack = pack_dir(key)
    if not (pack / "axquant_manifest.json").is_file():
        raise SystemExit(f"missing convert output {pack}")
    license_src = SOURCE / "LICENSE"
    if license_src.is_file() and not (pack / "LICENSE").is_file():
        shutil.copy2(license_src, pack / "LICENSE")
    write_readme(pack, key, measured_bpw=_measured_bpw(pack))
    repo = f"{HUB_OWNER}/{item['hub_name']}"
    marker = WORK / f"published-{key}.txt"
    if marker.is_file():
        log(f"already published {repo} ({marker.read_text(encoding='utf-8').strip()})")
        return
    py = str(venv_python())
    upload = f"""
from huggingface_hub import HfApi
api = HfApi()
api.create_repo({repo!r}, repo_type="model", exist_ok=True, private=False)
api.upload_large_folder(repo_id={repo!r}, folder_path={str(pack)!r}, repo_type="model")
print("uploaded", {repo!r})
"""
    run([py, "-c", upload], WORK / "logs" / f"publish-{key}.log")
    marker.write_text(repo + "\n", encoding="utf-8")
    log(f"published {repo}")


def cmd_remaining() -> None:
    """Convert and publish every pack that lacks a finished manifest.

    Shared PLE ``shard_N`` → ``shards.N`` binding applies to all bit modes.
    """
    cmd_inspect()
    failures: list[str] = []
    for key in PACKS:
        failed = WORK / "logs" / f"failed-{key}.txt"
        try:
            cmd_convert(key)
            cmd_publish(key)
            failed.unlink(missing_ok=True)
        except ConvertInterrupted:
            raise
        except SystemExit as exc:
            failures.append(f"{key}: {exc}")
            log(f"FAILED {key}: {exc}")
            failed.write_text(str(exc) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("packs failed:\n" + "\n".join(failures))
    log("all packs converted and published (development; not certified)")


def cmd_all() -> None:
    cmd_bootstrap()
    cmd_download()
    cmd_remaining()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("bootstrap", "download", "inspect", "convert", "publish", "remaining", "all"),
    )
    parser.add_argument("--pack", choices=tuple(PACKS), default=None)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    if args.stage == "bootstrap":
        cmd_bootstrap()
    elif args.stage == "download":
        cmd_download()
    elif args.stage == "inspect":
        cmd_inspect()
    elif args.stage == "convert":
        if args.pack is None:
            raise SystemExit("convert requires --pack")
        cmd_convert(args.pack)
    elif args.stage == "publish":
        if args.pack is None:
            raise SystemExit("publish requires --pack")
        cmd_publish(args.pack)
    elif args.stage == "remaining":
        cmd_remaining()
    else:
        cmd_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
