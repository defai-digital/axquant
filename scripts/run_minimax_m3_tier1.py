#!/usr/bin/env python3
"""Checkpoint Tier 1 for MiniMax-M3 Hub packs on tn-macstudio-m3.

512 GB M3 Ultra recert host. AX Engine 7.2.x requires macOS 26+. Local
MiniMax-capable binaries live under ``AX_ENGINE_ROOT`` (default
``~/opt/ax-engine-minimax``). Packs are already on the Hub. Quality is
**AX Engine native** (mlx-vlm 0.6.16 cannot load the AXQ expert layout).

  PYTHONPATH=src .venv/bin/python scripts/run_minimax_m3_tier1.py preflight
  PYTHONPATH=src .venv/bin/python scripts/run_minimax_m3_tier1.py --pack axq2 all
  PYTHONPATH=src .venv/bin/python scripts/run_minimax_m3_tier1.py both

Detach on the Studio with ``scripts/run_minimax_m3_tier1_detached.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import (  # noqa: E402
    LARGE_MEMORY_CERT_HOST_ID,
    require_large_memory_cert_host,
)

SOURCE_ID = "MiniMaxAI/MiniMax-M3"
SOURCE_REV = "f0e1c1e04d40177e4673a22097036854f536e9c0"
ARCHITECTURE = "MiniMaxM3SparseForConditionalGeneration"
ADAPTER_ID = "minimax-m3-v1"
ENGINE_RELEASE = os.environ.get("AX_ENGINE_RELEASE", "v7.2.0")
ENGINE_VERSION = ENGINE_RELEASE.lstrip("v")
MIN_MACOS_MAJOR = 26
MIN_QUALITY = 0.90
SEED = 20260728
MAX_TOKENS = 64
HEALTH_TIMEOUT_S = 3600
CHAT_TIMEOUT_S = 1800
HOME = Path.home()
DEFAULT_MODELS = HOME / "models"
DEFAULT_WORK = HOME / "axquant-certification" / "minimax-m3-tier1"
DEFAULT_ENGINE_ROOT = HOME / "opt" / f"ax-engine-{ENGINE_VERSION}"
DEFAULT_DATASETS = HOME / "axquant-certification" / "datasets"

PACKS: dict[str, dict[str, Any]] = {
    "axq2": {
        "hub_name": "AX-MiniMax-M3-MLX-AXQ-2bit",
        "cert_stem": "minimax-m3-axq2-tier1",
        "display_name": "MiniMax-M3 MLX AXQ 2-bit (exp.)",
        "recipe": ROOT / "examples" / "minimax-m3-experimental-2bit-v0.1.yaml",
        "product_class": "2bit-experimental",
        "sort_order": 240,
        "listed": True,
        "hub_commit": "9e682a4c236ea8d6ff37b89cfd065e223f6fbc25",
        "experimental_env": ("AX_ENGINE_2BIT_EXPERIMENTAL=1",),
    },
    "mxfp4": {
        "hub_name": "AX-MiniMax-M3-MLX-AXQ-MXFP4",
        "cert_stem": "minimax-m3-axq-mxfp4-tier1",
        "display_name": "MiniMax-M3 MLX AXQ MXFP4 (exp.)",
        "recipe": ROOT / "examples" / "minimax-m3-experimental-mxfp4-v0.1.yaml",
        "product_class": "MXFP4",
        "sort_order": 241,
        "listed": True,
        "hub_commit": "228f7b75249a92ed1abf43bc5b7b7d8c388c55b3",
        "experimental_env": (),
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def spec(key: str) -> dict[str, Any]:
    if key not in PACKS:
        raise SystemExit(f"unknown pack {key}; choose {sorted(PACKS)}")
    return PACKS[key]


def models_root() -> Path:
    return Path(os.environ.get("MINIMAX_M3_MODELS", DEFAULT_MODELS))


def work_dir() -> Path:
    return Path(os.environ.get("MINIMAX_M3_WORK", DEFAULT_WORK))


def pack_dir(key: str) -> Path:
    override = os.environ.get(f"MINIMAX_M3_{key.upper()}_PACK")
    if override:
        return Path(override)
    return models_root() / str(spec(key)["hub_name"])


def engine_root() -> Path:
    return Path(os.environ.get("AX_ENGINE_ROOT", DEFAULT_ENGINE_ROOT))


def engine_server() -> Path:
    root = engine_root()
    for cand in (root / "bin" / "ax-engine-server", root / "ax-engine-server"):
        if cand.is_file():
            return cand
    nested = list(root.rglob("ax-engine-server"))
    return nested[0] if nested else root / "bin" / "ax-engine-server"


def engine_bench() -> Path:
    root = engine_root()
    for cand in (root / "bin" / "ax-engine-bench", root / "ax-engine-bench"):
        if cand.is_file():
            return cand
    nested = list(root.rglob("ax-engine-bench"))
    return nested[0] if nested else root / "bin" / "ax-engine-bench"


def generate_manifest_bin() -> Path:
    root = engine_root()
    for cand in (root / "bin" / "generate-manifest", root / "generate-manifest"):
        if cand.is_file():
            return cand
    nested = list(root.rglob("generate-manifest"))
    return nested[0] if nested else root / "bin" / "generate-manifest"


def engine_port(key: str) -> int:
    return 8765 if key == "axq2" else 8766


def engine_env() -> dict[str, str]:
    env = hf_env()
    lib_dir = str(engine_server().parent)
    env["PATH"] = f"{lib_dir}:{env.get('PATH', '')}"
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    env["AX_ENGINE_2BIT_EXPERIMENTAL"] = "1"
    # MiniMax Super-class packs mark streaming required. On a 512 GB recert
    # host, keep every MoE layer stack after the first prefill (~3.6 GB each,
    # 57 layers) so decode does not re-read ~200 GB from SSD per token.
    env.setdefault("AX_STREAM_EXPERT_LAYERS", "64")
    return env


def manifest_cmd(pack: Path) -> list[str]:
    bench = engine_bench()
    gen = generate_manifest_bin()
    if bench.is_file():
        return [str(bench), "generate-manifest", "--force", "--validate", str(pack)]
    if gen.is_file():
        return [str(gen), "--force", "--validate", str(pack)]
    raise SystemExit("ax-engine-bench / generate-manifest not found")


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail[:800]}") from exc
    return json.loads(body) if body else {}


def wait_health(base: str, timeout: int = HEALTH_TIMEOUT_S) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (OSError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy: {last}")


def start_engine_server(key: str, pack: Path, log_path: Path) -> subprocess.Popen[str]:
    item = spec(key)
    server = engine_server()
    if not server.is_file():
        raise SystemExit(f"ax-engine-server not found: {server}")
    cmd = [
        str(server),
        "--host",
        "127.0.0.1",
        "--port",
        str(engine_port(key)),
        "--mlx",
        "--support-tier",
        "mlx-preview",
        "--mlx-model-artifacts-dir",
        str(pack),
        "--model-id",
        f"AutomatosX/{item['hub_name']}",
        "--deterministic",
        "--stream-experts",
        "auto",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    log("$ " + " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=engine_env(),
    )


def stop_engine_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def chat_complete(base: str, model_id: str, prompt: str, max_tokens: int, seed: int) -> str:
    payload = {
        "model": model_id,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = http_json("POST", base + "/v1/chat/completions", payload, timeout=CHAT_TIMEOUT_S)
    message = (data.get("choices") or [{}])[0].get("message") or {}
    return str(message.get("content") or "")


def datasets_dir() -> Path:
    return Path(os.environ.get("MINIMAX_M3_DATASETS", DEFAULT_DATASETS))


def axquant_cmd() -> list[str]:
    local = ROOT / ".venv" / "bin" / "axquant"
    if local.is_file():
        return [str(local)]
    which = shutil.which("axquant")
    if which:
        return [which]
    return [sys.executable, "-m", "axquant"]


def macos_major() -> int | None:
    raw = platform.mac_ver()[0]
    if not raw:
        return None
    try:
        return int(raw.split(".", 1)[0])
    except ValueError:
        return None


def hf_env(*, extra: dict[str, str] | None = None) -> dict[str, str]:
    hf_home = os.environ.get("HF_HOME", str(HOME / ".cache" / "huggingface"))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]).strip(
            os.pathsep
        ),
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", str(Path(hf_home) / "hub")),
        "HUGGINGFACE_HUB_CACHE": os.environ.get(
            "HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")
        ),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", str(Path(hf_home) / "xet")),
        "AX_ENGINE_2BIT_EXPERIMENTAL": "1",
        "AX_STREAM_EXPERT_LAYERS": os.environ.get("AX_STREAM_EXPERT_LAYERS", "64"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if extra:
        env.update(extra)
    return env


def run(
    cmd: list[str],
    log_path: Path | None = None,
    *,
    extra_env: dict[str, str] | None = None,
) -> None:
    log("$ " + " ".join(cmd))
    env = hf_env(extra=extra_env)
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


def safetensors_weight_bytes(model_dir: Path) -> int:
    total = 0
    for path in sorted(model_dir.rglob("*.safetensors")):
        if path.name.startswith("."):
            continue
        total += path.stat().st_size
    if total <= 0:
        raise SystemExit(f"no safetensors under {model_dir}")
    return total


def pack_ready(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        and (path / "axquant_manifest.json").is_file()
        and (path / "ax_expert_stream.json").is_file()
        and any(path.glob("model-*.safetensors"))
    )


def cmd_preflight() -> None:
    host = require_large_memory_cert_host(socket.gethostname())
    major = macos_major()
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mlx_ok = False
    mlx_note = "mlx-vlm not importable"
    try:
        from mlx_vlm.models.minimax_m3.minimax_m3 import Model
        from mlx_vlm.models.minimax_m3_vl.language import MiniMaxAttention

        _ = Model
        mlx_ok = hasattr(MiniMaxAttention, "_build_sparse_mask")
        mlx_note = (
            "mlx-vlm MiniMaxAttention MSA present"
            if mlx_ok
            else "mlx-vlm imported MiniMax modules but MiniMaxAttention MSA is incomplete"
        )
    except ImportError as exc:
        mlx_note = f"mlx-vlm MiniMax import failed: {exc}"
    payload = {
        "host_id": host,
        "observed_hostname": socket.gethostname(),
        "macos": platform.mac_ver()[0],
        "macos_major": major,
        "memory_bytes": mem,
        "engine_release": ENGINE_RELEASE,
        "latest_published": "v7.1.5",
        "local_minimax_engine": str(engine_root()),
        "engine_server": str(engine_server()),
        "engine_present": engine_server().is_file(),
        "os_ok": major is not None and major >= MIN_MACOS_MAJOR,
        "mlx_vlm_minimax": mlx_ok,
        "mlx_vlm_note": mlx_note,
    }
    write_json(work_dir() / "preflight.json", payload)
    log(json.dumps(payload, indent=2))
    if major is None or major < MIN_MACOS_MAJOR:
        raise SystemExit(
            f"AX Engine {ENGINE_VERSION} requires macOS {MIN_MACOS_MAJOR}+; "
            f"this host is {platform.mac_ver()[0]}. Upgrade the OS before cert."
        )
    if not payload["engine_present"]:
        raise SystemExit(f"ax-engine-server missing at {engine_server()}")
    if not mlx_ok:
        log(
            "mlx-vlm MiniMax generate is not required; cert quality is AX Engine native. "
            f"{mlx_note}"
        )


def cmd_install_engine() -> None:
    require_large_memory_cert_host(socket.gethostname())
    dest = engine_root()
    server = engine_server()
    if server.is_file():
        log(f"reuse engine {server}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/defai-digital/ax-engine/releases/download/"
        f"{ENGINE_RELEASE}/ax-engine-{ENGINE_RELEASE}-macos-arm64.tar.gz"
    )
    archive = dest.parent / f"ax-engine-{ENGINE_RELEASE}-macos-arm64.tar.gz"
    if not archive.is_file():
        log(f"download {url}")
        urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(path=dest)
    if not engine_server().is_file():
        nested = list(dest.rglob("ax-engine-server"))
        if not nested:
            raise SystemExit(f"ax-engine-server missing after extract in {dest}")
        log(f"engine binary at {nested[0]}")
    log(f"installed {ENGINE_RELEASE} under {dest}")


def cmd_download(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    item = spec(key)
    dest = pack_dir(key)
    if pack_ready(dest):
        log(f"reuse pack {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf = ROOT / ".venv" / "bin" / "hf"
    hf_bin = str(hf) if hf.is_file() else shutil.which("hf") or "hf"
    run(
        [hf_bin, "download", f"AutomatosX/{item['hub_name']}", "--local-dir", str(dest)],
        work_dir() / "logs" / f"download-{key}.log",
    )
    if not pack_ready(dest):
        raise SystemExit(f"Hub pack incomplete: {dest}")


def cmd_size(key: str) -> None:
    pack = pack_dir(key)
    if not pack_ready(pack):
        raise SystemExit(f"missing pack {pack}")
    man_path = pack / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    cand_bytes = int(man.get("weight_file_size_bytes") or 0)
    measured = man.get("measured_main_bpw") or man.get("measured_total_bpw")
    if cand_bytes <= 0:
        cand_bytes = safetensors_weight_bytes(pack)
    payload = {
        "candidate_bytes": cand_bytes,
        "candidate_measured_bpw": measured,
        "pass": cand_bytes > 0,
        "compare_mode": "total",
        "notes": (
            "MiniMax-M3 experimental Super-class track: size bound to measured "
            "bytes (no uniform 2-bit / MXFP4 reference on this host)."
        ),
        "stream_required": (pack / "ax_expert_stream.json").is_file(),
        "mtp_present": bool(man.get("mtp_present")),
    }
    write_json(work_dir() / key / "size.json", payload)
    log(f"size {key}: bytes={cand_bytes} bpw={measured}")


def _quality_score(path: Path) -> float | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("task_results") or []
    if tasks:
        scores = [float(task.get("score") or 0.0) for task in tasks]
        if scores:
            return sum(scores) / len(scores)
    metrics = payload.get("metrics") or {}
    task_scores = metrics.get("task_scores") or {}
    if task_scores:
        values = [float(value) for value in task_scores.values()]
        return sum(values) / len(values)
    for key in ("score", "viability", "aggregate_score"):
        if payload.get(key) is not None:
            return float(payload[key])
    return None


def _dataset_path(suite: str) -> Path:
    datasets = datasets_dir()
    names = (
        ("development-agent-coding", "coding")
        if suite == "agent-coding"
        else ("development-general", "instruction")
    )
    for name in names:
        for cand in (
            datasets / name / "dataset.jsonl",
            datasets / "datasets" / name / "dataset.jsonl",
            ROOT / "data" / "eval" / f"{name}.jsonl",
        ):
            if cand.is_file():
                return cand
    raise SystemExit(f"missing {suite} dataset under {datasets} or data/eval")


def cmd_generate_manifest(key: str) -> None:
    pack = pack_dir(key)
    cmd = manifest_cmd(pack)
    log_path = work_dir() / "logs" / f"generate-manifest-{key}.log"
    log("$ " + " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=engine_env(),
        )
    log(f"generate-manifest {key} exit={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(f"generate-manifest failed: see {log_path}")


def cmd_quality(key: str) -> None:
    """AX Engine native 15+15 generation viability. Does not use mlx-vlm."""

    require_large_memory_cert_host(socket.gethostname())
    from axquant.quality import load_quality_tasks, score_quality_task_output
    from axquant.serde import file_sha256

    pack = pack_dir(key)
    item = spec(key)
    model_id = f"AutomatosX/{item['hub_name']}"
    qdir = work_dir() / key / "quality"
    rdir = work_dir() / key / "runtime"
    qdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    force = os.environ.get("MINIMAX_M3_FORCE_QUALITY", "").strip() in {
        "1",
        "true",
        "yes",
    }
    reusable = (not force) and all(
        (qdir / f"{suite}.json").is_file() and (_quality_score(qdir / f"{suite}.json") or 0.0) > 0.0
        for suite in ("agent-coding", "general")
    )
    if reusable:
        for suite in ("agent-coding", "general"):
            log(f"quality {key} {suite}: reuse score={_quality_score(qdir / f'{suite}.json')}")
        return
    try:
        cmd_generate_manifest(key)
    except SystemExit:
        log_path = work_dir() / "logs" / f"generate-manifest-{key}.log"
        reason = (
            log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            if log_path.is_file()
            else "generate-manifest failed"
        )
        log(f"AX Engine cannot ingest {key}: {reason.strip()[:400]}")
        write_json(
            rdir / "ax-engine-smoke.json",
            {
                "passed": False,
                "error": reason.strip()[:1500],
                "runtime": "ax-engine",
            },
        )
        for suite in ("agent-coding", "general"):
            write_json(
                qdir / f"{suite}.json",
                {
                    "samples": 0,
                    "task_results": [],
                    "metrics": {"task_scores": {}},
                    "scoring": "experimental-generation-viability",
                    "runtime": "ax-engine-native",
                    "mean_score": None,
                    "pass": False,
                    "reason": reason.strip()[:1500],
                },
            )
        return
    log_path = work_dir() / "logs" / f"ax-engine-server-{key}.log"
    proc = start_engine_server(key, pack, log_path)
    base = f"http://127.0.0.1:{engine_port(key)}"
    try:
        log(f"waiting for engine health on {base} (up to {HEALTH_TIMEOUT_S}s)")
        wait_health(base)
        log(f"engine healthy; smoke chat {key}")
        smoke_text = chat_complete(base, model_id, "Say OK.", 8, SEED)
        write_json(
            rdir / "ax-engine-smoke.json",
            {"passed": bool(smoke_text.strip()), "text": smoke_text[:500], "runtime": "ax-engine"},
        )
        log(f"engine smoke {key}={smoke_text!r}")
        if not smoke_text.strip():
            raise SystemExit("AX Engine smoke returned empty text")
        for suite in ("agent-coding", "general"):
            out = qdir / f"{suite}.json"
            if out.is_file():
                log(f"quality {key} {suite}: reuse score={_quality_score(out)}")
                continue
            ds = _dataset_path(suite)
            tasks = list(load_quality_tasks(ds))
            results = []
            scores: list[float] = []
            for index, task in enumerate(tasks):
                err = None
                try:
                    text = chat_complete(base, model_id, task.prompt, MAX_TOKENS, SEED + index)
                except Exception as exc:
                    text, err = "", str(exc)
                score, checks = score_quality_task_output(task, text)
                scores.append(score)
                results.append(
                    {
                        "task_id": task.task_id,
                        "category": task.category,
                        "score": score,
                        "check_scores": checks,
                        "output": text[:2000],
                        "error": err,
                    }
                )
                log(f"quality {key} {suite} {index + 1}/{len(tasks)} {task.task_id} score={score}")
            mean = sum(scores) / len(scores) if scores else 0.0
            payload = {
                "samples": len(results),
                "dataset_sha256": file_sha256(ds),
                "task_results": results,
                "metrics": {
                    "task_scores": {
                        suite: mean,
                    }
                },
                "scoring": "experimental-generation-viability",
                "runtime": "ax-engine-native",
                "mean_score": mean,
            }
            write_json(out, payload)
            log(f"quality {key} {suite}: score={mean} (need >= {MIN_QUALITY})")
    except Exception:
        tail = ""
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        log(f"engine quality failed; server log tail:\n{tail}")
        raise
    finally:
        stop_engine_server(proc)


def cmd_runtime(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    pack = pack_dir(key)
    rdir = work_dir() / key / "runtime"
    rdir.mkdir(parents=True, exist_ok=True)
    smoke = rdir / "mlx-vlm-smoke.json"
    if not smoke.is_file():
        write_json(
            smoke,
            {
                "passed": False,
                "error": (
                    "skipped: mlx-vlm 0.6.16 cannot load this AXQ pack "
                    "(855 unexpected shared_experts/switch_mlp parameters). "
                    "Primary cert path is AX Engine 7.1.5 native."
                ),
            },
        )
    bench = engine_bench()
    doctor = rdir / "ax-engine-doctor.json"
    if bench.is_file() and not doctor.is_file():
        with doctor.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [str(bench), "doctor", "--mlx-model-artifacts-dir", str(pack), "--json"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=engine_env(),
            )
    if not (work_dir() / "logs" / f"generate-manifest-{key}.log").is_file():
        try:
            cmd_generate_manifest(key)
        except SystemExit as exc:
            log(f"generate-manifest {key} skipped/failed: {exc}")
    log(f"runtime {key} wrote {rdir}")


def cmd_write_certs(key: str) -> None:
    require_large_memory_cert_host(socket.gethostname())
    from axquant.modality_certification import (
        build_modalities_block,
        format_modalities_card_section,
        inspect_artifact_modalities,
        modalities_to_public_dict,
    )
    from axquant.schema.public_certification import load_public_checkpoint_certification

    item = spec(key)
    pack = pack_dir(key)
    work = work_dir() / key
    size = json.loads((work / "size.json").read_text(encoding="utf-8"))
    quality: dict[str, Any] = {}
    certified = bool(size.get("pass"))
    for suite in ("agent-coding", "general"):
        qpath = work / "quality" / f"{suite}.json"
        if not qpath.is_file():
            certified = False
            quality[suite] = {"pass": False, "reason": f"missing {qpath}"}
            continue
        payload = json.loads(qpath.read_text(encoding="utf-8"))
        score = _quality_score(qpath)
        if score is None or score < MIN_QUALITY:
            certified = False
        quality[suite] = {
            "candidate_score": score,
            "samples": int(payload.get("samples") or 15),
            "dataset_sha256": payload.get("dataset_sha256"),
            "scoring": "experimental-generation-viability",
            "pass": score is not None and score >= MIN_QUALITY,
        }
    mlx_path = work / "runtime" / "mlx-vlm-smoke.json"
    mlx_ok = False
    mlx_note = (
        "mlx-vlm 0.6.16 cannot load this AXQ expert layout (shared_experts/"
        "switch_mlp parameter mismatch); not a cert gate. Primary runtime is AX Engine."
    )
    if mlx_path.is_file():
        mlx = json.loads(mlx_path.read_text(encoding="utf-8"))
        mlx_ok = bool(mlx.get("passed"))
        mlx_note = mlx.get("error") or mlx.get("text") or mlx_note
    smoke_path = work / "runtime" / "ax-engine-smoke.json"
    engine_smoke = False
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        engine_smoke = bool(smoke.get("passed"))
        if not engine_smoke:
            certified = False
    else:
        certified = False
    doctor_path = work / "runtime" / "ax-engine-doctor.json"
    engine_ok = doctor_path.is_file() and engine_smoke
    man_path = pack / "axquant_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    inspect = inspect_artifact_modalities(pack)
    block = build_modalities_block(
        vision_supported=inspect.vision_supported,
        audio_supported=inspect.audio_supported,
        vision_smoke_passed=None,
        vision_reason=(
            "Vision tower is BF16-protected; language-path cert does not claim image/video quality."
            if inspect.vision_supported
            else None
        ),
    )
    hub_commit = os.environ.get(f"MINIMAX_M3_{key.upper()}_HUB_COMMIT", str(item["hub_commit"]))
    payload = {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": "certified" if certified else "not_certified",
        "certification_tier": "checkpoint",
        "certified_at" if certified else "evaluated_at": datetime.now(UTC).isoformat(),
        "host_id": LARGE_MEMORY_CERT_HOST_ID,
        "artifact": {
            "hub_repo_id": f"AutomatosX/{item['hub_name']}",
            "hub_commit": hub_commit,
            "product_class": item["product_class"],
            "architecture": ARCHITECTURE,
            "source_model_id": SOURCE_ID,
            "source_revision": SOURCE_REV,
            "candidate_manifest_sha256": sha256_file(man_path) if man_path.is_file() else None,
        },
        "plan": {
            "evidence_kind": "architecture_prior",
            "plan_source": "existing-pack",
            "recipe": str(Path(item["recipe"]).relative_to(ROOT)),
            "target_class": item["product_class"],
            "adapter_id": ADAPTER_ID,
            "measured_total_bpw": man.get("measured_total_bpw")
            or size.get("candidate_measured_bpw"),
            "measured_main_bpw": man.get("measured_main_bpw") or size.get("candidate_measured_bpw"),
        },
        "size": {
            "candidate_weight_bytes": size["candidate_bytes"],
            "candidate_measured_bpw": size.get("candidate_measured_bpw"),
            "pass": bool(size.get("pass")),
            "notes": size.get("notes"),
        },
        "quality": quality,
        "thresholds": {
            "minimum_generation_viability": MIN_QUALITY,
            "notes": (
                "Experimental Super-class track uses generation viability on the "
                "15+15 development suites, not BF16 retention. Seed 20260728, "
                "max gen 64."
            ),
        },
        "mtp_acceleration": {
            "status": "not-applicable",
            "reason": (
                "Config num_mtp_modules is not packaged MTP; Hub leaf has no "
                "-MTP and no mtp.safetensors."
            ),
        },
        "runtime": {
            "mlx_vlm": {
                "status": "pass" if mlx_ok else "fail",
                "notes": f"{mlx_note} on {LARGE_MEMORY_CERT_HOST_ID}",
            },
            "ax_engine": {
                "status": "pass" if engine_ok else "fail",
                "version": ENGINE_VERSION,
                "env_required": list(item["experimental_env"]),
                "notes": (
                    f"native chat smoke + doctor / generate-manifest on {LARGE_MEMORY_CERT_HOST_ID}"
                ),
            },
        },
        "toolchain": {
            "axquant": man.get("axquant_version", "1.9.0"),
            "ax_engine": ENGINE_VERSION,
            "host": LARGE_MEMORY_CERT_HOST_ID,
            "host_hardware": "Apple M3 Ultra 512 GB",
        },
        "notes": [
            (f"Checkpoint attempt on {LARGE_MEMORY_CERT_HOST_ID} with AX Engine {ENGINE_VERSION}."),
            "Language path only. Vision stays BF16; vision generate is not claimed.",
            (
                "ax_expert_stream.json is required. Engine Auto may keep the ~220 GB pack "
                "resident on 512 GB."
            ),
            "No AXQ 4-bit affine sibling.",
        ],
        "public_index": {
            "display_name": item["display_name"],
            "sort_order": item["sort_order"],
            "edition_label": f"m3@{ENGINE_VERSION}",
            "listed": bool(item["listed"]),
        },
        "modalities": modalities_to_public_dict(block),
    }
    if payload["artifact"]["candidate_manifest_sha256"] is None:
        del payload["artifact"]["candidate_manifest_sha256"]
    cert_json = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.json"
    write_json(cert_json, payload)
    load_public_checkpoint_certification(cert_json)
    verdict = "certified" if certified else "**not certified**"
    agent = quality.get("agent-coding", {})
    general = quality.get("general", {})
    md = ROOT / "docs" / "certifications" / f"{item['cert_stem']}.md"
    md.write_text(
        "\n".join(
            [
                f"# {item['display_name']} — checkpoint Tier 1",
                "",
                f"**Verdict:** {verdict} on `{LARGE_MEMORY_CERT_HOST_ID}` "
                f"with AX Engine `{ENGINE_VERSION}`.",
                "",
                f"This certificate covers [`AutomatosX/{item['hub_name']}`]"
                f"(https://huggingface.co/AutomatosX/{item['hub_name']}) "
                f"commit [`{hub_commit}`](https://huggingface.co/AutomatosX/"
                f"{item['hub_name']}/tree/{hub_commit}).",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Hub | [`AutomatosX/{item['hub_name']}`]"
                f"(https://huggingface.co/AutomatosX/{item['hub_name']}) |",
                f"| Source | `{SOURCE_ID}@{SOURCE_REV}` |",
                f"| Host | `{LARGE_MEMORY_CERT_HOST_ID}` |",
                f"| Product class | `{item['product_class']}` |",
                f"| Architecture | `{ARCHITECTURE}` |",
                f"| Measured main BPW | `{payload['plan']['measured_main_bpw']}` |",
                f"| Weight bytes | `{size['candidate_bytes']}` |",
                f"| Agent-coding viability | `{agent.get('candidate_score')}` "
                f"(need ≥ {MIN_QUALITY}) |",
                f"| General viability | `{general.get('candidate_score')}` "
                f"(need ≥ {MIN_QUALITY}) |",
                "| MTP acceleration | `not-applicable` (no packaged MTP) |",
                "| Stream | `ax_expert_stream.json` required |",
                "",
                "## Notes",
                "",
                "- Experimental Super-class track: generation viability, not BF16 retention.",
                "- Language path only. Vision is present/protected and **not** certified.",
                f"- Seed `{SEED}`, max gen {MAX_TOKENS}, AX Engine `{ENGINE_VERSION}`.",
                "",
                "## Related",
                "",
                "- Sibling 2-bit: [minimax-m3-axq2-tier1.md](minimax-m3-axq2-tier1.md)"
                if key == "mxfp4"
                else (
                    "- Sibling MXFP4: "
                    "[minimax-m3-axq-mxfp4-tier1.md](minimax-m3-axq-mxfp4-tier1.md)"
                ),
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


def cmd_all(key: str) -> None:
    cmd_preflight()
    cmd_install_engine()
    cmd_download(key)
    cmd_size(key)
    try:
        cmd_quality(key)
    except Exception as exc:
        log(f"quality {key} did not complete: {exc}")
        qdir = work_dir() / key / "quality"
        rdir = work_dir() / key / "runtime"
        qdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)
        if not (rdir / "ax-engine-smoke.json").is_file():
            write_json(
                rdir / "ax-engine-smoke.json",
                {"passed": False, "error": str(exc), "runtime": "ax-engine"},
            )
        for suite in ("agent-coding", "general"):
            if not (qdir / f"{suite}.json").is_file():
                write_json(
                    qdir / f"{suite}.json",
                    {
                        "samples": 0,
                        "task_results": [],
                        "pass": False,
                        "reason": str(exc),
                        "runtime": "ax-engine-native",
                    },
                )
    cmd_runtime(key)
    cmd_write_certs(key)


def cmd_both() -> None:
    for key in ("axq2", "mxfp4"):
        log(f"===== pack {key} =====")
        cmd_all(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", choices=sorted(PACKS), default="axq2")
    parser.add_argument(
        "step",
        choices=[
            "preflight",
            "install-engine",
            "download",
            "size",
            "quality",
            "runtime",
            "write-certs",
            "all",
            "both",
        ],
    )
    args = parser.parse_args()
    if args.step == "preflight":
        cmd_preflight()
    elif args.step == "install-engine":
        cmd_install_engine()
    elif args.step == "download":
        cmd_download(args.pack)
    elif args.step == "size":
        cmd_size(args.pack)
    elif args.step == "quality":
        cmd_quality(args.pack)
    elif args.step == "runtime":
        cmd_runtime(args.pack)
    elif args.step == "write-certs":
        cmd_write_certs(args.pack)
    elif args.step == "both":
        cmd_both()
    else:
        cmd_all(args.pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
