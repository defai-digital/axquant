#!/usr/bin/env python3
"""Native AX Engine QA + decode for Flash-0731 AXQ 2-bit.

Factory host only (df-macstudio-m2). Does **not** delegate to mlx-lm.

  PYTHONPATH=src python scripts/run_deepseek_v4_0731_axq2_axengine.py manifest
  PYTHONPATH=src python scripts/run_deepseek_v4_0731_axq2_axengine.py eval
  PYTHONPATH=src python scripts/run_deepseek_v4_0731_axq2_axengine.py all

7.1.x: set AX_ENGINE_SERVER / AX_ENGINE_GENERATE_MANIFEST to the 7.1.5
binaries (generate-manifest is `ax-engine-bench generate-manifest`). Stock
7.0.2 generate-manifest writes the fused packed role and fails validation;
use generate-manifest-split for that release.

QA protocol (env ``DSV4_QA_PROTOCOL``): default ``v-extract`` (fenced-Python
scoring, coding 256 / general 64, stop sequences). ``v256-strict`` is the
single-budget 256-token run; ``v64`` is the original 64-token user-only
measurement. Work dir includes the protocol so evidence is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from axquant.deepseek_v4_qa import (  # noqa: E402
    DEFAULT_PROTOCOL,
    build_qa_messages,
    normalize_qa_protocol,
    qa_protocol_record,
    qa_suite_config,
)

FACTORY_HOST_ID = "df-macstudio-m2"
FACTORY_DATASETS = "/Volumes/Ext12T/axquant-certification/datasets"
PACK = Path(
    os.environ.get(
        "DSV4_AXQ2",
        "/Volumes/Ext12T/models/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-v1.9.0",
    )
)
ENGINE_BIN = Path(
    os.environ.get(
        "AX_ENGINE_SERVER",
        "/Users/devop/opt/ax-engine-7.0.2/ax-engine-server",
    )
)
GENERATE_MANIFEST_BIN = Path(
    os.environ.get(
        "AX_ENGINE_GENERATE_MANIFEST",
        "/Users/devop/opt/ax-engine-7.0.2/generate-manifest-split",
    )
)
MODEL_ID = os.environ.get(
    "DSV4_AXQ2_MODEL_ID",
    "AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-MTP",
)
ENGINE_VERSION = os.environ.get("AX_ENGINE_VERSION", "7.0.2")
SOURCE_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
SOURCE_REV = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
SEED = 20260728
QA_PROTOCOL = normalize_qa_protocol(os.environ.get("DSV4_QA_PROTOCOL", DEFAULT_PROTOCOL))
_CODING_OVERRIDE = os.environ.get("DSV4_MAX_TOKENS_QA")
_GENERAL_OVERRIDE = os.environ.get("DSV4_MAX_TOKENS_QA_GENERAL")
MAX_TOKENS_DECODE = 128
PORT = int(os.environ.get("DSV4_AXENGINE_PORT", "8765"))


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def work_dir() -> Path:
    return Path(
        os.environ.get(
            "DSV4_AXENGINE_WORK",
            "/Volumes/Ext12T/axquant-certification/"
            f"deepseek-v4-0731-axq2-axengine-{ENGINE_VERSION}-{QA_PROTOCOL}",
        )
    )


def _manifest_cmd() -> list[str]:
    name = GENERATE_MANIFEST_BIN.name
    if name in {"ax-engine-bench", "ax-engine"}:
        return [
            str(GENERATE_MANIFEST_BIN),
            "generate-manifest",
            "--force",
            "--validate",
            str(PACK),
        ]
    return [str(GENERATE_MANIFEST_BIN), "--force", "--validate", str(PACK)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _require_factory_host() -> None:
    host = socket.gethostname().split(".", 1)[0]
    if host not in {"df-macstudio-m2", "devopsmacstudio"}:
        raise SystemExit(f"factory eval must run on df-macstudio-m2; observed {host}")


def _quality_lib():
    sys.path.insert(0, str(ROOT / "src"))
    from axquant.quality import load_quality_tasks, score_quality_task_output

    return load_quality_tasks, score_quality_task_output


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


def wait_health(base: str, timeout: int = 1800) -> None:
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


def engine_env() -> dict[str, str]:
    env = os.environ.copy()
    env["AX_ENGINE_2BIT_EXPERIMENTAL"] = "1"
    env["PATH"] = f"{ENGINE_BIN.parent}:{env.get('PATH', '')}"
    lib_dir = str(ENGINE_BIN.parent)
    existing = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return env


def cmd_manifest() -> None:
    _require_factory_host()
    if not GENERATE_MANIFEST_BIN.is_file():
        raise SystemExit(
            f"patched generate-manifest missing: {GENERATE_MANIFEST_BIN} "
            "(set AX_ENGINE_GENERATE_MANIFEST)"
        )
    if not (PACK / "config.json").is_file():
        raise SystemExit(f"missing pack {PACK}")
    cmd = _manifest_cmd()
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=engine_env())
    manifest = json.loads((PACK / "model-manifest.json").read_text(encoding="utf-8"))
    packed = up = gate = 0
    for tensor in manifest.get("tensors", []):
        role = tensor.get("role")
        if role == "ffn_gate_up_exps_packed":
            packed += 1
        elif role == "ffn_up_exps":
            up += 1
        elif role == "ffn_gate_exps":
            gate += 1
    log(f"manifest roles packed={packed} gate_exps={gate} up_exps={up}")
    if packed or gate == 0 or up == 0 or gate != up:
        raise SystemExit(f"split remapping failed: packed={packed} gate_exps={gate} up_exps={up}")


def start_server(log_path: Path) -> subprocess.Popen[str]:
    if not ENGINE_BIN.is_file():
        raise SystemExit(f"ax-engine-server not found: {ENGINE_BIN}")
    cmd = [
        str(ENGINE_BIN),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--mlx",
        "--support-tier",
        "mlx-preview",
        "--mlx-model-artifacts-dir",
        str(PACK),
        "--model-id",
        MODEL_ID,
        "--deterministic",
        "--stream-experts",
        "off",
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


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def chat_complete(
    base: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    *,
    messages: list[dict[str, str]] | None = None,
    stop: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }
    if stop:
        payload["stop"] = list(stop)
    started = time.perf_counter()
    data = http_json("POST", base + "/v1/chat/completions", payload, timeout=900)
    elapsed = time.perf_counter() - started
    message = (data.get("choices") or [{}])[0].get("message") or {}
    text = message.get("content") or ""
    usage = data.get("usage") or {}
    return {
        "text": text,
        "usage": usage,
        "elapsed": elapsed,
        "raw": data,
    }


def _run_quality(base: str) -> dict[str, Any]:
    load_quality_tasks, score_quality_task_output = _quality_lib()
    datasets = Path(os.environ.get("DSV4_DATASETS", FACTORY_DATASETS))
    suites = {
        "agent-coding": datasets / "development-agent-coding" / "dataset.jsonl",
        "general": datasets / "development-general" / "dataset.jsonl",
    }
    out: dict[str, Any] = {}
    for suite, path in suites.items():
        if not path.is_file():
            raise SystemExit(f"missing dataset {path}")
        tasks = list(load_quality_tasks(path))
        override_raw = _CODING_OVERRIDE if suite == "agent-coding" else _GENERAL_OVERRIDE
        suite_cfg = qa_suite_config(
            suite,
            QA_PROTOCOL,
            max_tokens_override=int(override_raw) if override_raw else None,
        )
        results = []
        scores: list[float] = []
        t0 = time.perf_counter()
        log(
            f"qa {suite} protocol={QA_PROTOCOL} max_tokens={suite_cfg.max_tokens} "
            f"stop={list(suite_cfg.stop)}"
        )
        for index, task in enumerate(tasks):
            try:
                reply = chat_complete(
                    base,
                    task.prompt,
                    suite_cfg.max_tokens,
                    SEED + index,
                    messages=build_qa_messages(suite, task.prompt, QA_PROTOCOL),
                    stop=suite_cfg.stop,
                )
                text, err = reply["text"], None
            except Exception as exc:
                text, err = "", str(exc)
            sc, checks = score_quality_task_output(task, text)
            scores.append(sc)
            results.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "score": sc,
                    "check_scores": checks,
                    "output": text[:2000],
                    "error": err,
                }
            )
            log(f"axengine {suite} {index + 1}/{len(tasks)} {task.task_id} score={sc}")
        passed = sum(1 for item in results if item["score"] == 1.0)
        out[suite] = {
            "n": len(results),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "pass_rate": passed / len(results) if results else None,
            "passed": passed,
            "seconds": time.perf_counter() - t0,
            "tasks": results,
        }
    return out


def _run_speed(base: str) -> dict[str, Any]:
    decode_prompt = "Continue listing incrementing integers, space-separated, starting at 1.\n"
    prefill_block = ("alpha bravo charlie delta echo foxtrot golf hotel " * 80).strip()
    cases = [
        ("decode-128", decode_prompt, MAX_TOKENS_DECODE),
        ("prefill-512-decode-8", prefill_block[:2000], 8),
        ("prefill-2k-decode-8", (prefill_block + " ") * 4, 8),
    ]
    chat_complete(base, "Say OK.", 8, SEED)
    rows = []
    for name, prompt, max_tokens in cases:
        reply = chat_complete(base, prompt, max_tokens, SEED)
        usage = reply["usage"]
        gen_tokens = int(usage.get("completion_tokens") or 0) or max(
            len((reply["text"] or "").split()), 1
        )
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        elapsed = float(reply["elapsed"])
        rows.append(
            {
                "case": name,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "elapsed_seconds": elapsed,
                "tok_per_s": gen_tokens / elapsed if elapsed > 0 else None,
                "preview": (reply["text"] or "")[:240],
            }
        )
        log(f"speed {name}: {rows[-1]['tok_per_s']:.3f} tok/s in {elapsed:.2f}s")
    return {"cases": rows}


def cmd_speed_only() -> None:
    _require_factory_host()
    if not (PACK / "model-manifest.json").is_file():
        raise SystemExit(f"missing {PACK / 'model-manifest.json'}; run manifest first")
    work = work_dir()
    out = work / "axq2-axengine-speed.json"
    log_path = work / "ax-engine-server.log"
    proc = start_server(log_path)
    base = f"http://127.0.0.1:{PORT}"
    t_load = time.perf_counter()
    try:
        wait_health(base, timeout=1800)
        load_s = time.perf_counter() - t_load
        log(f"server healthy in {load_s:.1f}s")
        smoke = chat_complete(base, "Say OK.", 8, SEED)
        log(f"smoke={smoke['text']!r}")
        if not (smoke["text"] or "").strip():
            raise SystemExit("native AX Engine smoke returned empty text")
        payload = {
            "key": "axq2-axengine-speed",
            "runtime": "ax-engine-native",
            "engine_bin": str(ENGINE_BIN),
            "path": str(PACK),
            "host_id": FACTORY_HOST_ID,
            "load_seconds": load_s,
            "speed": _run_speed(base),
            "measured_at": datetime.now(UTC).isoformat(),
            "not_delegated": True,
        }
        write_json(out, payload)
        log(f"wrote {out}")
    except Exception:
        tail = ""
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        log(f"speed failed; server log tail:\n{tail}")
        raise
    finally:
        stop_server(proc)


def cmd_eval() -> None:
    _require_factory_host()
    if not (PACK / "model-manifest.json").is_file():
        raise SystemExit(f"missing {PACK / 'model-manifest.json'}; run manifest first")
    work = work_dir()
    out = work / "axq2-axengine.json"
    if out.is_file() and os.environ.get("DSV4_FORCE_EVAL") != "1":
        log(f"reuse {out}")
        return
    log_path = work / "ax-engine-server.log"
    proc = start_server(log_path)
    base = f"http://127.0.0.1:{PORT}"
    t_load = time.perf_counter()
    try:
        wait_health(base, timeout=1800)
        load_s = time.perf_counter() - t_load
        log(f"server healthy in {load_s:.1f}s; log={log_path}")
        smoke = chat_complete(base, "Say OK.", 8, SEED)
        log(f"smoke={smoke['text']!r}")
        if not (smoke["text"] or "").strip():
            raise SystemExit("native AX Engine smoke returned empty text")
        quality = _run_quality(base)
        payload = {
            "key": "axq2-axengine",
            "label": f"DeepSeek V4 Flash-0731 AXQ 2-bit (AX Engine {ENGINE_VERSION} native)",
            "hub": MODEL_ID,
            "commit": os.environ.get("DSV4_AXQ2_HUB_COMMIT", ENGINE_VERSION),
            "runtime": f"ax-engine-{ENGINE_VERSION}-native",
            "engine_bin": str(ENGINE_BIN),
            "path": str(PACK),
            "host_id": FACTORY_HOST_ID,
            "source": f"{SOURCE_ID}@{SOURCE_REV}",
            "seed": SEED,
            "qa_protocol": qa_protocol_record(QA_PROTOCOL),
            "max_tokens_qa": qa_suite_config("agent-coding", QA_PROTOCOL).max_tokens,
            "load_seconds": load_s,
            "quality": quality,
            "speed": None,
            "measured_at": datetime.now(UTC).isoformat(),
            "not_delegated": True,
        }
        write_json(out, payload)
        try:
            payload["speed"] = _run_speed(base)
        except Exception as exc:
            payload["speed"] = {"error": f"{type(exc).__name__}: {exc}", "cases": []}
            log(f"speed failed: {exc}")
        payload["measured_at"] = datetime.now(UTC).isoformat()
        write_json(out, payload)
        log(f"wrote {out}")
    except Exception:
        tail = ""
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        log(f"eval failed; server log tail:\n{tail}")
        raise
    finally:
        stop_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("manifest", "eval", "speed", "all"))
    args = parser.parse_args()
    if args.cmd in {"manifest", "all"}:
        cmd_manifest()
    if args.cmd in {"eval", "all"}:
        cmd_eval()
    if args.cmd == "speed":
        cmd_speed_only()


if __name__ == "__main__":
    main()
